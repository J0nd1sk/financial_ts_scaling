#!/usr/bin/env python3
"""
Test Set Evaluation for a20 vs a50 Comparison.

Evaluates trained models on the held-out test set (2025+) to verify
whether validation-set patterns hold on unseen data.

This is critical for determining if observed differences are:
- Real, generalizable effects, OR
- Validation set overfitting artifacts

Usage:
    python experiments/phase6c/test_set_evaluation.py

Output:
    outputs/phase6c/test_set_evaluation.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import torch

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.patchtst import PatchTST, PatchTSTConfig
from src.data.dataset import FinancialDataset, SimpleSplitter

# ============================================================================
# CONFIGURATION
# ============================================================================

BUDGETS = ["2M", "20M", "200M"]
HORIZONS = [1, 2, 3, 5]

# Architecture configs for each budget
ARCHITECTURES = {
    "2M": {"d_model": 64, "n_layers": 4, "n_heads": 4, "d_ff": 256},
    "20M": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff": 512},
    "200M": {"d_model": 256, "n_layers": 8, "n_heads": 16, "d_ff": 1024},
}

# Paths
A20_CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "phase6a_final"
A50_CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "phase6c"

A20_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "v1" / "SPY_dataset_a20.parquet"
A50_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "v1" / "SPY_dataset_a50_combined.parquet"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "phase6c" / "test_set_evaluation.json"


# ============================================================================
# MODEL LOADING
# ============================================================================


def get_a20_checkpoint_path(budget: str, horizon: int) -> Path:
    """Get path to a20 (Phase 6A) checkpoint."""
    return A20_CHECKPOINT_DIR / f"phase6a_{budget.lower()}_h{horizon}" / "best_checkpoint.pt"


def get_a50_checkpoint_path(budget: str, horizon: int) -> Path:
    """Get path to a50 (Phase 6C) checkpoint."""
    if horizon == 1:
        idx = {"2M": "01", "20M": "02", "200M": "03"}[budget]
        return A50_CHECKPOINT_DIR / f"s1_{idx}_{budget.lower()}_h1_a50" / "best_checkpoint.pt"
    else:
        return A50_CHECKPOINT_DIR / f"s2_horizon_{budget.lower()}_h{horizon}_a50" / "best_checkpoint.pt"


def get_num_features_from_checkpoint(checkpoint_path: Path) -> int:
    """Infer num_features from checkpoint revin.gamma shape."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint and "revin.gamma" in checkpoint["model_state_dict"]:
        return checkpoint["model_state_dict"]["revin.gamma"].shape[0]
    raise ValueError(f"Cannot infer num_features from checkpoint: {checkpoint_path}")


def load_model(checkpoint_path: Path, budget: str, num_features: int = None, use_revin: bool = True) -> PatchTST:
    """Load a trained PatchTST model from checkpoint.

    If num_features is None, infers from checkpoint.
    """
    # Infer num_features if not provided
    if num_features is None:
        num_features = get_num_features_from_checkpoint(checkpoint_path)

    arch = ARCHITECTURES[budget]
    config = PatchTSTConfig(
        num_features=num_features,
        context_length=80,
        patch_length=16,
        stride=8,
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=0.5,
        head_dropout=0.0,
    )

    model = PatchTST(config, use_revin=use_revin)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def evaluate_model(
    model: PatchTST,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> dict:
    """Evaluate model with full metrics."""
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    binary_preds = (preds >= 0.5).astype(int)

    # Handle edge cases
    try:
        auc = float(roc_auc_score(labels, preds))
    except ValueError:
        auc = None

    return {
        "auc": auc,
        "accuracy": float(accuracy_score(labels, binary_preds)),
        "precision": float(precision_score(labels, binary_preds, zero_division=0)),
        "recall": float(recall_score(labels, binary_preds, zero_division=0)),
        "f1": float(f1_score(labels, binary_preds, zero_division=0)),
        "pred_min": float(preds.min()) if len(preds) > 0 else None,
        "pred_max": float(preds.max()) if len(preds) > 0 else None,
        "pred_mean": float(preds.mean()) if len(preds) > 0 else None,
        "pred_std": float(preds.std()) if len(preds) > 0 else None,
        "n_positive_preds": int((preds >= 0.5).sum()),
        "n_samples": len(labels),
        "class_balance": float(labels.mean()) if len(labels) > 0 else None,
        "n_actual_positives": int(labels.sum()),
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================


def run_test_evaluation(device: str = "mps") -> dict:
    """
    Run evaluation on test set (2025+) for all models.

    Returns:
        dict with evaluation results for both tiers
    """
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION (2025+ holdout)")
    print("=" * 70)

    results = {
        "a20": {},
        "a50": {},
        "comparison": [],
    }

    # Load datasets
    print("\nLoading datasets...")
    df_a20 = pd.read_parquet(A20_DATA_PATH)
    df_a50 = pd.read_parquet(A50_DATA_PATH)

    # Feature columns = all numeric columns except Date
    # Note: OHLCV are INCLUDED as features (this matches how models were trained)
    a20_feature_cols = [c for c in df_a20.columns if c != "Date"]
    a50_feature_cols = [c for c in df_a50.columns if c != "Date"]

    # Filter to numeric only
    a20_feature_cols = [c for c in a20_feature_cols if df_a20[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    a50_feature_cols = [c for c in a50_feature_cols if df_a50[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    num_features_a20 = len(a20_feature_cols)
    num_features_a50 = len(a50_feature_cols)

    print(f"  a20: {len(df_a20)} rows, {num_features_a20} features (OHLCV + indicators)")
    print(f"  a50: {len(df_a50)} rows, {num_features_a50} features (OHLCV + indicators)")

    # Check test set date range
    a20_dates = pd.to_datetime(df_a20["Date"])
    test_mask = a20_dates >= "2025-01-01"
    print(f"  Test set range: {a20_dates[test_mask].min()} to {a20_dates[test_mask].max()}")
    print(f"  Test set rows: {test_mask.sum()}")

    for horizon in HORIZONS:
        for budget in BUDGETS:
            print(f"\n--- {budget} / H{horizon} ---")

            # Get checkpoint paths
            a20_checkpoint = get_a20_checkpoint_path(budget, horizon)
            a50_checkpoint = get_a50_checkpoint_path(budget, horizon)

            # Process a20
            a20_key = f"{budget}_H{horizon}"
            if a20_checkpoint.exists():
                print(f"  Loading a20 model...")
                try:
                    model_a20 = load_model(a20_checkpoint, budget)  # Infers features from checkpoint

                    # Create test dataset
                    splitter_a20 = SimpleSplitter(
                        dates=df_a20["Date"],
                        context_length=80,
                        horizon=horizon,
                        val_start="2023-01-01",
                        test_start="2025-01-01",
                    )
                    split_a20 = splitter_a20.split()

                    dataset_a20 = FinancialDataset(
                        features_df=df_a20,
                        close_prices=df_a20["Close"].values,
                        context_length=80,
                        horizon=horizon,
                        threshold=0.01,
                        feature_columns=a20_feature_cols,
                        high_prices=df_a20["High"].values,
                    )

                    test_indices_a20 = split_a20.test_indices
                    test_dataset_a20 = torch.utils.data.Subset(dataset_a20, test_indices_a20)
                    test_loader_a20 = torch.utils.data.DataLoader(test_dataset_a20, batch_size=64, shuffle=False)

                    print(f"  Evaluating a20 on {len(test_indices_a20)} test samples...")
                    metrics_a20 = evaluate_model(model_a20, test_loader_a20, device)
                    results["a20"][a20_key] = metrics_a20

                    if metrics_a20["auc"]:
                        print(f"    a20 Test AUC: {metrics_a20['auc']:.4f}")
                    else:
                        print(f"    a20 Test AUC: N/A (single class)")

                except Exception as e:
                    print(f"  ERROR evaluating a20: {e}")
                    results["a20"][a20_key] = {"error": str(e)}
            else:
                print(f"  SKIP: a20 checkpoint not found")
                results["a20"][a20_key] = {"error": "checkpoint not found"}

            # Process a50
            a50_key = f"{budget}_H{horizon}"
            if a50_checkpoint.exists():
                print(f"  Loading a50 model...")
                try:
                    model_a50 = load_model(a50_checkpoint, budget)  # Infers features from checkpoint

                    # Create test dataset
                    splitter_a50 = SimpleSplitter(
                        dates=df_a50["Date"],
                        context_length=80,
                        horizon=horizon,
                        val_start="2023-01-01",
                        test_start="2025-01-01",
                    )
                    split_a50 = splitter_a50.split()

                    dataset_a50 = FinancialDataset(
                        features_df=df_a50,
                        close_prices=df_a50["Close"].values,
                        context_length=80,
                        horizon=horizon,
                        threshold=0.01,
                        feature_columns=a50_feature_cols,
                        high_prices=df_a50["High"].values,
                    )

                    test_indices_a50 = split_a50.test_indices
                    test_dataset_a50 = torch.utils.data.Subset(dataset_a50, test_indices_a50)
                    test_loader_a50 = torch.utils.data.DataLoader(test_dataset_a50, batch_size=64, shuffle=False)

                    print(f"  Evaluating a50 on {len(test_indices_a50)} test samples...")
                    metrics_a50 = evaluate_model(model_a50, test_loader_a50, device)
                    results["a50"][a50_key] = metrics_a50

                    if metrics_a50["auc"]:
                        print(f"    a50 Test AUC: {metrics_a50['auc']:.4f}")
                    else:
                        print(f"    a50 Test AUC: N/A (single class)")

                except Exception as e:
                    print(f"  ERROR evaluating a50: {e}")
                    results["a50"][a50_key] = {"error": str(e)}
            else:
                print(f"  SKIP: a50 checkpoint not found")
                results["a50"][a50_key] = {"error": "checkpoint not found"}

            # Comparison
            if (
                a20_key in results["a20"]
                and a50_key in results["a50"]
                and "error" not in results["a20"][a20_key]
                and "error" not in results["a50"][a50_key]
            ):
                auc_a20 = results["a20"][a20_key]["auc"]
                auc_a50 = results["a50"][a50_key]["auc"]

                if auc_a20 is not None and auc_a50 is not None:
                    delta = auc_a50 - auc_a20
                    results["comparison"].append({
                        "budget": budget,
                        "horizon": horizon,
                        "auc_a20_test": auc_a20,
                        "auc_a50_test": auc_a50,
                        "auc_delta_test": delta,
                        "n_samples_a20": results["a20"][a20_key]["n_samples"],
                        "n_samples_a50": results["a50"][a50_key]["n_samples"],
                    })
                    print(f"    Delta (a50-a20): {delta:+.4f}")

    return results


def load_validation_results() -> dict:
    """Load validation set comparison results for cross-reference."""
    comparison_path = PROJECT_ROOT / "outputs" / "phase6c" / "a20_vs_a50_comparison.json"
    if comparison_path.exists():
        with open(comparison_path) as f:
            return json.load(f)
    return {}


def print_comparison_summary(test_results: dict, val_results: dict) -> None:
    """Print summary comparing test vs validation performance."""
    print("\n" + "=" * 70)
    print("VALIDATION vs TEST SET COMPARISON")
    print("=" * 70)

    # Get validation AUC comparison
    val_auc = {(r["budget"], r["horizon"]): r for r in val_results.get("auc_comparison", [])}

    print("\n| Budget | Horizon | Val AUC Δ | Test AUC Δ | Consistent? |")
    print("|--------|---------|-----------|------------|-------------|")

    consistent_count = 0
    total_count = 0

    for comp in test_results["comparison"]:
        budget = comp["budget"]
        horizon = comp["horizon"]
        test_delta = comp["auc_delta_test"]

        val_data = val_auc.get((budget, horizon))
        if val_data:
            val_delta = val_data["auc_delta"]

            # Check if direction matches (both positive or both negative)
            consistent = (val_delta > 0 and test_delta > 0) or (val_delta < 0 and test_delta < 0)
            consistent_str = "YES" if consistent else "NO"

            if consistent:
                consistent_count += 1
            total_count += 1

            print(f"| {budget:6} | H{horizon} | {val_delta:+.4f} | {test_delta:+.4f} | {consistent_str} |")
        else:
            print(f"| {budget:6} | H{horizon} | N/A | {test_delta:+.4f} | N/A |")
            total_count += 1

    print(f"\nConsistency rate: {consistent_count}/{total_count} ({100*consistent_count/total_count:.0f}% if total_count > 0 else 0%)")

    # Test set summary
    print("\n" + "=" * 70)
    print("TEST SET PERFORMANCE SUMMARY")
    print("=" * 70)

    if test_results["comparison"]:
        test_deltas = [c["auc_delta_test"] for c in test_results["comparison"]]
        avg_delta = np.mean(test_deltas)
        h1_deltas = [c["auc_delta_test"] for c in test_results["comparison"] if c["horizon"] == 1]
        avg_h1_delta = np.mean(h1_deltas) if h1_deltas else None

        print(f"\nAverage AUC difference (a50 - a20) on test set: {avg_delta:+.4f}")
        if avg_h1_delta is not None:
            print(f"H1 average AUC difference on test set: {avg_h1_delta:+.4f}")

        # Count positive vs negative deltas
        pos_count = sum(1 for d in test_deltas if d > 0)
        neg_count = sum(1 for d in test_deltas if d < 0)
        print(f"\na50 better than a20: {pos_count}/{len(test_deltas)} cases")
        print(f"a20 better than a50: {neg_count}/{len(test_deltas)} cases")

    # Conclusion
    print("\n" + "=" * 70)
    print("TEST SET CONCLUSION")
    print("=" * 70)

    if total_count > 0 and consistent_count / total_count >= 0.75:
        print("\nPATTERN REPLICATED: Validation patterns hold on test set (>75% consistency)")
    elif total_count > 0 and consistent_count / total_count <= 0.25:
        print("\nPATTERN NOT REPLICATED: Test set shows opposite patterns (<25% consistency)")
        print("  -> Validation results likely due to overfitting")
    else:
        print("\nINCONCLUSIVE: Mixed patterns between validation and test sets")
        print("  -> No clear evidence for or against the hypothesis")


def main():
    """Main entry point."""
    print("=" * 70)
    print("TEST SET EVALUATION: a20 vs a50")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Check device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Run test evaluation
    test_results = run_test_evaluation(device)

    # Load validation results for comparison
    val_results = load_validation_results()

    # Print comparison summary
    print_comparison_summary(test_results, val_results)

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "test_results": test_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
