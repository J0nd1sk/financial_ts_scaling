#!/usr/bin/env python3
"""
Phase 6C: Comprehensive Threshold Sweep Analysis

Evaluates multiple S2 models at fine-grained decision thresholds to find
optimal operating points for production use (high precision with acceptable recall).

Focus: 0.60-0.80 threshold range with 0.03 granularity for precision optimization.
"""
import sys
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.models.patchtst import PatchTSTConfig, PatchTST
from src.data.dataset import SimpleSplitter, FinancialDataset
from torch.utils.data import DataLoader, Subset

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a50_combined.parquet"
NUM_FEATURES = 55  # 5 OHLCV + 50 indicators
CONTEXT_LENGTH = 80

# Fine-grained thresholds focused on high-precision operating points
THRESHOLDS = [0.50, 0.55, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68,
              0.70, 0.72, 0.74, 0.76, 0.78, 0.80]

# Experiments to analyze
EXPERIMENTS = [
    # Horizon experiments (H=2, H=3, H=5 at all budgets)
    {
        "name": "s2_horizon_2m_h2_a50",
        "horizon": 2,
        "d_model": 64, "n_layers": 4, "n_heads": 4, "d_ff": 256,
    },
    {
        "name": "s2_horizon_20m_h2_a50",
        "horizon": 2,
        "d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff": 512,
    },
    {
        "name": "s2_horizon_200m_h2_a50",
        "horizon": 2,
        "d_model": 256, "n_layers": 8, "n_heads": 8, "d_ff": 1024,
    },
    {
        "name": "s2_horizon_2m_h3_a50",
        "horizon": 3,
        "d_model": 64, "n_layers": 4, "n_heads": 4, "d_ff": 256,
    },
    {
        "name": "s2_horizon_20m_h3_a50",
        "horizon": 3,
        "d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff": 512,
    },
    {
        "name": "s2_horizon_200m_h3_a50",
        "horizon": 3,
        "d_model": 256, "n_layers": 8, "n_heads": 8, "d_ff": 1024,
    },
    {
        "name": "s2_horizon_2m_h5_a50",
        "horizon": 5,
        "d_model": 64, "n_layers": 4, "n_heads": 4, "d_ff": 256,
    },
    {
        "name": "s2_horizon_20m_h5_a50",
        "horizon": 5,
        "d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff": 512,
    },
    {
        "name": "s2_horizon_200m_h5_a50",
        "horizon": 5,
        "d_model": 256, "n_layers": 8, "n_heads": 8, "d_ff": 1024,
    },
    # Top architecture experiments (high AUC)
    {
        "name": "s2_arch_2m_h1_heads8",
        "horizon": 1,
        "d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff": 256,
    },
    {
        "name": "s2_arch_200m_h1_balanced",
        "horizon": 1,
        "d_model": 192, "n_layers": 10, "n_heads": 12, "d_ff": 768,
    },
    # Top training experiments
    {
        "name": "s2_train_200m_h1_wd1e3",
        "horizon": 1,
        "d_model": 256, "n_layers": 8, "n_heads": 8, "d_ff": 1024,
    },
]


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_model_and_get_predictions(exp_config, checkpoint_path, df, device):
    """Load model from checkpoint and get validation predictions."""

    horizon = exp_config["horizon"]

    # Create splitter for this horizon
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    # Create model config
    model_config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=exp_config["d_model"],
        n_heads=exp_config["n_heads"],
        n_layers=exp_config["n_layers"],
        d_ff=exp_config["d_ff"],
        dropout=0.5,
        head_dropout=0.0,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model and load weights
    model = PatchTST(model_config, use_revin=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Prepare data
    close_prices = df["Close"].values
    high_prices = df["High"].values

    # Create full dataset with 1% threshold
    full_dataset = FinancialDataset(
        features_df=df,
        close_prices=close_prices,
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
        threshold=0.01,  # 1% threshold
        high_prices=high_prices,
    )

    # Create validation subset
    val_subset = Subset(full_dataset, split_indices.val_indices.tolist())
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

    # Get predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())

    return np.array(all_preds), np.array(all_labels), len(split_indices.val_indices)


def compute_metrics_at_threshold(preds, labels, threshold):
    """Compute metrics at a specific decision threshold."""
    binary_preds = (preds >= threshold).astype(int)

    n_positive_preds = binary_preds.sum()
    n_negative_preds = len(binary_preds) - n_positive_preds

    if n_positive_preds == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    elif n_negative_preds == 0:
        precision = labels.mean()
        recall = 1.0
        f1 = 2 * precision / (precision + 1) if precision > 0 else 0
    else:
        precision = precision_score(labels, binary_preds, zero_division=0)
        recall = recall_score(labels, binary_preds, zero_division=0)
        f1 = f1_score(labels, binary_preds, zero_division=0)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, binary_preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_positive_preds": int(n_positive_preds),
        "n_samples": len(labels),
        "pred_positive_rate": float(n_positive_preds / len(binary_preds)),
    }


def find_production_operating_points(sweep_results):
    """Find operating points suitable for production use."""
    df = pd.DataFrame(sweep_results)

    # Filter to valid predictions (some in each class)
    valid = df[(df["n_positive_preds"] > 0) & (df["n_positive_preds"] < df["n_samples"])]

    if len(valid) == 0:
        return {}

    points = {}

    # High precision targets (production-worthy)
    for target_prec in [0.65, 0.70, 0.75, 0.80]:
        candidates = valid[valid["precision"] >= target_prec]
        if len(candidates) > 0:
            # Get the one with best recall at this precision level
            best_idx = candidates["recall"].idxmax()
            row = candidates.loc[best_idx]
            points[f"prec_{int(target_prec*100)}"] = {
                "threshold": float(row["threshold"]),
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "accuracy": float(row["accuracy"]),
                "n_positive_preds": int(row["n_positive_preds"]),
            }

    # Best F1 overall
    best_f1_idx = valid["f1"].idxmax()
    row = valid.loc[best_f1_idx]
    points["best_f1"] = {
        "threshold": float(row["threshold"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "accuracy": float(row["accuracy"]),
    }

    return points


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 6C: COMPREHENSIVE THRESHOLD SWEEP")
    print("=" * 80)
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Experiments: {len(EXPERIMENTS)}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data once
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data: {len(df)} rows")

    all_results = {}
    summary_rows = []

    for exp in EXPERIMENTS:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {exp['name']} (H={exp['horizon']})")
        print("=" * 80)

        checkpoint_path = PROJECT_ROOT / f"outputs/phase6c/{exp['name']}/best_checkpoint.pt"

        if not checkpoint_path.exists():
            print(f"  WARNING: Checkpoint not found at {checkpoint_path}")
            continue

        # Get predictions
        print("  Loading model and getting predictions...")
        try:
            preds, labels, n_val = load_model_and_get_predictions(exp, checkpoint_path, df, device)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        # Compute AUC
        auc = roc_auc_score(labels, preds)
        class_balance = labels.mean()
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Prediction range: [{preds.min():.4f}, {preds.max():.4f}]")
        print(f"  Class balance: {class_balance:.3f} ({int(labels.sum())}/{len(labels)} positives)")

        # Sweep thresholds
        print(f"\n  {'Thresh':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Acc':>6} | {'N_pos':>6}")
        print("  " + "-" * 55)

        sweep_results = []
        for thresh in THRESHOLDS:
            metrics = compute_metrics_at_threshold(preds, labels, thresh)
            sweep_results.append(metrics)
            print(f"  {thresh:>6.2f} | {metrics['precision']:>6.3f} | {metrics['recall']:>6.3f} | "
                  f"{metrics['f1']:>6.3f} | {metrics['accuracy']:>6.3f} | {metrics['n_positive_preds']:>6}")

        # Find production operating points
        operating_points = find_production_operating_points(sweep_results)

        print(f"\n  PRODUCTION OPERATING POINTS:")
        for name, point in operating_points.items():
            print(f"  {name}: thresh={point['threshold']:.2f}, "
                  f"prec={point['precision']:.3f}, rec={point['recall']:.3f}")

            # Add to summary for comparison
            if name.startswith("prec_"):
                summary_rows.append({
                    "experiment": exp['name'],
                    "horizon": exp['horizon'],
                    "auc": auc,
                    "target": name,
                    "threshold": point['threshold'],
                    "precision": point['precision'],
                    "recall": point['recall'],
                    "f1": point['f1'],
                })

        all_results[exp['name']] = {
            "horizon": exp['horizon'],
            "auc": float(auc),
            "class_balance": float(class_balance),
            "n_samples": len(labels),
            "pred_range": [float(preds.min()), float(preds.max())],
            "sweep": sweep_results,
            "operating_points": operating_points,
        }

    # Print summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY: PRODUCTION-WORTHY OPERATING POINTS")
    print("=" * 80)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)

        for target in ["prec_65", "prec_70", "prec_75", "prec_80"]:
            target_rows = summary_df[summary_df["target"] == target].sort_values("recall", ascending=False)
            if len(target_rows) > 0:
                print(f"\n{target.upper()} (sorted by recall):")
                print(f"{'Experiment':<35} | {'H':>2} | {'AUC':>5} | {'Thresh':>6} | {'Prec':>6} | {'Recall':>6}")
                print("-" * 80)
                for _, row in target_rows.head(5).iterrows():
                    print(f"{row['experiment']:<35} | {row['horizon']:>2} | {row['auc']:.3f} | "
                          f"{row['threshold']:.2f} | {row['precision']:.3f} | {row['recall']:.3f}")

    # Save results
    output_path = PROJECT_ROOT / "outputs/phase6c/comprehensive_threshold_sweep.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
