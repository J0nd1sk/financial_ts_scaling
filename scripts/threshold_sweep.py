#!/usr/bin/env python3
"""Threshold sweep for Phase 6A models.

Loads trained model checkpoints, runs inference on validation data,
and computes precision/recall/F1 at multiple classification thresholds.

Usage:
    # Sweep all 12 Phase 6A models
    python scripts/threshold_sweep.py

    # Sweep single model
    python scripts/threshold_sweep.py --budget 2M --horizon 1

    # Custom thresholds
    python scripts/threshold_sweep.py --thresholds 0.3,0.4,0.5,0.6,0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FinancialDataset, SimpleSplitter
from src.models.patchtst import PatchTST, PatchTSTConfig

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "v1" / "SPY_dataset_a20.parquet"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "phase6a_final"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "phase6a_final" / "threshold_sweep.csv"

# Feature columns (5 OHLCV + 20 indicators = 25 total, matching training)
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "dema_9", "dema_10", "sma_12", "dema_20", "dema_25",
    "sma_50", "dema_90", "sma_100", "sma_200",
    "rsi_daily", "rsi_weekly", "stochrsi_daily", "stochrsi_weekly",
    "macd_line", "obv", "adosc", "atr_14", "adx_14",
    "bb_percent_b", "vwap_20",
]

# Training constants (must match Phase 6A scripts)
CONTEXT_LENGTH = 80
PATCH_LENGTH = 16
STRIDE = 8
THRESHOLD_TARGET = 0.01  # 1% threshold for target (matches "threshold_1pct" task)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def compute_metrics_at_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute classification metrics at a specific threshold.

    Args:
        predictions: Array of predicted probabilities (0-1).
        labels: Array of binary labels (0 or 1).
        threshold: Classification threshold.

    Returns:
        Dict with precision, recall, f1, n_positive_preds, n_samples.
    """
    pred_binary = (predictions >= threshold).astype(int)
    labels_int = labels.astype(int)

    n_positive_preds = int(pred_binary.sum())
    n_samples = len(predictions)

    return {
        "precision": precision_score(labels_int, pred_binary, zero_division=0.0),
        "recall": recall_score(labels_int, pred_binary, zero_division=0.0),
        "f1": f1_score(labels_int, pred_binary, zero_division=0.0),
        "n_positive_preds": n_positive_preds,
        "n_samples": n_samples,
    }


def sweep_thresholds(
    predictions: np.ndarray,
    labels: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """Compute metrics at multiple thresholds.

    Args:
        predictions: Array of predicted probabilities (0-1).
        labels: Array of binary labels (0 or 1).
        thresholds: List of thresholds to evaluate.

    Returns:
        DataFrame with one row per threshold.
    """
    rows = []
    for t in thresholds:
        metrics = compute_metrics_at_threshold(predictions, labels, t)
        metrics["threshold"] = t
        rows.append(metrics)

    return pd.DataFrame(rows)


# =============================================================================
# MODEL LOADING AND INFERENCE
# =============================================================================


def load_model_and_config(checkpoint_dir: Path) -> tuple[PatchTST, dict]:
    """Load model from checkpoint directory.

    Args:
        checkpoint_dir: Directory containing best_checkpoint.pt and results.json.

    Returns:
        Tuple of (model, results_dict).
    """
    checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
    results_path = checkpoint_dir / "results.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Results not found: {results_path}")

    # Load results for architecture info
    with open(results_path) as f:
        results = json.load(f)

    arch = results["architecture"]
    hyper = results["hyperparameters"]

    # Create model config
    config = PatchTSTConfig(
        num_features=len(FEATURE_COLUMNS),
        context_length=hyper["context_length"],
        patch_length=PATCH_LENGTH,
        stride=STRIDE,
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=0.0,  # No dropout during inference
        head_dropout=0.0,
    )

    # Check if model was trained with RevIN
    use_revin = hyper.get("use_revin", False)

    # Load model weights
    model = PatchTST(config, use_revin=use_revin)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, results


def get_validation_data(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, DataLoader]:
    """Prepare validation data matching training splits.

    Args:
        df: Full dataset DataFrame.
        horizon: Prediction horizon.

    Returns:
        Tuple of (high_prices, close_prices, val_dataloader).
    """
    # Use SimpleSplitter with same params as training
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    splits = splitter.split()

    # Get prices for target calculation
    high_prices = df["High"].values
    close_prices = df["Close"].values

    # Create full dataset
    full_dataset = FinancialDataset(
        features_df=df,
        close_prices=close_prices,
        high_prices=high_prices,
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
        threshold=THRESHOLD_TARGET,
        feature_columns=FEATURE_COLUMNS,
    )

    # Create validation subset using val_indices
    val_subset = Subset(full_dataset, splits.val_indices.tolist())
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)

    return high_prices, close_prices, val_loader


def run_inference(model: PatchTST, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and collect predictions.

    Args:
        model: PatchTST model in eval mode.
        dataloader: DataLoader for validation data.

    Returns:
        Tuple of (predictions, labels) as numpy arrays.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())

    return np.array(all_preds), np.array(all_labels)


# =============================================================================
# MAIN
# =============================================================================


def sweep_single_model(
    budget: str,
    horizon: int,
    df: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    """Run threshold sweep for a single model.

    Args:
        budget: Parameter budget (2M, 20M, 200M).
        horizon: Prediction horizon (1, 2, 3, 5).
        df: Full dataset DataFrame.
        thresholds: Thresholds to evaluate.

    Returns:
        DataFrame with sweep results.
    """
    experiment_name = f"phase6a_{budget.lower()}_h{horizon}"
    checkpoint_dir = CHECKPOINT_DIR / experiment_name

    print(f"\n{'='*60}")
    print(f"Sweeping {budget}/H{horizon}")
    print(f"{'='*60}")

    # Load model
    model, results = load_model_and_config(checkpoint_dir)
    print(f"  Architecture: d={results['architecture']['d_model']}, "
          f"L={results['architecture']['n_layers']}, "
          f"h={results['architecture']['n_heads']}")

    # Get validation data
    _, _, val_loader = get_validation_data(df, horizon)
    print(f"  Validation samples: {len(val_loader.dataset)}")

    # Run inference
    predictions, labels = run_inference(model, val_loader)
    print(f"  Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"  Class balance: {labels.mean():.3f}")

    # Compute AUC (threshold-independent)
    try:
        auc = roc_auc_score(labels, predictions)
    except ValueError:
        auc = None
    print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A (single class)")

    # Sweep thresholds
    sweep_df = sweep_thresholds(predictions, labels, thresholds)
    sweep_df["budget"] = budget
    sweep_df["horizon"] = horizon
    sweep_df["auc"] = auc
    sweep_df["pred_min"] = predictions.min()
    sweep_df["pred_max"] = predictions.max()

    # Print summary table
    print(f"\n  {'Thresh':>6} {'Prec':>6} {'Recall':>6} {'F1':>6} {'N_pos':>6}")
    print(f"  {'-'*36}")
    for _, row in sweep_df.iterrows():
        print(f"  {row['threshold']:>6.2f} {row['precision']:>6.3f} "
              f"{row['recall']:>6.3f} {row['f1']:>6.3f} {row['n_positive_preds']:>6.0f}")

    return sweep_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Threshold sweep for Phase 6A models")
    parser.add_argument("--budget", type=str, help="Single budget (2M, 20M, 200M)")
    parser.add_argument("--horizon", type=int, help="Single horizon (1, 2, 3, 5)")
    parser.add_argument("--thresholds", type=str, help="Comma-separated thresholds")
    args = parser.parse_args()

    # Parse thresholds
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]
    else:
        thresholds = DEFAULT_THRESHOLDS

    # Load data
    print(f"Loading data from {DATA_PATH}")
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df)} rows")

    # Determine which models to sweep
    if args.budget and args.horizon:
        budgets = [args.budget]
        horizons = [args.horizon]
    else:
        budgets = ["2M", "20M", "200M"]
        horizons = [1, 2, 3, 5]

    # Collect results
    all_results = []

    for budget in budgets:
        for horizon in horizons:
            try:
                sweep_df = sweep_single_model(budget, horizon, df, thresholds)
                all_results.append(sweep_df)
            except Exception as e:
                print(f"  ERROR: {e}")

    # Combine and save
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        # Reorder columns
        cols = ["budget", "horizon", "threshold", "precision", "recall", "f1",
                "n_positive_preds", "n_samples", "auc", "pred_min", "pred_max"]
        combined = combined[cols]

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(OUTPUT_PATH, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to {OUTPUT_PATH}")
        print(f"Total rows: {len(combined)}")


if __name__ == "__main__":
    main()
