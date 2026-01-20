#!/usr/bin/env python3
"""Evaluate Phase 6A final trained models on 2025 holdout data.

Loads all 16 trained checkpoints and computes classification metrics
(accuracy, precision, recall, F1, AUC-ROC) on 2025+ test data.

Usage:
    # Evaluate all models
    python scripts/evaluate_final_models.py

    # Evaluate single model for debugging
    python scripts/evaluate_final_models.py --budget 2M --horizon 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FinancialDataset
from src.models.patchtst import PatchTST, PatchTSTConfig

if TYPE_CHECKING:
    from collections.abc import Iterable


# =============================================================================
# Architecture table from HPO results (must match generate_final_training_scripts.py)
# Key: (budget, horizon) -> (d_model, n_layers, n_heads)
# =============================================================================
ARCHITECTURES: dict[tuple[str, int], tuple[int, int, int]] = {
    # 2M budget
    ("2M", 1): (64, 48, 2),
    ("2M", 2): (64, 32, 2),
    ("2M", 3): (64, 32, 2),
    ("2M", 5): (64, 64, 16),
    # 20M budget
    ("20M", 1): (128, 180, 16),
    ("20M", 2): (256, 32, 2),
    ("20M", 3): (256, 32, 2),
    ("20M", 5): (384, 12, 4),
    # 200M budget
    ("200M", 1): (384, 96, 4),
    ("200M", 2): (768, 24, 16),
    ("200M", 3): (768, 24, 16),
    ("200M", 5): (256, 256, 16),
    # 2B budget
    ("2B", 1): (1024, 128, 2),
    ("2B", 2): (768, 256, 32),
    ("2B", 3): (768, 256, 32),
    ("2B", 5): (1024, 180, 4),
}

# Feature columns (25 features: 5 OHLCV + 20 indicators)
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "dema_9", "dema_10", "sma_12", "dema_20", "dema_25",
    "sma_50", "dema_90", "sma_100", "sma_200",
    "rsi_daily", "rsi_weekly", "stochrsi_daily", "stochrsi_weekly",
    "macd_line", "obv", "adosc", "atr_14", "adx_14",
    "bb_percent_b", "vwap_20",
]

# Model constants
CONTEXT_LENGTH = 60
PATCH_LENGTH = 16
STRIDE = 8
THRESHOLD = 0.01  # 1% threshold for binary classification

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "v1" / "SPY_dataset_a20.parquet"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "final_training"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "results" / "phase6a_backtest_2025.csv"


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    budget: str,
    horizon: int,
) -> PatchTST:
    """Load a trained PatchTST model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint .pt file.
        budget: Parameter budget (2M, 20M, 200M, 2B).
        horizon: Prediction horizon (1, 2, 3, 5).

    Returns:
        PatchTST model loaded with checkpoint weights, in eval mode.

    Raises:
        KeyError: If (budget, horizon) not found in ARCHITECTURES table.
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Look up architecture from table
    key = (budget, horizon)
    if key not in ARCHITECTURES:
        raise KeyError(f"Architecture not found for {key}. Valid keys: {list(ARCHITECTURES.keys())}")

    d_model, n_layers, n_heads = ARCHITECTURES[key]
    d_ff = 4 * d_model  # Standard transformer ratio

    # Create model config
    config = PatchTSTConfig(
        num_features=len(FEATURE_COLUMNS),
        context_length=CONTEXT_LENGTH,
        patch_length=PATCH_LENGTH,
        stride=STRIDE,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.0,  # No dropout during inference
        head_dropout=0.0,
    )

    # Create model and load weights
    model = PatchTST(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def prepare_test_data(
    df: pd.DataFrame,
    horizon: int,
    context_length: int = CONTEXT_LENGTH,
    test_start_date: str = "2025-01-01",
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """Prepare test data indices for 2025+ evaluation.

    Filters data to samples where the prediction date (last day of context window)
    is >= test_start_date.

    Args:
        df: DataFrame with Date column and feature columns.
        horizon: Prediction horizon in days.
        context_length: Number of days in input context window.
        test_start_date: Start date for test set (inclusive).

    Returns:
        Tuple of (test_indices, test_dates) where:
            - test_indices: Array of valid sample start indices
            - test_dates: List of prediction dates for each sample
    """
    # Ensure Date column is datetime
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    test_start = pd.Timestamp(test_start_date)
    n_rows = len(df)

    # Find valid test indices
    # For a sample starting at index i:
    # - Context window: [i, i + context_length)
    # - Prediction point: i + context_length - 1 (last day of context)
    # - Need horizon days after prediction point for target
    test_indices = []
    test_dates = []

    max_start = n_rows - context_length - horizon
    for i in range(max_start + 1):
        # Prediction point is last day of context window
        pred_idx = i + context_length - 1
        pred_date = df.iloc[pred_idx]["Date"]

        if pred_date >= test_start:
            test_indices.append(i)
            test_dates.append(pred_date)

    return np.array(test_indices, dtype=np.int64), test_dates


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics from predictions and targets.

    Args:
        predictions: Array of predicted probabilities (0-1).
        targets: Array of binary targets (0 or 1).
        threshold: Classification threshold for predictions.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, auc_roc.
    """
    # Convert predictions to binary using threshold
    pred_binary = (predictions >= threshold).astype(int)
    targets_int = targets.astype(int)

    # Compute metrics with zero_division handling
    metrics = {
        "accuracy": accuracy_score(targets_int, pred_binary),
        "precision": precision_score(targets_int, pred_binary, zero_division=0.0),
        "recall": recall_score(targets_int, pred_binary, zero_division=0.0),
        "f1": f1_score(targets_int, pred_binary, zero_division=0.0),
    }

    # AUC-ROC requires both classes present, handle gracefully
    unique_targets = np.unique(targets_int)
    if len(unique_targets) == 2:
        metrics["auc_roc"] = roc_auc_score(targets_int, predictions)
    else:
        # Single class - AUC undefined, use accuracy as proxy
        metrics["auc_roc"] = metrics["accuracy"]

    return metrics


def evaluate_model(
    model: PatchTST,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on model and collect predictions.

    Args:
        model: PatchTST model in eval mode.
        dataloader: DataLoader yielding (batch_x, batch_y) tuples.
        device: Device to run inference on.

    Returns:
        Tuple of (predictions, targets) as numpy arrays.
    """
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)

            # Forward pass - model already applies sigmoid in PredictionHead
            # DO NOT apply sigmoid again (was causing double-sigmoid bug)
            probs = model(batch_x).cpu().numpy()

            all_predictions.append(probs.flatten())
            all_targets.append(batch_y.numpy().flatten())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return predictions, targets


def evaluate_single_model(
    budget: str,
    horizon: int,
    df: pd.DataFrame,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate a single model and return metrics.

    Args:
        budget: Parameter budget (2M, 20M, 200M, 2B).
        horizon: Prediction horizon (1, 2, 3, 5).
        df: DataFrame with features and Date column.
        device: Device for inference.

    Returns:
        Dict with all metrics plus metadata (n_samples, checkpoint_val_loss).
    """
    # Load checkpoint
    checkpoint_dir = CHECKPOINT_DIR / f"train_{budget}_h{horizon}"
    checkpoint_path = checkpoint_dir / "best_checkpoint.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, budget, horizon)

    # Get checkpoint val_loss for reference
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    val_loss = checkpoint.get("val_loss", None)

    # Prepare test data
    test_indices, test_dates = prepare_test_data(df, horizon)

    if len(test_indices) == 0:
        raise ValueError(f"No test samples found for horizon={horizon}")

    # Create dataset for test indices only
    # We need to create samples for each test index
    close_prices = df["Close"].values

    # Build test dataset
    dataset = FinancialDataset(
        features_df=df,
        close_prices=close_prices,
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
        threshold=THRESHOLD,
        feature_columns=FEATURE_COLUMNS,
    )

    # Create a subset dataset for test indices only
    class TestSubset(torch.utils.data.Dataset):
        def __init__(self, full_dataset, indices):
            self.full_dataset = full_dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.full_dataset[self.indices[idx]]

    test_dataset = TestSubset(dataset, test_indices)
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Evaluate
    predictions, targets = evaluate_model(model, dataloader, device)

    # Compute metrics
    metrics = compute_metrics(predictions, targets)

    # Add metadata
    metrics["n_samples"] = len(test_indices)
    metrics["checkpoint_val_loss"] = val_loss
    metrics["avg_confidence"] = float(np.mean(np.abs(predictions - 0.5)) + 0.5)

    return metrics


def main():
    """Main evaluation loop for all 16 models."""
    parser = argparse.ArgumentParser(description="Evaluate Phase 6A models on 2025 data")
    parser.add_argument("--budget", type=str, help="Single budget to evaluate (2M, 20M, 200M, 2B)")
    parser.add_argument("--horizon", type=int, help="Single horizon to evaluate (1, 2, 3, 5)")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {DATA_PATH}")
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        sys.exit(1)

    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")

    # Determine which models to evaluate
    if args.budget and args.horizon:
        budgets = [args.budget]
        horizons = [args.horizon]
    else:
        budgets = ["2M", "20M", "200M", "2B"]
        horizons = [1, 2, 3, 5]

    # Collect results
    results = []

    for budget in budgets:
        for horizon in horizons:
            print(f"\nEvaluating {budget}/h{horizon}...")

            try:
                metrics = evaluate_single_model(budget, horizon, df, args.device)

                results.append({
                    "budget": budget,
                    "horizon": horizon,
                    **metrics,
                })

                print(f"  n_samples: {metrics['n_samples']}")
                print(f"  accuracy:  {metrics['accuracy']:.4f}")
                print(f"  precision: {metrics['precision']:.4f}")
                print(f"  recall:    {metrics['recall']:.4f}")
                print(f"  f1:        {metrics['f1']:.4f}")
                print(f"  auc_roc:   {metrics['auc_roc']:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    "budget": budget,
                    "horizon": horizon,
                    "error": str(e),
                })

    # Save results
    if results:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nResults saved to {OUTPUT_PATH}")

        # Print summary table
        print("\n" + "=" * 70)
        print("SUMMARY: 2025 Backtest Results")
        print("=" * 70)
        if "accuracy" in results_df.columns:
            summary = results_df.pivot_table(
                index="budget",
                columns="horizon",
                values="accuracy",
                aggfunc="first",
            )
            print("\nAccuracy by Budget x Horizon:")
            print(summary.to_string())


if __name__ == "__main__":
    main()
