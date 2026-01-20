#!/usr/bin/env python3
"""
RevIN vs Z-score Normalization Comparison Test

Compares three normalization strategies for PatchTST:
1. zscore_only:  Z-score preprocessing (global stats from train), no RevIN
2. revin_only:   No Z-score, RevIN in model (per-instance normalization)
3. zscore_revin: Both Z-score preprocessing and RevIN

Uses 2M model with fixed architecture for controlled comparison.
Outputs: val_loss, AUC-ROC, prediction spread for each config.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.config.experiment import ExperimentConfig
from src.data.dataset import (
    SimpleSplitter,
    FinancialDataset,
    compute_normalization_params,
    normalize_dataframe,
)
from src.models.patchtst import PatchTST, PatchTSTConfig
from src.training.trainer import Trainer

# ============================================================
# CONFIGURATION
# ============================================================

# Data settings
DATA_PATH = "data/processed/v1/SPY_dataset_a20.parquet"
HORIZON = 1  # 1-day ahead prediction
TASK = "threshold_1pct"
SEED = 42

# Training settings (fixed across all configs for fair comparison)
EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

# 2M architecture (from best HPO results)
D_MODEL = 64
N_LAYERS = 4  # Reduced from extreme values per research gap analysis
N_HEADS = 2
D_FF = 256
DROPOUT = 0.2

# Feature columns (a20 tier)
FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "dema_9", "dema_10", "sma_12", "dema_20", "dema_25",
    "sma_50", "dema_90", "sma_100", "sma_200",
    "rsi_daily", "rsi_weekly", "stochrsi_daily", "stochrsi_weekly",
    "macd_line", "obv",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def load_data():
    """Load and validate data."""
    data_path = PROJECT_ROOT / DATA_PATH
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    # Filter to feature columns that exist
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    print(f"Using {len(available_features)} features")

    return df, available_features


def get_split_indices(df):
    """Create train/val/test split indices using SimpleSplitter.

    Uses date-based contiguous splits:
    - Train: before 2023 (maximize training data)
    - Val: 2023-2024 (for validation/early stopping)
    - Test: 2025+ (most recent for backtesting)
    """
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=60,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    return splitter.split()


def create_model_config(num_features: int) -> PatchTSTConfig:
    """Create PatchTST config."""
    return PatchTSTConfig(
        num_features=num_features,
        context_length=60,
        patch_length=16,
        stride=8,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        head_dropout=0.0,
        num_classes=1,
    )


def create_experiment_config(data_path: str) -> ExperimentConfig:
    """Create experiment config."""
    return ExperimentConfig(
        task=TASK,
        timescale="daily",
        data_path=data_path,
        horizon=HORIZON,
        seed=SEED,
    )


def evaluate_predictions(model, dataloader, device):
    """Evaluate model and return predictions + labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())

    return np.array(all_preds), np.array(all_labels)


def run_config(
    config_name: str,
    df: pd.DataFrame,
    features: list[str],
    split_indices,
    use_zscore: bool,
    use_revin: bool,
    output_dir: Path,
) -> dict:
    """Run a single configuration and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"  Z-score: {use_zscore}, RevIN: {use_revin}")
    print(f"{'='*60}")

    # Set seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Prepare data
    df_train = df.copy()
    norm_params = None

    if use_zscore:
        # Compute normalization params from training data only
        train_end_row = split_indices.train_indices.max() + 60 + HORIZON
        norm_params = compute_normalization_params(df_train, train_end_row)
        df_train = normalize_dataframe(df_train, norm_params)
        print(f"  Applied Z-score normalization ({len(norm_params)} features)")

    # Save normalized data to temp file for Trainer
    temp_data_path = output_dir / f"temp_{config_name}.parquet"
    df_train.to_parquet(temp_data_path)

    # Create configs
    model_config = create_model_config(len(features))
    exp_config = create_experiment_config(str(temp_data_path))

    # Create trainer with RevIN setting
    trainer = Trainer(
        experiment_config=exp_config,
        model_config=model_config,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        checkpoint_dir=output_dir / config_name,
        split_indices=split_indices,
        use_revin=use_revin,
    )

    # Train
    start_time = time.time()
    result = trainer.train()
    train_time = time.time() - start_time

    val_loss = result.get("val_loss", result.get("train_loss"))
    print(f"  Training completed in {train_time:.1f}s")
    print(f"  Val loss: {val_loss:.4f}")

    # Evaluate on validation set
    preds, labels = evaluate_predictions(
        trainer.model, trainer.val_dataloader, trainer.device
    )

    # Compute metrics
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.5  # If only one class present

    pred_min = preds.min()
    pred_max = preds.max()
    pred_std = preds.std()
    pred_mean = preds.mean()

    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Predictions: min={pred_min:.4f}, max={pred_max:.4f}, std={pred_std:.4f}")

    # Clean up temp file
    temp_data_path.unlink()

    return {
        "config": config_name,
        "use_zscore": use_zscore,
        "use_revin": use_revin,
        "val_loss": val_loss,
        "auc": auc,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "pred_std": pred_std,
        "pred_mean": pred_mean,
        "train_time_s": train_time,
    }


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 60)
    print("RevIN vs Z-score Normalization Comparison Test")
    print("=" * 60)

    # Setup
    output_dir = PROJECT_ROOT / "outputs" / "revin_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df, features = load_data()

    # Get split indices
    split_indices = get_split_indices(df)
    print(f"Split: {len(split_indices.train_indices)} train, "
          f"{len(split_indices.val_indices)} val samples")

    # Configurations to test
    configs = [
        ("zscore_only", True, False),   # Z-score preprocessing, no RevIN
        ("revin_only", False, True),    # No Z-score, RevIN in model
        ("zscore_revin", True, True),   # Both Z-score and RevIN
    ]

    # Run all configurations
    results = []
    for config_name, use_zscore, use_revin in configs:
        result = run_config(
            config_name=config_name,
            df=df,
            features=features,
            split_indices=split_indices,
            use_zscore=use_zscore,
            use_revin=use_revin,
            output_dir=output_dir,
        )
        results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # Determine winner
    best_idx = results_df["auc"].idxmax()
    best_config = results_df.loc[best_idx, "config"]
    best_auc = results_df.loc[best_idx, "auc"]

    print(f"\n🏆 Best config: {best_config} (AUC: {best_auc:.4f})")

    # Check prediction spread
    for _, row in results_df.iterrows():
        spread = row["pred_max"] - row["pred_min"]
        status = "✅" if spread > 0.1 else "⚠️"
        print(f"  {status} {row['config']}: spread={spread:.4f}")


if __name__ == "__main__":
    main()
