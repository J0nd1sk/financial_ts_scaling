#!/usr/bin/env python3
"""
Loss Function Comparison Test

Compares four loss functions for PatchTST binary classification:
1. BCE (baseline): Standard binary cross-entropy
2. SoftAUCLoss: Direct AUC optimization via pairwise ranking
3. FocalLoss: Focus on hard examples, down-weight easy ones
4. LabelSmoothingBCELoss: Reduce overconfidence via soft targets

Uses established foundation:
- SimpleSplitter (442 val samples) for reliable validation
- RevIN only (no Z-score) based on previous comparison results

Outputs: val_loss, AUC-ROC, accuracy, prediction spread for each config.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score

from src.config.experiment import ExperimentConfig
from src.data.dataset import SimpleSplitter, FinancialDataset
from src.models.patchtst import PatchTST, PatchTSTConfig
from src.training.trainer import Trainer
from src.training.losses import SoftAUCLoss, FocalLoss, LabelSmoothingBCELoss

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

# 2M architecture (reduced for controlled comparison)
D_MODEL = 64
N_LAYERS = 4
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

# Loss function configurations: (name, criterion_instance)
# None means use default BCE
LOSS_CONFIGS = [
    ("bce", None),  # Baseline: standard BCE
    ("softauc", SoftAUCLoss(gamma=2.0)),  # Direct AUC optimization
    ("focal", FocalLoss(gamma=2.0, alpha=0.25)),  # Focus on hard examples
    ("label_smooth", LabelSmoothingBCELoss(epsilon=0.1)),  # Reduce overconfidence
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
    criterion: nn.Module | None,
    df: pd.DataFrame,
    features: list[str],
    split_indices,
    output_dir: Path,
) -> dict:
    """Run a single loss configuration and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    criterion_name = criterion.__class__.__name__ if criterion else "BCELoss"
    print(f"  Criterion: {criterion_name}")
    print(f"{'='*60}")

    # Set seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Save data to temp file for Trainer (no normalization - RevIN handles it)
    temp_data_path = output_dir / f"temp_{config_name}.parquet"
    df.to_parquet(temp_data_path)

    # Create configs
    model_config = create_model_config(len(features))
    exp_config = create_experiment_config(str(temp_data_path))

    # Create trainer with RevIN and custom criterion
    trainer = Trainer(
        experiment_config=exp_config,
        model_config=model_config,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        checkpoint_dir=output_dir / config_name,
        split_indices=split_indices,
        use_revin=True,  # Always use RevIN (best from previous comparison)
        criterion=criterion,  # Custom loss function
    )

    # Train
    start_time = time.time()
    try:
        result = trainer.train()
        train_time = time.time() - start_time
        val_loss = result.get("val_loss", result.get("train_loss"))
        status = "success"
    except Exception as e:
        print(f"  ERROR: {e}")
        train_time = time.time() - start_time
        val_loss = float("nan")
        status = f"error: {e}"

    print(f"  Training completed in {train_time:.1f}s")
    print(f"  Val loss: {val_loss:.4f}" if not np.isnan(val_loss) else "  Val loss: NaN")

    # Evaluate on validation set
    if status == "success":
        preds, labels = evaluate_predictions(
            trainer.model, trainer.val_dataloader, trainer.device
        )

        # Compute metrics
        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = 0.5  # If only one class present

        # Accuracy at threshold 0.5
        binary_preds = (preds >= 0.5).astype(int)
        accuracy = accuracy_score(labels, binary_preds)

        pred_min = preds.min()
        pred_max = preds.max()
        pred_std = preds.std()
        pred_mean = preds.mean()

        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Predictions: min={pred_min:.4f}, max={pred_max:.4f}, std={pred_std:.4f}")
    else:
        auc = 0.5
        accuracy = 0.5
        pred_min = pred_max = pred_std = pred_mean = float("nan")

    # Clean up temp file
    temp_data_path.unlink()

    return {
        "config": config_name,
        "criterion": criterion.__class__.__name__ if criterion else "BCELoss",
        "val_loss": val_loss,
        "auc": auc,
        "accuracy": accuracy,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "pred_std": pred_std,
        "pred_mean": pred_mean,
        "pred_spread": pred_max - pred_min,
        "train_time_s": train_time,
        "status": status,
    }


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 60)
    print("Loss Function Comparison Test")
    print("=" * 60)
    print(f"Configs to test: {[c[0] for c in LOSS_CONFIGS]}")
    print(f"Foundation: SimpleSplitter + RevIN only")

    # Setup
    output_dir = PROJECT_ROOT / "outputs" / "loss_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df, features = load_data()

    # Get split indices
    split_indices = get_split_indices(df)
    print(f"Split: {len(split_indices.train_indices)} train, "
          f"{len(split_indices.val_indices)} val samples")

    # Run all configurations
    results = []
    for config_name, criterion in LOSS_CONFIGS:
        result = run_config(
            config_name=config_name,
            criterion=criterion,
            df=df,
            features=features,
            split_indices=split_indices,
            output_dir=output_dir,
        )
        results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_path = output_dir / "comparison_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Select columns for display
    display_cols = ["config", "val_loss", "auc", "accuracy", "pred_spread", "train_time_s"]
    print(results_df[display_cols].to_string(index=False))

    # Determine winners
    successful = results_df[results_df["status"] == "success"]

    if len(successful) > 0:
        # Best by AUC
        best_auc_idx = successful["auc"].idxmax()
        best_auc_config = successful.loc[best_auc_idx, "config"]
        best_auc = successful.loc[best_auc_idx, "auc"]

        # Best by accuracy
        best_acc_idx = successful["accuracy"].idxmax()
        best_acc_config = successful.loc[best_acc_idx, "config"]
        best_acc = successful.loc[best_acc_idx, "accuracy"]

        # Best by val_loss (lower is better)
        best_loss_idx = successful["val_loss"].idxmin()
        best_loss_config = successful.loc[best_loss_idx, "config"]
        best_loss = successful.loc[best_loss_idx, "val_loss"]

        print(f"\n🏆 Winners:")
        print(f"   Best AUC:      {best_auc_config} (AUC: {best_auc:.4f})")
        print(f"   Best Accuracy: {best_acc_config} (Acc: {best_acc:.4f})")
        print(f"   Best Val Loss: {best_loss_config} (Loss: {best_loss:.4f})")

        # Check if same winner for AUC and accuracy
        if best_auc_config == best_acc_config:
            print(f"\n✅ Clear winner: {best_auc_config} (best on both AUC and accuracy)")
        else:
            print(f"\n⚠️ Different winners for AUC vs accuracy - may need multi-objective approach")

    # Check prediction spread (healthy spread indicates model is making varied predictions)
    print(f"\nPrediction Spread Analysis:")
    for _, row in results_df.iterrows():
        spread = row["pred_spread"]
        if np.isnan(spread):
            status = "❌ ERROR"
        elif spread > 0.3:
            status = "✅ Good spread"
        elif spread > 0.1:
            status = "⚠️ Moderate spread"
        else:
            status = "🔴 Low spread (possible collapse)"
        print(f"  {status}: {row['config']} (spread={spread:.4f})")


if __name__ == "__main__":
    main()
