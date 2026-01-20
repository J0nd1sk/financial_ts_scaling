#!/usr/bin/env python3
"""
Test 1: BCE vs SoftAUC AUC-ROC Comparison on 2025 Test Data

Purpose: Verify that SoftAUCLoss improves ranking performance (AUC-ROC)
on held-out test data, not just prediction spread.

Method:
1. Train 2M_h1 model with SoftAUCLoss (same architecture/hyperparams as BCE)
2. Load existing BCE checkpoint
3. Evaluate both on identical 2025 test samples
4. Compare AUC-ROC scores

Known limitation: Early stopping uses val_loss, not val_AUC (addressed in Test 2)

Usage:
    python experiments/compare_bce_vs_soft_auc.py
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
from torch.utils.data import DataLoader

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTST, PatchTSTConfig
from src.models.arch_grid import get_memory_safe_batch_config
from src.data.dataset import ChunkSplitter, FinancialDataset
from src.training.trainer import Trainer
from src.training.losses import SoftAUCLoss

# =============================================================================
# CONFIGURATION (matches train_2M_h1.py exactly)
# =============================================================================

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "v1" / "SPY_dataset_a20.parquet"
BCE_CHECKPOINT = PROJECT_ROOT / "outputs" / "final_training" / "train_2M_h1" / "best_checkpoint.pt"
SOFT_AUC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "soft_auc_validation"

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "dema_9", "dema_10", "sma_12", "dema_20", "dema_25",
    "sma_50", "dema_90", "sma_100", "sma_200",
    "rsi_daily", "rsi_weekly", "stochrsi_daily", "stochrsi_weekly",
    "macd_line", "obv", "adosc", "atr_14", "adx_14",
    "bb_percent_b", "vwap_20",
]

# Architecture (from HPO - 2M_h1)
D_MODEL = 64
N_LAYERS = 48
N_HEADS = 2
D_FF = 256

# Training params (from HPO)
LEARNING_RATE = 0.0008
DROPOUT = 0.12
WEIGHT_DECAY = 0.001
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Model params
CONTEXT_LENGTH = 60
PATCH_LENGTH = 16
STRIDE = 8
HORIZON = 1
THRESHOLD = 0.01


# =============================================================================
# HELPERS
# =============================================================================

def prepare_test_data(df: pd.DataFrame, test_start_date: str = "2025-01-01"):
    """Get test indices for 2025+ data (same logic as evaluate_final_models.py)."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"])

    test_start = pd.Timestamp(test_start_date)
    n_rows = len(df)
    test_indices = []
    test_dates = []

    max_start = n_rows - CONTEXT_LENGTH - HORIZON
    for i in range(max_start + 1):
        pred_idx = i + CONTEXT_LENGTH - 1
        pred_date = df.iloc[pred_idx]["Date"]
        if pred_date >= test_start:
            test_indices.append(i)
            test_dates.append(pred_date)

    return np.array(test_indices, dtype=np.int64), test_dates


def evaluate_model(model: PatchTST, dataloader, device: str = "mps"):
    """Run inference and return predictions + targets."""
    model = model.to(device)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.append(preds.flatten())
            all_targets.append(batch_y.numpy().flatten())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray):
    """Compute AUC-ROC and spread metrics."""
    auc = roc_auc_score(targets, predictions) if len(np.unique(targets)) == 2 else 0.5
    return {
        "auc_roc": auc,
        "spread": float(predictions.max() - predictions.min()),
        "std": float(predictions.std()),
        "mean": float(predictions.mean()),
        "min": float(predictions.min()),
        "max": float(predictions.max()),
    }


def load_model_from_checkpoint(checkpoint_path: Path) -> PatchTST:
    """Load model from checkpoint."""
    config = PatchTSTConfig(
        num_features=len(FEATURE_COLUMNS),
        context_length=CONTEXT_LENGTH,
        patch_length=PATCH_LENGTH,
        stride=STRIDE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=0.0,  # No dropout at inference
        head_dropout=0.0,
    )
    model = PatchTST(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Test 1: BCE vs SoftAUC AUC-ROC Comparison")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"  {len(df)} rows, date range: {df['Date'].min()} to {df['Date'].max()}")

    # Create contiguous splits (same as final training)
    splitter = ChunkSplitter(
        total_days=len(df),
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="contiguous",
    )
    split_indices = splitter.split()
    print(f"  Splits: train={len(split_indices.train_indices)}, "
          f"val={len(split_indices.val_indices)}, test={len(split_indices.test_indices)}")

    # Prepare test data (2025+)
    test_indices, test_dates = prepare_test_data(df)
    print(f"  2025 test samples: {len(test_indices)}")

    # Create test dataloader
    dataset = FinancialDataset(
        features_df=df,
        close_prices=df["Close"].values,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        threshold=THRESHOLD,
        feature_columns=FEATURE_COLUMNS,
    )

    class TestSubset(torch.utils.data.Dataset):
        def __init__(self, full_dataset, indices):
            self.full_dataset = full_dataset
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.full_dataset[self.indices[idx]]

    test_dataset = TestSubset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # =========================================================================
    # TRAIN SOFT AUC MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("Training SoftAUC Model (same architecture as BCE)")
    print("=" * 70)

    experiment_config = ExperimentConfig(
        data_path=str(DATA_PATH),
        task="threshold_1pct",
        timescale="daily",
        horizon=HORIZON,
        context_length=CONTEXT_LENGTH,
    )

    model_config = PatchTSTConfig(
        num_features=len(FEATURE_COLUMNS),
        context_length=CONTEXT_LENGTH,
        patch_length=PATCH_LENGTH,
        stride=STRIDE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT,
        head_dropout=0.0,
    )

    batch_config = get_memory_safe_batch_config(d_model=D_MODEL, n_layers=N_LAYERS)

    SOFT_AUC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=batch_config["micro_batch"],
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device="mps",
        checkpoint_dir=SOFT_AUC_OUTPUT_DIR,
        split_indices=split_indices,
        accumulation_steps=batch_config["accumulation_steps"],
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_metric="val_auc",  # Use AUC for early stopping with SoftAUCLoss
        criterion=SoftAUCLoss(gamma=2.0),
    )

    start_time = time.time()
    result = trainer.train(verbose=True)
    train_duration = time.time() - start_time

    print(f"\nSoftAUC training complete in {train_duration/60:.1f} min")
    print(f"  Epochs: {result.get('epochs_trained', 'N/A')}")
    print(f"  Val loss: {result.get('val_loss', 'N/A'):.4f}")
    print(f"  Val AUC: {result.get('val_auc', 'N/A')}")
    print(f"  Early stopped: {result.get('stopped_early', False)}")
    print(f"  Early stopping metric: val_auc")

    # =========================================================================
    # EVALUATE BOTH MODELS ON 2025 TEST DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("Evaluating Both Models on 2025 Test Data")
    print("=" * 70)

    # Load BCE model
    print("\nLoading BCE model...")
    bce_model, bce_checkpoint = load_model_from_checkpoint(BCE_CHECKPOINT)
    bce_preds, targets = evaluate_model(bce_model, test_loader)
    bce_metrics = compute_metrics(bce_preds, targets)

    # Load SoftAUC model (fresh from training)
    print("Loading SoftAUC model...")
    soft_auc_checkpoint_path = SOFT_AUC_OUTPUT_DIR / "best_checkpoint.pt"
    soft_auc_model, soft_auc_checkpoint = load_model_from_checkpoint(soft_auc_checkpoint_path)
    soft_auc_preds, _ = evaluate_model(soft_auc_model, test_loader)
    soft_auc_metrics = compute_metrics(soft_auc_preds, targets)

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS: BCE vs SoftAUC Comparison")
    print("=" * 70)

    print(f"\nTest set: {len(test_indices)} samples (2025+)")
    print(f"Positive rate: {targets.mean():.2%}")

    print("\n--- BCE Model ---")
    print(f"  Checkpoint: {BCE_CHECKPOINT.name}")
    print(f"  Training epochs: {bce_checkpoint.get('epoch', 'N/A')}")
    print(f"  Val loss (BCE): {bce_checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Test AUC-ROC: {bce_metrics['auc_roc']:.4f}")
    print(f"  Test spread: {bce_metrics['spread']:.4f}")
    print(f"  Prediction range: [{bce_metrics['min']:.4f}, {bce_metrics['max']:.4f}]")

    print("\n--- SoftAUC Model ---")
    print(f"  Checkpoint: {soft_auc_checkpoint_path.name}")
    print(f"  Training epochs: {soft_auc_checkpoint.get('epoch', 'N/A')}")
    print(f"  Val loss (SoftAUC): {soft_auc_checkpoint.get('val_loss', 'N/A'):.4f}")
    print(f"  Test AUC-ROC: {soft_auc_metrics['auc_roc']:.4f}")
    print(f"  Test spread: {soft_auc_metrics['spread']:.4f}")
    print(f"  Prediction range: [{soft_auc_metrics['min']:.4f}, {soft_auc_metrics['max']:.4f}]")

    # Comparison
    auc_diff = soft_auc_metrics['auc_roc'] - bce_metrics['auc_roc']
    auc_pct = (auc_diff / bce_metrics['auc_roc']) * 100 if bce_metrics['auc_roc'] > 0 else 0
    spread_ratio = soft_auc_metrics['spread'] / bce_metrics['spread'] if bce_metrics['spread'] > 0 else float('inf')

    print("\n--- Comparison ---")
    print(f"  AUC-ROC difference: {auc_diff:+.4f} ({auc_pct:+.1f}%)")
    print(f"  Spread ratio: {spread_ratio:.1f}x (SoftAUC / BCE)")

    if auc_diff > 0.01:
        verdict = "SoftAUC BETTER - meaningful AUC improvement"
    elif auc_diff < -0.01:
        verdict = "BCE BETTER - SoftAUC hurt performance"
    else:
        verdict = "COMPARABLE - AUC within 1% (need more data or different approach)"

    print(f"\n  VERDICT: {verdict}")

    print("\n" + "=" * 70)
    print("Recommendation:")
    if auc_diff > 0:
        print("  Proceed with Test 2 (AUC-based early stopping)")
    else:
        print("  Investigate: Check if BCE hyperparams suboptimal for SoftAUC")
    print("=" * 70)
