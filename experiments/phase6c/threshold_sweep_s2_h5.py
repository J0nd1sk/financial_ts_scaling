#!/usr/bin/env python3
"""
Phase 6C Stage 2: Threshold Sweep for s2_horizon_2m_h5_a50

Evaluates the H=5 horizon model at different decision thresholds to understand
precision/recall trade-offs and find optimal operating points.

Context:
- Model: 2M params, H=5 horizon, 55 features (50 indicators + 5 OHLCV)
- Task: threshold_1pct (1% price rise within 5 days)
- Original metrics: AUC=0.594, Acc=61.0%, Prec=65.7%, Rec=73.3%
- Class balance: 60% positive (at H=5, more samples hit 1% threshold)
"""
import sys
import json
from pathlib import Path

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

EXPERIMENT_NAME = "s2_horizon_2m_h5_a50"
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a50_combined.parquet"
CHECKPOINT_PATH = PROJECT_ROOT / f"outputs/phase6c/{EXPERIMENT_NAME}/best_checkpoint.pt"
OUTPUT_PATH = PROJECT_ROOT / "outputs/phase6c/s2_horizon_2m_h5_threshold_sweep.json"

# Model configuration (2M budget)
NUM_FEATURES = 55  # 5 OHLCV + 50 indicators
CONTEXT_LENGTH = 80
HORIZON = 5  # Key difference: 5-day horizon
D_MODEL = 64
N_LAYERS = 4
N_HEADS = 4
D_FF = 256

# Task: 1% threshold (threshold_1pct task, same as S1)
TASK_THRESHOLD = 0.01

# Thresholds to sweep (focused range around default 0.5)
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def load_model_and_get_predictions(checkpoint_path, df, split_indices, device):
    """Load model from checkpoint and get validation predictions."""

    # Create model config
    model_config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=0.5,
        head_dropout=0.0,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model and load weights (use_revin=True as training used RevIN)
    model = PatchTST(model_config, use_revin=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Prepare data
    close_prices = df["Close"].values
    high_prices = df["High"].values

    # Create full dataset with 2% threshold for H=5
    full_dataset = FinancialDataset(
        features_df=df,
        close_prices=close_prices,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        threshold=TASK_THRESHOLD,
        high_prices=high_prices,
    )

    # Create validation subset using indices
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

    return np.array(all_preds), np.array(all_labels)


def compute_metrics_at_threshold(preds, labels, threshold):
    """Compute metrics at a specific decision threshold."""
    binary_preds = (preds >= threshold).astype(int)

    # Handle edge cases
    n_positive_preds = binary_preds.sum()
    n_negative_preds = len(binary_preds) - n_positive_preds

    if n_positive_preds == 0:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    elif n_negative_preds == 0:
        precision = labels.mean()  # All positive predictions
        recall = 1.0
        f1 = 2 * precision / (precision + 1) if precision > 0 else 0
    else:
        precision = precision_score(labels, binary_preds, zero_division=0)
        recall = recall_score(labels, binary_preds, zero_division=0)
        f1 = f1_score(labels, binary_preds, zero_division=0)

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, binary_preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_positive_preds": int(n_positive_preds),
        "n_negative_preds": int(n_negative_preds),
        "pred_positive_rate": float(n_positive_preds / len(binary_preds)),
    }


def find_optimal_thresholds(sweep_results):
    """Find optimal thresholds for different objectives."""
    df = pd.DataFrame(sweep_results)

    # Filter to reasonable thresholds (at least some predictions in each class)
    valid = df[(df["n_positive_preds"] > 0) & (df["n_negative_preds"] > 0)]

    if len(valid) == 0:
        return {}

    optimal = {}

    # Best F1
    best_f1_idx = valid["f1"].idxmax()
    optimal["best_f1"] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                          for k, v in valid.loc[best_f1_idx].to_dict().items()}

    # Best precision (with at least 10% recall - more realistic for H=5 with 60% class balance)
    high_precision = valid[valid["recall"] >= 0.10]
    if len(high_precision) > 0:
        best_prec_idx = high_precision["precision"].idxmax()
        optimal["best_precision_10pct_recall"] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                                                   for k, v in high_precision.loc[best_prec_idx].to_dict().items()}

    # Best recall (with at least 50% precision - realistic for high positive rate)
    high_recall = valid[valid["precision"] >= 0.50]
    if len(high_recall) > 0:
        best_recall_idx = high_recall["recall"].idxmax()
        optimal["best_recall_50pct_precision"] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                                                   for k, v in high_recall.loc[best_recall_idx].to_dict().items()}

    # Best accuracy
    best_acc_idx = valid["accuracy"].idxmax()
    optimal["best_accuracy"] = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                                 for k, v in valid.loc[best_acc_idx].to_dict().items()}

    return optimal


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("PHASE 6C STAGE 2: THRESHOLD SWEEP FOR s2_horizon_2m_h5_a50")
    print("=" * 80)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Verify checkpoint exists
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        return None

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data: {len(df)} rows")

    # Create splitter with H=5
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()
    print(f"Validation samples: {len(split_indices.val_indices)}")

    # Load model and get predictions
    print("\nLoading model and getting predictions...")
    preds, labels = load_model_and_get_predictions(CHECKPOINT_PATH, df, split_indices, device)

    # Compute AUC (threshold-independent)
    auc = roc_auc_score(labels, preds)
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Prediction range: [{preds.min():.4f}, {preds.max():.4f}]")
    print(f"Class balance: {labels.mean():.3f} ({int(labels.sum())}/{len(labels)} positives)")

    # Sweep thresholds
    print(f"\nThreshold Sweep:")
    print(f"{'Thresh':>6} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Pos Preds':>10}")
    print("-" * 60)

    sweep_results = []
    for thresh in THRESHOLDS:
        metrics = compute_metrics_at_threshold(preds, labels, thresh)
        sweep_results.append(metrics)
        print(f"{thresh:>6.2f} | {metrics['accuracy']:>6.3f} | {metrics['precision']:>6.3f} | "
              f"{metrics['recall']:>6.3f} | {metrics['f1']:>6.3f} | {metrics['n_positive_preds']:>10}")

    # Find optimal thresholds
    optimal = find_optimal_thresholds(sweep_results)

    print(f"\nOPTIMAL THRESHOLDS:")
    print("-" * 60)
    for objective, result in optimal.items():
        print(f"{objective}:")
        print(f"  Threshold: {result['threshold']:.2f}")
        print(f"  Accuracy: {result['accuracy']:.3f}, Precision: {result['precision']:.3f}, "
              f"Recall: {result['recall']:.3f}, F1: {result['f1']:.3f}")

    # Prepare results
    results = {
        "experiment": EXPERIMENT_NAME,
        "horizon": HORIZON,
        "task_threshold": TASK_THRESHOLD,
        "budget": "2M",
        "auc": float(auc),
        "pred_range": [float(preds.min()), float(preds.max())],
        "class_balance": float(labels.mean()),
        "n_samples": len(labels),
        "sweep": sweep_results,
        "optimal": optimal,
    }

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    main()
