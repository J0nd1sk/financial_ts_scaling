#!/usr/bin/env python3
"""
0.5% Threshold with HIGH-based Targets: 2M Narrow-Deep h=8
Purpose: Test if h=8 improves over h=2 at 2M scale
Architecture: d=64, L=32, h=8 (~2M params)

Comparison experiment: At 20M scale, h=8 performed second-best. Does this hold at 2M?
Note: d_k = 64/8 = 8 (smallest head dimension tested)
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "2M_narrow_h8_threshold_05pct_HIGH"
TASK = "threshold_0.5pct"
HORIZON = 1

# Architecture: 2M narrow-deep with h=8
D_MODEL = 64
N_LAYERS = 32
N_HEADS = 8  # Changed from h=2
D_FF = 256  # 4 * d_model
CONTEXT_LENGTH = 80
DROPOUT = 0.5  # Same as baseline for comparison

# Training (same as baseline for fair comparison)
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 50

# Data
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
NUM_FEATURES = 20

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/threshold_05pct_high" / EXPERIMENT_NAME

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print("=" * 70)
    print("TARGET: max(HIGH[t+1:t+horizon]) >= CLOSE[t] * 1.005")
    print(f"Architecture: 2M narrow-deep (d={D_MODEL}, L={N_LAYERS}, h={N_HEADS})")
    print(f"Note: d_k = {D_MODEL // N_HEADS} (head dimension)")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data: {len(df)} rows, {df['Date'].min()} to {df['Date'].max()}")

    # Extract prices for target calculation
    close_prices = df["Close"].values
    high_prices = df["High"].values

    # Create experiment config
    experiment_config = ExperimentConfig(
        data_path=str(DATA_PATH),
        task=TASK,
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
    )

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
        dropout=DROPOUT,
        head_dropout=0.0,
    )

    # Estimate parameters
    est_params = (NUM_FEATURES * D_MODEL +  # input projection
                  N_LAYERS * (4 * D_MODEL * D_MODEL + 2 * D_MODEL * D_FF) +  # transformer
                  D_MODEL * 1)  # output head
    print(f"\nModel Config:")
    print(f"  d_model: {D_MODEL}, n_layers: {N_LAYERS}, n_heads: {N_HEADS}")
    print(f"  d_ff: {D_FF}, dropout: {DROPOUT}")
    print(f"  Estimated params: ~{est_params/1e6:.1f}M")
    print(f"  context_length: {CONTEXT_LENGTH}, num_patches: {model_config.num_patches}")

    # Create splits
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    print(f"\nSplit sizes:")
    print(f"  Train: {len(split_indices.train_indices)} samples")
    print(f"  Val: {len(split_indices.val_indices)} samples")
    print(f"  Test: {len(split_indices.test_indices)} samples")

    # Calculate class balance
    from src.data.dataset import FinancialDataset

    full_dataset = FinancialDataset(
        features_df=df,
        close_prices=close_prices,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        threshold=0.005,
        high_prices=high_prices,
    )

    train_labels = [full_dataset[i][1].item() for i in split_indices.train_indices]
    val_labels = [full_dataset[i][1].item() for i in split_indices.val_indices]
    test_labels = [full_dataset[i][1].item() for i in split_indices.test_indices]

    print(f"\nClass balance (using HIGH-based targets):")
    print(f"  Train: {sum(train_labels)/len(train_labels)*100:.1f}% positive")
    print(f"  Val: {sum(val_labels)/len(val_labels)*100:.1f}% positive")
    print(f"  Test: {sum(test_labels)/len(test_labels)*100:.1f}% positive")

    # Train
    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        checkpoint_dir=OUTPUT_DIR / "checkpoints",
        split_indices=split_indices,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_metric="val_auc",
        use_revin=True,
        high_prices=high_prices,
    )

    print(f"\nTraining for up to {EPOCHS} epochs (early stopping on val_auc)...")
    start_time = time.time()
    result = trainer.train(verbose=True)
    elapsed = time.time() - start_time

    print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    # Get predictions for full metrics
    print("\nEvaluating on test set...")
    trainer.model.eval()

    test_subset = torch.utils.data.Subset(full_dataset, split_indices.test_indices.tolist())
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(trainer.device)
            outputs = trainer.model(features)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    binary_preds = (all_preds >= 0.5).astype(int)

    test_accuracy = accuracy_score(all_labels, binary_preds)
    test_precision = precision_score(all_labels, binary_preds, zero_division=0)
    test_recall = recall_score(all_labels, binary_preds, zero_division=0)
    test_f1 = f1_score(all_labels, binary_preds, zero_division=0)
    test_auc = roc_auc_score(all_labels, all_preds)

    pred_min, pred_max = all_preds.min(), all_preds.max()
    pred_spread = pred_max - pred_min
    pred_std = all_preds.std()

    # Save results
    results = {
        "experiment_name": EXPERIMENT_NAME,
        "task": TASK,
        "threshold": 0.005,
        "target_type": "HIGH-based (correct)",
        "context_length": CONTEXT_LENGTH,
        "architecture": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "use_revin": True,
            "estimated_params_M": est_params / 1e6,
        },
        "training": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs_requested": EPOCHS,
            "epochs_completed": result.get("epochs_completed", EPOCHS),
            "stopped_early": result.get("stopped_early", False),
        },
        "class_balance": {
            "train_positive_pct": sum(train_labels)/len(train_labels)*100,
            "val_positive_pct": sum(val_labels)/len(val_labels)*100,
            "test_positive_pct": sum(test_labels)/len(test_labels)*100,
        },
        "val_metrics": {
            "auc": result.get("best_val_auc") or result.get("val_auc"),
            "loss": result.get("best_val_loss") or result.get("val_loss"),
        },
        "test_metrics": {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "auc_roc": test_auc,
        },
        "prediction_distribution": {
            "min": float(pred_min),
            "max": float(pred_max),
            "spread": float(pred_spread),
            "std": float(pred_std),
        },
        "splits": {
            "train_samples": len(split_indices.train_indices),
            "val_samples": len(split_indices.val_indices),
            "test_samples": len(split_indices.test_indices),
        },
        "training_time_minutes": elapsed / 60,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Architecture: 2M narrow (d={D_MODEL}, L={N_LAYERS}, h={N_HEADS})")
    print(f"\nValidation Metrics:")
    print(f"  AUC: {results['val_metrics']['auc']:.4f}")
    print(f"  Loss: {results['val_metrics']['loss']:.4f}")
    print(f"\nTest Metrics:")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1: {test_f1:.4f}")
    print(f"  AUC-ROC: {test_auc:.4f}")
    print(f"\nPrediction Distribution:")
    print(f"  Range: [{pred_min:.4f}, {pred_max:.4f}]")
    print(f"  Spread: {pred_spread:.4f}")

    # Compare with baseline
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH 2M h=2 BASELINE")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'Test AUC':<12} {'Test Acc':<12} {'Test F1':<12}")
    print("-" * 56)
    print(f"{'2M h=8 (this)':<20} {test_auc:<12.4f} {test_accuracy:<12.4f} {test_f1:<12.4f}")
    print(f"{'2M h=2 (baseline)':<20} {'0.7044':<12} {'0.6630':<12} {'0.5344':<12}")


if __name__ == "__main__":
    main()
