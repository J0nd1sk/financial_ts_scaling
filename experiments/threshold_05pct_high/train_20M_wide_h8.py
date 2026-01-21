#!/usr/bin/env python3
"""
0.5% Threshold with HIGH-based Targets: 20M_wide h=8
Purpose: First correct experiment with HIGH-based threshold targets
Architecture: d=512, L=6, h=8 (20M_wide - best from previous experiments)
Target: max(HIGH[t+1:t+horizon]) >= CLOSE[t] * 1.005

This is the FIRST experiment with correctly wired HIGH-based targets.
Previous experiments all used CLOSE-based targets due to Trainer bug.
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

EXPERIMENT_NAME = "20M_wide_h8_threshold_05pct_HIGH"
TASK = "threshold_0.5pct"  # 0.5% threshold - balanced classes (~50% positive with HIGH)
HORIZON = 1

# Architecture: 20M_wide (best from dropout scaling experiment)
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
D_FF = 2048  # 4 * d_model
CONTEXT_LENGTH = 80  # Optimal from context length ablation
DROPOUT = 0.5  # Best from LR/dropout tuning

# Training
LEARNING_RATE = 1e-4  # Best from LR/dropout tuning
BATCH_SIZE = 32  # Conservative for 20M model
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
    print("This uses HIGH-based targets (correct formulation)")
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
    high_prices = df["High"].values  # CRITICAL: Use HIGH for target

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

    print(f"\nModel Config:")
    print(f"  d_model: {D_MODEL}, n_layers: {N_LAYERS}, n_heads: {N_HEADS}")
    print(f"  d_ff: {D_FF}, dropout: {DROPOUT}")
    print(f"  context_length: {CONTEXT_LENGTH}, num_patches: {model_config.num_patches}")

    # Create splits (SimpleSplitter - proper sample sizes)
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

    # Calculate class balance for each split
    from src.data.dataset import FinancialDataset

    full_dataset = FinancialDataset(
        features_df=df,
        close_prices=close_prices,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        threshold=0.005,
        high_prices=high_prices,  # CRITICAL: Pass high prices
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
        use_revin=True,  # Best from normalization comparison
        high_prices=high_prices,  # CRITICAL: Pass high prices to Trainer
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

    # Binary predictions at 0.5 threshold
    binary_preds = (all_preds >= 0.5).astype(int)

    # Compute metrics
    test_accuracy = accuracy_score(all_labels, binary_preds)
    test_precision = precision_score(all_labels, binary_preds, zero_division=0)
    test_recall = recall_score(all_labels, binary_preds, zero_division=0)
    test_f1 = f1_score(all_labels, binary_preds, zero_division=0)
    test_auc = roc_auc_score(all_labels, all_preds)

    # Prediction spread (check for probability collapse)
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
    print(f"Target: HIGH-based (correct formulation)")
    print(f"\nClass Balance:")
    print(f"  Train: {results['class_balance']['train_positive_pct']:.1f}% positive")
    print(f"  Val: {results['class_balance']['val_positive_pct']:.1f}% positive")
    print(f"  Test: {results['class_balance']['test_positive_pct']:.1f}% positive")
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
    print(f"  Std: {pred_std:.4f}")

    if pred_spread < 0.1:
        print(f"\n⚠️ WARNING: Prediction spread < 0.1 indicates probability collapse!")
    else:
        print(f"\n✅ Prediction spread looks healthy")


if __name__ == "__main__":
    main()
