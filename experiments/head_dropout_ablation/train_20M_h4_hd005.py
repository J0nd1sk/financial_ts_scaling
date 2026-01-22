#!/usr/bin/env python3
"""
Head Dropout Ablation: 20M h=4 with head_dropout=0.05
Purpose: Test effect of light head dropout on best 20M architecture
Architecture: d=512, L=6, h=4 (~20M params)
Baseline: head_dropout=0.0 achieved AUC 0.712
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

EXPERIMENT_NAME = "20M_h4_hd005"
TASK = "threshold_0.5pct"
HORIZON = 1

# Architecture: 20M wide-shallow with h=4
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 4
D_FF = 2048
CONTEXT_LENGTH = 80
DROPOUT = 0.5  # Encoder dropout
HEAD_DROPOUT = 0.05  # <-- VARIABLE UNDER TEST

# Training
LEARNING_RATE = 1e-4
BATCH_SIZE = 32  # Smaller batch for 20M
EPOCHS = 50

# Data
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
NUM_FEATURES = 20

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/head_dropout_ablation" / EXPERIMENT_NAME

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print("=" * 70)
    print(f"HEAD DROPOUT ABLATION: head_dropout={HEAD_DROPOUT}")
    print(f"Architecture: 20M (d={D_MODEL}, L={N_LAYERS}, h={N_HEADS})")
    print(f"Baseline (hd=0.0): AUC 0.712")
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
        head_dropout=HEAD_DROPOUT,
    )

    # Estimate parameters
    est_params = (NUM_FEATURES * D_MODEL +
                  N_LAYERS * (4 * D_MODEL * D_MODEL + 2 * D_MODEL * D_FF) +
                  D_MODEL * 1)
    print(f"\nModel Config:")
    print(f"  d_model: {D_MODEL}, n_layers: {N_LAYERS}, n_heads: {N_HEADS}")
    print(f"  d_ff: {D_FF}, dropout: {DROPOUT}, head_dropout: {HEAD_DROPOUT}")
    print(f"  Estimated params: ~{est_params/1e6:.1f}M")

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

    print(f"\nClass balance (HIGH-based targets):")
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

    # Evaluate on test set
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

    # Save results
    results = {
        "experiment_name": EXPERIMENT_NAME,
        "ablation": "head_dropout",
        "head_dropout": HEAD_DROPOUT,
        "baseline_auc": 0.712,
        "architecture": {
            "scale": "20M",
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "head_dropout": HEAD_DROPOUT,
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
        },
        "training": {
            "epochs_completed": result.get("epochs_completed", EPOCHS),
            "stopped_early": result.get("stopped_early", False),
            "time_minutes": elapsed / 60,
        },
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
    print(f"head_dropout: {HEAD_DROPOUT} (baseline: 0.0)")
    print(f"Val AUC: {results['val_metrics']['auc']:.4f}")
    print(f"Test AUC: {test_auc:.4f} (baseline: 0.712)")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Prediction spread: {pred_spread:.4f}")
    delta = test_auc - 0.712
    print(f"Delta vs baseline: {delta:+.4f} ({delta/0.712*100:+.1f}%)")


if __name__ == "__main__":
    main()
