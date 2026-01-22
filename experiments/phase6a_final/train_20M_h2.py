#!/usr/bin/env python3
"""
Phase 6A Training: 20M parameters, horizon=2

Architecture (Option A - PatchTST-standard):
    d_model=128, n_layers=6, n_heads=8, d_ff=512

Hyperparameters (ablation-validated):
    dropout=0.5, lr=1e-4, context=80, RevIN=True

Data splits (SimpleSplitter):
    Train: through 2022
    Val: 2023-2024 (442 samples)
    Test: 2025+
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "phase6a_20m_h2"
BUDGET = "20M"
HORIZON = 2

# Architecture (Option A - PatchTST-standard)
D_MODEL = 128
N_LAYERS = 6
N_HEADS = 8
D_FF = 512

# Hyperparameters (ablation-validated - DO NOT CHANGE without new evidence)
LEARNING_RATE = 1e-4
DROPOUT = 0.5
CONTEXT_LENGTH = 80

# Training
BATCH_SIZE = 128
EPOCHS = 50

# Data
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
NUM_FEATURES = 20

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/phase6a_final" / EXPERIMENT_NAME


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, dataloader, device):
    """Evaluate model with full metrics."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            preds = model(batch_x).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(batch_y.numpy().flatten())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    binary_preds = (preds >= 0.5).astype(int)

    # Handle edge case where only one class in labels
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = None

    return {
        "auc": auc,
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "f1": f1_score(labels, binary_preds, zero_division=0),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "n_positive_preds": int((preds >= 0.5).sum()),
        "n_samples": len(labels),
        "class_balance": float(labels.mean()),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"PHASE 6A: {BUDGET} / horizon={HORIZON}")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    high_prices = df["High"].values
    print(f"Data: {len(df)} rows")

    # Experiment config
    experiment_config = ExperimentConfig(
        data_path=str(DATA_PATH.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        wandb_project=None,
        mlflow_experiment=None,
    )

    # Model config
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

    print(f"\nArchitecture: d={D_MODEL}, L={N_LAYERS}, h={N_HEADS}, d_ff={D_FF}")
    print(f"Training: lr={LEARNING_RATE}, dropout={DROPOUT}, ctx={CONTEXT_LENGTH}")

    # SimpleSplitter for proper validation
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    print(f"\nSplits: train={len(split_indices.train_indices)}, "
          f"val={len(split_indices.val_indices)}, test={len(split_indices.test_indices)}")

    # Trainer with RevIN
    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        checkpoint_dir=OUTPUT_DIR,
        split_indices=split_indices,
        early_stopping_patience=10,
        early_stopping_min_delta=0.001,
        early_stopping_metric="val_auc",
        use_revin=True,
        high_prices=high_prices,
    )

    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    start_time = time.time()
    result = trainer.train(verbose=True)
    elapsed = time.time() - start_time

    # Evaluate on validation (test evaluation requires separate dataloader creation)
    val_metrics = evaluate_model(trainer.model, trainer.val_dataloader, trainer.device)

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Training time: {elapsed/60:.1f} min")
    print(f"Stopped early: {result.get('stopped_early', False)}")
    print(f"\nValidation (2023-2024, {val_metrics['n_samples']} samples):")
    print(f"  AUC: {val_metrics['auc']:.4f}" if val_metrics['auc'] else "  AUC: N/A")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")
    print(f"  Pred Range: [{val_metrics['pred_min']:.4f}, {val_metrics['pred_max']:.4f}]")
    print(f"  Class Balance: {val_metrics['class_balance']:.3f} ({int(val_metrics['class_balance']*val_metrics['n_samples'])} positives)")

    # Save results
    results = {
        "experiment": EXPERIMENT_NAME,
        "budget": BUDGET,
        "horizon": HORIZON,
        "architecture": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
        },
        "hyperparameters": {
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "context_length": CONTEXT_LENGTH,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "use_revin": True,
        },
        "splits": {
            "train": len(split_indices.train_indices),
            "val": len(split_indices.val_indices),
            "test": len(split_indices.test_indices),
        },
        "training": {
            "train_loss": result.get("train_loss"),
            "val_loss": result.get("val_loss"),
            "val_auc": result.get("val_auc"),
            "stopped_early": result.get("stopped_early", False),
            "training_time_min": elapsed / 60,
        },
        "val_metrics": val_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return val_metrics["auc"]


if __name__ == "__main__":
    main()
