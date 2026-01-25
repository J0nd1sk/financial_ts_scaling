#!/usr/bin/env python3
"""
Phase 6C Stage 2: Horizon Experiment - 200M parameters, horizon=3

Tests if 55 features help more at longer horizons.

Architecture (inherited from Phase 6A 200M):
    d_model=256, n_layers=8, n_heads=8, d_ff=1024

Baseline (Phase 6A a20 H3): AUC=0.622
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

EXPERIMENT_NAME = "s2_horizon_200m_h3_a50"
BUDGET = "200M"
HORIZON = 3

# Architecture (inherited from Phase 6A 200M)
D_MODEL = 256
N_LAYERS = 8
N_HEADS = 8
D_FF = 1024

# Hyperparameters (ablation-validated)
LEARNING_RATE = 1e-4
DROPOUT = 0.5
CONTEXT_LENGTH = 80

# Training
BATCH_SIZE = 128
EPOCHS = 50

# Data - a50 combined (55 features = 5 OHLCV + 50 indicators)
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a50_combined.parquet"
NUM_FEATURES = 50

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/phase6c" / EXPERIMENT_NAME

# Baseline for comparison
BASELINE_AUC = 0.622


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
    print(f"PHASE 6C STAGE 2: {BUDGET} / horizon={HORIZON} / features=55 (a50)")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    high_prices = df["High"].values
    print(f"Data: {len(df)} rows, {len(df.columns)} columns")

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
    print(f"Features: {NUM_FEATURES} (a50 tier)")

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

    # Evaluate on validation
    val_metrics = evaluate_model(trainer.model, trainer.val_dataloader, trainer.device)

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Training time: {elapsed/60:.1f} min")
    print(f"Stopped early: {result.get('stopped_early', False)}")
    print(f"\nValidation ({val_metrics['n_samples']} samples):")
    print(f"  AUC: {val_metrics['auc']:.4f}" if val_metrics['auc'] else "  AUC: N/A")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")
    print(f"  Pred Range: [{val_metrics['pred_min']:.4f}, {val_metrics['pred_max']:.4f}]")
    print(f"  Class Balance: {val_metrics['class_balance']:.3f}")

    # Save results
    results = {
        "experiment": EXPERIMENT_NAME,
        "phase": "6C",
        "stage": "S2",
        "track": "T1_horizon",
        "budget": BUDGET,
        "horizon": HORIZON,
        "feature_tier": "a50",
        "num_features": NUM_FEATURES,
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
        "baseline_auc": BASELINE_AUC,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Comparison baseline
    print(f"\n" + "-" * 70)
    print(f"COMPARISON (Phase 6A a20 H{HORIZON} baseline):")
    print(f"  Baseline AUC: {BASELINE_AUC}")
    if val_metrics['auc']:
        print(f"  Current AUC: {val_metrics['auc']:.3f}")
        delta = (val_metrics['auc'] - BASELINE_AUC) / BASELINE_AUC * 100
        print(f"  Delta: {delta:+.1f}%")
    print("-" * 70)

    return val_metrics["auc"]


if __name__ == "__main__":
    main()
