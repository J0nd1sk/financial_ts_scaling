#!/usr/bin/env python3
"""
Threshold Comparison: 0.5% (balanced classes)
Purpose: Test if balanced classes improve model discrimination
Architecture: d_model=64, n_layers=4, n_heads=4, ctx=80 (optimal from ablation)
"""

import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "threshold_0.5pct"
TASK = "threshold_0.5pct"  # 0.5% threshold - balanced classes (~50% positive)
HORIZON = 1

# Architecture (from context length ablation winner)
D_MODEL = 64
N_LAYERS = 4
N_HEADS = 4
D_FF = 256
CONTEXT_LENGTH = 80  # Optimal from context length ablation

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 50
DROPOUT = 0.20

# Data
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
NUM_FEATURES = 20

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/threshold_comparison" / EXPERIMENT_NAME

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"THRESHOLD COMPARISON: {EXPERIMENT_NAME}")
    print(f"Task: {TASK} (expecting ~50% positive class)")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data: {len(df)} rows")

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

    # Train
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            device=device,
            checkpoint_dir=Path(tmp_dir),
            split_indices=split_indices,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            early_stopping_metric="val_auc",
            use_revin=True,
        )

        print(f"\nTraining for {EPOCHS} epochs...")
        start_time = time.time()
        result = trainer.train(verbose=True)
        elapsed = time.time() - start_time

        print(f"\nTraining completed in {elapsed/60:.1f} minutes")

    # Save results
    results = {
        "experiment_name": EXPERIMENT_NAME,
        "task": TASK,
        "threshold": 0.005,
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
            "epochs": EPOCHS,
        },
        "results": {
            "val_auc": result.get("best_val_auc") or result.get("val_auc"),
            "val_loss": result.get("best_val_loss") or result.get("val_loss"),
            "train_loss": result.get("train_loss"),
            "stopped_early": result.get("stopped_early", False),
            "training_time_minutes": elapsed / 60,
        },
        "splits": {
            "train_samples": len(split_indices.train_indices),
            "val_samples": len(split_indices.val_indices),
            "test_samples": len(split_indices.test_indices),
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
    print(f"Task: {TASK}")
    print(f"Val AUC: {results['results']['val_auc']:.4f}")
    print(f"Val Loss: {results['results']['val_loss']:.4f}")


if __name__ == "__main__":
    main()
