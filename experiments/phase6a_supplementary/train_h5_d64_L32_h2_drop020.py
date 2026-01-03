#!/usr/bin/env python3
"""
Supplementary Training: h5_d64_L32_h2_drop020
Purpose: Dropout sensitivity - dropout=0.20 on h5
Architecture: d_model=64, n_layers=32, n_heads=2, d_ff=256
Horizon: 5-day prediction
Dropout: 0.20
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import ChunkSplitter
from src.training.trainer import Trainer

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "h5_d64_L32_h2_drop020"
HORIZON = 5

# Architecture (fixed for all supplementary: d=64, L=32)
D_MODEL = 64
N_LAYERS = 32
N_HEADS = 2
D_FF = 256  # 4 * d_model

# Training (h3-optimal baseline)
LEARNING_RATE = 0.001
BATCH_SIZE = 256
EPOCHS = 50
DROPOUT = 0.20

# Data
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a25.parquet"
CONTEXT_LENGTH = 60
NUM_FEATURES = 25

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/supplementary" / EXPERIMENT_NAME

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"SUPPLEMENTARY: {EXPERIMENT_NAME}")
    print("=" * 70)

    # Check MPS
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data to get row count for splitter
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data: {len(df)} rows")

    # Create experiment config
    experiment_config = ExperimentConfig(
        data_path=str(DATA_PATH.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        wandb_project=None,
        mlflow_experiment=None,
    )

    # Create model config
    model_config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=10,
        stride=5,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        head_dropout=0.0,
    )

    print(f"\nArchitecture: d={D_MODEL}, L={N_LAYERS}, h={N_HEADS}, d_ff={D_FF}")
    print(f"Horizon: {HORIZON}-day, Dropout: {DROPOUT}")

    # Create splits
    splitter = ChunkSplitter(
        total_days=len(df),
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    splits = splitter.split()
    print(f"Splits: train={len(splits.train_indices)}, val={len(splits.val_indices)}, test={len(splits.test_indices)}")

    # Create trainer
    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device=device,
        checkpoint_dir=OUTPUT_DIR,
        split_indices=splits,
    )

    # Train
    print(f"\nTraining for {EPOCHS} epochs...")
    start_time = time.time()
    result = trainer.train(verbose=True)
    elapsed = time.time() - start_time

    # Report results
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {EXPERIMENT_NAME}")
    print(f"{'=' * 70}")
    print(f"Train loss: {result['train_loss']:.6f}")
    print(f"Val loss:   {result.get('val_loss', 'N/A')}")
    print(f"Time:       {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if result.get('stopped_early'):
        print(f"Early stop: {result.get('stop_reason')}")

    # Save results
    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump({
            "experiment": EXPERIMENT_NAME,
            "horizon": HORIZON,
            "architecture": {
                "d_model": D_MODEL,
                "n_layers": N_LAYERS,
                "n_heads": N_HEADS,
                "d_ff": D_FF,
                "dropout": DROPOUT,
            },
            "training": {
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
            },
            "results": {
                "train_loss": result["train_loss"],
                "val_loss": result.get("val_loss"),
                "stopped_early": result.get("stopped_early", False),
                "stop_reason": result.get("stop_reason"),
            },
            "runtime_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    return result


if __name__ == "__main__":
    main()
