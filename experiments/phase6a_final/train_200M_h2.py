#!/usr/bin/env python3
"""
6A Final Training: 200M parameters, horizon=2
Type: Final Evaluation (fixed architecture and training params from HPO)
Generated: 2026-01-19T20:53:44.196136+00:00

Architecture (from HPO):
    d_model=768, n_layers=24, n_heads=16, d_ff=3072

Training params (from HPO):
    lr=0.00065, dropout=0.25, weight_decay=0.0003

Data splits (contiguous mode):
    Train: 1993 - Sept 2024
    Val: Oct - Dec 2024 (early stopping)
    Test: 2025 (final evaluation)
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.models.arch_grid import get_memory_safe_batch_config
from src.data.dataset import ChunkSplitter
from src.training.trainer import Trainer
from src.training.thermal import ThermalCallback
from src.experiments.runner import update_experiment_log

# ============================================================
# EXPERIMENT CONFIGURATION (all parameters visible)
# ============================================================

EXPERIMENT = "train_200M_h2"
PHASE = "6A"
BUDGET = "200M"
HORIZON = 2
DATA_PATH = "data/processed/v1/SPY_dataset_a20.parquet"
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'dema_9', 'dema_10', 'sma_12', 'dema_20', 'dema_25', 'sma_50', 'dema_90', 'sma_100', 'sma_200', 'rsi_daily', 'rsi_weekly', 'stochrsi_daily', 'stochrsi_weekly', 'macd_line', 'obv', 'adosc', 'atr_14', 'adx_14', 'bb_percent_b', 'vwap_20']

# Architecture (fixed from HPO)
D_MODEL = 768
N_LAYERS = 24
N_HEADS = 16
D_FF = 3072

# Training params (fixed from HPO)
LEARNING_RATE = 0.00065
DROPOUT = 0.25
WEIGHT_DECAY = 0.0003
WARMUP_STEPS = 200
EPOCHS = 50

# Model params
CONTEXT_LENGTH = 60
PATCH_LENGTH = 16
STRIDE = 8

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data():
    """Validate data file before running experiment."""
    df = pd.read_parquet(PROJECT_ROOT / DATA_PATH)
    assert len(df) > 1000, f"Insufficient data: {len(df)} rows"
    assert all(col in df.columns for col in FEATURE_COLUMNS), "Missing feature columns"
    print(f"✓ Data validated: {len(df)} rows, {len(FEATURE_COLUMNS)} features")
    return df

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    start_time = time.time()

    # Validate data
    df = validate_data()
    high_prices = df["High"].values

    # Create contiguous splits (production-realistic)
    splitter = ChunkSplitter(
        total_days=len(df),
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_ratio=0.15,
        test_ratio=0.15,
        mode="contiguous",
    )
    split_indices = splitter.split()
    print(f"✓ Contiguous splits: train={len(split_indices.train_indices)}, "
          f"val={len(split_indices.val_indices)}, test={len(split_indices.test_indices)}")

    # Create experiment config
    experiment_config = ExperimentConfig(
        data_path=str(PROJECT_ROOT / DATA_PATH),
        task="threshold_1pct",
        timescale="daily",
        horizon=HORIZON,
        context_length=CONTEXT_LENGTH,
    )

    # Create model config (fixed architecture from HPO)
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

    # Get memory-safe batch config
    batch_config = get_memory_safe_batch_config(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
    )
    print(f"✓ Batch config: batch_size={batch_config['micro_batch']}, "
          f"accumulation={batch_config['accumulation_steps']}")

    # Create thermal callback
    thermal_callback = ThermalCallback()
    print("✓ Thermal monitoring enabled")

    # Output directory
    output_dir = PROJECT_ROOT / "outputs" / "final_training" / EXPERIMENT
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=batch_config["micro_batch"],
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        device="mps",
        checkpoint_dir=output_dir,
        thermal_callback=thermal_callback,
        split_indices=split_indices,
        accumulation_steps=batch_config["accumulation_steps"],
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        high_prices=high_prices,
    )

    # Train
    print(f"\nStarting final training: {EXPERIMENT}")
    print(f"  Architecture: d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}")
    print(f"  Training: lr={LEARNING_RATE}, epochs={EPOCHS}, dropout={DROPOUT}")
    result = trainer.train(verbose=True)

    duration = time.time() - start_time

    # Log to experiment CSV
    log_result = {
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "budget": BUDGET,
        "task": "threshold_1pct",
        "horizon": HORIZON,
        "timescale": "daily",
        "script_path": __file__,
        "run_type": "final_training",
        "status": "success" if result.get("val_loss") is not None else "completed",
        "val_loss": result.get("val_loss"),
        "duration_seconds": duration,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "d_ff": D_FF,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "epochs": EPOCHS,
        },
    }
    update_experiment_log(log_result, PROJECT_ROOT / "docs" / "experiment_results.csv")

    print(f"\n✓ Final training complete in {duration/60:.1f} min")
    print(f"  Val loss: {result.get('val_loss', 'N/A')}")
    print(f"  Stopped early: {result.get('stopped_early', False)}")
    print(f"  Checkpoint: {output_dir / 'best_checkpoint.pt'}")
