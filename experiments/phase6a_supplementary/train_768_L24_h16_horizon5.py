#!/usr/bin/env python3
"""
Supplementary Training Experiment: 768_L24_h16_horizon5

Architecture:
  d_model: 768
  n_layers: 24
  n_heads: 16
  d_ff: 3072

Horizon: 5-day prediction
Task: threshold_1pct (binary classification: >1% move)

Rationale: Cross-horizon validation: h3 winner (0.3081) on h5

Training params borrowed from best-performing trial with similar architecture.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import pandas as pd
import numpy as np

from src.models.arch_grid import estimate_param_count
from src.data.dataset import TimeSeriesDataset, ChunkSplitter
from src.training.trainer import train_model
from src.training.thermal import ThermalMonitor, ThermalCallback

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "768_L24_h16_horizon5"
HORIZON = 5
TASK = "threshold_1pct"

# Architecture (fixed)
D_MODEL = 768
N_LAYERS = 24
N_HEADS = 16
D_FF = 3072

# Training hyperparameters (from best trial)
LEARNING_RATE = 0.000112
BATCH_SIZE = 256
EPOCHS = 75
WEIGHT_DECAY = 0.000155
WARMUP_STEPS = 500

# Data settings
DATA_PATH = "data/processed/SPY_features_a20.parquet"
SEQUENCE_LENGTH = 60
NUM_FEATURES = 25
FEATURE_COLUMNS = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
    'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'atr_14', 'obv', 'vwap', 'roc_10', 'willr_14', 'cci_20', 'stoch_k'
]

# Output paths
OUTPUT_DIR = Path("outputs/supplementary") / EXPERIMENT_NAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# THERMAL MONITORING
# ============================================================================

thermal_monitor = ThermalMonitor()
thermal_callback = ThermalCallback(
    warning_threshold=85,
    critical_threshold=95,
    pause_duration=60,
)

def check_thermal():
    """Check thermal status before/during training."""
    temp = thermal_monitor.get_temperature()
    if temp and temp > 95:
        print(f"🚨 CRITICAL: Temperature {temp}°C - aborting")
        return False
    elif temp and temp > 85:
        print(f"⚠️  WARNING: Temperature {temp}°C - will pause if needed")
    return True

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print(f"SUPPLEMENTARY EXPERIMENT: {EXPERIMENT_NAME}")
    print("=" * 70)
    
    # Pre-flight checks
    if not check_thermal():
        return {"status": "aborted", "reason": "thermal"}
    
    if not torch.backends.mps.is_available():
        print("⚠️  MPS not available, using CPU")
    
    # Calculate actual param count
    param_count = estimate_param_count(D_MODEL, N_LAYERS, N_HEADS, D_FF, NUM_FEATURES)
    print(f"\nArchitecture: d={D_MODEL}, L={N_LAYERS}, h={N_HEADS}, d_ff={D_FF}")
    print(f"Estimated parameters: {param_count:,} ({param_count/1e6:.1f}M)")
    print(f"Horizon: {HORIZON}-day prediction")
    
    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    
    # Filter to numeric feature columns only
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    df_features = df[feature_cols].copy()
    
    # Create dataset
    dataset = TimeSeriesDataset(
        data=df_features,
        sequence_length=SEQUENCE_LENGTH,
        horizon=HORIZON,
        task=TASK,
        threshold=0.01,
    )
    print(f"Dataset size: {len(dataset)} samples")
    
    # Create train/val/test splits
    splitter = ChunkSplitter(
        total_samples=len(dataset),
        chunk_size=20,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )
    splits = splitter.get_splits()
    print(f"Train: {len(splits.train_indices)}, Val: {len(splits.val_indices)}, Test: {len(splits.test_indices)}")
    
    # Build model config
    model_config = {
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "d_ff": D_FF,
        "num_features": NUM_FEATURES,
        "seq_len": SEQUENCE_LENGTH,
        "pred_len": HORIZON,
        "dropout": 0.1,
    }
    
    # Training config
    train_config = {
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
    }
    
    print(f"\nTraining config: {train_config}")
    
    # Train
    start_time = time.time()
    print(f"\nStarting training at {datetime.now().strftime('%H:%M:%S')}...")
    
    result = train_model(
        dataset=dataset,
        model_config=model_config,
        train_config=train_config,
        split_indices=splits,
        callbacks=[thermal_callback],
    )
    
    duration = time.time() - start_time
    
    # Save results
    result_data = {
        "experiment": EXPERIMENT_NAME,
        "horizon": HORIZON,
        "task": TASK,
        "architecture": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
            "param_count": param_count,
        },
        "training_params": train_config,
        "results": {
            "train_loss": result.get("train_loss"),
            "val_loss": result.get("val_loss"),
            "train_accuracy": result.get("train_accuracy"),
            "val_accuracy": result.get("val_accuracy"),
        },
        "duration_seconds": duration,
        "timestamp": datetime.now().isoformat(),
    }
    
    output_file = OUTPUT_DIR / f"{EXPERIMENT_NAME}_result.json"
    with open(output_file, "w") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Val Loss: {result.get('val_loss', 'N/A')}")
    print(f"Val Accuracy: {result.get('val_accuracy', 'N/A')}")
    print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
    print(f"Results saved to: {output_file}")
    
    return result_data

if __name__ == "__main__":
    main()
