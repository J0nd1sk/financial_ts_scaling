#!/usr/bin/env python3
"""
PHASE6A Experiment: 2M parameters, threshold_1pct task
Type: HPO (Hyperparameter Optimization)
Generated: 2025-12-11T20:33:12.337025+00:00
"""
import logging
import sys
from pathlib import Path

# Configure logging for verbose output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
# Ensure stdout is unbuffered for real-time output
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from src.experiments.runner import run_hpo_experiment, update_experiment_log

# ============================================================
# EXPERIMENT CONFIGURATION (all parameters visible)
# ============================================================

EXPERIMENT = "phase6a_2M_threshold_1pct"
PHASE = "phase6a"
BUDGET = "2M"
TASK = "threshold_1pct"
HORIZON = 1
TIMESCALE = "daily"
DATA_PATH = "data/processed/v1/SPY_dataset_a25.parquet"
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'dema_9', 'dema_10', 'sma_12', 'dema_20', 'dema_25', 'sma_50', 'dema_90', 'sma_100', 'sma_200', 'rsi_daily', 'rsi_weekly', 'stochrsi_daily', 'stochrsi_weekly', 'macd_line', 'obv', 'adosc', 'atr_14', 'adx_14', 'bb_percent_b', 'vwap_20']

# HPO settings
N_TRIALS = 50
TIMEOUT_HOURS = 4.0

# ============================================================
# DATA VALIDATION
# ============================================================

def validate_data():
    """Validate data file before running experiment."""
    df = pd.read_parquet(PROJECT_ROOT / DATA_PATH)
    assert len(df) > 1000, f"Insufficient data: {len(df)} rows"
    assert all(col in df.columns for col in FEATURE_COLUMNS), "Missing feature columns"
    assert not df[FEATURE_COLUMNS].isna().any().any(), "NaN values in features"
    print(f"✓ Data validated: {len(df)} rows, {len(FEATURE_COLUMNS)} features")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    # Show device info
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Phase 6A HPO Experiment ===")
    logger.info(f"Device: {device} (MPS available: {torch.backends.mps.is_available()})")
    logger.info(f"Experiment: {EXPERIMENT}")
    logger.info(f"Budget: {BUDGET}, Task: {TASK}")
    logger.info(f"N_TRIALS: {N_TRIALS}, TIMEOUT: {TIMEOUT_HOURS}h")

    validate_data()

    logger.info("Starting HPO...")
    result = run_hpo_experiment(
        experiment=EXPERIMENT,
        budget=BUDGET,
        task=TASK,
        data_path=PROJECT_ROOT / DATA_PATH,
        output_dir=PROJECT_ROOT / "outputs" / "hpo" / EXPERIMENT,
        n_trials=N_TRIALS,
        timeout_hours=TIMEOUT_HOURS,
    )

    # Log result
    result.update({
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "budget": BUDGET,
        "task": TASK,
        "horizon": HORIZON,
        "timescale": TIMESCALE,
        "script_path": __file__,
        "run_type": "hpo",
    })
    update_experiment_log(result, PROJECT_ROOT / "outputs" / "results" / "experiment_log.csv")

    logger.info(f"=== HPO Complete ===")
    logger.info(f"Status: {result['status']}")
    if result.get('val_loss'):
        logger.info(f"Best val_loss: {result['val_loss']:.6f}")
    if result.get('hyperparameters'):
        logger.info(f"Best params: {result['hyperparameters']}")
