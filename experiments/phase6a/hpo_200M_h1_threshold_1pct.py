#!/usr/bin/env python3
"""
PHASE6A Experiment: 200M parameters, threshold_1pct task
Type: HPO (Hyperparameter Optimization) with Architectural Search
Generated: 2025-12-12T21:27:32.056330+00:00

This script searches both model ARCHITECTURE (d_model, n_layers, n_heads, d_ff)
and TRAINING parameters (lr, epochs, batch_size) to find optimal configuration.
"""
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml
from src.models.arch_grid import get_architectures_for_budget
from src.training.hpo import (
    create_architectural_objective,
    create_study,
    save_best_params,
)
from src.data.dataset import ChunkSplitter
from src.experiments.runner import update_experiment_log

# ============================================================
# EXPERIMENT CONFIGURATION (all parameters visible)
# ============================================================

EXPERIMENT = "phase6a_200M_h1_threshold_1pct"
PHASE = "phase6a"
BUDGET = "200M"
TASK = "threshold_1pct"
HORIZON = 1
TIMESCALE = "daily"
DATA_PATH = "data/processed/v1/SPY_dataset_a25.parquet"
FEATURE_COLUMNS = ['dema_9', 'dema_10', 'sma_12', 'dema_20', 'dema_25', 'sma_50', 'dema_90', 'sma_100', 'sma_200', 'rsi_daily', 'rsi_weekly', 'stochrsi_daily', 'stochrsi_weekly', 'macd_line', 'obv', 'adosc', 'atr_14', 'adx_14', 'bb_percent_b', 'vwap_20']

# HPO settings
N_TRIALS = 50
TIMEOUT_HOURS = 4.0
SEARCH_SPACE_PATH = "configs/hpo/architectural_search.yaml"
CONFIG_PATH = f"configs/experiments/{TASK}.yaml"

# ============================================================
# ARCHITECTURE GRID (pre-computed valid architectures for budget)
# ============================================================

ARCHITECTURES = get_architectures_for_budget(
    budget=BUDGET,
    num_features=len(FEATURE_COLUMNS),
)
print(f"✓ Architecture grid: {len(ARCHITECTURES)} valid configs for {BUDGET}")

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
    return df

# ============================================================
# HPO CONFIGURATION
# ============================================================

def load_training_search_space():
    """Load training parameter search space from YAML."""
    config_path = PROJECT_ROOT / SEARCH_SPACE_PATH
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("training_search_space", {})

def get_split_indices(df):
    """Get train/val/test split indices."""
    splitter = ChunkSplitter(
        total_days=len(df),
        context_length=60,  # PatchTST context window
        horizon=HORIZON,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    return splitter.split()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    start_time = time.time()

    # Validate data and get splits
    df = validate_data()
    split_indices = get_split_indices(df)
    print(f"✓ Splits: train={len(split_indices.train_indices)}, val={len(split_indices.val_indices)}, test={len(split_indices.test_indices)}")

    # Load training search space
    training_search_space = load_training_search_space()
    print(f"✓ Training params: {list(training_search_space.keys())}")

    # Create Optuna study
    study = create_study(
        experiment_name=EXPERIMENT,
        budget=BUDGET,
        direction="minimize",
    )

    # Create architectural objective
    objective = create_architectural_objective(
        config_path=str(PROJECT_ROOT / CONFIG_PATH),
        budget=BUDGET,
        architectures=ARCHITECTURES,
        training_search_space=training_search_space,
        split_indices=split_indices,
        num_features=len(FEATURE_COLUMNS),
    )

    # Run optimization
    print(f"\nStarting HPO: {N_TRIALS} trials, {len(ARCHITECTURES)} architectures...")
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT_HOURS * 3600 if TIMEOUT_HOURS else None,
    )

    # Save best params (includes architecture info)
    output_dir = PROJECT_ROOT / "outputs" / "hpo" / EXPERIMENT
    output_path = save_best_params(
        study=study,
        experiment_name=EXPERIMENT,
        budget=BUDGET,
        output_dir=output_dir,
        architectures=ARCHITECTURES,
    )

    duration = time.time() - start_time

    # Get best architecture info
    best_arch = ARCHITECTURES[study.best_params.get("arch_idx", 0)]

    # Prepare result for logging
    result = {
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "budget": BUDGET,
        "task": TASK,
        "horizon": HORIZON,
        "timescale": TIMESCALE,
        "script_path": __file__,
        "run_type": "hpo",
        "status": "success",
        "val_loss": study.best_value,
        "hyperparameters": study.best_params,
        "duration_seconds": duration,
        "d_model": best_arch["d_model"],
        "n_layers": best_arch["n_layers"],
        "n_heads": best_arch["n_heads"],
        "d_ff": best_arch["d_ff"],
        "param_count": best_arch["param_count"],
    }

    # Log to experiment CSV
    update_experiment_log(result, PROJECT_ROOT / "docs" / "experiment_results.csv")

    print(f"\n✓ HPO complete in {duration/60:.1f} min")
    print(f"  Best val_loss: {study.best_value:.6f}")
    print(f"  Best arch: d_model={best_arch['d_model']}, n_layers={best_arch['n_layers']}, params={best_arch['param_count']:,}")
    print(f"  Results saved to: {output_path}")
