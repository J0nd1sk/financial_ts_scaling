#!/usr/bin/env python3
"""
PHASE6A Experiment: 2M parameters, threshold_1pct task
Type: HPO (Hyperparameter Optimization) with Architectural Search
Generated: 2026-01-20T18:13:43.761751+00:00

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
    save_trial_result,
    update_best_params,
    save_all_trials,
)
from src.training.thermal import ThermalCallback
from src.data.dataset import SimpleSplitter
from src.experiments.runner import update_experiment_log
from src.experiments.trial_logger import TrialLogger

# Thermal pause duration in seconds when warning threshold exceeded
THERMAL_PAUSE_SECONDS = 60

# ============================================================
# EXPERIMENT CONFIGURATION (all parameters visible)
# ============================================================

EXPERIMENT = "phase6a_2M_h3_threshold_1pct"
PHASE = "phase6a"
BUDGET = "2M"
TASK = "threshold_1pct"
HORIZON = 3
TIMESCALE = "daily"
DATA_PATH = "data/processed/v1/SPY_dataset_a20.parquet"
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'dema_9', 'dema_10', 'sma_12', 'dema_20', 'dema_25', 'sma_50', 'dema_90', 'sma_100', 'sma_200', 'rsi_daily', 'rsi_weekly', 'stochrsi_daily', 'stochrsi_weekly', 'macd_line', 'obv', 'adosc', 'atr_14', 'adx_14', 'bb_percent_b', 'vwap_20']

# HPO settings
N_TRIALS = 50
TIMEOUT_HOURS = None  # No timeout - experiments run to completion
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
    """Get train/val/test split indices using SimpleSplitter.

    Uses date-based contiguous splits:
    - Train: before 2023-01-01
    - Val: 2023-01-01 to 2024-12-31
    - Test: 2025-01-01 onwards
    """
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=60,  # PatchTST context window
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    return splitter.split()

# ============================================================
# OUTPUT DIRECTORY (defined early for incremental logging)
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hpo" / EXPERIMENT

# ============================================================
# THERMAL MONITORING
# ============================================================

def thermal_check_callback(study, trial):
    """Check thermal status between trials.

    Pauses on warning, stops on critical.
    """
    global stopped_early, stop_reason

    status = thermal_callback.check()

    if status.status == "critical":
        stopped_early = True
        stop_reason = "thermal"
        print(f"\n🚨 THERMAL CRITICAL: {status.message}")
        study.stop()
    elif status.status == "warning":
        print(f"\n⚠️ THERMAL WARNING: {status.message}")
        print(f"   Pausing {THERMAL_PAUSE_SECONDS}s to cool down...")
        time.sleep(THERMAL_PAUSE_SECONDS)

# ============================================================
# INCREMENTAL LOGGING CALLBACK
# ============================================================

def incremental_logging_callback(study, trial):
    """Save trial results incrementally after each trial completes.

    Saves:
    - Individual trial JSON (trials/trial_NNNN.json)
    - Updated best params (experiment_budget_best.json)
    - Updated all trials summary (experiment_all_trials.json)
    """
    # Save individual trial result
    save_trial_result(
        trial=trial,
        output_dir=OUTPUT_DIR,
        architectures=ARCHITECTURES,
    )

    # Update best params file
    update_best_params(
        study=study,
        experiment_name=EXPERIMENT,
        budget=BUDGET,
        output_dir=OUTPUT_DIR,
        architectures=ARCHITECTURES,
    )

    # Update all trials summary
    save_all_trials(
        study=study,
        experiment_name=EXPERIMENT,
        budget=BUDGET,
        output_dir=OUTPUT_DIR,
        architectures=ARCHITECTURES,
    )

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    start_time = time.time()
    stopped_early = False
    stop_reason = None

    # Validate data and get splits
    df = validate_data()
    split_indices = get_split_indices(df)
    print(f"✓ Splits: train={len(split_indices.train_indices)}, val={len(split_indices.val_indices)}, test={len(split_indices.test_indices)}")

    # Load training search space
    training_search_space = load_training_search_space()
    print(f"✓ Training params: {list(training_search_space.keys())}")

    # Create thermal callback for monitoring
    thermal_callback = ThermalCallback()
    print("✓ Thermal monitoring enabled")

    # Create Optuna study
    study = create_study(
        experiment_name=EXPERIMENT,
        budget=BUDGET,
        direction="minimize",
    )

    # Create architectural objective with verbose logging
    objective = create_architectural_objective(
        config_path=str(PROJECT_ROOT / CONFIG_PATH),
        budget=BUDGET,
        architectures=ARCHITECTURES,
        training_search_space=training_search_space,
        split_indices=split_indices,
        num_features=len(FEATURE_COLUMNS),
        verbose=True,  # Enable detailed per-trial metrics
        use_revin=True,  # RevIN alone is best for non-stationary financial data
    )

    # Run optimization with thermal monitoring and incremental logging
    print(f"\nStarting HPO: {N_TRIALS} trials, {len(ARCHITECTURES)} architectures...")
    print(f"  Output dir: {OUTPUT_DIR}")
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        timeout=TIMEOUT_HOURS * 3600 if TIMEOUT_HOURS else None,
        callbacks=[thermal_check_callback, incremental_logging_callback],
    )

    # Save final best params (already updated incrementally, but ensure final state)
    output_path = save_best_params(
        study=study,
        experiment_name=EXPERIMENT,
        budget=BUDGET,
        output_dir=OUTPUT_DIR,
        architectures=ARCHITECTURES,
    )

    duration = time.time() - start_time

    # Generate comprehensive study summary with all trials, arch analysis, etc.
    trial_logger = TrialLogger(
        output_dir=OUTPUT_DIR,
        experiment_name=EXPERIMENT,
    )
    summary_paths = trial_logger.generate_study_summary(
        study=study,
        architectures=ARCHITECTURES,
        training_search_space=training_search_space,
        split_indices=split_indices,
    )
    print(f"\n✓ Study summary saved:")
    print(f"    JSON: {summary_paths['json']}")
    print(f"    Markdown: {summary_paths['markdown']}")

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
