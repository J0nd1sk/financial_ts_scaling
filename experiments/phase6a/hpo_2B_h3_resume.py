#!/usr/bin/env python3
"""
PHASE6A Experiment: 2B parameters, threshold_1pct task, 3-day horizon - RESUME SCRIPT
Type: HPO (Hyperparameter Optimization) with Architectural Search

This script RESUMES from saved trials:
- Loads trials 0-31 from outputs/hpo/phase6a_2B_h3_threshold_1pct/trials/
- Injects them into the Optuna study
- Continues from trial 32 to completion

Generated: 2026-01-06
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
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
from src.data.dataset import ChunkSplitter
from src.experiments.runner import update_experiment_log
from src.experiments.trial_logger import TrialLogger

# Thermal pause duration in seconds when warning threshold exceeded
THERMAL_PAUSE_SECONDS = 60

# ============================================================
# EXPERIMENT CONFIGURATION (all parameters visible)
# ============================================================

EXPERIMENT = "phase6a_2B_h3_threshold_1pct"
PHASE = "phase6a"
BUDGET = "2B"
TASK = "threshold_1pct"
HORIZON = 3
TIMESCALE = "daily"
DATA_PATH = "data/processed/v1/SPY_dataset_a25.parquet"
FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'dema_9', 'dema_10', 'sma_12', 'dema_20', 'dema_25', 'sma_50', 'dema_90', 'sma_100', 'sma_200', 'rsi_daily', 'rsi_weekly', 'stochrsi_daily', 'stochrsi_weekly', 'macd_line', 'obv', 'adosc', 'atr_14', 'adx_14', 'bb_percent_b', 'vwap_20']

# HPO settings - resuming from trial 32
N_TRIALS_TOTAL = 50
N_TRIALS_LOADED = 32  # Trials 0-31 will be loaded
N_TRIALS_REMAINING = N_TRIALS_TOTAL - N_TRIALS_LOADED  # 18 more trials
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
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hpo" / EXPERIMENT
TRIALS_DIR = OUTPUT_DIR / "trials"

# ============================================================
# LOAD EXISTING TRIALS
# ============================================================

def load_existing_trials():
    """Load completed trials from JSON files."""
    trials = []
    for trial_file in sorted(TRIALS_DIR.glob("trial_*.json")):
        with open(trial_file) as f:
            trial_data = json.load(f)
            trials.append(trial_data)
    print(f"✓ Found {len(trials)} existing trial files")
    return trials

def inject_trials_into_study(study, trial_data_list):
    """Inject loaded trials into Optuna study using add_trial."""
    for trial_data in trial_data_list:
        # Get arch_idx from user_attrs if not in params
        params = trial_data["params"].copy()
        user_attrs = trial_data.get("user_attrs", {})

        if "arch_idx" not in params and "arch_idx" in user_attrs:
            params["arch_idx"] = user_attrs["arch_idx"]

        # Create distributions matching the params we have
        distributions = {
            "learning_rate": optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True),
            "epochs": optuna.distributions.IntDistribution(25, 100),
            "weight_decay": optuna.distributions.FloatDistribution(1e-6, 1e-2, log=True),
            "warmup_steps": optuna.distributions.IntDistribution(100, 500),  # Broader range to accept historical values
            "dropout": optuna.distributions.FloatDistribution(0.1, 0.3),
        }

        # Only add arch_idx distribution if we have arch_idx in params
        if "arch_idx" in params:
            distributions["arch_idx"] = optuna.distributions.IntDistribution(0, len(ARCHITECTURES) - 1)

        # Create FrozenTrial
        frozen_trial = optuna.trial.create_trial(
            params=params,
            distributions=distributions,
            values=[trial_data["value"]],
            user_attrs=user_attrs,
            state=optuna.trial.TrialState.COMPLETE,
        )

        study.add_trial(frozen_trial)

    print(f"✓ Injected {len(trial_data_list)} trials into study")
    print(f"  Current best: val_loss={study.best_value:.6f}")

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
# THERMAL MONITORING
# ============================================================

def thermal_check_callback(study, trial):
    """Check thermal status between trials."""
    global stopped_early, stop_reason

    status = thermal_callback.check()

    if status.status == "critical":
        stopped_early = True
        stop_reason = "thermal"
        print(f"\n THERMAL CRITICAL: {status.message}")
        study.stop()
    elif status.status == "warning":
        print(f"\n⚠️ THERMAL WARNING: {status.message}")
        print(f"   Pausing {THERMAL_PAUSE_SECONDS}s to cool down...")
        time.sleep(THERMAL_PAUSE_SECONDS)

# ============================================================
# INCREMENTAL LOGGING CALLBACK
# ============================================================

def incremental_logging_callback(study, trial):
    """Save trial results incrementally after each trial completes."""
    save_trial_result(
        trial=trial,
        output_dir=OUTPUT_DIR,
        architectures=ARCHITECTURES,
    )
    update_best_params(
        study=study,
        experiment_name=EXPERIMENT,
        budget=BUDGET,
        output_dir=OUTPUT_DIR,
        architectures=ARCHITECTURES,
    )
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

    print("=" * 60)
    print("RESUME MODE: Loading existing trials and continuing")
    print(f"Experiment: {EXPERIMENT}")
    print(f"Horizon: {HORIZON}-day")
    print("=" * 60)

    # Load existing trials
    existing_trials = load_existing_trials()
    if len(existing_trials) < N_TRIALS_LOADED:
        print(f"⚠️ Warning: Expected {N_TRIALS_LOADED} trials, found {len(existing_trials)}")
        N_TRIALS_REMAINING = N_TRIALS_TOTAL - len(existing_trials)
        print(f"  Adjusted: will run {N_TRIALS_REMAINING} more trials")

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

    # Inject existing trials
    inject_trials_into_study(study, existing_trials)

    # Create architectural objective
    objective = create_architectural_objective(
        config_path=str(PROJECT_ROOT / CONFIG_PATH),
        budget=BUDGET,
        architectures=ARCHITECTURES,
        training_search_space=training_search_space,
        split_indices=split_indices,
        num_features=len(FEATURE_COLUMNS),
        verbose=True,
    )

    # Run optimization for remaining trials
    print(f"\nResuming HPO: {N_TRIALS_REMAINING} more trials (starting from {len(existing_trials)})")
    print(f"  Output dir: {OUTPUT_DIR}")
    study.optimize(
        objective,
        n_trials=N_TRIALS_REMAINING,
        timeout=TIMEOUT_HOURS * 3600 if TIMEOUT_HOURS else None,
        callbacks=[thermal_check_callback, incremental_logging_callback],
    )

    # Save final best params
    output_path = save_best_params(
        study=study,
        experiment_name=EXPERIMENT,
        budget=BUDGET,
        output_dir=OUTPUT_DIR,
        architectures=ARCHITECTURES,
    )

    duration = time.time() - start_time

    # Generate comprehensive study summary
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
    best_trial = study.best_trial
    if "architecture" in best_trial.user_attrs:
        best_arch = best_trial.user_attrs["architecture"]
    else:
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
    print(f"  Total trials: {len(study.trials)} ({len(existing_trials)} loaded + {N_TRIALS_REMAINING} new)")
    print(f"  Best val_loss: {study.best_value:.6f}")
    print(f"  Best arch: d_model={best_arch['d_model']}, n_layers={best_arch['n_layers']}, params={best_arch['param_count']:,}")
    print(f"  Results saved to: {output_path}")
