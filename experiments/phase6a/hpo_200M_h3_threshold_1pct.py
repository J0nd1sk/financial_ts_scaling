#!/usr/bin/env python3
"""
PHASE6A Experiment: 200M parameters, threshold_1pct task, 3-day horizon
Type: HPO (Hyperparameter Optimization)
Generated: 2025-12-11

CRITICAL: This script uses ChunkSplitter for proper data splits.
HPO optimizes val_loss, NOT train_loss.
"""
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
from src.data.dataset import ChunkSplitter, SplitIndices
from src.experiments.runner import update_experiment_log
from src.training.hpo import load_search_space, create_study, create_objective
from src.training.thermal import ThermalCallback

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENT = "phase6a_200M_h3_threshold_1pct"
PHASE = "phase6a"
BUDGET = "200M"
TASK = "threshold_1pct"
HORIZON = 3
TIMESCALE = "daily"
DATA_PATH = "data/processed/v1/SPY_dataset_a25.parquet"
CONFIG_PATH = "configs/experiments/threshold_1pct.yaml"

N_TRIALS = 50
TIMEOUT_HOURS = None
SEARCH_SPACE_PATH = "configs/hpo/default_search.yaml"

CONTEXT_LENGTH = 60
VAL_RATIO = 0.15
TEST_RATIO = 0.15
HPO_TRAIN_FRACTION = 0.3
SEED = 42

# ============================================================
# DATA VALIDATION AND SPLIT CREATION
# ============================================================

def validate_and_create_splits() -> tuple[pd.DataFrame, SplitIndices]:
    logger = logging.getLogger(__name__)
    df = pd.read_parquet(PROJECT_ROOT / DATA_PATH)
    total_days = len(df)
    assert total_days > 1000, f"Insufficient data: {total_days} rows"
    assert not df.isna().any().any(), "NaN values in data"
    logger.info(f"✓ Data loaded: {total_days} rows")

    splitter = ChunkSplitter(
        total_days=total_days,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )
    splits = splitter.split()
    logger.info(f"✓ Splits: train={len(splits.train_indices)}, val={len(splits.val_indices)}, test={len(splits.test_indices)}")

    hpo_train_indices = splitter.get_hpo_subset(splits, fraction=HPO_TRAIN_FRACTION)
    hpo_splits = SplitIndices(
        train_indices=hpo_train_indices,
        val_indices=splits.val_indices,
        test_indices=splits.test_indices,
        chunk_size=splits.chunk_size,
    )
    logger.info(f"✓ HPO subset: {len(hpo_train_indices)} train samples ({HPO_TRAIN_FRACTION*100:.0f}%)")
    return df, hpo_splits

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    start_time = time.time()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== {EXPERIMENT} ===")
    logger.info(f"Device: {device}, Budget: {BUDGET}, Horizon: {HORIZON}d, N_TRIALS: {N_TRIALS}")

    thermal = ThermalCallback()
    thermal_status = thermal.check()
    logger.info(f"Thermal: {thermal_status.status} ({thermal_status.temperature}°C)")

    if thermal_status.status == "critical":
        logger.error("THERMAL ABORT")
        result = {"status": "thermal_abort", "val_loss": None, "error_message": f"Thermal: {thermal_status.temperature}°C", "duration_seconds": time.time() - start_time}
    else:
        df, hpo_splits = validate_and_create_splits()
        search_config = load_search_space(PROJECT_ROOT / SEARCH_SPACE_PATH)
        search_space = search_config.get("search_space", {})

        study = create_study(experiment_name=EXPERIMENT, budget=BUDGET, direction="minimize")
        objective = create_objective(config_path=str(PROJECT_ROOT / CONFIG_PATH), budget=BUDGET, search_space=search_space, split_indices=hpo_splits)

        logger.info(f"Starting HPO ({N_TRIALS} trials)...")
        try:
            study.optimize(objective, n_trials=N_TRIALS, timeout=TIMEOUT_HOURS * 3600 if TIMEOUT_HOURS else None, show_progress_bar=False)

            output_dir = PROJECT_ROOT / "outputs" / "hpo" / EXPERIMENT
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / "best_params.json", "w") as f:
                json.dump({"experiment": EXPERIMENT, "budget": BUDGET, "horizon": HORIZON, "best_params": study.best_params, "best_value": study.best_value, "n_trials": len(study.trials), "timestamp": datetime.now(timezone.utc).isoformat()}, f, indent=2)

            result = {"status": "success", "val_loss": study.best_value, "hyperparameters": study.best_params, "duration_seconds": time.time() - start_time}
        except Exception as e:
            logger.exception(f"HPO failed: {e}")
            result = {"status": "failed", "val_loss": None, "error_message": str(e), "duration_seconds": time.time() - start_time}

    result.update({"timestamp": datetime.now(timezone.utc).isoformat(), "experiment": EXPERIMENT, "phase": PHASE, "budget": BUDGET, "task": TASK, "horizon": HORIZON, "timescale": TIMESCALE, "script_path": str(Path(__file__).relative_to(PROJECT_ROOT)), "run_type": "hpo", "thermal_max_temp": thermal_status.temperature})
    update_experiment_log(result, PROJECT_ROOT / "docs" / "experiment_results.csv")

    logger.info(f"=== Complete: {result['status']}, {result['duration_seconds']:.1f}s ===")
    if result.get('val_loss'): logger.info(f"Best val_loss: {result['val_loss']:.6f}")
    if result.get('hyperparameters'): logger.info(f"Best params: {result['hyperparameters']}")
