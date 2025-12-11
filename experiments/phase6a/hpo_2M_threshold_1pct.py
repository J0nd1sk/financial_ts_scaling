#!/usr/bin/env python3
"""
PHASE6A Experiment: 2M parameters, threshold_1pct task
Type: HPO (Hyperparameter Optimization)
Generated: 2025-12-11 (with proper train/val/test splits)

CRITICAL: This script uses ChunkSplitter for proper data splits.
HPO optimizes val_loss, NOT train_loss.
"""
import json
import logging
import sys
import time
from datetime import datetime, timezone
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
from src.data.dataset import ChunkSplitter, SplitIndices
from src.experiments.runner import update_experiment_log
from src.training.hpo import load_search_space, create_study, create_objective
from src.training.thermal import ThermalCallback

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
CONFIG_PATH = "configs/experiments/threshold_1pct.yaml"

# HPO settings
N_TRIALS = 50
TIMEOUT_HOURS = None  # No timeout - let HPO run to completion
SEARCH_SPACE_PATH = "configs/hpo/default_search.yaml"

# Split settings (MANDATORY for proper val_loss optimization)
CONTEXT_LENGTH = 60
VAL_RATIO = 0.15
TEST_RATIO = 0.15
HPO_TRAIN_FRACTION = 0.3  # Use 30% of train for faster HPO
SEED = 42

# ============================================================
# DATA VALIDATION AND SPLIT CREATION
# ============================================================

def validate_and_create_splits() -> tuple[pd.DataFrame, SplitIndices]:
    """Validate data file and create train/val/test splits."""
    logger = logging.getLogger(__name__)

    df = pd.read_parquet(PROJECT_ROOT / DATA_PATH)
    total_days = len(df)

    # Validate data
    assert total_days > 1000, f"Insufficient data: {total_days} rows"
    assert not df.isna().any().any(), "NaN values in data"

    logger.info(f"✓ Data loaded: {total_days} rows")

    # Create ChunkSplitter for proper splits
    splitter = ChunkSplitter(
        total_days=total_days,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
    )

    splits = splitter.split()

    logger.info(f"✓ Splits created:")
    logger.info(f"  Train samples: {len(splits.train_indices)}")
    logger.info(f"  Val chunks: {len(splits.val_indices)}")
    logger.info(f"  Test chunks: {len(splits.test_indices)}")
    logger.info(f"  Chunk size: {splits.chunk_size}")

    # Create HPO subset (30% of train for faster iteration)
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

    # Show device info
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"=== Phase 6A HPO Experiment ===")
    logger.info(f"Device: {device} (MPS available: {torch.backends.mps.is_available()})")
    logger.info(f"Experiment: {EXPERIMENT}")
    logger.info(f"Budget: {BUDGET}, Task: {TASK}")
    logger.info(f"N_TRIALS: {N_TRIALS}, TIMEOUT: {TIMEOUT_HOURS}h")

    # Pre-flight thermal check
    thermal = ThermalCallback()
    thermal_status = thermal.check()
    logger.info(f"Thermal status: {thermal_status.status} ({thermal_status.temperature}°C)")

    if thermal_status.status == "critical":
        logger.error("THERMAL ABORT: Temperature critical, cannot start HPO")
        result = {
            "status": "thermal_abort",
            "val_loss": None,
            "error_message": f"Thermal abort: {thermal_status.temperature}°C",
            "duration_seconds": time.time() - start_time,
        }
    else:
        # Validate data and create splits
        df, hpo_splits = validate_and_create_splits()

        # Load search space
        search_config = load_search_space(PROJECT_ROOT / SEARCH_SPACE_PATH)
        search_space = search_config.get("search_space", {})
        logger.info(f"Search space: {list(search_space.keys())}")

        # Create study
        study = create_study(
            experiment_name=EXPERIMENT,
            budget=BUDGET,
            direction="minimize",
        )
        logger.info(f"Study created: {study.study_name}")

        # Create objective with split_indices (CRITICAL for val_loss)
        objective = create_objective(
            config_path=str(PROJECT_ROOT / CONFIG_PATH),
            budget=BUDGET,
            search_space=search_space,
            split_indices=hpo_splits,  # MANDATORY
        )
        logger.info("Objective created with val_loss optimization")

        # Run optimization
        timeout_msg = f"{TIMEOUT_HOURS}h timeout" if TIMEOUT_HOURS else "no timeout"
        logger.info(f"Starting HPO ({N_TRIALS} trials, {timeout_msg})...")
        try:
            study.optimize(
                objective,
                n_trials=N_TRIALS,
                timeout=TIMEOUT_HOURS * 3600 if TIMEOUT_HOURS else None,
                show_progress_bar=False,
            )

            # Save best params
            output_dir = PROJECT_ROOT / "outputs" / "hpo" / EXPERIMENT
            output_dir.mkdir(parents=True, exist_ok=True)

            best_params_path = output_dir / "best_params.json"
            with open(best_params_path, "w") as f:
                json.dump({
                    "experiment": EXPERIMENT,
                    "budget": BUDGET,
                    "best_params": study.best_params,
                    "best_value": study.best_value,
                    "n_trials": len(study.trials),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2)

            result = {
                "status": "success",
                "val_loss": study.best_value,
                "hyperparameters": study.best_params,
                "duration_seconds": time.time() - start_time,
            }

        except Exception as e:
            logger.exception(f"HPO failed: {e}")
            result = {
                "status": "failed",
                "val_loss": None,
                "error_message": str(e),
                "duration_seconds": time.time() - start_time,
            }

    # Log result
    result.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT,
        "phase": PHASE,
        "budget": BUDGET,
        "task": TASK,
        "horizon": HORIZON,
        "timescale": TIMESCALE,
        "script_path": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "run_type": "hpo",
        "thermal_max_temp": thermal_status.temperature,
    })
    update_experiment_log(result, PROJECT_ROOT / "docs" / "experiment_results.csv")

    logger.info(f"=== HPO Complete ===")
    logger.info(f"Status: {result['status']}")
    logger.info(f"Duration: {result['duration_seconds']:.1f}s")
    if result.get('val_loss'):
        logger.info(f"Best val_loss: {result['val_loss']:.6f}")
    if result.get('hyperparameters'):
        logger.info(f"Best params: {result['hyperparameters']}")
