#!/usr/bin/env python3
"""Supplementary HPO trials with fixed hyperparameters.

Runs targeted trials to explore gaps in the hyperparameter search space
identified from analysis of initial HPO trials.

Usage:
    ./venv/bin/python scripts/run_supplementary_hpo.py --tier a50
    ./venv/bin/python scripts/run_supplementary_hpo.py --tier a100
    ./venv/bin/python scripts/run_supplementary_hpo.py --tier a50 --trials A1,A2,B1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Project root for imports and data paths
PROJECT_ROOT = Path(__file__).parent.parent

# Type checking imports (not executed at runtime)
if TYPE_CHECKING:
    import pandas as pd
    from src.data.dataset import SimpleSplitter
    from src.models.patchtst import PatchTSTConfig
    from src.config.experiment import ExperimentConfig
    from src.training.trainer import Trainer


# =============================================================================
# a50 Supplementary Configs (27 trials)
# Based on best: d_model=128, n_layers=6, dropout=0.3, lr=5e-5, wd=1e-4
# =============================================================================

SUPPLEMENTARY_CONFIGS_A50 = {
    # Phase 1: Dropout exploration around 0.3 best (6 trials) - A1-A6
    "A1": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.35, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "A2": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.40, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "A3": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.45, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "A4": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.50, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "A5": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.55, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "A6": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.60, "learning_rate": 5e-5, "weight_decay": 1e-4},

    # Phase 1: LR fine-tuning around 5e-5 best (5 trials) - B1-B5
    "B1": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 7e-5, "weight_decay": 1e-4},
    "B2": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 8e-5, "weight_decay": 1e-4},
    "B3": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 1e-4, "weight_decay": 1e-4},
    "B4": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 1.2e-4, "weight_decay": 1e-4},
    "B5": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 1.5e-4, "weight_decay": 1e-4},

    # Phase 1: Weight decay around 1e-4 best (4 trials) - C1-C4
    "C1": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 3e-4},
    "C2": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 5e-4},
    "C3": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-3},
    "C4": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 2e-3},

    # Phase 2: Shallow architecture variants (4 trials) - D1-D4
    "D1": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "D2": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.4, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "D3": {"d_model": 96, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "D4": {"d_model": 96, "n_layers": 5, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-4},

    # Phase 2: Deep architecture variants (4 trials) - E1-E4
    "E1": {"d_model": 128, "n_layers": 7, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "E2": {"d_model": 128, "n_layers": 8, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "E3": {"d_model": 128, "n_layers": 7, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.4, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "E4": {"d_model": 160, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.3, "learning_rate": 5e-5, "weight_decay": 1e-4},

    # Phase 3: Combined optimization (4 trials) - F1-F4
    "F1": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.35, "learning_rate": 7e-5, "weight_decay": 3e-4},
    "F2": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.40, "learning_rate": 8e-5, "weight_decay": 5e-4},
    "F3": {"d_model": 128, "n_layers": 5, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.35, "learning_rate": 7e-5, "weight_decay": 1e-4},
    "F4": {"d_model": 128, "n_layers": 7, "n_heads": 8, "d_ff_ratio": 4, "dropout": 0.35, "learning_rate": 5e-5, "weight_decay": 3e-4},
}

# =============================================================================
# a100 Supplementary Configs (24 trials)
# Based on best: d_model=64, n_layers=4, dropout=0.7, lr=5e-4, wd=1e-3
# Note: a100 optimal region is VERY different from a50 - needs MORE regularization
# =============================================================================

SUPPLEMENTARY_CONFIGS_A100 = {
    # Phase 1: Very high dropout beyond current 0.7 max (4 trials) - A1-A4
    "A1": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.75, "learning_rate": 5e-4, "weight_decay": 1e-3},
    "A2": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.80, "learning_rate": 5e-4, "weight_decay": 1e-3},
    "A3": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.85, "learning_rate": 5e-4, "weight_decay": 1e-3},
    "A4": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.90, "learning_rate": 5e-4, "weight_decay": 1e-3},

    # Phase 1: Moderate dropout range (gap between 0.3 and 0.7) (4 trials) - B1-B4
    "B1": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.45, "learning_rate": 5e-4, "weight_decay": 1e-3},
    "B2": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.50, "learning_rate": 5e-4, "weight_decay": 1e-3},
    "B3": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.55, "learning_rate": 5e-4, "weight_decay": 1e-3},
    "B4": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.60, "learning_rate": 5e-4, "weight_decay": 1e-3},

    # Phase 2: LR around 5e-4 optimum (4 trials) - C1-C4
    "C1": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 3e-4, "weight_decay": 1e-3},
    "C2": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 4e-4, "weight_decay": 1e-3},
    "C3": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 6e-4, "weight_decay": 1e-3},
    "C4": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 8e-4, "weight_decay": 1e-3},

    # Phase 2: Weight decay around 1e-3 (4 trials) - D1-D4
    "D1": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 5e-4, "weight_decay": 5e-4},
    "D2": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 5e-4, "weight_decay": 2e-3},
    "D3": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 5e-4, "weight_decay": 3e-3},
    "D4": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 5e-4, "weight_decay": 5e-3},

    # Phase 3: Tiny architecture variants (4 trials) - E1-E4
    "E1": {"d_model": 48, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 5e-4, "weight_decay": 1e-3},
    "E2": {"d_model": 48, "n_layers": 3, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 5e-4, "weight_decay": 1e-3},
    "E3": {"d_model": 64, "n_layers": 3, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 5e-4, "weight_decay": 1e-3},
    "E4": {"d_model": 80, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.7, "learning_rate": 5e-4, "weight_decay": 1e-3},

    # Phase 3: Combined optimization (4 trials) - F1-F4
    "F1": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.75, "learning_rate": 4e-4, "weight_decay": 2e-3},
    "F2": {"d_model": 64, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.80, "learning_rate": 3e-4, "weight_decay": 2e-3},
    "F3": {"d_model": 48, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.75, "learning_rate": 5e-4, "weight_decay": 2e-3},
    "F4": {"d_model": 64, "n_layers": 3, "n_heads": 8, "d_ff_ratio": 2, "dropout": 0.80, "learning_rate": 4e-4, "weight_decay": 1e-3},
}


# =============================================================================
# Follow-Up a50 Configs - Round 2 (13 trials)
# Based on Round 1 findings:
#   - D2 (n_layers=4, dropout=0.4): 27.6% recall (2x better than any other)
#   - E4 (d_model=160): Best balance - 22.4% recall, 51.5% precision
#   - F1 (lr=7e-5, wd=3e-4): 56.3% precision
# =============================================================================

FOLLOWUP_CONFIGS_A50 = {
    # Direction G: Maximize Recall (D2-style optimization)
    # D2 achieved 27.6% recall with n_layers=4, dropout=0.4
    "G1": {"d_model": 160, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.40, "learning_rate": 5e-5, "weight_decay": 1e-4},  # D2 + wider
    "G2": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.45, "learning_rate": 5e-5, "weight_decay": 1e-4},  # D2 + more dropout
    "G3": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.40, "learning_rate": 7e-5, "weight_decay": 1e-4},  # D2 + faster LR
    "G4": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.40, "learning_rate": 5e-5, "weight_decay": 3e-4},  # D2 + more WD
    "G5": {"d_model": 128, "n_layers": 3, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.40, "learning_rate": 5e-5, "weight_decay": 1e-4},  # Even shallower

    # Direction H: Maximize Precision (F1-style optimization)
    # F1 achieved 56.3% precision with dropout=0.35, lr=7e-5, wd=3e-4
    "H1": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.35, "learning_rate": 7e-5, "weight_decay": 5e-4},  # F1 + more WD
    "H2": {"d_model": 128, "n_layers": 5, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.35, "learning_rate": 7e-5, "weight_decay": 3e-4},  # F1 + shallower
    "H3": {"d_model": 160, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.35, "learning_rate": 7e-5, "weight_decay": 3e-4},  # F1 + wider
    "H4": {"d_model": 128, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.35, "learning_rate": 7e-5, "weight_decay": 3e-4},  # F1 + much shallower

    # Direction I: Best Balance (E4-style optimization)
    # E4 achieved 51.5% precision, 22.4% recall with d_model=160
    "I1": {"d_model": 160, "n_layers": 5, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.35, "learning_rate": 5e-5, "weight_decay": 1e-4},  # E4 + shallower
    "I2": {"d_model": 192, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.30, "learning_rate": 5e-5, "weight_decay": 1e-4},  # E4 + wider
    "I3": {"d_model": 160, "n_layers": 4, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.35, "learning_rate": 5e-5, "weight_decay": 1e-4},  # E4 + much shallower
    "I4": {"d_model": 160, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.40, "learning_rate": 5e-5, "weight_decay": 1e-4},  # E4 + more dropout
}


def get_configs_for_tier(tier: str, round_num: int = 1) -> dict:
    """Get supplementary configs for a given tier and round.

    Args:
        tier: Feature tier ("a50" or "a100")
        round_num: Round number (1 for original A-F configs, 2 for follow-up G-I configs)

    Returns:
        Dictionary of config_name -> config dict

    Raises:
        ValueError: If tier is not recognized or round 2 requested for a100
    """
    if round_num == 2:
        if tier == "a50":
            return FOLLOWUP_CONFIGS_A50
        else:
            raise ValueError("Round 2 only available for a50 tier")

    if tier == "a50":
        return SUPPLEMENTARY_CONFIGS_A50
    elif tier == "a100":
        return SUPPLEMENTARY_CONFIGS_A100
    else:
        raise ValueError(f"Unknown tier: {tier}. Must be 'a50' or 'a100'")


def get_thermal_reading() -> float | None:
    """Get current CPU temperature using powermetrics (requires prior sudo auth).

    Returns:
        Temperature in Celsius, or None if unable to read.

    Note:
        This uses sudo powermetrics. To avoid password prompts during runs,
        either run `sudo -v` before starting, or the thermal check will
        silently return None and continue without thermal monitoring.
    """
    try:
        # Use -n flag to avoid password prompt - will fail silently if no cached credentials
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "smc", "-i", "1", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.split("\n"):
            if "CPU die temperature" in line:
                temp_str = line.split(":")[1].strip().replace(" C", "")
                return float(temp_str)
    except Exception:
        pass
    return None


def train_single_config(
    config_name: str,
    config: dict,
    tier: str,
    output_dir: Path,
    budget: str = "20M",
    horizon: int = 1,
) -> dict:
    """Train a single config and return full metrics.

    Args:
        config_name: Name of the config (e.g., "A1")
        config: Hyperparameter dictionary
        tier: Feature tier (e.g., "a50")
        output_dir: Directory to save results
        budget: Parameter budget (default "20M")
        horizon: Prediction horizon in days (default 1)

    Returns:
        Dictionary with all metrics and metadata
    """
    import tempfile

    import pandas as pd
    import torch

    # Lazy imports to avoid import errors in tests
    # Add project root so 'src.X' imports work
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.data.dataset import SimpleSplitter
    from src.models.patchtst import PatchTSTConfig
    from src.config.experiment import ExperimentConfig
    from src.training.trainer import Trainer

    # Feature counts by tier
    NUM_FEATURES_BY_TIER = {
        "a20": 25,
        "a50": 55,
        "a100": 105,
        "a200": 211,
    }

    # Fixed parameters
    CONTEXT_LENGTH = 80
    EPOCHS = 50

    start_time = time.time()

    # Load dataset
    data_path = PROJECT_ROOT / f"data/processed/v1/SPY_dataset_{tier}_combined.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_parquet(data_path)
    high_prices = df["High"].values
    num_features = NUM_FEATURES_BY_TIER.get(tier, 105)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Determine batch size based on model size
    d_model = config["d_model"]
    if d_model >= 256:
        batch_size = 32
    elif d_model >= 128:
        batch_size = 64
    else:
        batch_size = 128

    # Build experiment config
    exp_config = ExperimentConfig(
        data_path=str(data_path.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
        wandb_project=None,
        mlflow_experiment=None,
    )

    # Build model config
    model_config = PatchTSTConfig(
        num_features=num_features,
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        d_ff=config["d_model"] * config["d_ff_ratio"],
        dropout=config["dropout"],
        head_dropout=0.0,
    )

    # Create splitter
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=horizon,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    # Create temp directory for checkpoints
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = Trainer(
            experiment_config=exp_config,
            model_config=model_config,
            batch_size=batch_size,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            epochs=EPOCHS,
            device=device,
            checkpoint_dir=Path(tmp_dir),
            split_indices=split_indices,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            early_stopping_metric="val_auc",
            use_revin=True,
            high_prices=high_prices,
        )

        # Train with verbose=True to get all metrics
        result = trainer.train(verbose=True)

    duration = time.time() - start_time

    # Build result dict
    metrics = {
        "config_name": config_name,
        "tier": tier,
        "budget": budget,
        "horizon": horizon,
        "params": config,
        "duration_sec": duration,
        "epochs_run": result.get("epochs_run", EPOCHS),
        "stopped_early": result.get("stopped_early", False),
        # Validation metrics
        "val_auc": result.get("val_auc"),
        "val_accuracy": result.get("val_accuracy"),
        "val_precision": result.get("val_precision"),
        "val_recall": result.get("val_recall"),
        "val_pred_range": result.get("val_pred_range"),
        "val_confusion": result.get("val_confusion"),
        # Training metrics
        "train_auc": result.get("train_auc"),
        "train_precision": result.get("train_precision"),
        "train_recall": result.get("train_recall"),
        "train_pred_range": result.get("train_pred_range"),
        "timestamp": datetime.now().isoformat(),
    }

    return metrics


def run_supplementary_trials(
    tier: str,
    trial_names: list[str] | None = None,
    output_base: Path | None = None,
    thermal_pause_threshold: float = 85.0,
    round_num: int = 1,
) -> dict:
    """Run supplementary HPO trials for a tier.

    Args:
        tier: Feature tier ("a50" or "a100")
        trial_names: Optional list of specific trials to run (e.g., ["A1", "A2"])
        output_base: Base output directory (default: outputs/phase6c_{tier}/supplementary_trials)
        thermal_pause_threshold: Pause if temperature exceeds this (Celsius)
        round_num: Round number (1 for original A-F configs, 2 for follow-up G-I configs)

    Returns:
        Dictionary with all results
    """
    configs = get_configs_for_tier(tier, round_num=round_num)

    # Filter to specific trials if requested
    if trial_names:
        configs = {k: v for k, v in configs.items() if k in trial_names}
        if not configs:
            raise ValueError(f"No matching trials found. Available: {list(get_configs_for_tier(tier).keys())}")

    # Setup output directory
    if output_base is None:
        if round_num == 2:
            output_base = PROJECT_ROOT / f"outputs/phase6c_{tier}/supplementary_trials_round2"
        else:
            output_base = PROJECT_ROOT / f"outputs/phase6c_{tier}/supplementary_trials"
    output_base.mkdir(parents=True, exist_ok=True)

    round_label = f" (Round {round_num})" if round_num > 1 else ""
    print(f"\n{'='*60}")
    print(f"Supplementary HPO Trials - Tier {tier}{round_label}")
    print(f"{'='*60}")
    print(f"Total trials: {len(configs)}")
    print(f"Output: {output_base}")
    print(f"Thermal threshold: {thermal_pause_threshold}°C")
    print(f"{'='*60}\n")

    all_results = []
    results_file = output_base / "all_results.json"

    for i, (name, config) in enumerate(configs.items(), 1):
        print(f"\n[{i}/{len(configs)}] Running trial {name}...")
        print(f"  Config: {config}")

        # Thermal check between trials
        temp = get_thermal_reading()
        if temp is not None:
            print(f"  Temperature: {temp:.1f}°C")
            if temp > thermal_pause_threshold:
                print(f"  ⚠️  Temperature above {thermal_pause_threshold}°C, pausing 60s...")
                time.sleep(60)

        try:
            result = train_single_config(name, config, tier, output_base)
            all_results.append(result)

            # Report key metrics
            print(f"  ✓ AUC: {result['val_auc']:.4f}")
            prec_str = f"{result['val_precision']:.4f}" if result.get('val_precision') else 'N/A'
            recall_str = f"{result['val_recall']:.4f}" if result.get('val_recall') else 'N/A'
            print(f"    Precision: {prec_str}")
            print(f"    Recall: {recall_str}")
            print(f"    Pred range: {result['val_pred_range']}")
            print(f"    Duration: {result['duration_sec']:.1f}s")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            all_results.append({
                "config_name": name,
                "tier": tier,
                "params": config,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })

        # Incremental save after each trial
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Generate summary
    summary = generate_summary(all_results, tier, output_base)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Completed: {summary['completed']}/{summary['total']}")
    print(f"Best AUC: {summary['best_auc']:.4f} ({summary['best_trial']})")
    print(f"Results saved to: {output_base}")
    print(f"{'='*60}\n")

    return {"results": all_results, "summary": summary}


def generate_summary(results: list[dict], tier: str, output_dir: Path) -> dict:
    """Generate summary files from results.

    Args:
        results: List of result dictionaries
        tier: Feature tier
        output_dir: Output directory

    Returns:
        Summary dictionary
    """
    import pandas as pd

    # Filter successful results
    successful = [r for r in results if "val_auc" in r and r["val_auc"] is not None]

    if not successful:
        return {"completed": 0, "total": len(results), "best_auc": 0, "best_trial": "N/A"}

    # Find best
    best = max(successful, key=lambda x: x["val_auc"])

    summary = {
        "tier": tier,
        "total": len(results),
        "completed": len(successful),
        "failed": len(results) - len(successful),
        "best_trial": best["config_name"],
        "best_auc": best["val_auc"],
        "best_params": best["params"],
        "timestamp": datetime.now().isoformat(),
    }

    # Save summary JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV
    df_data = []
    for r in successful:
        row = {
            "config_name": r["config_name"],
            "val_auc": r["val_auc"],
            "val_precision": r.get("val_precision"),
            "val_recall": r.get("val_recall"),
            "pred_min": r["val_pred_range"][0] if r.get("val_pred_range") else None,
            "pred_max": r["val_pred_range"][1] if r.get("val_pred_range") else None,
            "epochs_run": r.get("epochs_run"),
            "duration_sec": r.get("duration_sec"),
            **r["params"],
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    df = df.sort_values("val_auc", ascending=False)
    df.to_csv(output_dir / "results_summary.csv", index=False)

    # Save markdown report
    md_lines = [
        f"# Supplementary HPO Results - Tier {tier}",
        f"\nGenerated: {datetime.now().isoformat()}",
        f"\n## Summary",
        f"- Total trials: {summary['total']}",
        f"- Completed: {summary['completed']}",
        f"- Failed: {summary['failed']}",
        f"- Best AUC: {summary['best_auc']:.4f} ({summary['best_trial']})",
        f"\n## Best Config",
        f"```json",
        json.dumps(summary['best_params'], indent=2),
        f"```",
        f"\n## All Results (sorted by AUC)",
        f"\n| Config | AUC | Precision | Recall | Pred Range |",
        f"|--------|-----|-----------|--------|------------|",
    ]

    for _, row in df.iterrows():
        pred_range = f"[{row['pred_min']:.3f}, {row['pred_max']:.3f}]" if row['pred_min'] is not None else "N/A"
        precision_str = f"{row['val_precision']:.3f}" if row['val_precision'] is not None else "N/A"
        recall_str = f"{row['val_recall']:.3f}" if row['val_recall'] is not None else "N/A"
        md_lines.append(
            f"| {row['config_name']} | {row['val_auc']:.4f} | "
            f"{precision_str} | {recall_str} | {pred_range} |"
        )

    with open(output_dir / "results_summary.md", "w") as f:
        f.write("\n".join(md_lines))

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run supplementary HPO trials")
    parser.add_argument("--tier", required=True, choices=["a50", "a100"], help="Feature tier")
    parser.add_argument("--trials", type=str, help="Comma-separated list of specific trials (e.g., A1,A2,B1)")
    parser.add_argument("--thermal-threshold", type=float, default=85.0, help="Thermal pause threshold (°C)")
    parser.add_argument("--round", type=int, default=1, choices=[1, 2],
                        help="Round number: 1 for original A-F configs, 2 for follow-up G-I configs")
    args = parser.parse_args()

    trial_names = None
    if args.trials:
        trial_names = [t.strip() for t in args.trials.split(",")]

    run_supplementary_trials(
        tier=args.tier,
        trial_names=trial_names,
        thermal_pause_threshold=args.thermal_threshold,
        round_num=args.round,
    )


if __name__ == "__main__":
    main()
