#!/usr/bin/env python3
"""
HPO Template - Enhanced Hyperparameter Optimization

METHODOLOGY:
Two-phase HPO strategy with full metrics capture:

Phase A: Forced Extremes (6 trials)
- Trial 0: Min d_model (smallest embedding, middle other params)
- Trial 1: Max d_model (largest embedding, middle other params)
- Trial 2: Min n_layers (shallowest, middle d_model/n_heads)
- Trial 3: Max n_layers (deepest, middle d_model/n_heads)
- Trial 4: Min n_heads (fewest heads, middle d_model)
- Trial 5: Max n_heads (most heads, middle d_model)

Phase B: TPE Exploration (remaining trials)
- Random exploration with TPE sampler
- Ensures coverage of promising regions

FEATURES:
- Captures ALL metrics: AUC, precision, recall, pred_range
- Two-phase strategy: forced extremes + TPE
- SQLite study storage for resumability
- Incremental result saving
- Comprehensive error handling

USAGE:
    # Configure the parameters below, then run:
    python experiments/templates/hpo_template.py [--dry-run] [--resume]

CONFIGURATION:
    Edit the CONFIGURATION section below for your experiment.
"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import torch
import numpy as np
import pandas as pd

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer
from src.training.hpo_coverage import CoverageTracker

# ============================================================================
# CONFIGURATION - Edit this section for your experiment
# ============================================================================

# Experiment identification
EXPERIMENT_NAME = "hpo_{budget}_{horizon}_{tier}"  # Will be formatted
BUDGET = "2M"  # "2M", "20M", or "200M"
HORIZON = 1
TIER = "a100"

# Number of trials
N_TRIALS = 50  # Total trials including forced extremes
N_FORCED_EXTREME_TRIALS = 6  # First 6 trials test architecture extremes

# Search space - configure per budget
SEARCH_SPACES = {
    "2M": {
        "d_model": [32, 48, 64, 80, 96],
        "n_layers": [2, 3, 4, 5, 6],
        "n_heads": [2, 4, 8],
        "d_ff_ratio": [2, 4],
        "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
        "dropout": [0.1, 0.3, 0.5, 0.7],
        "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    },
    "20M": {
        "d_model": [64, 96, 128, 160, 192],
        "n_layers": [4, 5, 6, 7, 8],
        "n_heads": [4, 8],
        "d_ff_ratio": [2, 4],
        "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
        "dropout": [0.1, 0.3, 0.5, 0.7],
        "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    },
    "200M": {
        "d_model": [128, 192, 256, 320, 384],
        "n_layers": [6, 8, 10, 12],
        "n_heads": [8, 16],
        "d_ff_ratio": [2, 4],
        "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4],
        "dropout": [0.1, 0.3, 0.5, 0.7],
        "weight_decay": [0.0, 1e-5, 1e-4, 1e-3],
    },
}

# Fixed hyperparameters (ablation-validated)
CONTEXT_LENGTH = 80
EPOCHS = 50

# Data configuration
DATA_PATH_TEMPLATE = "data/processed/v1/SPY_dataset_{tier}_combined.parquet"
NUM_FEATURES_BY_TIER = {
    "a20": 25,
    "a50": 55,
    "a100": 105,
    "a200": 211,
}

# ============================================================================
# FORCED EXTREME TRIALS CONFIGURATION
# ============================================================================


def compute_forced_extreme_configs(search_space: dict) -> list[dict]:
    """Compute the 6 forced extreme configurations.

    Returns configs for:
    0: Min d_model
    1: Max d_model
    2: Min n_layers
    3: Max n_layers
    4: Min n_heads
    5: Max n_heads
    """
    d_models = sorted(search_space["d_model"])
    n_layers_vals = sorted(search_space["n_layers"])
    n_heads_vals = sorted(search_space["n_heads"])

    # Middle values for non-extreme dimensions
    middle_d = d_models[len(d_models) // 2]
    middle_L = n_layers_vals[len(n_layers_vals) // 2]
    middle_h = 8 if 8 in n_heads_vals else n_heads_vals[len(n_heads_vals) // 2]

    # Middle training HPs
    middle_lr = 5e-5
    middle_dropout = 0.3
    middle_wd = 1e-4
    middle_ff = 4

    extreme_configs = [
        # 0: Min d_model
        {"d_model": min(d_models), "n_layers": middle_L, "n_heads": middle_h,
         "d_ff_ratio": middle_ff, "learning_rate": middle_lr,
         "dropout": middle_dropout, "weight_decay": middle_wd},
        # 1: Max d_model
        {"d_model": max(d_models), "n_layers": middle_L, "n_heads": middle_h,
         "d_ff_ratio": middle_ff, "learning_rate": middle_lr,
         "dropout": middle_dropout, "weight_decay": middle_wd},
        # 2: Min n_layers
        {"d_model": middle_d, "n_layers": min(n_layers_vals), "n_heads": middle_h,
         "d_ff_ratio": middle_ff, "learning_rate": middle_lr,
         "dropout": middle_dropout, "weight_decay": middle_wd},
        # 3: Max n_layers
        {"d_model": middle_d, "n_layers": max(n_layers_vals), "n_heads": middle_h,
         "d_ff_ratio": middle_ff, "learning_rate": middle_lr,
         "dropout": middle_dropout, "weight_decay": middle_wd},
        # 4: Min n_heads
        {"d_model": middle_d, "n_layers": middle_L, "n_heads": min(n_heads_vals),
         "d_ff_ratio": middle_ff, "learning_rate": middle_lr,
         "dropout": middle_dropout, "weight_decay": middle_wd},
        # 5: Max n_heads
        {"d_model": middle_d, "n_layers": middle_L, "n_heads": max(n_heads_vals),
         "d_ff_ratio": middle_ff, "learning_rate": middle_lr,
         "dropout": middle_dropout, "weight_decay": middle_wd},
    ]

    # Validate d_model divisible by n_heads
    for config in extreme_configs:
        if config["d_model"] % config["n_heads"] != 0:
            # Adjust n_heads to be valid
            valid_heads = [h for h in n_heads_vals if config["d_model"] % h == 0]
            if valid_heads:
                config["n_heads"] = valid_heads[len(valid_heads) // 2]

    return extreme_configs


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================


def create_objective(
    search_space: dict,
    data_path: Path,
    num_features: int,
    output_dir: Path,
    extreme_configs: list[dict],
    dry_run: bool = False,
):
    """Create Optuna objective function with full metrics capture."""
    # Coverage tracker for architecture exploration (shared across trials)
    coverage_tracker = CoverageTracker(search_space)

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function."""
        trial_start = time.time()

        # Determine if this is a forced extreme trial
        is_forced_extreme = trial.number < len(extreme_configs)

        if is_forced_extreme:
            # Use pre-computed extreme config
            config = extreme_configs[trial.number].copy()
            trial.set_user_attr("forced_extreme", True)
            trial.set_user_attr("extreme_type", [
                "min_d_model", "max_d_model",
                "min_n_layers", "max_n_layers",
                "min_n_heads", "max_n_heads"
            ][trial.number])

            # Set params via user_attrs (not suggest to avoid conflicts)
            for k, v in config.items():
                trial.set_user_attr(f"forced_{k}", v)
        else:
            # TPE sampling for remaining trials
            trial.set_user_attr("forced_extreme", False)
            config = {
                "d_model": trial.suggest_categorical("d_model", search_space["d_model"]),
                "n_layers": trial.suggest_categorical("n_layers", search_space["n_layers"]),
                "n_heads": trial.suggest_categorical("n_heads", search_space["n_heads"]),
                "d_ff_ratio": trial.suggest_categorical("d_ff_ratio", search_space["d_ff_ratio"]),
                "learning_rate": trial.suggest_categorical("learning_rate", search_space["learning_rate"]),
                "dropout": trial.suggest_categorical("dropout", search_space["dropout"]),
                "weight_decay": trial.suggest_categorical("weight_decay", search_space["weight_decay"]),
            }

            # Coverage-aware redirect: avoid testing duplicate architecture combos
            # Reconstruct tracker state from study to handle resume correctly
            tracker = CoverageTracker.from_study(trial.study, search_space)
            original_config = config.copy()
            config = tracker.suggest_coverage_config(config)
            if config != original_config:
                trial.set_user_attr("coverage_redirect", True)
                trial.set_user_attr("original_d_model", original_config["d_model"])
                trial.set_user_attr("original_n_layers", original_config["n_layers"])
                trial.set_user_attr("original_n_heads", original_config["n_heads"])
                for k in ["d_model", "n_layers", "n_heads"]:
                    trial.set_user_attr(f"actual_{k}", config[k])

        # Validate n_heads divides d_model
        if config["d_model"] % config["n_heads"] != 0:
            trial.set_user_attr("pruned_reason", "d_model not divisible by n_heads")
            raise optuna.TrialPruned("d_model not divisible by n_heads")

        # Compute derived values
        d_ff = config["d_model"] * config["d_ff_ratio"]

        # Determine batch size based on model size
        d_model = config["d_model"]
        if d_model >= 256:
            batch_size = 32
        elif d_model >= 128:
            batch_size = 64
        else:
            batch_size = 128

        if dry_run:
            # Return dummy result for dry run
            print(f"[DRY RUN] Trial {trial.number}: {config}")
            return 0.5 + np.random.random() * 0.1

        device = "mps" if torch.backends.mps.is_available() else "cpu"

        # Load data
        df = pd.read_parquet(data_path)
        high_prices = df["High"].values

        experiment_config = ExperimentConfig(
            data_path=str(data_path.relative_to(PROJECT_ROOT)),
            task="threshold_1pct",
            timescale="daily",
            context_length=CONTEXT_LENGTH,
            horizon=HORIZON,
            wandb_project=None,
            mlflow_experiment=None,
        )

        model_config = PatchTSTConfig(
            num_features=num_features,
            context_length=CONTEXT_LENGTH,
            patch_length=16,
            stride=8,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            d_ff=d_ff,
            dropout=config["dropout"],
            head_dropout=0.0,
        )

        splitter = SimpleSplitter(
            dates=df["Date"],
            context_length=CONTEXT_LENGTH,
            horizon=HORIZON,
            val_start="2023-01-01",
            test_start="2025-01-01",
        )
        split_indices = splitter.split()

        trial_dir = output_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=batch_size,
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            epochs=EPOCHS,
            device=device,
            checkpoint_dir=trial_dir,
            split_indices=split_indices,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            early_stopping_metric="val_auc",
            use_revin=True,
            high_prices=high_prices,
        )

        try:
            result = trainer.train(verbose=False)

            # Extract ALL metrics
            val_auc = result.get("val_auc", 0.5)
            val_precision = result.get("val_precision")
            val_recall = result.get("val_recall")
            val_pred_range = result.get("val_pred_range")

            # Store all metrics as user attributes
            trial.set_user_attr("val_auc", val_auc)
            if val_precision is not None:
                trial.set_user_attr("val_precision", float(val_precision))
            if val_recall is not None:
                trial.set_user_attr("val_recall", float(val_recall))
            if val_pred_range is not None:
                trial.set_user_attr("val_pred_min", float(val_pred_range[0]))
                trial.set_user_attr("val_pred_max", float(val_pred_range[1]))

            # Store training metrics
            trial.set_user_attr("epochs_run", result.get("epochs_run", EPOCHS))
            if "train_pred_range" in result and result["train_pred_range"]:
                trial.set_user_attr("train_pred_min", float(result["train_pred_range"][0]))
                trial.set_user_attr("train_pred_max", float(result["train_pred_range"][1]))

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            error_msg = str(e)
            if "out of memory" in error_msg.lower() or "mps" in error_msg.lower():
                trial.set_user_attr("error", f"OOM: {error_msg[:200]}")
                raise optuna.TrialPruned(f"OOM error: {error_msg[:100]}")
            trial.set_user_attr("error", f"RuntimeError: {error_msg[:200]}")
            print(f"Trial {trial.number} failed (RuntimeError): {e}")
            val_auc = 0.5

        except ValueError as e:
            error_msg = str(e)
            trial.set_user_attr("error", f"ValueError: {error_msg[:200]}")
            if "nan" in error_msg.lower() or "inf" in error_msg.lower():
                raise optuna.TrialPruned(f"NaN/Inf error: {error_msg[:100]}")
            print(f"Trial {trial.number} failed (ValueError): {e}")
            val_auc = 0.5

        except KeyboardInterrupt:
            raise

        except Exception as e:
            error_msg = str(e)
            trial.set_user_attr("error", f"Unexpected: {type(e).__name__}: {error_msg[:200]}")
            print(f"Trial {trial.number} failed ({type(e).__name__}): {e}")
            val_auc = 0.5

        # Store timing
        trial.set_user_attr("duration_sec", time.time() - trial_start)

        return val_auc if val_auc else 0.5

    return objective


# ============================================================================
# RESULTS SAVING
# ============================================================================


def save_trial_results(study: optuna.Study, output_dir: Path, search_space: dict) -> None:
    """Save comprehensive trial results."""
    trials_data = []
    for trial in study.trials:
        trial_data = {
            "number": trial.number,
            "value": trial.value,
            "state": str(trial.state),
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        trials_data.append(trial_data)

    # Save all trials
    with open(output_dir / "all_trials.json", "w") as f:
        json.dump(trials_data, f, indent=2, default=str)

    # Save detailed metrics CSV
    metrics_data = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        row = {
            "trial": trial.number,
            "auc": trial.value,
            "precision": trial.user_attrs.get("val_precision"),
            "recall": trial.user_attrs.get("val_recall"),
            "pred_min": trial.user_attrs.get("val_pred_min"),
            "pred_max": trial.user_attrs.get("val_pred_max"),
            "forced_extreme": trial.user_attrs.get("forced_extreme", False),
            "extreme_type": trial.user_attrs.get("extreme_type", ""),
            "duration_sec": trial.user_attrs.get("duration_sec"),
            **trial.params,
        }
        # Add forced params if present
        for k in ["d_model", "n_layers", "n_heads", "d_ff_ratio",
                  "learning_rate", "dropout", "weight_decay"]:
            if f"forced_{k}" in trial.user_attrs:
                row[k] = trial.user_attrs[f"forced_{k}"]
        metrics_data.append(row)

    df = pd.DataFrame(metrics_data)
    df.to_csv(output_dir / "trial_metrics.csv", index=False)

    # Save best params
    best_results = {
        "experiment": EXPERIMENT_NAME.format(
            budget=BUDGET.lower(), horizon=HORIZON, tier=TIER
        ),
        "budget": BUDGET,
        "horizon": HORIZON,
        "tier": TIER,
        "n_trials": len(study.trials),
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": study.best_trial.user_attrs,
        "search_space": search_space,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(best_results, f, indent=2, default=str)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="HPO with full metrics capture")
    parser.add_argument("--dry-run", action="store_true", help="Test without training")
    parser.add_argument("--resume", action="store_true", help="Resume existing study")
    parser.add_argument("--budget", type=str, default=BUDGET, help="Budget (2M, 20M, 200M)")
    parser.add_argument("--horizon", type=int, default=HORIZON, help="Prediction horizon")
    parser.add_argument("--tier", type=str, default=TIER, help="Feature tier")
    parser.add_argument("--trials", type=int, default=N_TRIALS, help="Number of trials")
    args = parser.parse_args()

    budget = args.budget.upper()
    horizon = args.horizon
    tier = args.tier.lower()
    n_trials = args.trials

    experiment_name = EXPERIMENT_NAME.format(
        budget=budget.lower(), horizon=horizon, tier=tier
    )

    print("=" * 70)
    print(f"HPO: {budget} / horizon={horizon} / tier={tier}")
    print("=" * 70)

    if budget not in SEARCH_SPACES:
        print(f"Error: Unknown budget {budget}. Use 2M, 20M, or 200M.")
        sys.exit(1)

    search_space = SEARCH_SPACES[budget]
    num_features = NUM_FEATURES_BY_TIER.get(tier, 105)
    data_path = PROJECT_ROOT / DATA_PATH_TEMPLATE.format(tier=tier)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    output_dir = PROJECT_ROOT / "outputs" / f"phase6c_{tier}" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute forced extreme configs
    extreme_configs = compute_forced_extreme_configs(search_space)
    print(f"\nForced extreme trials ({len(extreme_configs)}):")
    for i, cfg in enumerate(extreme_configs):
        print(f"  {i}: d={cfg['d_model']}, L={cfg['n_layers']}, h={cfg['n_heads']}")

    # Create/load study
    storage_path = f"sqlite:///{output_dir}/study.db"

    if args.resume:
        try:
            study = optuna.load_study(
                study_name=experiment_name,
                storage=storage_path,
            )
            print(f"\nResuming study with {len(study.trials)} existing trials")
        except KeyError:
            print("\nNo existing study found, creating new one")
            study = optuna.create_study(
                direction="maximize",
                study_name=experiment_name,
                storage=storage_path,
                sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
            )
    else:
        study = optuna.create_study(
            direction="maximize",
            study_name=experiment_name,
            storage=storage_path,
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
            load_if_exists=False,
        )

    # Create objective
    objective = create_objective(
        search_space=search_space,
        data_path=data_path,
        num_features=num_features,
        output_dir=output_dir,
        extreme_configs=extreme_configs,
        dry_run=args.dry_run,
    )

    print(f"\nStarting HPO with {n_trials} trials...")
    print(f"  - {len(extreme_configs)} forced extreme trials")
    print(f"  - {n_trials - len(extreme_configs)} TPE exploration trials")
    print(f"  - Output: {output_dir}")
    start_time = time.time()

    # Callback to save results after each trial
    def save_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        save_trial_results(study, output_dir, search_space)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        callbacks=[save_callback],
    )

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("HPO RESULTS")
    print("=" * 70)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best AUC: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Print additional best metrics
    best_attrs = study.best_trial.user_attrs
    if "val_precision" in best_attrs:
        print(f"Best precision: {best_attrs['val_precision']:.4f}")
    if "val_recall" in best_attrs:
        print(f"Best recall: {best_attrs['val_recall']:.4f}")
    if "val_pred_min" in best_attrs and "val_pred_max" in best_attrs:
        print(f"Best pred_range: [{best_attrs['val_pred_min']:.4f}, {best_attrs['val_pred_max']:.4f}]")

    print(f"\nTotal time: {elapsed/60:.1f} min")
    print(f"Results saved to: {output_dir}")

    return study.best_value


if __name__ == "__main__":
    main()
