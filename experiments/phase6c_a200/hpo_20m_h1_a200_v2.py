#!/usr/bin/env python3
"""
Phase 6C A200 HPO v2: 20M parameters, horizon=1

Improved HPO with:
1. Forced extreme trials (first 6) to systematically test architecture bounds
2. CoverageTracker integration to redirect duplicate architectures
3. Expanded search space with additional dropout/LR/weight_decay values
4. 75d context length (per ablation finding: 66.7% precision, 7.8% recall)
5. pred_range logging to detect probability collapse

Trials: 50 (6 forced + 44 TPE with coverage tracking)
Metric: val_auc (maximize)
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import torch
import pandas as pd

from src.config.experiment import ExperimentConfig
from src.models.patchtst import PatchTSTConfig
from src.data.dataset import SimpleSplitter
from src.training.trainer import Trainer
from src.training.hpo_coverage import CoverageTracker

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "hpo_20m_h1_a200_v2"
BUDGET = "20M"
HORIZON = 1
N_TRIALS = 50

# Expanded search space (v2)
SEARCH_SPACE_V2 = {
    # Architecture (unchanged from v1)
    "d_model": [64, 96, 128, 160, 192],
    "n_layers": [4, 5, 6, 7, 8],
    "n_heads": [4, 8],
    "d_ff_ratio": [2, 4],
    # Training - EXPANDED
    "learning_rate": [5e-5, 7e-5, 8e-5, 9e-5, 1e-4, 1.5e-4],
    "dropout": [0.3, 0.4, 0.5, 0.6, 0.7],
    "weight_decay": [1e-5, 1e-4, 3e-4, 5e-4, 1e-3],
}

# Fixed hyperparameters (ablation-validated)
# 75d from context ablation: best precision/recall tradeoff
CONTEXT_LENGTH = 75
EPOCHS = 50

# Data - A200 tier (206 features)
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a200_combined.parquet"
NUM_FEATURES = 211  # 206 indicators + 5 OHLCV (auto-adjusted by Trainer)

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/phase6c_a200" / EXPERIMENT_NAME

# Extreme type labels for the 6 forced trials
EXTREME_TYPES = [
    "min_d_model",
    "max_d_model",
    "min_n_layers",
    "max_n_layers",
    "min_n_heads",
    "max_n_heads",
]


# ============================================================================
# FORCED EXTREME CONFIGURATION
# ============================================================================


def compute_forced_extreme_configs(search_space: dict) -> list[dict]:
    """Compute 6 forced extreme configs to test architecture bounds.

    Returns configs for:
    - min/max d_model (with middle n_layers, safe n_heads)
    - min/max n_layers (with middle d_model, safe n_heads)
    - min/max n_heads (with middle d_model, middle n_layers)

    Args:
        search_space: Dict with lists of values for d_model, n_layers, n_heads

    Returns:
        List of 6 config dicts with d_model, n_layers, n_heads keys
    """
    d_models = sorted(search_space["d_model"])
    n_layers_list = sorted(search_space["n_layers"])
    n_heads_list = sorted(search_space["n_heads"])

    # Middle values for non-extreme params
    mid_d = d_models[len(d_models) // 2]  # 128
    mid_L = n_layers_list[len(n_layers_list) // 2]  # 6
    mid_h = 4  # Safe default (divides all d_models in our search space)

    return [
        # min/max d_model
        {"d_model": d_models[0], "n_layers": mid_L, "n_heads": mid_h},   # min d
        {"d_model": d_models[-1], "n_layers": mid_L, "n_heads": mid_h},  # max d
        # min/max n_layers
        {"d_model": mid_d, "n_layers": n_layers_list[0], "n_heads": mid_h},   # min L
        {"d_model": mid_d, "n_layers": n_layers_list[-1], "n_heads": mid_h},  # max L
        # min/max n_heads
        {"d_model": mid_d, "n_layers": mid_L, "n_heads": n_heads_list[0]},    # min h
        {"d_model": mid_d, "n_layers": mid_L, "n_heads": n_heads_list[-1]},   # max h
    ]


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================


def objective(trial):
    """Optuna objective function with forced extremes and coverage tracking."""
    extreme_configs = compute_forced_extreme_configs(SEARCH_SPACE_V2)
    is_forced = trial.number < len(extreme_configs)

    if is_forced:
        # Forced extreme trial: use predetermined architecture
        config = extreme_configs[trial.number].copy()
        d_model = config["d_model"]
        n_layers = config["n_layers"]
        n_heads = config["n_heads"]

        # Mark as forced extreme in user_attrs
        trial.set_user_attr("forced_extreme", True)
        trial.set_user_attr("extreme_type", EXTREME_TYPES[trial.number])
        trial.set_user_attr("forced_d_model", d_model)
        trial.set_user_attr("forced_n_layers", n_layers)
        trial.set_user_attr("forced_n_heads", n_heads)

        # Sample training params normally
        d_ff_ratio = trial.suggest_categorical("d_ff_ratio", SEARCH_SPACE_V2["d_ff_ratio"])
        learning_rate = trial.suggest_categorical("learning_rate", SEARCH_SPACE_V2["learning_rate"])
        dropout = trial.suggest_categorical("dropout", SEARCH_SPACE_V2["dropout"])
        weight_decay = trial.suggest_categorical("weight_decay", SEARCH_SPACE_V2["weight_decay"])

    else:
        # TPE trial with coverage tracking
        tracker = CoverageTracker.from_study(trial.study, SEARCH_SPACE_V2)

        # Sample architecture
        d_model = trial.suggest_categorical("d_model", SEARCH_SPACE_V2["d_model"])
        n_layers = trial.suggest_categorical("n_layers", SEARCH_SPACE_V2["n_layers"])
        n_heads = trial.suggest_categorical("n_heads", SEARCH_SPACE_V2["n_heads"])
        d_ff_ratio = trial.suggest_categorical("d_ff_ratio", SEARCH_SPACE_V2["d_ff_ratio"])

        # Check for coverage redirect
        proposed = {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads}
        redirected = tracker.suggest_coverage_config(proposed)

        if redirected != proposed:
            # Architecture was redirected to untested combo
            trial.set_user_attr("coverage_redirect", True)
            trial.set_user_attr("original_d_model", d_model)
            trial.set_user_attr("original_n_layers", n_layers)
            trial.set_user_attr("original_n_heads", n_heads)
            d_model = redirected["d_model"]
            n_layers = redirected["n_layers"]
            n_heads = redirected["n_heads"]
            trial.set_user_attr("redirected_d_model", d_model)
            trial.set_user_attr("redirected_n_layers", n_layers)
            trial.set_user_attr("redirected_n_heads", n_heads)

        # Sample training params
        learning_rate = trial.suggest_categorical("learning_rate", SEARCH_SPACE_V2["learning_rate"])
        dropout = trial.suggest_categorical("dropout", SEARCH_SPACE_V2["dropout"])
        weight_decay = trial.suggest_categorical("weight_decay", SEARCH_SPACE_V2["weight_decay"])

    # Validate n_heads divides d_model
    if d_model % n_heads != 0:
        raise optuna.TrialPruned("d_model not divisible by n_heads")

    d_ff = d_model * d_ff_ratio
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Load data
    df = pd.read_parquet(DATA_PATH)
    high_prices = df["High"].values

    # Batch size based on model size
    if d_model >= 192:
        batch_size = 32
    elif d_model >= 128:
        batch_size = 64
    else:
        batch_size = 128

    experiment_config = ExperimentConfig(
        data_path=str(DATA_PATH.relative_to(PROJECT_ROOT)),
        task="threshold_1pct",
        timescale="daily",
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        wandb_project=None,
        mlflow_experiment=None,
    )

    model_config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=16,
        stride=8,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
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

    trial_dir = OUTPUT_DIR / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        experiment_config=experiment_config,
        model_config=model_config,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
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
        val_auc = result.get("val_auc", 0.5)

        # Log metrics in user_attrs
        trial.set_user_attr("val_auc", val_auc)
        trial.set_user_attr("val_precision", result.get("val_precision", 0))
        trial.set_user_attr("val_recall", result.get("val_recall", 0))

        # Log pred_range for probability collapse detection
        pred_range = result.get("val_pred_range")
        if pred_range:
            trial.set_user_attr("pred_min", pred_range[0])
            trial.set_user_attr("pred_max", pred_range[1])

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

    return val_auc if val_auc else 0.5


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 70)
    print(f"PHASE 6C A200 HPO v2: {BUDGET} / horizon={HORIZON}")
    print("=" * 70)
    print(f"Context length: {CONTEXT_LENGTH}d (from ablation)")
    print(f"Features: {NUM_FEATURES} (a200 tier)")
    print(f"Forced extremes: {len(compute_forced_extreme_configs(SEARCH_SPACE_V2))} trials")
    print(f"Total trials: {N_TRIALS}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=EXPERIMENT_NAME,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print("Starting HPO with", N_TRIALS, "trials...")
    print("  - Trials 0-5: Forced extreme architectures")
    print("  - Trials 6+: TPE with coverage tracking")
    print()
    start_time = time.time()

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("HPO RESULTS")
    print("=" * 70)
    print("Best trial:", study.best_trial.number)
    print("Best AUC:", round(study.best_value, 4))
    print()
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Show best trial user_attrs (includes architecture for forced trials)
    best_attrs = study.best_trial.user_attrs
    if best_attrs.get("forced_extreme"):
        print()
        print("Best trial architecture (forced extreme):")
        print(f"  d_model: {best_attrs.get('forced_d_model')}")
        print(f"  n_layers: {best_attrs.get('forced_n_layers')}")
        print(f"  n_heads: {best_attrs.get('forced_n_heads')}")
        print(f"  extreme_type: {best_attrs.get('extreme_type')}")

    if "val_precision" in best_attrs:
        print()
        print("Best trial metrics:")
        print(f"  precision: {best_attrs.get('val_precision', 'N/A')}")
        print(f"  recall: {best_attrs.get('val_recall', 'N/A')}")
        if "pred_min" in best_attrs:
            print(f"  pred_range: [{best_attrs.get('pred_min'):.3f}, {best_attrs.get('pred_max'):.3f}]")

    print()
    print("Total time:", round(elapsed / 60, 1), "min")

    # Coverage stats
    tracker = CoverageTracker.from_study(study, SEARCH_SPACE_V2)
    stats = tracker.coverage_stats()
    print()
    print("Coverage stats:")
    print(f"  Valid arch combos: {stats['total_valid_combos']}")
    print(f"  Tested: {stats['tested_combos']}")
    print(f"  Coverage: {stats['coverage_pct']:.1f}%")

    # Save results
    results = {
        "experiment": EXPERIMENT_NAME,
        "budget": BUDGET,
        "horizon": HORIZON,
        "n_trials": N_TRIALS,
        "context_length": CONTEXT_LENGTH,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_user_attrs": dict(best_attrs),
        "search_space": SEARCH_SPACE_V2,
        "coverage_stats": stats,
        "total_time_min": elapsed / 60,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "best_params.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("Results saved to", results_path)

    # Save all trials with full details
    trials_data = []
    for trial in study.trials:
        trial_info = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": dict(trial.user_attrs),
            "state": str(trial.state),
        }
        trials_data.append(trial_info)

    trials_path = OUTPUT_DIR / "all_trials.json"
    with open(trials_path, "w") as f:
        json.dump(trials_data, f, indent=2)

    return study.best_value


if __name__ == "__main__":
    main()
