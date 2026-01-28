#!/usr/bin/env python3
"""
HPO v2 for NeuralForecast Alternative Architectures (iTransformer, Informer).

METHODOLOGY:
Two-phase HPO strategy matching PatchTST methodology for fair comparison:

Phase A: Forced Extremes (6 trials)
- Trial 0: Min hidden_size (smallest capacity)
- Trial 1: Max hidden_size (largest capacity)
- Trial 2: Min n_layers (shallowest)
- Trial 3: Max n_layers (deepest)
- Trial 4: Min n_heads (fewest heads)
- Trial 5: Max n_heads (most heads)

Phase B: TPE Exploration (remaining trials)
- TPE sampling from expanded search space
- Ensures coverage of promising regions

EXPANDED SEARCH SPACE (vs v1):
- learning_rate: Added 1e-5, 2e-5 (PatchTST found 1e-5 optimal!)
- dropout: Added 0.6, 0.7 (PatchTST found 0.7 optimal for 20M!)
- weight_decay: NEW parameter [0.0, 1e-4, 1e-3]
- hidden_size: Expanded to [128, 256, 384, 512]
- n_layers: Expanded to [2, 3, 4, 5, 6]
- max_steps: Extended to [2000, 3000, 5000]

Usage:
    # Full run (fresh start - do NOT resume from v1)
    python experiments/architectures/hpo_neuralforecast_v2.py --model itransformer --trials 50

    # Dry run (3 trials, no file output)
    python experiments/architectures/hpo_neuralforecast_v2.py --model itransformer --trials 3 --dry-run

    # Resume interrupted v2 study
    python experiments/architectures/hpo_neuralforecast_v2.py --model itransformer --trials 50 --resume

Success criteria: AUC >= 0.65 (within striking distance of PatchTST 0.718)
"""
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

# Check NeuralForecast availability
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import iTransformer, Informer
    from neuralforecast.losses.pytorch import MSE
    NEURALFORECAST_AVAILABLE = True
except ImportError as e:
    NEURALFORECAST_AVAILABLE = False
    IMPORT_ERROR = str(e)

from experiments.architectures.common import (
    DATA_PATH_A20,
    OUTPUT_BASE,
    evaluate_forecasting_model,
    compute_precision_recall_curve,
    compare_to_baseline,
)


# ============================================================================
# HPO CONFIGURATION - V2 EXPANDED SEARCH SPACE
# ============================================================================

# Expanded search space based on PatchTST findings
SEARCH_SPACE_V2 = {
    "dropout": [0.3, 0.5, 0.6, 0.7],           # Added 0.6, 0.7 (PatchTST found 0.7 best for 20M)
    "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4],  # Added 1e-5, 2e-5 (PatchTST found 1e-5 optimal!)
    "hidden_size": [128, 256, 384, 512],       # Expanded range (previous best was 256)
    "n_layers": [2, 3, 4, 5, 6],               # More depth options
    "n_heads": [2, 4, 8],                      # Same as v1
    "max_steps": [2000, 3000, 5000],           # Longer training (v1 max was 2000)
    "batch_size": [32, 64],                    # Keep what worked in v1
    "weight_decay": [0.0, 1e-4, 1e-3],         # NEW: regularization parameter
}

# Number of forced extreme trials
N_FORCED_EXTREME_TRIALS = 6

# Fixed parameters
FIXED_PARAMS = {
    "input_size": 80,  # Context length
    "h": 1,  # Forecast horizon
    "random_seed": 42,
}

# Model-specific parameter mappings
MODEL_CONFIGS = {
    "itransformer": {
        "class": lambda: iTransformer,
        "layer_param": "e_layers",  # iTransformer uses e_layers
        "extra_fixed": {"n_series": 1},  # Univariate forecasting
    },
    "informer": {
        "class": lambda: Informer,
        "layer_param": "encoder_layers",  # Informer uses encoder_layers
        "extra_fixed": {
            "decoder_layers": 1,
            "conv_hidden_size": 32,
            "factor": 5,  # ProbSparse attention factor
        },
    },
}

HPO_OUTPUT_BASE = PROJECT_ROOT / "outputs/hpo/architectures"


# ============================================================================
# FORCED EXTREME TRIALS CONFIGURATION
# ============================================================================

def compute_forced_extreme_configs(search_space: dict) -> list[dict]:
    """Compute the 6 forced extreme configurations.

    Returns configs for:
    0: Min hidden_size (smallest capacity)
    1: Max hidden_size (largest capacity)
    2: Min n_layers (shallowest)
    3: Max n_layers (deepest)
    4: Min n_heads (fewest heads)
    5: Max n_heads (most heads)

    "Middle params" for non-extreme dimensions:
    - lr=5e-5, dropout=0.5, wd=1e-4, steps=3000
    """
    hidden_sizes = sorted(search_space["hidden_size"])
    n_layers_vals = sorted(search_space["n_layers"])
    n_heads_vals = sorted(search_space["n_heads"])

    # Middle values for non-extreme dimensions
    middle_hidden = hidden_sizes[len(hidden_sizes) // 2]
    middle_layers = n_layers_vals[len(n_layers_vals) // 2]
    middle_heads = 4 if 4 in n_heads_vals else n_heads_vals[len(n_heads_vals) // 2]

    # Middle training HPs (conservative choices)
    middle_lr = 5e-5
    middle_dropout = 0.5
    middle_wd = 1e-4
    middle_steps = 3000
    middle_batch = 64

    extreme_configs = [
        # 0: Min hidden_size
        {
            "hidden_size": min(hidden_sizes),
            "n_layers": middle_layers,
            "n_heads": middle_heads,
            "learning_rate": middle_lr,
            "dropout": middle_dropout,
            "weight_decay": middle_wd,
            "max_steps": middle_steps,
            "batch_size": middle_batch,
        },
        # 1: Max hidden_size
        {
            "hidden_size": max(hidden_sizes),
            "n_layers": middle_layers,
            "n_heads": middle_heads,
            "learning_rate": middle_lr,
            "dropout": middle_dropout,
            "weight_decay": middle_wd,
            "max_steps": middle_steps,
            "batch_size": middle_batch,
        },
        # 2: Min n_layers
        {
            "hidden_size": middle_hidden,
            "n_layers": min(n_layers_vals),
            "n_heads": middle_heads,
            "learning_rate": middle_lr,
            "dropout": middle_dropout,
            "weight_decay": middle_wd,
            "max_steps": middle_steps,
            "batch_size": middle_batch,
        },
        # 3: Max n_layers
        {
            "hidden_size": middle_hidden,
            "n_layers": max(n_layers_vals),
            "n_heads": middle_heads,
            "learning_rate": middle_lr,
            "dropout": middle_dropout,
            "weight_decay": middle_wd,
            "max_steps": middle_steps,
            "batch_size": middle_batch,
        },
        # 4: Min n_heads
        {
            "hidden_size": middle_hidden,
            "n_layers": middle_layers,
            "n_heads": min(n_heads_vals),
            "learning_rate": middle_lr,
            "dropout": middle_dropout,
            "weight_decay": middle_wd,
            "max_steps": middle_steps,
            "batch_size": middle_batch,
        },
        # 5: Max n_heads
        {
            "hidden_size": middle_hidden,
            "n_layers": middle_layers,
            "n_heads": max(n_heads_vals),
            "learning_rate": middle_lr,
            "dropout": middle_dropout,
            "weight_decay": middle_wd,
            "max_steps": middle_steps,
            "batch_size": middle_batch,
        },
    ]

    # Validate hidden_size divisible by n_heads (for attention)
    for config in extreme_configs:
        if config["hidden_size"] % config["n_heads"] != 0:
            # Adjust n_heads to be valid
            valid_heads = [h for h in n_heads_vals if config["hidden_size"] % h == 0]
            if valid_heads:
                config["n_heads"] = valid_heads[len(valid_heads) // 2]

    return extreme_configs


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_hpo_data():
    """Prepare data for HPO experiments.

    Returns data in NeuralForecast panel format with evaluation metadata.
    """
    df = pd.read_parquet(DATA_PATH_A20)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Feature columns (exclude Date, High)
    exclude_cols = {"Date", "High"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Calculate next-day return as target
    df["return"] = df["Close"].pct_change().shift(-1)

    # Calculate threshold target (for evaluation)
    # True if next-day high reaches +1% from current close
    df["threshold_target"] = (df["High"].shift(-1) >= df["Close"] * 1.01).astype(float)

    # Drop rows with NaN
    df = df.dropna(subset=["return", "threshold_target"]).reset_index(drop=True)

    # Panel format with single series
    df_nf = pd.DataFrame({
        "unique_id": "SPY",
        "ds": df["Date"],
        "y": df["return"],
    })

    # Store metadata for evaluation
    metadata = {
        "threshold_targets": df["threshold_target"].values,
        "actual_returns": df["return"].values,
    }

    # Split by date
    val_start = pd.Timestamp("2023-01-01")
    test_start = pd.Timestamp("2025-01-01")

    df_train = df_nf[df_nf["ds"] < val_start].copy()
    df_val = df_nf[(df_nf["ds"] >= val_start) & (df_nf["ds"] < test_start)].copy()
    df_test = df_nf[df_nf["ds"] >= test_start].copy()

    # Get corresponding metadata indices
    val_mask = (df["Date"] >= val_start) & (df["Date"] < test_start)
    test_mask = df["Date"] >= test_start

    return {
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "df_full": df_nf,
        "feature_cols": feature_cols,
        "val_targets": metadata["threshold_targets"][val_mask],
        "val_returns": metadata["actual_returns"][val_mask],
        "test_targets": metadata["threshold_targets"][test_mask],
        "test_returns": metadata["actual_returns"][test_mask],
    }


# ============================================================================
# OBJECTIVE FUNCTION
# ============================================================================

def create_objective(
    model_type: str,
    data: dict,
    extreme_configs: list[dict],
    verbose: bool = False,
):
    """Create Optuna objective function with two-phase methodology.

    Phase A (trials 0-5): Forced extreme configurations
    Phase B (trials 6+): TPE sampling from expanded search space

    Args:
        model_type: 'itransformer' or 'informer'
        data: Prepared data dict from prepare_hpo_data()
        extreme_configs: List of 6 forced extreme configurations
        verbose: If True, print progress during trials

    Returns:
        Optuna objective function
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'itransformer' or 'informer'")

    model_config = MODEL_CONFIGS[model_type]
    model_class = model_config["class"]()
    layer_param = model_config["layer_param"]
    extra_fixed = model_config.get("extra_fixed", {})

    def objective(trial: optuna.Trial) -> float:
        """Objective function that trains model and returns negative AUC (for maximization)."""
        start_time = time.time()

        # Determine if this is a forced extreme trial
        is_forced_extreme = trial.number < len(extreme_configs)

        if is_forced_extreme:
            # Use pre-computed extreme config
            config = extreme_configs[trial.number].copy()
            trial.set_user_attr("forced_extreme", True)
            trial.set_user_attr("extreme_type", [
                "min_hidden", "max_hidden",
                "min_layers", "max_layers",
                "min_heads", "max_heads"
            ][trial.number])

            # Store forced params as user attributes
            for k, v in config.items():
                trial.set_user_attr(f"forced_{k}", v)

            if verbose:
                print(f"  Trial {trial.number} [FORCED {trial.user_attrs['extreme_type']}]: "
                      f"hidden={config['hidden_size']}, layers={config['n_layers']}, "
                      f"heads={config['n_heads']}, lr={config['learning_rate']}")
        else:
            # TPE sampling for remaining trials
            trial.set_user_attr("forced_extreme", False)
            config = {
                "dropout": trial.suggest_categorical("dropout", SEARCH_SPACE_V2["dropout"]),
                "learning_rate": trial.suggest_categorical("learning_rate", SEARCH_SPACE_V2["learning_rate"]),
                "hidden_size": trial.suggest_categorical("hidden_size", SEARCH_SPACE_V2["hidden_size"]),
                "n_layers": trial.suggest_categorical("n_layers", SEARCH_SPACE_V2["n_layers"]),
                "n_heads": trial.suggest_categorical("n_heads", SEARCH_SPACE_V2["n_heads"]),
                "max_steps": trial.suggest_categorical("max_steps", SEARCH_SPACE_V2["max_steps"]),
                "batch_size": trial.suggest_categorical("batch_size", SEARCH_SPACE_V2["batch_size"]),
                "weight_decay": trial.suggest_categorical("weight_decay", SEARCH_SPACE_V2["weight_decay"]),
            }

            if verbose:
                print(f"  Trial {trial.number} [TPE]: dropout={config['dropout']}, "
                      f"lr={config['learning_rate']}, hidden={config['hidden_size']}, "
                      f"layers={config['n_layers']}, heads={config['n_heads']}, "
                      f"steps={config['max_steps']}, wd={config['weight_decay']}")

        # Validate hidden_size divisible by n_heads
        if config["hidden_size"] % config["n_heads"] != 0:
            trial.set_user_attr("pruned_reason", "hidden_size not divisible by n_heads")
            if verbose:
                print(f"    -> Pruned: hidden_size {config['hidden_size']} not divisible by n_heads {config['n_heads']}")
            raise optuna.TrialPruned("hidden_size not divisible by n_heads")

        # Build model kwargs
        model_kwargs = {
            "h": FIXED_PARAMS["h"],
            "input_size": FIXED_PARAMS["input_size"],
            "hidden_size": config["hidden_size"],
            "n_heads": config["n_heads"] if model_type == "itransformer" else None,
            "n_head": config["n_heads"] if model_type == "informer" else None,
            layer_param: config["n_layers"],
            "learning_rate": config["learning_rate"],
            "max_steps": config["max_steps"],
            "batch_size": config["batch_size"],
            "dropout": config["dropout"],
            "loss": MSE(),
            "random_seed": FIXED_PARAMS["random_seed"],
            **extra_fixed,
        }

        # Remove None values
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        # Note: NeuralForecast doesn't support weight_decay directly in all models
        # We track it for consistency but may not be applied
        trial.set_user_attr("weight_decay_config", config.get("weight_decay", 0.0))

        try:
            # Initialize model
            model = model_class(**model_kwargs)

            # Create NeuralForecast wrapper
            nf = NeuralForecast(models=[model], freq="D")

            # Train
            nf.fit(df=data["df_train"])

            # Get validation predictions via cross-validation
            cv_results = nf.cross_validation(
                df=data["df_full"],
                step_size=1,
                n_windows=len(data["df_val"]) + len(data["df_test"]),
                refit=False,
            )

            # Extract val predictions
            cv_results["ds"] = pd.to_datetime(cv_results["ds"])
            val_start = pd.Timestamp("2023-01-01")
            test_start = pd.Timestamp("2025-01-01")

            model_col = "iTransformer" if model_type == "itransformer" else "Informer"
            val_preds = cv_results[
                (cv_results["ds"] >= val_start) & (cv_results["ds"] < test_start)
            ][model_col].values

            # Ensure alignment
            val_preds = val_preds[:len(data["val_targets"])]

            # Evaluate
            metrics = evaluate_forecasting_model(
                predicted_returns=val_preds,
                actual_returns=data["val_returns"][:len(val_preds)],
                threshold_targets=data["val_targets"][:len(val_preds)],
                return_threshold=0.01,
            )

            auc = metrics.get("auc")
            if auc is None:
                auc = 0.5  # Default if AUC computation fails

            training_time = (time.time() - start_time) / 60

            # Store ALL metrics in trial user_attrs
            trial.set_user_attr("val_auc", auc)
            trial.set_user_attr("val_accuracy", metrics.get("accuracy", 0))
            trial.set_user_attr("val_precision", metrics.get("precision", 0))
            trial.set_user_attr("val_recall", metrics.get("recall", 0))
            trial.set_user_attr("val_f1", metrics.get("f1", 0))
            trial.set_user_attr("pred_min", metrics.get("pred_min", 0))
            trial.set_user_attr("pred_max", metrics.get("pred_max", 0))
            trial.set_user_attr("pred_mean", metrics.get("pred_mean", 0))
            trial.set_user_attr("pred_std", metrics.get("pred_std", 0))
            trial.set_user_attr("training_time_min", training_time)
            trial.set_user_attr("n_positive_preds", metrics.get("n_positive_preds", 0))

            if verbose:
                print(f"    -> AUC={auc:.4f}, Acc={metrics.get('accuracy', 0):.4f}, "
                      f"Recall={metrics.get('recall', 0):.4f}, "
                      f"Pred=[{metrics.get('pred_min', 0):.4f}, {metrics.get('pred_max', 0):.4f}], "
                      f"Time={training_time:.1f}min")

            # Return negative AUC for maximization (Optuna minimizes by default)
            return -auc

        except Exception as e:
            error_msg = str(e)[:200]
            trial.set_user_attr("error", f"{type(e).__name__}: {error_msg}")
            if verbose:
                print(f"    -> Trial failed: {type(e).__name__}: {error_msg}")
            # Return worst possible value
            return 0.0  # Negative of AUC=0 means trial failed

    return objective


# ============================================================================
# RESULT SAVING
# ============================================================================

def save_trial_result(
    trial: optuna.trial.FrozenTrial,
    output_dir: Path,
    model_type: str,
) -> Path:
    """Save individual trial result to JSON file."""
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Extract config from params or forced attributes
    if trial.user_attrs.get("forced_extreme"):
        config = {
            k.replace("forced_", ""): v
            for k, v in trial.user_attrs.items()
            if k.startswith("forced_") and k != "forced_extreme"
        }
    else:
        config = trial.params

    trial_data = {
        "trial_number": trial.number,
        "value": trial.value,
        "auc": -trial.value if trial.value is not None else None,  # Convert back from negative
        "params": config,
        "state": trial.state.name,
        "forced_extreme": trial.user_attrs.get("forced_extreme", False),
        "extreme_type": trial.user_attrs.get("extreme_type", ""),
        "user_attrs": {k: v for k, v in trial.user_attrs.items() if not k.startswith("forced_")},
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        "model_type": model_type,
    }

    trial_path = trials_dir / f"trial_{trial.number:04d}.json"
    with open(trial_path, "w") as f:
        json.dump(trial_data, f, indent=2, default=str)

    return trial_path


def update_best_params(
    study: optuna.Study,
    output_dir: Path,
    model_type: str,
) -> Path:
    """Update best parameters file after each trial."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not study.best_trial:
        return output_dir / "best_params.json"

    best_auc = -study.best_value if study.best_value is not None else None
    best_trial = study.best_trial

    # Extract config from best trial
    if best_trial.user_attrs.get("forced_extreme"):
        best_config = {
            k.replace("forced_", ""): v
            for k, v in best_trial.user_attrs.items()
            if k.startswith("forced_") and k != "forced_extreme"
        }
    else:
        best_config = best_trial.params

    # Count trial states
    n_complete = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    n_failed = sum(1 for t in study.trials if t.state.name == "FAIL")

    # Baseline comparison
    comparison = compare_to_baseline(best_auc) if best_auc else {}

    result = {
        "model_type": model_type,
        "methodology": "two_phase_v2",
        "best_params": best_config,
        "best_auc": best_auc,
        "best_trial_number": best_trial.number,
        "best_trial_forced_extreme": best_trial.user_attrs.get("forced_extreme", False),
        "best_trial_extreme_type": best_trial.user_attrs.get("extreme_type", ""),
        "best_trial_attrs": {
            k: v for k, v in best_trial.user_attrs.items()
            if not k.startswith("forced_")
        },
        "n_trials_completed": n_complete,
        "n_trials_pruned": n_pruned,
        "n_trials_failed": n_failed,
        "n_forced_extreme_trials": N_FORCED_EXTREME_TRIALS,
        "n_tpe_trials": n_complete - min(N_FORCED_EXTREME_TRIALS, n_complete),
        "baseline_comparison": comparison,
        "search_space": SEARCH_SPACE_V2,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "study_name": study.study_name,
        "optuna_version": optuna.__version__,
    }

    output_path = output_dir / "best_params.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return output_path


def save_trial_metrics_csv(
    study: optuna.Study,
    output_dir: Path,
) -> Path:
    """Save all trial metrics as CSV for analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_data = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        # Extract config
        if trial.user_attrs.get("forced_extreme"):
            config = {
                k.replace("forced_", ""): v
                for k, v in trial.user_attrs.items()
                if k.startswith("forced_") and k != "forced_extreme"
            }
        else:
            config = trial.params

        row = {
            "trial": trial.number,
            "auc": -trial.value if trial.value is not None else None,
            "accuracy": trial.user_attrs.get("val_accuracy"),
            "precision": trial.user_attrs.get("val_precision"),
            "recall": trial.user_attrs.get("val_recall"),
            "f1": trial.user_attrs.get("val_f1"),
            "pred_min": trial.user_attrs.get("pred_min"),
            "pred_max": trial.user_attrs.get("pred_max"),
            "pred_mean": trial.user_attrs.get("pred_mean"),
            "n_positive_preds": trial.user_attrs.get("n_positive_preds"),
            "forced_extreme": trial.user_attrs.get("forced_extreme", False),
            "extreme_type": trial.user_attrs.get("extreme_type", ""),
            "training_time_min": trial.user_attrs.get("training_time_min"),
            **config,
        }
        metrics_data.append(row)

    df = pd.DataFrame(metrics_data)
    output_path = output_dir / "trial_metrics.csv"
    df.to_csv(output_path, index=False)

    return output_path


def save_study_summary(
    study: optuna.Study,
    output_dir: Path,
    model_type: str,
) -> Path:
    """Save study summary as markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    best_auc = -study.best_value if study.best_value is not None else None
    comparison = compare_to_baseline(best_auc) if best_auc else {}

    n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
    n_forced = min(N_FORCED_EXTREME_TRIALS, n_complete)
    n_tpe = n_complete - n_forced

    lines = [
        f"# {model_type.title()} HPO v2 Results",
        "",
        f"**Methodology:** Two-phase (6 forced extremes + TPE)",
        f"**Timestamp:** {datetime.now().isoformat()}",
        f"**Trials Completed:** {n_complete} ({n_forced} forced + {n_tpe} TPE)",
        "",
        "## Best Configuration",
        "",
        f"**AUC:** {best_auc:.4f}" if best_auc else "**AUC:** N/A",
        f"**Target (0.65):** {'✅ Met' if best_auc and best_auc >= 0.65 else '❌ Not met'}",
        f"**Beats Baseline (0.718):** {'✅ Yes' if comparison.get('beats_baseline') else '❌ No'}",
        "",
    ]

    # Best trial info
    if study.best_trial:
        bt = study.best_trial
        lines.append(f"**Best Trial:** {bt.number}")
        if bt.user_attrs.get("forced_extreme"):
            lines.append(f"**Type:** Forced Extreme ({bt.user_attrs.get('extreme_type', '')})")
        else:
            lines.append(f"**Type:** TPE")

        lines.extend([
            "",
            "### Best Parameters",
            "",
        ])

        # Extract config
        if bt.user_attrs.get("forced_extreme"):
            config = {
                k.replace("forced_", ""): v
                for k, v in bt.user_attrs.items()
                if k.startswith("forced_") and k != "forced_extreme"
            }
        else:
            config = bt.params

        for param, value in config.items():
            if isinstance(value, float):
                lines.append(f"- **{param}:** {value:.6g}")
            else:
                lines.append(f"- **{param}:** {value}")

        # Best trial metrics
        lines.extend([
            "",
            "### Best Trial Metrics",
            "",
            f"- **Precision:** {bt.user_attrs.get('val_precision', 0):.4f}",
            f"- **Recall:** {bt.user_attrs.get('val_recall', 0):.4f}",
            f"- **Pred Range:** [{bt.user_attrs.get('pred_min', 0):.4f}, {bt.user_attrs.get('pred_max', 0):.4f}]",
        ])

    # Forced extreme results
    lines.extend([
        "",
        "## Forced Extreme Trials",
        "",
        "| Trial | Type | AUC | Hidden | Layers | Heads |",
        "|-------|------|-----|--------|--------|-------|",
    ])

    for trial in study.trials[:N_FORCED_EXTREME_TRIALS]:
        if trial.state.name != "COMPLETE":
            continue
        auc = -trial.value if trial.value is not None else 0
        extreme_type = trial.user_attrs.get("extreme_type", "")
        hidden = trial.user_attrs.get("forced_hidden_size", "-")
        layers = trial.user_attrs.get("forced_n_layers", "-")
        heads = trial.user_attrs.get("forced_n_heads", "-")
        lines.append(f"| {trial.number} | {extreme_type} | {auc:.4f} | {hidden} | {layers} | {heads} |")

    # Top TPE trials
    tpe_trials = [t for t in study.trials[N_FORCED_EXTREME_TRIALS:] if t.state.name == "COMPLETE"]
    sorted_tpe = sorted(tpe_trials, key=lambda t: t.value if t.value is not None else float('inf'))[:5]

    if sorted_tpe:
        lines.extend([
            "",
            "## Top 5 TPE Trials",
            "",
            "| Trial | AUC | Dropout | LR | Hidden | Layers | Heads | Steps | WD |",
            "|-------|-----|---------|-----|--------|--------|-------|-------|-----|",
        ])

        for t in sorted_tpe:
            auc = -t.value if t.value is not None else 0
            p = t.params
            lines.append(
                f"| {t.number} | {auc:.4f} | {p.get('dropout', '-')} | "
                f"{p.get('learning_rate', 0):.0e} | {p.get('hidden_size', '-')} | "
                f"{p.get('n_layers', '-')} | {p.get('n_heads', '-')} | "
                f"{p.get('max_steps', '-')} | {p.get('weight_decay', 0):.0e} |"
            )

    # Search space
    lines.extend([
        "",
        "## Search Space (v2 Expanded)",
        "",
    ])
    for param, values in SEARCH_SPACE_V2.items():
        lines.append(f"- **{param}:** {values}")

    output_path = output_dir / "study_summary.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    return output_path


# ============================================================================
# MAIN HPO RUNNER
# ============================================================================

def run_hpo(
    model_type: str,
    n_trials: int = 50,
    resume: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run two-phase HPO for specified NeuralForecast model.

    Args:
        model_type: 'itransformer' or 'informer'
        n_trials: Number of trials to run
        resume: If True, resume existing study
        dry_run: If True, don't save results to disk
        verbose: If True, print progress

    Returns:
        Dict with best params and metrics
    """
    print("=" * 70)
    print(f"HPO v2 for {model_type.upper()} (Two-Phase Methodology)")
    print("=" * 70)

    if not NEURALFORECAST_AVAILABLE:
        print(f"ERROR: NeuralForecast not available")
        print(f"Import error: {IMPORT_ERROR}")
        print("\nInstall with: pip install neuralforecast>=1.7.0")
        return {"error": IMPORT_ERROR}

    # Setup output directory (v2 suffix to distinguish from v1)
    output_dir = HPO_OUTPUT_BASE / f"{model_type}_v2"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("\nPreparing data...")
    data = prepare_hpo_data()
    print(f"  Train: {len(data['df_train'])} samples")
    print(f"  Val: {len(data['df_val'])} samples")

    # Compute forced extreme configs
    extreme_configs = compute_forced_extreme_configs(SEARCH_SPACE_V2)
    print(f"\nForced extreme trials ({len(extreme_configs)}):")
    for i, cfg in enumerate(extreme_configs):
        print(f"  {i}: hidden={cfg['hidden_size']}, layers={cfg['n_layers']}, heads={cfg['n_heads']}")

    # Create study (fresh start, don't load from v1)
    study_name = f"arch_hpo_{model_type}_v2"
    storage = f"sqlite:///{output_dir / 'study.db'}" if not dry_run else None

    # TPE sampler with n_startup_trials > N_FORCED_EXTREME_TRIALS
    # This ensures TPE uses forced results to inform sampling
    sampler = TPESampler(n_startup_trials=max(10, N_FORCED_EXTREME_TRIALS + 4))

    if resume and storage:
        try:
            study = optuna.load_study(
                study_name=study_name,
                storage=storage,
                sampler=sampler,
            )
            print(f"\nResumed study with {len(study.trials)} existing trials")
        except KeyError:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="minimize",  # Minimizing negative AUC
                sampler=sampler,
            )
            print("\nCreated new study")
    else:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="minimize",
            sampler=sampler,
            load_if_exists=resume,
        )
        print(f"\nCreated study: {study_name}")

    # Create objective
    objective = create_objective(model_type, data, extreme_configs, verbose=verbose)

    # Trial callback for incremental saving
    def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if dry_run:
            return

        # Save trial result
        save_trial_result(trial, output_dir, model_type)

        # Update best params
        update_best_params(study, output_dir, model_type)

        # Update CSV
        save_trial_metrics_csv(study, output_dir)

        if verbose and trial.state.name == "COMPLETE":
            best_auc = -study.best_value if study.best_value is not None else 0
            print(f"  [Best AUC so far: {best_auc:.4f}]")

    # Run optimization
    print(f"\nStarting HPO with {n_trials} trials...")
    print(f"  - {N_FORCED_EXTREME_TRIALS} forced extreme trials (Phase A)")
    print(f"  - {n_trials - N_FORCED_EXTREME_TRIALS} TPE exploration trials (Phase B)")
    print(f"\nExpanded search space:")
    for param, values in SEARCH_SPACE_V2.items():
        print(f"  {param}: {values}")
    print()

    start_time = time.time()

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[trial_callback],
            show_progress_bar=not verbose,  # Show progress bar if not verbose
        )
    except KeyboardInterrupt:
        print("\nHPO interrupted by user")

    total_time = (time.time() - start_time) / 60

    # Final results
    print("\n" + "=" * 70)
    print("HPO v2 COMPLETE")
    print("=" * 70)

    n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
    best_auc = -study.best_value if study.best_value is not None else None

    print(f"\nTrials completed: {n_complete}")
    print(f"Total time: {total_time:.1f} min")

    if best_auc is not None:
        print(f"\nBest AUC: {best_auc:.4f}")
        comparison = compare_to_baseline(best_auc)
        print(f"Target (0.65): {'✅ Met' if best_auc >= 0.65 else '❌ Not met'}")
        print(f"Beats baseline (0.718): {'✅ Yes' if comparison['beats_baseline'] else '❌ No'}")

        print(f"\nBest trial: {study.best_trial.number}")
        if study.best_trial.user_attrs.get("forced_extreme"):
            print(f"Type: Forced Extreme ({study.best_trial.user_attrs.get('extreme_type', '')})")
        else:
            print("Type: TPE")

        # Extract best config
        if study.best_trial.user_attrs.get("forced_extreme"):
            best_config = {
                k.replace("forced_", ""): v
                for k, v in study.best_trial.user_attrs.items()
                if k.startswith("forced_") and k != "forced_extreme"
            }
        else:
            best_config = study.best_params

        print("\nBest parameters:")
        for param, value in best_config.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.6g}")
            else:
                print(f"  {param}: {value}")

    # Save final summary
    if not dry_run:
        save_study_summary(study, output_dir, model_type)
        print(f"\nResults saved to: {output_dir}")

    return {
        "model_type": model_type,
        "methodology": "two_phase_v2",
        "best_params": best_config if best_auc else {},
        "best_auc": best_auc,
        "n_trials_completed": n_complete,
        "total_time_min": total_time,
        "output_dir": str(output_dir) if not dry_run else None,
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run two-phase HPO v2 for NeuralForecast alternative architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["itransformer", "informer"],
        help="Model type to optimize",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of HPO trials",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing v2 study if available",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without saving results (for testing)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    result = run_hpo(
        model_type=args.model,
        n_trials=args.trials,
        resume=args.resume,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    if "error" in result:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
