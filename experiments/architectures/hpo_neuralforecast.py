#!/usr/bin/env python3
"""
HPO for NeuralForecast Alternative Architectures (iTransformer, Informer).

This script runs proper hyperparameter optimization on alternative architectures
to enable fair comparison with PatchTST which went through 50+ trials of HPO.

Key insight from PatchTST: dropout=0.5 was critical for preventing probability collapse.
We need to test high dropout values on alternative architectures.

Usage:
    # Dry run (3 trials, no file output)
    python experiments/architectures/hpo_neuralforecast.py --model itransformer --trials 3 --dry-run

    # Full run
    python experiments/architectures/hpo_neuralforecast.py --model itransformer --trials 50

    # Resume interrupted study
    python experiments/architectures/hpo_neuralforecast.py --model itransformer --trials 50 --resume

Success criteria: AUC >= 0.70 (comparable to PatchTST 0.718)
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
# HPO CONFIGURATION
# ============================================================================

# Search space based on plan
SEARCH_SPACE = {
    "dropout": [0.3, 0.4, 0.5],  # PatchTST found 0.5 best
    "learning_rate": [5e-5, 1e-4, 2e-4],  # Include slower LR
    "hidden_size": [64, 128, 256],  # Test capacity
    "n_layers": [2, 3, 4],  # Test depth
    "n_heads": [2, 4, 8],  # Test attention
    "max_steps": [1000, 2000],  # Current 500 too short
    "batch_size": [16, 32, 64],  # Test batch effects
}

# Fixed parameters
FIXED_PARAMS = {
    "input_size": 80,  # Context length
    "h": 1,  # Forecast horizon
    "random_seed": 42,
    "early_stop_patience_steps": 10,
    "val_size": 100,  # Validation samples for early stopping
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

def create_objective(model_type: str, data: dict, verbose: bool = False):
    """Create Optuna objective function for NeuralForecast model HPO.

    Args:
        model_type: 'itransformer' or 'informer'
        data: Prepared data dict from prepare_hpo_data()
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

        # Sample hyperparameters
        dropout = trial.suggest_categorical("dropout", SEARCH_SPACE["dropout"])
        learning_rate = trial.suggest_categorical("learning_rate", SEARCH_SPACE["learning_rate"])
        hidden_size = trial.suggest_categorical("hidden_size", SEARCH_SPACE["hidden_size"])
        n_layers = trial.suggest_categorical("n_layers", SEARCH_SPACE["n_layers"])
        n_heads = trial.suggest_categorical("n_heads", SEARCH_SPACE["n_heads"])
        max_steps = trial.suggest_categorical("max_steps", SEARCH_SPACE["max_steps"])
        batch_size = trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"])

        if verbose:
            print(f"  Trial {trial.number}: dropout={dropout}, lr={learning_rate}, "
                  f"hidden={hidden_size}, layers={n_layers}, heads={n_heads}, "
                  f"steps={max_steps}, batch={batch_size}")

        # Build model kwargs
        model_kwargs = {
            "h": FIXED_PARAMS["h"],
            "input_size": FIXED_PARAMS["input_size"],
            "hidden_size": hidden_size,
            "n_heads": n_heads if model_type == "itransformer" else None,
            "n_head": n_heads if model_type == "informer" else None,
            layer_param: n_layers,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "dropout": dropout,
            "loss": MSE(),
            "random_seed": FIXED_PARAMS["random_seed"],
            "early_stop_patience_steps": FIXED_PARAMS["early_stop_patience_steps"],
            **extra_fixed,
        }

        # Remove None values
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

        try:
            # Initialize model
            model = model_class(**model_kwargs)

            # Create NeuralForecast wrapper
            nf = NeuralForecast(models=[model], freq="D")

            # Train (val_size required for early stopping)
            nf.fit(df=data["df_train"], val_size=FIXED_PARAMS["val_size"])

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

            # Store metrics in trial user_attrs
            trial.set_user_attr("val_auc", auc)
            trial.set_user_attr("val_accuracy", metrics.get("accuracy", 0))
            trial.set_user_attr("val_precision", metrics.get("precision", 0))
            trial.set_user_attr("val_recall", metrics.get("recall", 0))
            trial.set_user_attr("val_f1", metrics.get("f1", 0))
            trial.set_user_attr("pred_range", [metrics.get("pred_min", 0), metrics.get("pred_max", 0)])
            trial.set_user_attr("training_time_min", training_time)

            if verbose:
                print(f"    -> AUC={auc:.4f}, Acc={metrics.get('accuracy', 0):.4f}, "
                      f"Recall={metrics.get('recall', 0):.4f}, Time={training_time:.1f}min")

            # Return negative AUC for maximization (Optuna minimizes by default)
            return -auc

        except Exception as e:
            if verbose:
                print(f"    -> Trial failed: {e}")
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

    trial_data = {
        "trial_number": trial.number,
        "value": trial.value,
        "auc": -trial.value if trial.value is not None else None,  # Convert back from negative
        "params": trial.params,
        "state": trial.state.name,
        "user_attrs": dict(trial.user_attrs) if trial.user_attrs else {},
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

    # Count trial states
    n_complete = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    n_failed = sum(1 for t in study.trials if t.state.name == "FAIL")

    # Baseline comparison
    comparison = compare_to_baseline(best_auc) if best_auc else {}

    result = {
        "model_type": model_type,
        "best_params": study.best_params,
        "best_auc": best_auc,
        "best_trial_number": study.best_trial.number,
        "best_trial_attrs": dict(study.best_trial.user_attrs) if study.best_trial.user_attrs else {},
        "n_trials_completed": n_complete,
        "n_trials_pruned": n_pruned,
        "n_trials_failed": n_failed,
        "baseline_comparison": comparison,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "study_name": study.study_name,
        "optuna_version": optuna.__version__,
    }

    output_path = output_dir / "best_params.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

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

    lines = [
        f"# {model_type.title()} HPO Results",
        "",
        f"**Timestamp:** {datetime.now().isoformat()}",
        f"**Trials Completed:** {len([t for t in study.trials if t.state.name == 'COMPLETE'])}",
        "",
        "## Best Configuration",
        "",
        f"**AUC:** {best_auc:.4f}" if best_auc else "**AUC:** N/A",
        f"**Meets Threshold (0.70):** {'Yes' if comparison.get('meets_threshold') else 'No'}",
        f"**Beats Baseline (0.718):** {'Yes' if comparison.get('beats_baseline') else 'No'}",
        "",
        "### Best Parameters",
        "",
    ]

    if study.best_params:
        for param, value in study.best_params.items():
            if isinstance(value, float):
                lines.append(f"- **{param}:** {value:.6g}")
            else:
                lines.append(f"- **{param}:** {value}")

    # Top 5 trials
    completed_trials = [t for t in study.trials if t.state.name == "COMPLETE"]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value if t.value is not None else float('inf'))[:5]

    lines.extend([
        "",
        "## Top 5 Trials",
        "",
        "| Trial | AUC | Dropout | LR | Hidden | Layers | Heads | Steps |",
        "|-------|-----|---------|-----|--------|--------|-------|-------|",
    ])

    for t in sorted_trials:
        auc = -t.value if t.value is not None else 0
        p = t.params
        lines.append(
            f"| {t.number} | {auc:.4f} | {p.get('dropout', '-')} | "
            f"{p.get('learning_rate', 0):.0e} | {p.get('hidden_size', '-')} | "
            f"{p.get('n_layers', '-')} | {p.get('n_heads', '-')} | {p.get('max_steps', '-')} |"
        )

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
    """Run HPO for specified NeuralForecast model.

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
    print(f"HPO for {model_type.upper()}")
    print("=" * 70)

    if not NEURALFORECAST_AVAILABLE:
        print(f"ERROR: NeuralForecast not available")
        print(f"Import error: {IMPORT_ERROR}")
        print("\nInstall with: pip install neuralforecast>=1.7.0")
        return {"error": IMPORT_ERROR}

    # Setup output directory
    output_dir = HPO_OUTPUT_BASE / model_type
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("\nPreparing data...")
    data = prepare_hpo_data()
    print(f"  Train: {len(data['df_train'])} samples")
    print(f"  Val: {len(data['df_val'])} samples")

    # Create study
    study_name = f"arch_hpo_{model_type}"
    storage = f"sqlite:///{output_dir / 'study.db'}" if not dry_run else None

    sampler = TPESampler(n_startup_trials=min(20, n_trials // 2))

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
    objective = create_objective(model_type, data, verbose=verbose)

    # Trial callback for incremental saving
    def trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if dry_run:
            return

        # Save trial result
        save_trial_result(trial, output_dir, model_type)

        # Update best params
        update_best_params(study, output_dir, model_type)

        if verbose and trial.state.name == "COMPLETE":
            best_auc = -study.best_value if study.best_value is not None else 0
            print(f"  Trial {trial.number} complete. Best AUC so far: {best_auc:.4f}")

    # Run optimization
    print(f"\nStarting HPO with {n_trials} trials...")
    print(f"Search space: {SEARCH_SPACE}")
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
    print("HPO COMPLETE")
    print("=" * 70)

    n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
    best_auc = -study.best_value if study.best_value is not None else None

    print(f"\nTrials completed: {n_complete}")
    print(f"Total time: {total_time:.1f} min")

    if best_auc is not None:
        print(f"\nBest AUC: {best_auc:.4f}")
        comparison = compare_to_baseline(best_auc)
        print(f"Meets threshold (0.70): {'Yes' if comparison['meets_threshold'] else 'No'}")
        print(f"Beats baseline (0.718): {'Yes' if comparison['beats_baseline'] else 'No'}")

        print("\nBest parameters:")
        for param, value in study.best_params.items():
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
        "best_params": study.best_params if study.best_trial else {},
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
        description="Run HPO for NeuralForecast alternative architectures",
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
        help="Resume existing study if available",
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
