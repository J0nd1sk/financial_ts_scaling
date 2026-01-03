"""Optuna HPO integration for scaling experiments.

Provides hyperparameter optimization with thermal monitoring,
W&B/MLflow tracking integration, and persistent study storage.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import optuna
import torch
import yaml

from src.config.experiment import load_experiment_config
from src.data.dataset import SplitIndices
from src.models.arch_grid import get_memory_safe_batch_config
from src.models.configs import load_patchtst_config
from src.training.thermal import ThermalCallback
from src.training.trainer import Trainer


# Thermal pause duration in seconds when warning threshold exceeded
THERMAL_PAUSE_SECONDS = 60


def load_search_space(path: Path | str) -> dict:
    """Load search space definition from YAML.

    Args:
        path: Path to YAML file with search space definition.

    Returns:
        Dictionary with keys: n_trials, timeout_hours, direction, search_space

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If YAML is missing required 'search_space' key.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Search space file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    if "search_space" not in data:
        raise ValueError(
            f"Invalid search space config: missing 'search_space' key in {path}"
        )

    return data


def create_study(
    experiment_name: str,
    budget: str,
    storage: str | None = None,
    direction: str = "minimize",
    n_startup_trials: int = 20,
) -> optuna.Study:
    """Create Optuna study with optional persistent storage.

    Args:
        experiment_name: Name for the study (e.g., 'spy_daily_threshold_1pct')
        budget: Parameter budget ('2M', '20M', '200M', '2B')
        storage: SQLite URL for persistence, or None for in-memory
        direction: 'minimize' for loss, 'maximize' for accuracy
        n_startup_trials: Number of random trials before TPE kicks in (default 20)

    Returns:
        Optuna Study object with study_name = f"{experiment_name}_{budget}"
    """
    study_name = f"{experiment_name}_{budget}"

    # Use TPESampler with more startup trials for broader exploration
    sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        sampler=sampler,
    )

    return study


def _sample_hyperparameter(
    trial: optuna.Trial,
    name: str,
    spec: dict,
) -> Any:
    """Sample a single hyperparameter based on its specification.

    Args:
        trial: Optuna trial object
        name: Parameter name
        spec: Parameter specification with 'type' and either:
            - 'low'/'high' for numeric types (uniform, log_uniform, int)
            - 'choices' for categorical type

    Returns:
        Sampled parameter value
    """
    param_type = spec.get("type", "uniform")

    # Handle categorical separately since it uses 'choices' not 'low'/'high'
    if param_type == "categorical":
        choices = spec["choices"]
        return trial.suggest_categorical(name, choices)

    # All other types use low/high
    low = spec["low"]
    high = spec["high"]

    if param_type == "log_uniform":
        return trial.suggest_float(name, low, high, log=True)
    elif param_type == "uniform":
        return trial.suggest_float(name, low, high)
    elif param_type == "int":
        step = spec.get("step", 1)
        return trial.suggest_int(name, low, high, step=step)
    else:
        # Default to uniform
        return trial.suggest_float(name, low, high)


def get_extreme_architecture_indices(architectures: list[dict]) -> list[int]:
    """Get indices of extreme architectures to force-test first.

    Returns up to 6 indices testing extremes of d_model, n_layers, n_heads
    while keeping other dimensions at middle values.

    Args:
        architectures: List of architecture dicts with d_model, n_layers, n_heads

    Returns:
        List of architecture indices (may be fewer than 6 if some extremes
        don't exist in the grid)
    """
    if not architectures:
        return []

    # Find unique values and middle points
    d_models = sorted(set(a['d_model'] for a in architectures))
    n_layers_vals = sorted(set(a['n_layers'] for a in architectures))
    n_heads_vals = sorted(set(a['n_heads'] for a in architectures))

    middle_d = d_models[len(d_models) // 2]
    middle_L = n_layers_vals[len(n_layers_vals) // 2]
    middle_h = 8 if 8 in n_heads_vals else n_heads_vals[len(n_heads_vals) // 2]

    extreme_indices: list[int] = []

    def find_arch(d_model: int | None = None, n_layers: int | None = None,
                  n_heads: int | None = None) -> int | None:
        """Find first architecture matching criteria."""
        for i, a in enumerate(architectures):
            if d_model is not None and a['d_model'] != d_model:
                continue
            if n_layers is not None and a['n_layers'] != n_layers:
                continue
            if n_heads is not None and a['n_heads'] != n_heads:
                continue
            return i
        return None

    # 1. Min d_model (with middle h, any L)
    idx = find_arch(d_model=min(d_models), n_heads=middle_h)
    if idx is not None and idx not in extreme_indices:
        extreme_indices.append(idx)

    # 2. Max d_model (with middle h, any L)
    idx = find_arch(d_model=max(d_models), n_heads=middle_h)
    if idx is not None and idx not in extreme_indices:
        extreme_indices.append(idx)

    # 3. Min n_layers (with middle d, middle h)
    idx = find_arch(n_layers=min(n_layers_vals), d_model=middle_d, n_heads=middle_h)
    if idx is not None and idx not in extreme_indices:
        extreme_indices.append(idx)

    # 4. Max n_layers (with middle d, middle h)
    idx = find_arch(n_layers=max(n_layers_vals), d_model=middle_d, n_heads=middle_h)
    if idx is not None and idx not in extreme_indices:
        extreme_indices.append(idx)

    # 5. Min n_heads (h=2 with middle d, any L)
    idx = find_arch(n_heads=min(n_heads_vals), d_model=middle_d)
    if idx is not None and idx not in extreme_indices:
        extreme_indices.append(idx)

    # 6. Max n_heads (h=32 with middle d, any L)
    idx = find_arch(n_heads=max(n_heads_vals), d_model=middle_d)
    if idx is not None and idx not in extreme_indices:
        extreme_indices.append(idx)

    return extreme_indices


def create_architectural_objective(
    config_path: str,
    budget: str,
    architectures: list[dict],
    training_search_space: dict,
    split_indices: SplitIndices | None = None,
    num_features: int | None = None,
    verbose: bool = False,
    force_extreme_trials: bool = True,
) -> Callable[[optuna.Trial], float]:
    """Create Optuna objective function for architectural HPO.

    Searches both model architecture AND training parameters simultaneously.
    Architecture is sampled from a pre-computed list of valid configurations.

    Args:
        config_path: Path to experiment config YAML
        budget: Parameter budget
        architectures: Pre-computed list of valid architecture dicts from arch_grid
        training_search_space: Dict defining training parameter ranges
        split_indices: Optional SplitIndices for train/val/test splits
        num_features: Number of input features (required for model config)
        verbose: If True, capture and store detailed metrics per trial
        force_extreme_trials: If True, first 6 trials test architecture extremes

    Returns:
        Objective function for study.optimize()
    """
    if num_features is None:
        raise ValueError("num_features is required for architectural HPO")
    from src.models.patchtst import PatchTSTConfig

    # Pre-compute extreme indices if forcing extremes
    extreme_indices = get_extreme_architecture_indices(architectures) if force_extreme_trials else []

    def objective(trial: optuna.Trial) -> float:
        """Objective function that searches architecture and training params."""
        import logging
        logger = logging.getLogger("optuna")

        # Force extreme architectures for first N trials
        if trial.number < len(extreme_indices):
            arch_idx = extreme_indices[trial.number]
            # Record forced arch_idx without suggesting (avoids CategoricalDistribution error)
            trial.set_user_attr("arch_idx", arch_idx)
        else:
            # Random sampling for remaining trials
            arch_idx = trial.suggest_int("arch_idx", 0, len(architectures) - 1)
        arch = architectures[arch_idx]

        # Sample training params from narrow ranges
        sampled_params = {}
        for param_name, param_spec in training_search_space.items():
            sampled_params[param_name] = _sample_hyperparameter(
                trial, param_name, param_spec
            )

        # Load experiment config for dataset/task info
        experiment_config = load_experiment_config(config_path)

        # Extract training params with defaults
        learning_rate = sampled_params.get("learning_rate", 0.001)
        epochs = sampled_params.get("epochs", 50)
        dropout = sampled_params.get("dropout", 0.1)

        # Force variation when same architecture is reused with similar params
        # This ensures we learn something new from each trial
        study = trial.study
        same_arch_trials = []
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE:
                prev_arch_idx = t.params.get("arch_idx")
                if prev_arch_idx is None:
                    prev_arch_idx = t.user_attrs.get("arch_idx")
                if prev_arch_idx == arch_idx:
                    same_arch_trials.append(t)

        if same_arch_trials:
            prev = same_arch_trials[-1]  # Most recent trial with this arch
            prev_dropout = prev.params.get("dropout", 0.1)
            prev_epochs = prev.params.get("epochs", 50)

            dropout_similar = abs(dropout - prev_dropout) < 0.08
            epochs_similar = abs(epochs - prev_epochs) < 20

            if dropout_similar and epochs_similar:
                # Force dropout to opposite extreme for meaningful variation
                if prev_dropout < 0.2:
                    dropout = 0.27  # Push to high end
                else:
                    dropout = 0.12  # Push to low end
                logger.info(
                    f"  → Forced dropout={dropout:.2f} for diversity "
                    f"(was {sampled_params.get('dropout', 0.1):.2f}, prev={prev_dropout:.2f})"
                )

        # Get memory-safe batch config based on architecture size
        batch_config = get_memory_safe_batch_config(
            d_model=arch["d_model"],
            n_layers=arch["n_layers"],
        )

        # Build PatchTSTConfig dynamically from sampled architecture
        # Fixed params from design doc: patch_length=10, stride=5, context_length=60
        model_config = PatchTSTConfig(
            num_features=num_features,
            context_length=experiment_config.context_length,
            patch_length=10,  # Fixed per design doc
            stride=5,  # Fixed per design doc
            d_model=arch["d_model"],
            n_heads=arch["n_heads"],
            n_layers=arch["n_layers"],
            d_ff=arch["d_ff"],
            dropout=dropout,  # Sampled from training_search_space
            head_dropout=0.0,  # Default
            num_classes=1,  # Binary classification
        )

        # Select best available device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Log trial start with architecture info (including n_heads for recovery)
        logger.info(
            f"Trial {trial.number}: arch_idx={arch_idx}, "
            f"d_model={arch['d_model']}, n_layers={arch['n_layers']}, "
            f"n_heads={arch['n_heads']}, params={arch['param_count']:,}, "
            f"lr={learning_rate:.6f}, epochs={epochs}, "
            f"batch={batch_config['micro_batch']}x{batch_config['accumulation_steps']}, "
            f"dropout={dropout:.2f}"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                experiment_config=experiment_config,
                model_config=model_config,
                batch_size=batch_config["micro_batch"],
                learning_rate=learning_rate,
                epochs=epochs,
                device=device,
                checkpoint_dir=Path(tmp_dir),
                thermal_callback=None,  # Thermal handled at study level
                tracking_manager=None,  # Tracking handled at study level
                split_indices=split_indices,
                accumulation_steps=batch_config["accumulation_steps"],
                early_stopping_patience=10,
                early_stopping_min_delta=0.001,
            )

            result = trainer.train(verbose=verbose)

        # Store detailed metrics in trial user_attrs if verbose
        if verbose:
            trial.set_user_attr("architecture", arch)
            trial.set_user_attr("training_params", sampled_params)
            if result.get("learning_curve"):
                trial.set_user_attr("learning_curve", result["learning_curve"])
            if result.get("val_accuracy") is not None:
                trial.set_user_attr("val_accuracy", result["val_accuracy"])
            if result.get("val_confusion"):
                trial.set_user_attr("val_confusion", result["val_confusion"])
            if result.get("train_accuracy") is not None:
                trial.set_user_attr("train_accuracy", result["train_accuracy"])
            if result.get("split_stats"):
                trial.set_user_attr("split_stats", result["split_stats"])

        # Return val_loss if splits provided, otherwise train_loss
        if split_indices is not None:
            return result["val_loss"]
        return result["train_loss"]

    return objective


def create_objective(
    config_path: str,
    budget: str,
    search_space: dict,
    split_indices: SplitIndices | None = None,
) -> Callable[[optuna.Trial], float]:
    """Create Optuna objective function for HPO.

    The objective function:
    1. Samples hyperparameters from search space
    2. Creates and trains model with sampled params
    3. Returns validation loss if splits provided, else training loss

    Args:
        config_path: Path to experiment config YAML
        budget: Parameter budget
        search_space: Dict defining parameter ranges
        split_indices: Optional SplitIndices for train/val/test splits.
            If provided, objective returns val_loss instead of train_loss.

    Returns:
        Objective function for study.optimize()
    """

    def objective(trial: optuna.Trial) -> float:
        """Objective function that trains model and returns loss."""
        import logging
        logger = logging.getLogger("optuna")

        # Sample hyperparameters from search space
        sampled_params = {}
        for param_name, param_spec in search_space.items():
            sampled_params[param_name] = _sample_hyperparameter(
                trial, param_name, param_spec
            )

        # Load configs
        experiment_config = load_experiment_config(config_path)
        model_config = load_patchtst_config(budget)

        # Extract training params with defaults
        learning_rate = sampled_params.get("learning_rate", 0.001)
        epochs = sampled_params.get("epochs", 50)
        batch_size = sampled_params.get("batch_size", 32)

        # Create temporary checkpoint directory for this trial
        # Select best available device: MPS (Apple Silicon GPU) > CUDA > CPU
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # Log trial start with key parameters
        logger.info(
            f"Trial {trial.number} starting: device={device}, "
            f"lr={learning_rate:.6f}, epochs={epochs}, batch_size={batch_size}"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                experiment_config=experiment_config,
                model_config=model_config,
                batch_size=batch_size,
                learning_rate=learning_rate,
                epochs=epochs,
                device=device,
                checkpoint_dir=Path(tmp_dir),
                thermal_callback=None,  # Thermal handled at study level
                tracking_manager=None,  # Tracking handled at study level
                split_indices=split_indices,  # Pass splits to trainer
            )

            result = trainer.train()

        # Return val_loss if splits provided, otherwise train_loss
        if split_indices is not None:
            return result["val_loss"]
        return result["train_loss"]

    return objective


def save_trial_result(
    trial: optuna.trial.FrozenTrial,
    output_dir: Path | str,
    architectures: list[dict] | None = None,
) -> Path:
    """Save individual trial result to JSON file.

    Called after each trial completes for incremental logging.

    Args:
        trial: Completed Optuna trial
        output_dir: Directory to save results
        architectures: Optional list of architecture dicts

    Returns:
        Path to saved JSON file
    """
    output_dir = Path(output_dir)
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    # Build trial data
    trial_data: dict[str, Any] = {
        "trial_number": trial.number,
        "value": trial.value,
        "params": trial.params,
        "state": trial.state.name,
        "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
        "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
        "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
    }

    # Include architecture if available
    # Check user_attrs first (forced extreme trials store arch there)
    # Then fall back to architectures list lookup via arch_idx in params
    if "architecture" in trial.user_attrs:
        trial_data["architecture"] = trial.user_attrs["architecture"]
    elif architectures is not None and "arch_idx" in trial.params:
        arch_idx = trial.params["arch_idx"]
        if arch_idx < len(architectures):
            trial_data["architecture"] = architectures[arch_idx]

    # Include user_attrs (verbose data: learning_curve, confusion matrix, etc.)
    if trial.user_attrs:
        trial_data["user_attrs"] = trial.user_attrs

    # Save to individual trial file
    trial_path = trials_dir / f"trial_{trial.number:04d}.json"
    with open(trial_path, "w") as f:
        json.dump(trial_data, f, indent=2)

    return trial_path


def update_best_params(
    study: optuna.Study,
    experiment_name: str,
    budget: str,
    output_dir: Path | str,
    architectures: list[dict] | None = None,
) -> Path:
    """Update best parameters file after each trial.

    Called incrementally so current best is always saved to disk.

    Args:
        study: Optuna study (may still be running)
        experiment_name: Name of the experiment
        budget: Parameter budget
        output_dir: Directory to save results
        architectures: Optional list of architecture dicts

    Returns:
        Path to saved JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not study.best_trial:
        # No completed trials yet
        return output_dir / f"{experiment_name}_{budget}_best.json"

    # Count trial states
    n_complete = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")
    n_running = sum(1 for t in study.trials if t.state.name == "RUNNING")

    result: dict[str, Any] = {
        "experiment": experiment_name,
        "budget": budget,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "n_trials_completed": n_complete,
        "n_trials_pruned": n_pruned,
        "n_trials_running": n_running,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "study_name": study.study_name,
        "optuna_version": optuna.__version__,
    }

    # Include architecture info if available
    # Check user_attrs first (forced extreme trials store arch there)
    # Then fall back to architectures list lookup via arch_idx in params
    best_trial = study.best_trial
    if best_trial and "architecture" in best_trial.user_attrs:
        result["architecture"] = best_trial.user_attrs["architecture"]
    elif architectures is not None and "arch_idx" in study.best_params:
        arch_idx = study.best_params["arch_idx"]
        if arch_idx < len(architectures):
            result["architecture"] = architectures[arch_idx]

    output_path = output_dir / f"{experiment_name}_{budget}_best.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return output_path


def save_all_trials(
    study: optuna.Study,
    experiment_name: str,
    budget: str,
    output_dir: Path | str,
    architectures: list[dict] | None = None,
) -> Path:
    """Save summary of all trials to a single JSON file.

    Called after each trial for incremental updates.

    Args:
        study: Optuna study
        experiment_name: Name of the experiment
        budget: Parameter budget
        output_dir: Directory to save results
        architectures: Optional list of architecture dicts

    Returns:
        Path to saved JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trials_summary = []
    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue

        trial_info: dict[str, Any] = {
            "trial_number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "duration_seconds": trial.duration.total_seconds() if trial.duration else None,
        }

        # Include architecture
        # Check user_attrs first (forced extreme trials store arch there)
        # Then fall back to architectures list lookup via arch_idx in params
        arch = None
        if "architecture" in trial.user_attrs:
            arch = trial.user_attrs["architecture"]
        elif architectures is not None and "arch_idx" in trial.params:
            arch_idx = trial.params["arch_idx"]
            if arch_idx < len(architectures):
                arch = architectures[arch_idx]

        if arch is not None:
            trial_info["d_model"] = arch.get("d_model")
            trial_info["n_layers"] = arch.get("n_layers")
            trial_info["n_heads"] = arch.get("n_heads")
            trial_info["param_count"] = arch.get("param_count")

        # Include key metrics from user_attrs
        if trial.user_attrs:
            if "val_accuracy" in trial.user_attrs:
                trial_info["val_accuracy"] = trial.user_attrs["val_accuracy"]
            if "train_accuracy" in trial.user_attrs:
                trial_info["train_accuracy"] = trial.user_attrs["train_accuracy"]

        trials_summary.append(trial_info)

    # Sort by value (val_loss)
    trials_summary.sort(key=lambda x: x["value"] if x["value"] is not None else float("inf"))

    result = {
        "experiment": experiment_name,
        "budget": budget,
        "n_trials": len(trials_summary),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trials": trials_summary,
    }

    output_path = output_dir / f"{experiment_name}_all_trials.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return output_path


def save_best_params(
    study: optuna.Study,
    experiment_name: str,
    budget: str,
    output_dir: Path | str = Path("outputs/hpo"),
    architectures: list[dict] | None = None,
) -> Path:
    """Save best hyperparameters to JSON file.

    Args:
        study: Completed Optuna study
        experiment_name: Name of the experiment
        budget: Parameter budget
        output_dir: Directory to save results
        architectures: Optional list of architecture dicts for architectural HPO.
            If provided and arch_idx in best_params, includes architecture in output.

    Returns:
        Path to saved JSON file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count trial states
    n_complete = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    n_pruned = sum(1 for t in study.trials if t.state.name == "PRUNED")

    result: dict[str, Any] = {
        "experiment": experiment_name,
        "budget": budget,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials_completed": n_complete,
        "n_trials_pruned": n_pruned,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "study_name": study.study_name,
        "optuna_version": optuna.__version__,
    }

    # Include architecture info if available
    # Check user_attrs first (forced extreme trials store arch there)
    # Then fall back to architectures list lookup via arch_idx in params
    best_trial = study.best_trial
    if best_trial and "architecture" in best_trial.user_attrs:
        result["architecture"] = best_trial.user_attrs["architecture"]
    elif architectures is not None and "arch_idx" in study.best_params:
        arch_idx = study.best_params["arch_idx"]
        result["architecture"] = architectures[arch_idx]

    output_path = output_dir / f"{experiment_name}_{budget}_best.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return output_path


def run_hpo(
    config_path: str,
    budget: str,
    n_trials: int = 50,
    timeout_hours: float | None = None,
    search_space_path: str = "configs/hpo/default_search.yaml",
    thermal_callback: ThermalCallback | None = None,
) -> dict[str, Any]:
    """Run complete HPO workflow.

    1. Loads search space
    2. Creates study
    3. Runs optimization with thermal monitoring
    4. Saves best params to JSON

    Args:
        config_path: Experiment config path
        budget: Parameter budget
        n_trials: Max trials to run
        timeout_hours: Max time in hours (None = no timeout, default)
        search_space_path: Path to search space YAML
        thermal_callback: Optional thermal monitoring callback

    Returns:
        Dict with best hyperparameters, metrics, and status
    """
    # Load search space config
    search_config = load_search_space(search_space_path)
    search_space = search_config.get("search_space", {})
    direction = search_config.get("direction", "minimize")

    # Derive experiment name from config path
    config_name = Path(config_path).stem
    experiment_name = config_name.replace("spy_daily_", "").replace(".yaml", "")

    # Create study
    study = create_study(
        experiment_name=experiment_name,
        budget=budget,
        direction=direction,
    )

    # Create objective function
    objective = create_objective(
        config_path=config_path,
        budget=budget,
        search_space=search_space,
    )

    # Track early stopping
    stopped_early = False
    stop_reason = None

    # Check thermal status before starting (early abort if critical)
    # Note: Only abort on "critical" status, not "unknown" (temp read failure)
    # Unknown status should log warning but allow HPO to proceed
    if thermal_callback is not None:
        status = thermal_callback.check()
        if status.status == "critical":
            stopped_early = True
            stop_reason = "thermal"
            # Save empty results and return early
            output_path = save_best_params(
                study=study,
                experiment_name=experiment_name,
                budget=budget,
            )
            return {
                "best_params": {},
                "best_value": None,
                "n_trials": 0,
                "output_path": str(output_path),
                "stopped_early": stopped_early,
                "stop_reason": stop_reason,
            }

    # Custom callback to check thermal status between trials
    def thermal_check_callback(
        study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        nonlocal stopped_early, stop_reason

        if thermal_callback is None:
            return

        status = thermal_callback.check()

        # Only abort on critical status (not unknown/should_pause from read failures)
        if status.status == "critical":
            # Critical temperature - abort
            stopped_early = True
            stop_reason = "thermal"
            study.stop()
        elif status.status == "warning":
            # Warning temperature - pause before next trial
            time.sleep(THERMAL_PAUSE_SECONDS)

    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_hours * 3600 if timeout_hours else None,
            callbacks=[thermal_check_callback] if thermal_callback else None,
        )
    except optuna.exceptions.OptunaError:
        # Study was stopped (e.g., by thermal callback)
        pass

    # Save best params (even if stopped early)
    output_path = save_best_params(
        study=study,
        experiment_name=experiment_name,
        budget=budget,
    )

    return {
        "best_params": study.best_params if study.best_trial else {},
        "best_value": study.best_value if study.best_trial else None,
        "n_trials": len(study.trials),
        "output_path": str(output_path),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }
