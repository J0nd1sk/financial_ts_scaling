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
    from neuralforecast.losses.pytorch import DistributionLoss
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
    get_data_path,
)

from src.training.hpo_budget_extremes import (
    BUDGET_CONFIGS,
    DEFAULT_REGULARIZATION,
    generate_forced_configs,
    check_early_stopping_convergence,
    compute_budget_aware_extremes,
)

# Import torch for NeuralForecast-compatible FocalLoss
import torch


class NFCompatibleFocalLoss(torch.nn.Module):
    """NeuralForecast-compatible Focal Loss for binary classification.

    Wraps focal loss logic with the interface NeuralForecast expects.

    Args:
        gamma: Focusing parameter. Higher values down-weight easy examples more.
        alpha: Weight for positive class.
        eps: Small constant for numerical stability.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        # Required NeuralForecast attributes
        self.is_distribution_output = False
        self.outputsize_multiplier = 1
        self.output_names = [""]

    def domain_map(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Map predictions to valid probability range."""
        return torch.sigmoid(y_hat)

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        mask: torch.Tensor = None,
        **kwargs,  # Accept extra NeuralForecast args like y_insample
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            y: Ground truth, shape [B, H, 1] or [B, H]
            y_hat: Predictions (logits), shape [B, H, 1] or [B, H]
            mask: Optional mask for valid entries
            **kwargs: Extra arguments from NeuralForecast (ignored)

        Returns:
            Scalar loss tensor.
        """
        # Map logits to probabilities
        y_hat = self.domain_map(y_hat)

        # Flatten
        y = y.view(-1)
        y_hat = y_hat.view(-1)

        # Clamp for numerical stability
        y_hat = y_hat.clamp(self.eps, 1 - self.eps)

        # Compute p_t (probability of correct class)
        p_t = y_hat * y + (1 - y_hat) * (1 - y)

        # Compute alpha_t (class weight)
        alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)

        # Focal term and cross-entropy
        focal_weight = (1 - p_t) ** self.gamma
        ce = -torch.log(p_t)

        loss = alpha_t * focal_weight * ce

        # Apply mask if provided
        if mask is not None:
            mask = mask.view(-1)
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1)

        return loss.mean()


# ============================================================================
# HPO CONFIGURATION
# ============================================================================

# Default search space for architecture HPO
SEARCH_SPACE = {
    "dropout": [0.3, 0.4, 0.5],  # PatchTST found 0.5 best
    "learning_rate": [5e-5, 1e-4, 2e-4],  # Include slower LR
    "hidden_size": [64, 128, 256],  # Test capacity
    "n_layers": [2, 3, 4],  # Test depth
    "n_heads": [2, 4, 8],  # Test attention
    "max_steps": [1000, 2000],  # Current 500 too short
    "batch_size": [16, 32, 64],  # Test batch effects
}

# Extended search space with extreme values (for Stage B joint HPO)
SEARCH_SPACE_EXTENDED = {
    "dropout": [0.3, 0.4, 0.5, 0.6, 0.7],  # Higher dropout exploration
    "learning_rate": [1e-5, 2e-5, 5e-5, 1e-4, 2e-4],  # Lower LRs
    "hidden_size": [32, 64, 128, 256, 384],  # Extremes: 32, 384
    "n_layers": [2, 3, 4, 5, 6, 8],  # Extreme: 8
    "n_heads": [2, 4, 8],  # Same as default
    "max_steps": [1000, 2000, 3000],  # Longer training
    "batch_size": [16, 32, 64],  # Same as default
}

# Loss-only search space for Stage A (fixed architecture, vary gamma/alpha)
# 4 gamma × 7 alpha = 28 combinations
LOSS_SEARCH_SPACE = {
    "focal_gamma": [0.0, 0.5, 1.0, 2.0],  # gamma=0 is weighted BCE
    "focal_alpha": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Class weight
}

# Narrowed loss search for Stage B joint HPO (updated after Stage A analysis)
# Based on Stage A: only gamma=0.5, alpha=0.9 prevented collapse
JOINT_LOSS_SEARCH = {
    "focal_gamma": [0.5],   # Best from Stage A
    "focal_alpha": [0.9],   # Only value that prevented collapse
}

# Fixed architecture for loss-only HPO (from prior HPO best results)
FIXED_ARCH_FOR_LOSS_HPO = {
    "hidden_size": 128,
    "n_layers": 6,
    "n_heads": 8,
    "dropout": 0.45,
    "learning_rate": 5e-5,
    "max_steps": 2000,
    "batch_size": 32,
}

# Default focal loss parameters
FOCAL_DEFAULTS = {
    "gamma": 2.0,  # Standard focal loss gamma
    "alpha": 0.25,  # Standard focal loss alpha
}

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
# SEARCH SPACE SELECTION
# ============================================================================

def get_search_space(extended: bool = False, loss_only: bool = False) -> dict:
    """Get the appropriate search space based on mode.

    Args:
        extended: If True, use SEARCH_SPACE_EXTENDED with extreme values
        loss_only: If True, use LOSS_SEARCH_SPACE (only gamma/alpha)

    Returns:
        Search space dictionary
    """
    if loss_only:
        return LOSS_SEARCH_SPACE
    elif extended:
        return SEARCH_SPACE_EXTENDED
    else:
        return SEARCH_SPACE


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_hpo_data(data_path: Path | None = None):
    """Prepare data for HPO experiments.

    Args:
        data_path: Path to parquet file. Defaults to DATA_PATH_A20.

    Returns data in NeuralForecast panel format with evaluation metadata.
    """
    if data_path is None:
        data_path = DATA_PATH_A20

    df = pd.read_parquet(data_path)
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
        "y": df["threshold_target"],
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
    verbose: bool = False,
    loss_type: str = "bernoulli",
    pos_weight: float = 4.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    search_space: dict | None = None,
    loss_only: bool = False,
    joint_hpo: bool = False,
    input_size: int = 80,
):
    """Create Optuna objective function for NeuralForecast model HPO.

    Args:
        model_type: 'itransformer' or 'informer'
        data: Prepared data dict from prepare_hpo_data()
        verbose: If True, print progress during trials
        loss_type: 'bernoulli', 'weighted_bce', or 'focal'
        pos_weight: Positive class weight for weighted_bce loss
        focal_gamma: Gamma parameter for focal loss
        focal_alpha: Alpha parameter for focal loss
        search_space: Search space dict (if None, uses SEARCH_SPACE)
        loss_only: If True, use fixed architecture and only search loss params
        joint_hpo: If True, search both architecture and loss params
        input_size: Context length in days (default: 80)

    Returns:
        Optuna objective function
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'itransformer' or 'informer'")

    model_config = MODEL_CONFIGS[model_type]
    model_class = model_config["class"]()
    layer_param = model_config["layer_param"]
    extra_fixed = model_config.get("extra_fixed", {})

    # Use provided search space or default
    if search_space is None:
        search_space = SEARCH_SPACE

    def objective(trial: optuna.Trial) -> float:
        """Objective function that trains model and returns negative AUC (for maximization)."""
        start_time = time.time()

        # Determine architecture parameters based on mode
        if loss_only:
            # Use fixed architecture for loss-only HPO
            dropout = FIXED_ARCH_FOR_LOSS_HPO["dropout"]
            learning_rate = FIXED_ARCH_FOR_LOSS_HPO["learning_rate"]
            hidden_size = FIXED_ARCH_FOR_LOSS_HPO["hidden_size"]
            n_layers = FIXED_ARCH_FOR_LOSS_HPO["n_layers"]
            n_heads = FIXED_ARCH_FOR_LOSS_HPO["n_heads"]
            max_steps = FIXED_ARCH_FOR_LOSS_HPO["max_steps"]
            batch_size = FIXED_ARCH_FOR_LOSS_HPO["batch_size"]

            # Search only loss parameters
            trial_gamma = trial.suggest_categorical("focal_gamma", LOSS_SEARCH_SPACE["focal_gamma"])
            trial_alpha = trial.suggest_categorical("focal_alpha", LOSS_SEARCH_SPACE["focal_alpha"])
        else:
            # Sample architecture hyperparameters from search space
            dropout = trial.suggest_categorical("dropout", search_space["dropout"])
            learning_rate = trial.suggest_categorical("learning_rate", search_space["learning_rate"])
            hidden_size = trial.suggest_categorical("hidden_size", search_space["hidden_size"])
            n_layers = trial.suggest_categorical("n_layers", search_space["n_layers"])
            n_heads = trial.suggest_categorical("n_heads", search_space["n_heads"])
            max_steps = trial.suggest_categorical("max_steps", search_space["max_steps"])
            batch_size = trial.suggest_categorical("batch_size", search_space["batch_size"])

            if joint_hpo:
                # Also search loss parameters in joint mode
                trial_gamma = trial.suggest_categorical("focal_gamma", JOINT_LOSS_SEARCH["focal_gamma"])
                trial_alpha = trial.suggest_categorical("focal_alpha", JOINT_LOSS_SEARCH["focal_alpha"])
            else:
                # Use fixed focal params
                trial_gamma = focal_gamma
                trial_alpha = focal_alpha

        if verbose:
            if loss_only:
                print(f"  Trial {trial.number}: gamma={trial_gamma}, alpha={trial_alpha} "
                      f"(fixed arch: hidden={hidden_size}, layers={n_layers})")
            elif joint_hpo:
                print(f"  Trial {trial.number}: dropout={dropout}, lr={learning_rate}, "
                      f"hidden={hidden_size}, layers={n_layers}, heads={n_heads}, "
                      f"gamma={trial_gamma}, alpha={trial_alpha}")
            else:
                print(f"  Trial {trial.number}: dropout={dropout}, lr={learning_rate}, "
                      f"hidden={hidden_size}, layers={n_layers}, heads={n_heads}, "
                      f"steps={max_steps}, batch={batch_size}")

        # Select loss function
        if loss_type == "focal" or loss_only or joint_hpo:
            # Use NeuralForecast-compatible focal loss wrapper
            loss_fn = NFCompatibleFocalLoss(gamma=trial_gamma, alpha=trial_alpha)
        elif loss_type == "weighted_bce":
            from src.training.losses import WeightedBCELoss
            loss_fn = WeightedBCELoss(pos_weight=pos_weight)
        else:
            loss_fn = DistributionLoss(distribution='Bernoulli')

        # Build model kwargs
        model_kwargs = {
            "h": FIXED_PARAMS["h"],
            "input_size": input_size,
            "hidden_size": hidden_size,
            "n_heads": n_heads if model_type == "itransformer" else None,
            "n_head": n_heads if model_type == "informer" else None,
            layer_param: n_layers,
            "learning_rate": learning_rate,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "dropout": dropout,
            "loss": loss_fn,
            "random_seed": FIXED_PARAMS["random_seed"],
            **extra_fixed,
        }

        # Remove None values
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

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

            # Evaluate (is_classification=True because model outputs Bernoulli probabilities)
            metrics = evaluate_forecasting_model(
                predicted_returns=val_preds,
                actual_returns=data["val_returns"][:len(val_preds)],
                threshold_targets=data["val_targets"][:len(val_preds)],
                is_classification=True,
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
    loss_type: str = "bernoulli",
    pos_weight: float = 4.0,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    extended: bool = False,
    loss_only: bool = False,
    joint_hpo: bool = False,
    input_size: int = 80,
    data_tier: str = "a20",
    forced_extremes: bool = False,
    budgets: list[str] | None = None,
    early_stop_patience: int = 20,
    early_stop_threshold: float = 0.02,
    supplementary: bool = False,
    param_budget: str | None = None,
) -> dict[str, Any]:
    """Run HPO for specified NeuralForecast model.

    Args:
        model_type: 'itransformer' or 'informer'
        n_trials: Number of trials to run
        resume: If True, resume existing study
        dry_run: If True, don't save results to disk
        verbose: If True, print progress
        loss_type: 'bernoulli', 'weighted_bce', or 'focal'
        pos_weight: Positive class weight for weighted_bce loss
        focal_gamma: Gamma parameter for focal loss (if not searching)
        focal_alpha: Alpha parameter for focal loss (if not searching)
        extended: If True, use extended search space with extremes
        loss_only: If True, use fixed architecture, only search loss params
        joint_hpo: If True, search both architecture and loss params
        input_size: Context length in days (default: 80)
        data_tier: Data tier to use ('a20', 'a100', 'a200'). Default: 'a20'
        forced_extremes: If True, use forced extreme configs (18 configs)
        budgets: List of parameter budgets to include (default: all 4)
        early_stop_patience: Minimum trials before checking early stopping
        early_stop_threshold: AUC range threshold for convergence
        supplementary: If True, Phase 2 supplementary mode
        param_budget: Target budget for supplementary mode

    Returns:
        Dict with best params and metrics
    """
    # Determine mode string for display
    mode_str = "LOSS-ONLY" if loss_only else ("JOINT" if joint_hpo else "ARCHITECTURE")
    search_space = get_search_space(extended=extended, loss_only=loss_only)

    print("=" * 70)
    print(f"HPO for {model_type.upper()} ({mode_str} mode)")
    if loss_type == "focal" or loss_only or joint_hpo:
        if loss_only or joint_hpo:
            print(f"Loss: focal (searching gamma/alpha)")
        else:
            print(f"Loss: focal (gamma={focal_gamma}, alpha={focal_alpha})")
    elif loss_type == "weighted_bce":
        print(f"Loss: weighted_bce (pos_weight={pos_weight})")
    else:
        print(f"Loss: {loss_type}")
    if extended:
        print("Search space: EXTENDED (with extremes)")
    print("=" * 70)

    if not NEURALFORECAST_AVAILABLE:
        print(f"ERROR: NeuralForecast not available")
        print(f"Import error: {IMPORT_ERROR}")
        print("\nInstall with: pip install neuralforecast>=1.7.0")
        return {"error": IMPORT_ERROR}

    # Setup output directory (include context length and data tier in path)
    output_dir = HPO_OUTPUT_BASE / f"{model_type}_ctx{input_size}_{data_tier}"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Override FIXED_PARAMS input_size with provided value
    effective_input_size = input_size

    # Get data path for the selected tier
    data_path = get_data_path(data_tier)

    # Prepare data
    print("\nPreparing data...")
    print(f"  Data tier: {data_tier}")
    print(f"  Data path: {data_path}")
    print(f"  Context length: {effective_input_size} days")
    data = prepare_hpo_data(data_path=data_path)
    print(f"  Train: {len(data['df_train'])} samples")
    print(f"  Val: {len(data['df_val'])} samples")

    # Create study with mode-specific name (include context length)
    if loss_only:
        study_name = f"loss_hpo_{model_type}_ctx{input_size}"
    elif joint_hpo:
        study_name = f"joint_hpo_{model_type}_ctx{input_size}"
    else:
        study_name = f"arch_hpo_{model_type}_ctx{input_size}"
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
    objective = create_objective(
        model_type,
        data,
        verbose=verbose,
        loss_type=loss_type,
        pos_weight=pos_weight,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
        search_space=search_space,
        loss_only=loss_only,
        joint_hpo=joint_hpo,
        input_size=effective_input_size,
    )

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
    if loss_only:
        print(f"Fixed architecture: {FIXED_ARCH_FOR_LOSS_HPO}")
        print(f"Loss search space: {LOSS_SEARCH_SPACE}")
    else:
        print(f"Search space: {search_space}")
        if joint_hpo:
            print(f"Joint loss search: {JOINT_LOSS_SEARCH}")
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
        "data_tier": data_tier,
        "loss_type": loss_type if not (loss_only or joint_hpo) else "focal",
        "pos_weight": pos_weight if loss_type == "weighted_bce" else None,
        "focal_gamma": focal_gamma if loss_type == "focal" else None,
        "focal_alpha": focal_alpha if loss_type == "focal" else None,
        "mode": "loss_only" if loss_only else ("joint_hpo" if joint_hpo else "architecture"),
        "extended": extended,
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
    parser.add_argument(
        "--loss-type",
        type=str,
        default="bernoulli",
        choices=["bernoulli", "weighted_bce", "focal"],
        help="Loss function type: bernoulli, weighted_bce, or focal",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=4.0,
        help="Positive class weight for weighted_bce loss (default: 4.0 for ~20%% positive rate)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=FOCAL_DEFAULTS["gamma"],
        help=f"Focal loss gamma parameter (default: {FOCAL_DEFAULTS['gamma']})",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=FOCAL_DEFAULTS["alpha"],
        help=f"Focal loss alpha parameter (default: {FOCAL_DEFAULTS['alpha']})",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Use extended search space with extreme values (hidden=32-384, layers=2-8)",
    )
    parser.add_argument(
        "--loss-only",
        action="store_true",
        help="Loss-only HPO: fix architecture, search only gamma/alpha (Stage A)",
    )
    parser.add_argument(
        "--joint-hpo",
        action="store_true",
        help="Joint HPO: search both architecture and loss params (Stage B)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=80,
        help="Context length in days (default: 80)",
    )
    parser.add_argument(
        "--data-tier",
        type=str,
        default="a20",
        choices=["a20", "a100", "a200"],
        help="Data tier to use (default: a20)",
    )
    # Budget-aware forced extremes flags
    parser.add_argument(
        "--forced-extremes",
        action="store_true",
        help="Use forced extreme configurations (18 configs covering all budgets)",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        nargs="+",
        default=["750k", "2M", "20M", "200M"],
        help="Parameter budgets to include (default: all 4)",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Minimum trials before checking early stopping (default: 20)",
    )
    parser.add_argument(
        "--early-stop-threshold",
        type=float,
        default=0.02,
        help="AUC range threshold for early stopping convergence (default: 0.02)",
    )
    # Phase 2 supplementary mode flags
    parser.add_argument(
        "--supplementary",
        action="store_true",
        help="Phase 2 supplementary mode: focus on specific budget",
    )
    parser.add_argument(
        "--param-budget",
        type=str,
        choices=["750k", "2M", "20M", "200M"],
        help="Target parameter budget for supplementary mode",
    )

    args = parser.parse_args()

    result = run_hpo(
        model_type=args.model,
        n_trials=args.trials,
        resume=args.resume,
        dry_run=args.dry_run,
        verbose=not args.quiet,
        loss_type=args.loss_type,
        pos_weight=args.pos_weight,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        extended=args.extended,
        loss_only=args.loss_only,
        joint_hpo=args.joint_hpo,
        input_size=args.input_size,
        data_tier=args.data_tier,
        forced_extremes=args.forced_extremes,
        budgets=args.budgets,
        early_stop_patience=args.early_stop_patience,
        early_stop_threshold=args.early_stop_threshold,
        supplementary=args.supplementary,
        param_budget=args.param_budget,
    )

    if "error" in result:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
