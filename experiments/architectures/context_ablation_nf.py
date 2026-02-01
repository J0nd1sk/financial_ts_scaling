#!/usr/bin/env python3
"""
Context Length Ablation for NeuralForecast Architectures (iTransformer, Informer).

This script runs single-model evaluations with fixed hyperparameters at specified
context lengths to study the effect of context window size on model performance.

Uses best hyperparameters from HPO runs (at 80d context) and evaluates at different
context lengths (60d, 80d, 120d) to understand context sensitivity.

Usage:
    # Run iTransformer at 60-day context
    python experiments/architectures/context_ablation_nf.py --model itransformer --context-length 60

    # Run Informer at 120-day context
    python experiments/architectures/context_ablation_nf.py --model informer --context-length 120

    # Dry run (no file output)
    python experiments/architectures/context_ablation_nf.py --model itransformer --context-length 60 --dry-run

    # Validate 80d matches HPO results
    python experiments/architectures/context_ablation_nf.py --model itransformer --context-length 80
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

# Check NeuralForecast availability
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import iTransformer, Informer
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

# Import focal loss from hpo_neuralforecast
from experiments.architectures.hpo_neuralforecast import NFCompatibleFocalLoss

# ============================================================================
# BEST HYPERPARAMETERS FROM HPO (at 80-day context)
# ============================================================================

# iTransformer best config from HPO
# Source: outputs/hpo/architectures/itransformer/best_params.json
ITRANSFORMER_BEST = {
    "hidden_size": 32,
    "learning_rate": 1e-5,
    "max_steps": 3000,
    "dropout": 0.4,
    "n_layers": 6,
    "n_heads": 4,
    "batch_size": 32,
    "focal_gamma": 0.5,
    "focal_alpha": 0.9,
}

# Informer best config from HPO
# Source: outputs/hpo/architectures/informer/best_params.json
INFORMER_BEST = {
    "hidden_size": 256,
    "learning_rate": 1e-4,
    "max_steps": 1000,
    "dropout": 0.4,
    "n_layers": 2,
    "n_heads": 2,
    "batch_size": 16,
    "focal_gamma": 0.5,
    "focal_alpha": 0.9,
}

# Model-specific parameter mappings
MODEL_CONFIGS = {
    "itransformer": {
        "class": lambda: iTransformer,
        "layer_param": "e_layers",
        "extra_fixed": {"n_series": 1},
        "best_params": ITRANSFORMER_BEST,
    },
    "informer": {
        "class": lambda: Informer,
        "layer_param": "encoder_layers",
        "extra_fixed": {
            "decoder_layers": 1,
            "conv_hidden_size": 32,
            "factor": 5,
        },
        "best_params": INFORMER_BEST,
    },
}

# Fixed parameters
FIXED_PARAMS = {
    "h": 1,  # Forecast horizon
    "random_seed": 42,
}

# Output directory for context ablation experiments
CONTEXT_ABLATION_OUTPUT = OUTPUT_BASE / "context_ablation"


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_hpo_data():
    """Prepare data for evaluation in NeuralForecast panel format.

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
# SINGLE EVALUATION
# ============================================================================

def run_single_eval(
    model_type: str,
    context_length: int,
    data: dict,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run single model evaluation with fixed hyperparameters.

    Args:
        model_type: 'itransformer' or 'informer'
        context_length: Context window size in days
        data: Prepared data dict from prepare_hpo_data()
        verbose: If True, print progress

    Returns:
        Dictionary with evaluation metrics
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'itransformer' or 'informer'")

    model_config = MODEL_CONFIGS[model_type]
    model_class = model_config["class"]()
    layer_param = model_config["layer_param"]
    extra_fixed = model_config.get("extra_fixed", {})
    best_params = model_config["best_params"]

    if verbose:
        print(f"\nRunning {model_type} with context_length={context_length}")
        print(f"Best params: {best_params}")

    # Create focal loss with best params
    loss_fn = NFCompatibleFocalLoss(
        gamma=best_params["focal_gamma"],
        alpha=best_params["focal_alpha"],
    )

    # Build model kwargs
    model_kwargs = {
        "h": FIXED_PARAMS["h"],
        "input_size": context_length,
        "hidden_size": best_params["hidden_size"],
        "n_heads": best_params["n_heads"] if model_type == "itransformer" else None,
        "n_head": best_params["n_heads"] if model_type == "informer" else None,
        layer_param: best_params["n_layers"],
        "learning_rate": best_params["learning_rate"],
        "max_steps": best_params["max_steps"],
        "batch_size": best_params["batch_size"],
        "dropout": best_params["dropout"],
        "loss": loss_fn,
        "random_seed": FIXED_PARAMS["random_seed"],
        **extra_fixed,
    }

    # Remove None values
    model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None}

    start_time = time.time()

    try:
        # Initialize model
        model = model_class(**model_kwargs)

        # Create NeuralForecast wrapper
        nf = NeuralForecast(models=[model], freq="D")

        if verbose:
            print("Training model...")

        # Train
        nf.fit(df=data["df_train"])

        if verbose:
            print("Running cross-validation...")

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
            is_classification=True,
        )

        training_time = (time.time() - start_time) / 60

        # Add context-specific info
        metrics["context_length"] = context_length
        metrics["model_type"] = model_type
        metrics["training_time_min"] = training_time
        metrics["best_params"] = best_params

        if verbose:
            print(f"\nResults:")
            print(f"  AUC: {metrics.get('auc', 'N/A'):.4f}" if metrics.get('auc') else "  AUC: N/A")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall: {metrics.get('recall', 0):.4f}")
            print(f"  Pred Range: [{metrics.get('pred_min', 0):.4f}, {metrics.get('pred_max', 0):.4f}]")
            print(f"  Training time: {training_time:.1f} min")

        return metrics

    except Exception as e:
        if verbose:
            print(f"Error during evaluation: {e}")
        return {
            "error": str(e),
            "context_length": context_length,
            "model_type": model_type,
        }


# ============================================================================
# RESULT SAVING
# ============================================================================

def save_results(
    metrics: dict,
    output_dir: Path,
    model_type: str,
    context_length: int,
) -> Path:
    """Save evaluation results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        **metrics,
        "baseline_comparison": compare_to_baseline(metrics.get("auc", 0)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    return results_path


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_context_ablation(
    model_type: str,
    context_length: int = 80,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run context length ablation experiment.

    Args:
        model_type: 'itransformer' or 'informer'
        context_length: Context window size in days
        dry_run: If True, don't save results to disk
        verbose: If True, print progress

    Returns:
        Dict with evaluation metrics
    """
    print("=" * 70)
    print(f"CONTEXT ABLATION: {model_type.upper()} @ {context_length}d context")
    print("=" * 70)

    if not NEURALFORECAST_AVAILABLE:
        print(f"ERROR: NeuralForecast not available")
        print(f"Import error: {IMPORT_ERROR}")
        print("\nInstall with: pip install neuralforecast>=1.7.0")
        return {"error": IMPORT_ERROR}

    # Setup output directory
    output_dir = CONTEXT_ABLATION_OUTPUT / model_type / f"ctx{context_length}"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("\nPreparing data...")
    data = prepare_hpo_data()
    print(f"  Train: {len(data['df_train'])} samples")
    print(f"  Val: {len(data['df_val'])} samples")

    # Run evaluation
    metrics = run_single_eval(
        model_type=model_type,
        context_length=context_length,
        data=data,
        verbose=verbose,
    )

    # Save results
    if not dry_run and "error" not in metrics:
        save_results(metrics, output_dir, model_type, context_length)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    if "error" not in metrics:
        auc = metrics.get("auc")
        comparison = compare_to_baseline(auc) if auc else {}
        print(f"\nAUC: {auc:.4f}" if auc else "\nAUC: N/A")
        print(f"Meets threshold (0.70): {'Yes' if comparison.get('meets_threshold') else 'No'}")
        print(f"Beats baseline (0.718): {'Yes' if comparison.get('beats_baseline') else 'No'}")

    return metrics


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run context length ablation for NeuralForecast architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["itransformer", "informer"],
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=80,
        help="Context window size in days",
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

    result = run_context_ablation(
        model_type=args.model,
        context_length=args.context_length,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    if "error" in result:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
