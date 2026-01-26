#!/usr/bin/env python3
"""
Alternative Architecture Investigation: iTransformer (ARCH-01)

iTransformer uses "inverted" attention - attending across features/variables
instead of across time steps. This may capture cross-feature relationships
better than PatchTST's temporal attention.

Approach: Forecasting → Threshold
- Train iTransformer to forecast next-day returns
- At inference: prediction > threshold → positive class
- Sweep thresholds for full precision-recall curve

Baseline to beat:
    PatchTST 200M H1: AUC 0.718
    P@90% precision: 4% recall
    P@75% precision: 23% recall

Success criteria:
    AUC >= 0.70 AND better precision-recall tradeoff

Reference:
    Liu et al., "iTransformer: Inverted Transformers Are Effective for
    Time Series Forecasting", ICLR 2024
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# Check NeuralForecast availability
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import iTransformer
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
    format_results_summary,
    save_results,
)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "itransformer_forecast"
MODEL_NAME = "iTransformer"
EXPERIMENT_ID = "ARCH-01"

# Model configuration
# Note: loss is set at model initialization, not in CONFIG
# early_stop disabled - conflicts with cross_validation
CONFIG = {
    "input_size": 80,      # Context length (matches PatchTST)
    "h": 1,                # Forecast horizon
    "n_series": 20,        # Number of input features (a20 tier)
    "hidden_size": 128,    # d_model equivalent
    "n_heads": 4,          # Attention heads
    "e_layers": 3,         # Encoder layers
    "learning_rate": 1e-4,
    "max_steps": 500,      # Training steps (reduced for faster iteration)
    "batch_size": 32,
}

OUTPUT_DIR = OUTPUT_BASE / EXPERIMENT_NAME


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_neuralforecast_data():
    """Prepare data in NeuralForecast panel format.

    NeuralForecast requires panel data with:
    - unique_id: time series identifier
    - ds: datetime
    - y: target (return for forecasting)
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
    df["threshold_target"] = (df["High"].shift(-1) >= df["Close"] * 1.01).astype(float)

    # Drop rows with NaN
    df = df.dropna(subset=["return", "threshold_target"]).reset_index(drop=True)

    # NeuralForecast format: each feature becomes a separate time series
    # OR use multivariate with exogenous variables
    # For simplicity, use Close price as y with features as static/dynamic exog

    # Panel format with single series
    df_nf = pd.DataFrame({
        "unique_id": "SPY",
        "ds": df["Date"],
        "y": df["return"],
    })

    # Store metadata for evaluation
    metadata = {
        "threshold_targets": df["threshold_target"].values,
        "close_prices": df["Close"].values,
        "dates": df["Date"].values,
        "actual_returns": df["return"].values,
    }

    # Split by date
    val_start = pd.Timestamp("2023-01-01")
    test_start = pd.Timestamp("2025-01-01")

    df_train = df_nf[df_nf["ds"] < val_start].copy()
    df_val = df_nf[(df_nf["ds"] >= val_start) & (df_nf["ds"] < test_start)].copy()
    df_test = df_nf[df_nf["ds"] >= test_start].copy()

    # Get corresponding metadata indices
    train_mask = df["Date"] < val_start
    val_mask = (df["Date"] >= val_start) & (df["Date"] < test_start)
    test_mask = df["Date"] >= test_start

    print(f"Data loaded from {DATA_PATH_A20}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(df_train)} samples")
    print(f"  Val: {len(df_val)} samples")
    print(f"  Test: {len(df_test)} samples")
    print(f"  Return range: [{df_nf['y'].min():.4f}, {df_nf['y'].max():.4f}]")

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
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run iTransformer forecasting experiment."""
    print("=" * 70)
    print(f"ARCH-01: {MODEL_NAME} Forecasting Experiment")
    print("=" * 70)

    if not NEURALFORECAST_AVAILABLE:
        print(f"ERROR: NeuralForecast not available")
        print(f"Import error: {IMPORT_ERROR}")
        print("\nInstall with: pip install neuralforecast>=1.7.0")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare data
    print("\nPreparing data...")
    data = prepare_neuralforecast_data()

    # Initialize model
    print(f"\nInitializing {MODEL_NAME}...")
    model = iTransformer(
        h=CONFIG["h"],
        input_size=CONFIG["input_size"],
        n_series=1,  # Univariate forecasting
        hidden_size=CONFIG["hidden_size"],
        n_heads=CONFIG["n_heads"],
        e_layers=CONFIG["e_layers"],
        learning_rate=CONFIG["learning_rate"],
        max_steps=CONFIG["max_steps"],
        batch_size=CONFIG["batch_size"],
        loss=MSE(),  # NeuralForecast requires loss objects, not strings
        random_seed=42,
    )

    print(f"Model config: {CONFIG}")

    # Create NeuralForecast wrapper
    nf = NeuralForecast(
        models=[model],
        freq="D",  # Daily frequency
    )

    # Train
    print(f"\nTraining for up to {CONFIG['max_steps']} steps...")
    start_time = time.time()

    # NeuralForecast fit uses all data up to val period
    nf.fit(df=data["df_train"])

    training_time = (time.time() - start_time) / 60
    print(f"Training completed in {training_time:.1f} min")

    # Predict on validation period
    print("\nGenerating predictions...")

    # Cross-validation prediction for val set
    val_predictions = nf.predict(df=data["df_train"])

    # For test, need to refit or use cross-validation
    # Simpler approach: predict step by step or use cross_validation

    # Use cross_validation for proper evaluation
    # refit=False uses the already-trained model for prediction
    cv_results = nf.cross_validation(
        df=data["df_full"],
        step_size=1,
        n_windows=len(data["df_val"]) + len(data["df_test"]),
        refit=False,
    )

    # Extract predictions aligned with val/test periods
    cv_results["ds"] = pd.to_datetime(cv_results["ds"])

    val_start = pd.Timestamp("2023-01-01")
    test_start = pd.Timestamp("2025-01-01")

    val_preds = cv_results[(cv_results["ds"] >= val_start) & (cv_results["ds"] < test_start)]["iTransformer"].values
    test_preds = cv_results[cv_results["ds"] >= test_start]["iTransformer"].values

    # Ensure alignment (may need to truncate)
    val_preds = val_preds[:len(data["val_targets"])]
    test_preds = test_preds[:len(data["test_targets"])]

    print(f"  Val predictions: {len(val_preds)}")
    print(f"  Test predictions: {len(test_preds)}")
    print(f"  Val pred range: [{val_preds.min():.4f}, {val_preds.max():.4f}]")

    # Evaluate using threshold classification
    print("\nEvaluating via threshold classification...")

    val_metrics = evaluate_forecasting_model(
        predicted_returns=val_preds,
        actual_returns=data["val_returns"][:len(val_preds)],
        threshold_targets=data["val_targets"][:len(val_preds)],
        return_threshold=0.01,
    )

    test_metrics = evaluate_forecasting_model(
        predicted_returns=test_preds,
        actual_returns=data["test_returns"][:len(test_preds)],
        threshold_targets=data["test_targets"][:len(test_preds)],
        return_threshold=0.01,
    ) if len(test_preds) > 0 else None

    # Compute precision-recall curve
    pr_curve = compute_precision_recall_curve(
        predictions=val_preds,
        targets=data["val_targets"][:len(val_preds)],
    )

    # Print results
    summary = format_results_summary(
        experiment_name=f"{EXPERIMENT_ID}: {MODEL_NAME}",
        model_name=MODEL_NAME,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        pr_curve=pr_curve,
        training_time_min=training_time,
    )
    print(summary)

    # Save results
    save_results(
        output_dir=OUTPUT_DIR,
        experiment_name=EXPERIMENT_NAME,
        model_name=MODEL_NAME,
        config=CONFIG,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        pr_curve=pr_curve,
        training_time_min=training_time,
    )

    # Save predictions for further analysis
    np.savez(
        OUTPUT_DIR / "predictions.npz",
        val_preds=val_preds,
        val_targets=data["val_targets"][:len(val_preds)],
        test_preds=test_preds,
        test_targets=data["test_targets"][:len(test_preds)] if len(test_preds) > 0 else np.array([]),
    )

    return val_metrics["auc"]


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_experiment()
