#!/usr/bin/env python3
"""
Common utilities for Alternative Architecture Investigation.

Provides:
- Data loading for NeuralForecast format (panel data)
- Return calculation for forecasting approach
- Threshold-based classification evaluation
- Precision-recall curve analysis
"""
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve
)


# ============================================================================
# DATA PATHS
# ============================================================================

DATA_PATH_A100 = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a100_combined.parquet"
DATA_PATH_A20 = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
OUTPUT_BASE = PROJECT_ROOT / "outputs/architectures"


# ============================================================================
# DATA LOADING FOR NEURALFORECAST
# ============================================================================

def load_data_for_neuralforecast(data_path: Path | None = None, use_returns: bool = True):
    """Load data in NeuralForecast panel format.

    NeuralForecast expects panel data with columns:
    - unique_id: identifier for time series
    - ds: datetime column
    - y: target variable (returns or price)

    Additional features can be included as exogenous variables.

    Args:
        data_path: Path to parquet file. Defaults to a20 tier.
        use_returns: If True, y is next-day return. If False, y is price.

    Returns:
        Tuple of (df_train, df_val, df_test, feature_cols)
    """
    if data_path is None:
        data_path = DATA_PATH_A20

    df = pd.read_parquet(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Calculate returns (target for forecasting)
    df["return"] = df["Close"].pct_change()

    # Calculate threshold target (for comparison)
    # True if next-day high reaches +1% from current close
    df["future_high"] = df["High"].shift(-1)
    df["threshold_target"] = (df["future_high"] >= df["Close"] * 1.01).astype(int)

    # Drop first row (NaN from pct_change) and last row (NaN from shift)
    df = df.iloc[1:-1].reset_index(drop=True)

    # Feature columns (exclude Date, High, Open for purity, and calculated columns)
    exclude_cols = {"Date", "High", "Open", "return", "future_high", "threshold_target"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # NeuralForecast format
    df_nf = pd.DataFrame({
        "unique_id": "SPY",
        "ds": df["Date"],
        "y": df["return"] if use_returns else df["Close"],
    })

    # Add features as exogenous variables
    for col in feature_cols:
        df_nf[col] = df[col].values

    # Add threshold target for evaluation
    df_nf["_threshold_target"] = df["threshold_target"].values
    df_nf["_close"] = df["Close"].values
    df_nf["_high"] = df["High"].values

    # Split by date
    val_start = pd.Timestamp("2023-01-01")
    test_start = pd.Timestamp("2025-01-01")

    df_train = df_nf[df_nf["ds"] < val_start].copy()
    df_val = df_nf[(df_nf["ds"] >= val_start) & (df_nf["ds"] < test_start)].copy()
    df_test = df_nf[df_nf["ds"] >= test_start].copy()

    print(f"Data loaded from {data_path}")
    print(f"  Date range: {df_nf['ds'].min()} to {df_nf['ds'].max()}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(df_train)} samples")
    print(f"  Val: {len(df_val)} samples")
    print(f"  Test: {len(df_test)} samples")

    return df_train, df_val, df_test, feature_cols


def prepare_data_simple(data_path: Path | None = None, context_length: int = 80):
    """Prepare data in simple format for manual model usage.

    Returns numpy arrays suitable for direct training loops.

    Args:
        data_path: Path to parquet file
        context_length: Context window size

    Returns:
        Dictionary with train/val/test splits as numpy arrays
    """
    if data_path is None:
        data_path = DATA_PATH_A20

    df = pd.read_parquet(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Feature columns
    exclude_cols = {"Date", "High"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    features = df[feature_cols].values.astype(np.float32)
    high_prices = df["High"].values
    close_prices = df["Close"].values
    dates = df["Date"]

    # Normalize using training stats
    val_start_idx = (dates < "2023-01-01").sum()
    train_features = features[:val_start_idx]

    feature_mean = train_features.mean(axis=0)
    feature_std = train_features.std(axis=0)
    feature_std[feature_std < 1e-8] = 1.0

    features_norm = (features - feature_mean) / feature_std

    # Create sliding window samples
    val_start = pd.Timestamp("2023-01-01")
    test_start = pd.Timestamp("2025-01-01")

    train_X, train_y, train_returns = [], [], []
    val_X, val_y, val_returns = [], [], []
    test_X, test_y, test_returns = [], [], []

    horizon = 1
    for i in range(context_length, len(df) - horizon):
        x = features_norm[i - context_length:i]

        # Target: Did high price exceed +1% threshold within horizon?
        future_highs = high_prices[i:i + horizon]
        current_close = close_prices[i - 1]
        target = 1 if np.max(future_highs) >= current_close * 1.01 else 0

        # Return (for forecasting models)
        next_close = close_prices[i]
        ret = (next_close - current_close) / current_close

        target_date = dates.iloc[i]

        if target_date < val_start:
            train_X.append(x)
            train_y.append(target)
            train_returns.append(ret)
        elif target_date < test_start:
            val_X.append(x)
            val_y.append(target)
            val_returns.append(ret)
        else:
            test_X.append(x)
            test_y.append(target)
            test_returns.append(ret)

    return {
        "train_X": np.array(train_X),
        "train_y": np.array(train_y),
        "train_returns": np.array(train_returns),
        "val_X": np.array(val_X),
        "val_y": np.array(val_y),
        "val_returns": np.array(val_returns),
        "test_X": np.array(test_X),
        "test_y": np.array(test_y),
        "test_returns": np.array(test_returns),
        "feature_cols": feature_cols,
        "norm_params": {"mean": feature_mean, "std": feature_std},
    }


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def evaluate_forecasting_model(
    predicted_returns: np.ndarray,
    actual_returns: np.ndarray,
    threshold_targets: np.ndarray,
    return_threshold: float = 0.01,
) -> dict:
    """Evaluate forecasting model via threshold classification.

    Converts return forecasts to binary predictions:
    - positive if predicted_return > return_threshold
    - negative otherwise

    Args:
        predicted_returns: Model's return forecasts
        actual_returns: Actual returns
        threshold_targets: Binary targets (1 if high reached +1%)
        return_threshold: Threshold for classifying as positive

    Returns:
        Dictionary with classification and forecasting metrics
    """
    # Convert forecasts to binary predictions
    binary_preds = (predicted_returns > return_threshold).astype(int)

    # Classification metrics
    try:
        auc = roc_auc_score(threshold_targets, predicted_returns)
    except ValueError:
        auc = None

    metrics = {
        "auc": auc,
        "accuracy": accuracy_score(threshold_targets, binary_preds),
        "precision": precision_score(threshold_targets, binary_preds, zero_division=0),
        "recall": recall_score(threshold_targets, binary_preds, zero_division=0),
        "f1": f1_score(threshold_targets, binary_preds, zero_division=0),
        "n_positive_preds": int(binary_preds.sum()),
        "n_samples": len(threshold_targets),
        "class_balance": float(threshold_targets.mean()),
        "pred_min": float(predicted_returns.min()),
        "pred_max": float(predicted_returns.max()),
        "pred_mean": float(predicted_returns.mean()),
        "pred_std": float(predicted_returns.std()),
    }

    # Forecasting metrics
    mse = np.mean((predicted_returns - actual_returns) ** 2)
    mae = np.mean(np.abs(predicted_returns - actual_returns))

    # Direction accuracy
    pred_direction = (predicted_returns > 0).astype(int)
    actual_direction = (actual_returns > 0).astype(int)
    direction_acc = accuracy_score(actual_direction, pred_direction)

    metrics["mse"] = float(mse)
    metrics["mae"] = float(mae)
    metrics["direction_accuracy"] = direction_acc

    return metrics


def compute_precision_recall_curve(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict:
    """Compute full precision-recall curve for threshold sweep.

    Returns precision at specific recall levels for comparison.
    """
    precisions, recalls, thresholds = precision_recall_curve(targets, predictions)

    # Find precision at specific recall levels
    recall_levels = [0.10, 0.20, 0.30, 0.40, 0.50]
    precision_at_recall = {}

    for target_recall in recall_levels:
        # Find threshold that achieves closest recall
        idx = np.argmin(np.abs(recalls - target_recall))
        precision_at_recall[f"precision_at_recall_{int(target_recall*100)}"] = float(precisions[idx])
        precision_at_recall[f"actual_recall_{int(target_recall*100)}"] = float(recalls[idx])

    return {
        "precision_recall_curve": {
            "precisions": precisions.tolist(),
            "recalls": recalls.tolist(),
            "thresholds": thresholds.tolist() if len(thresholds) > 0 else [],
        },
        **precision_at_recall,
        "auc_pr": float(np.trapz(precisions, recalls)),
    }


def compare_to_baseline(val_auc: float, baseline_auc: float = 0.718) -> dict:
    """Compare model performance to PatchTST baseline.

    Baseline: PatchTST 200M H1 = 0.718 AUC
    Target: >= 0.70 AUC with better precision-recall
    """
    improvement = (val_auc - baseline_auc) / baseline_auc * 100 if val_auc else None

    return {
        "baseline_auc": baseline_auc,
        "model_auc": val_auc,
        "improvement_pct": improvement,
        "meets_threshold": val_auc >= 0.70 if val_auc else False,
        "beats_baseline": val_auc > baseline_auc if val_auc else False,
    }


def format_results_summary(
    experiment_name: str,
    model_name: str,
    val_metrics: dict,
    test_metrics: dict | None = None,
    pr_curve: dict | None = None,
    training_time_min: float | None = None,
) -> str:
    """Format results as readable summary string."""
    lines = [
        "=" * 70,
        f"RESULTS: {experiment_name}",
        "=" * 70,
        f"Model: {model_name}",
    ]

    if training_time_min:
        lines.append(f"Training time: {training_time_min:.1f} min")

    lines.extend([
        "",
        f"Validation Metrics:",
        f"  AUC: {val_metrics.get('auc', 'N/A'):.4f}" if val_metrics.get('auc') else "  AUC: N/A",
        f"  Accuracy: {val_metrics.get('accuracy', 0):.4f}",
        f"  Precision: {val_metrics.get('precision', 0):.4f}",
        f"  Recall: {val_metrics.get('recall', 0):.4f}",
        f"  F1: {val_metrics.get('f1', 0):.4f}",
        f"  Direction Accuracy: {val_metrics.get('direction_accuracy', 0):.4f}",
        f"  Pred Range: [{val_metrics.get('pred_min', 0):.4f}, {val_metrics.get('pred_max', 0):.4f}]",
    ])

    if pr_curve:
        lines.extend([
            "",
            "Precision-Recall Analysis:",
            f"  P@R20%: {pr_curve.get('precision_at_recall_20', 0):.4f}",
            f"  P@R30%: {pr_curve.get('precision_at_recall_30', 0):.4f}",
            f"  P@R50%: {pr_curve.get('precision_at_recall_50', 0):.4f}",
        ])

    if test_metrics:
        lines.extend([
            "",
            f"Test/Backtest Metrics:",
            f"  AUC: {test_metrics.get('auc', 'N/A'):.4f}" if test_metrics.get('auc') else "  AUC: N/A",
            f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}",
            f"  Precision: {test_metrics.get('precision', 0):.4f}",
            f"  Recall: {test_metrics.get('recall', 0):.4f}",
        ])

    comparison = compare_to_baseline(val_metrics.get('auc', 0))
    lines.extend([
        "",
        f"Comparison to PatchTST Baseline (AUC=0.718):",
        f"  Improvement: {comparison['improvement_pct']:+.1f}%" if comparison['improvement_pct'] else "  N/A",
        f"  Meets threshold (0.70): {'Yes' if comparison['meets_threshold'] else 'No'}",
        f"  Beats baseline: {'Yes' if comparison['beats_baseline'] else 'No'}",
    ])

    return "\n".join(lines)


# ============================================================================
# RESULT SAVING
# ============================================================================

def save_results(
    output_dir: Path,
    experiment_name: str,
    model_name: str,
    config: dict,
    val_metrics: dict,
    test_metrics: dict | None = None,
    pr_curve: dict | None = None,
    training_time_min: float | None = None,
):
    """Save experiment results to JSON."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment": experiment_name,
        "model": model_name,
        "config": config,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "precision_recall": pr_curve,
        "baseline_comparison": compare_to_baseline(val_metrics.get("auc", 0)),
        "training_time_min": training_time_min,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {results_path}")
    return results_path


# ============================================================================
# HPO UTILITIES
# ============================================================================

def prepare_hpo_data(data_path: Path | None = None):
    """Prepare data for HPO experiments with NeuralForecast models.

    Returns data in NeuralForecast panel format with evaluation metadata.

    Args:
        data_path: Path to parquet file. Defaults to DATA_PATH_A20.

    Returns:
        Dictionary with train/val/test data and evaluation metadata.
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


def format_hpo_results_table(trials_data: list[dict]) -> str:
    """Format HPO trial results as a markdown table.

    Args:
        trials_data: List of trial dicts with params and metrics.

    Returns:
        Markdown table string.
    """
    if not trials_data:
        return "No trials completed."

    # Sort by AUC (descending)
    sorted_trials = sorted(
        trials_data,
        key=lambda t: t.get("auc", 0) or 0,
        reverse=True
    )

    lines = [
        "| Trial | AUC | Acc | Recall | Dropout | LR | Hidden | Layers |",
        "|-------|-----|-----|--------|---------|------|--------|--------|",
    ]

    for t in sorted_trials[:10]:  # Top 10
        auc = t.get("auc", 0) or 0
        acc = t.get("val_accuracy", 0) or 0
        recall = t.get("val_recall", 0) or 0
        p = t.get("params", {})

        lines.append(
            f"| {t.get('trial_number', '-')} | {auc:.4f} | {acc:.4f} | "
            f"{recall:.4f} | {p.get('dropout', '-')} | "
            f"{p.get('learning_rate', 0):.0e} | {p.get('hidden_size', '-')} | "
            f"{p.get('n_layers', '-')} |"
        )

    return "\n".join(lines)
