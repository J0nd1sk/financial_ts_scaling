#!/usr/bin/env python3
"""
Foundation Investigation: Lag-Llama H1, Forecast Mode (FD-01d)

Train Lag-Llama on its NATIVE TASK (forecasting) instead of classification.
Then threshold predictions at inference to compute classification metrics.

Key insight: Classification approach (FD-01b) produced near-constant predictions.
Lag-Llama was pretrained for forecasting, not classification. Using its native
task may give better results.

Baseline to beat (PatchTST):
    H1 2M: AUC 0.706
    H1 200M: AUC 0.718

Success criteria: AUC >= 0.74 (5% improvement)
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.models.foundation.lag_llama import LagLlamaWrapper

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "lagllama_h1_forecast"
EXPERIMENT_ID = "FD-01d"
HORIZON = 1

# Lag-Llama requires minimum context of 1124 (max_lag=1092 + 32)
CONTEXT_LENGTH = 1150
BATCH_SIZE = 4  # Smaller batch for stability
EPOCHS = 30
LEARNING_RATE = 1e-4

# Data - using close returns only (univariate, native to Lag-Llama)
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
CHECKPOINT_PATH = PROJECT_ROOT / "models/pretrained/lag-llama.ckpt"
NUM_FEATURES = 1  # Univariate - close returns only

# Threshold for converting forecasts to classification
THRESHOLD = 0.01  # 1%

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/foundation" / EXPERIMENT_NAME


# ============================================================================
# DATA LOADING
# ============================================================================

def create_forecast_dataset(df, context_length, horizon):
    """Create dataset for forecasting task.

    Target: Next-day return (close[t+1]/close[t] - 1)
    Input: Sequence of past returns (univariate)

    For classification evaluation, we threshold forecasts at THRESHOLD.
    """
    close_prices = df["Close"].values.astype(np.float32)
    high_prices = df["High"].values.astype(np.float32)
    dates = pd.to_datetime(df["Date"])

    # Compute returns
    returns = np.zeros_like(close_prices)
    returns[1:] = close_prices[1:] / close_prices[:-1] - 1

    # Define region boundaries by target date
    val_start = pd.Timestamp("2023-01-01")
    test_start = pd.Timestamp("2025-01-01")

    # Find training region for normalization stats
    train_mask = dates < val_start
    train_returns = returns[train_mask]

    # Compute normalization params from training data only
    returns_mean = train_returns.mean()
    returns_std = train_returns.std()
    if returns_std < 1e-8:
        returns_std = 1.0

    # Normalize returns
    returns_norm = (returns - returns_mean) / returns_std

    print(f"  Return stats (train): mean={returns_mean:.6f}, std={returns_std:.6f}")
    print(f"  Normalized range: [{returns_norm.min():.2f}, {returns_norm.max():.2f}]")

    train_X, train_y, train_binary = [], [], []
    val_X, val_y, val_binary = [], [], []
    test_X, test_y, test_binary = [], [], []

    # Iterate through all possible prediction points
    for pred_idx in range(context_length, len(df) - horizon):
        context_start = pred_idx - context_length
        target_date = dates.iloc[pred_idx]

        # Input: Normalized returns for context window (univariate)
        x = returns_norm[context_start:pred_idx].reshape(-1, 1)  # (context_length, 1)

        # Target for forecasting: next return (normalized)
        next_return = returns_norm[pred_idx]

        # Binary target for evaluation: did high exceed threshold?
        future_highs = high_prices[pred_idx:pred_idx + horizon]
        current_close = close_prices[pred_idx - 1]
        binary_target = 1 if np.max(future_highs) >= current_close * (1 + THRESHOLD) else 0

        # Assign to appropriate split based on target date
        if target_date < val_start:
            train_X.append(x)
            train_y.append(next_return)
            train_binary.append(binary_target)
        elif target_date < test_start:
            val_X.append(x)
            val_y.append(next_return)
            val_binary.append(binary_target)
        else:
            test_X.append(x)
            test_y.append(next_return)
            test_binary.append(binary_target)

    return (
        np.array(train_X), np.array(train_y), np.array(train_binary),
        np.array(val_X), np.array(val_y), np.array(val_binary),
        np.array(test_X), np.array(test_y), np.array(test_binary),
        {"mean": returns_mean, "std": returns_std},
    )


def create_dataloader(X, y, batch_size, shuffle=True):
    """Create PyTorch DataLoader for forecast targets."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch using NLL loss (native Lag-Llama loss)."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # Use NLL loss - native loss for Lag-Llama
        loss = model.compute_nll_loss(batch_x, batch_y)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_model(model, dataloader, binary_labels, device, norm_params):
    """Evaluate model - get forecasts and compute classification metrics via thresholding.

    For AUC, we use the raw forecast as a "score" - higher predicted return
    should correlate with higher probability of exceeding threshold.
    """
    model.eval()
    all_forecasts = []
    total_nll = 0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Get raw forecast (in normalized space)
            forecast = model(batch_x)  # mode="forecast" returns loc directly

            # Compute NLL for monitoring
            nll = model.compute_nll_loss(batch_x, batch_y)

            all_forecasts.extend(forecast.cpu().numpy().flatten())
            total_nll += nll.item()
            n_batches += 1

    forecasts = np.array(all_forecasts)

    # Denormalize forecasts to get predicted returns
    pred_returns = forecasts * norm_params["std"] + norm_params["mean"]

    # For classification metrics, threshold the predicted returns
    binary_preds = (pred_returns >= THRESHOLD).astype(int)

    # For AUC, use raw predicted returns as scores
    # Higher predicted return = higher "probability" of exceeding threshold
    try:
        auc = roc_auc_score(binary_labels, pred_returns)
    except ValueError:
        auc = None

    return {
        "nll": total_nll / n_batches,
        "auc": auc,
        "accuracy": accuracy_score(binary_labels, binary_preds),
        "precision": precision_score(binary_labels, binary_preds, zero_division=0),
        "recall": recall_score(binary_labels, binary_preds, zero_division=0),
        "f1": f1_score(binary_labels, binary_preds, zero_division=0),
        "forecast_min": float(forecasts.min()),
        "forecast_max": float(forecasts.max()),
        "forecast_mean": float(forecasts.mean()),
        "forecast_std": float(forecasts.std()),
        "pred_return_min": float(pred_returns.min()),
        "pred_return_max": float(pred_returns.max()),
        "pred_return_mean": float(pred_returns.mean()),
        "pred_return_std": float(pred_returns.std()),
        "n_positive_preds": int(binary_preds.sum()),
        "n_samples": len(binary_labels),
        "class_balance": float(binary_labels.mean()),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"FOUNDATION INVESTIGATION: Lag-Llama / H{HORIZON} / Forecast Mode ({EXPERIMENT_ID})")
    print("=" * 70)
    print("\nKey insight: Train on native forecasting task, threshold at inference.")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Check checkpoint exists
    if not CHECKPOINT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Download from: https://huggingface.co/time-series-foundation-models/Lag-Llama")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data: {len(df)} rows, {df['Date'].min()} to {df['Date'].max()}")

    # Create datasets
    print(f"\nCreating forecast datasets (univariate, context={CONTEXT_LENGTH})...")
    (train_X, train_y, train_binary,
     val_X, val_y, val_binary,
     test_X, test_y, test_binary,
     norm_params) = create_forecast_dataset(df, CONTEXT_LENGTH, HORIZON)

    print(f"Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")
    print(f"Train positive rate: {train_binary.mean():.3f}")
    print(f"Val positive rate: {val_binary.mean():.3f}")
    print(f"Test positive rate: {test_binary.mean():.3f}")

    train_loader = create_dataloader(train_X, train_y, BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_X, val_y, BATCH_SIZE, shuffle=False)
    test_loader = create_dataloader(test_X, test_y, BATCH_SIZE, shuffle=False)

    # Initialize model in FORECAST mode
    print(f"\nInitializing Lag-Llama wrapper in FORECAST mode...")
    model = LagLlamaWrapper(
        context_length=CONTEXT_LENGTH,
        prediction_length=1,
        threshold=THRESHOLD,
        num_features=NUM_FEATURES,  # Univariate
        fine_tune_mode="full",
        dropout=0.1,
        mode="forecast",  # KEY: Use forecast mode, not classification
    )

    # Load pretrained weights
    print(f"Loading pretrained weights from {CHECKPOINT_PATH}...")
    model.load_pretrained(str(CHECKPOINT_PATH))
    model = model.to(device)

    config = model.get_config()
    print(f"Model config: {config}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print(f"\nTraining for {EPOCHS} epochs (using NLL loss)...")
    start_time = time.time()

    best_val_auc = 0
    best_val_nll = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_nll = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, val_binary, device, norm_params)

        val_auc = val_metrics["auc"] or 0
        val_nll = val_metrics["nll"]

        print(f"Epoch {epoch+1:3d}: train_nll={train_nll:.4f}, "
              f"val_nll={val_nll:.4f}, val_auc={val_auc:.4f}, "
              f"forecast_range=[{val_metrics['forecast_min']:.3f}, {val_metrics['forecast_max']:.3f}]")

        # Early stopping based on NLL (native loss)
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    elapsed = time.time() - start_time

    # Load best model and final evaluation
    model.load_state_dict(torch.load(OUTPUT_DIR / "best_model.pt", weights_only=True))
    val_metrics = evaluate_model(model, val_loader, val_binary, device, norm_params)
    test_metrics = evaluate_model(model, test_loader, test_binary, device, norm_params)

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Training time: {elapsed/60:.1f} min")

    print(f"\nValidation (2023-2024, {val_metrics['n_samples']} samples):")
    print(f"  NLL: {val_metrics['nll']:.4f}")
    print(f"  AUC: {val_metrics['auc']:.4f}" if val_metrics['auc'] else "  AUC: N/A")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")
    print(f"  Forecast Range (norm): [{val_metrics['forecast_min']:.4f}, {val_metrics['forecast_max']:.4f}]")
    print(f"  Pred Return Range: [{val_metrics['pred_return_min']:.4f}, {val_metrics['pred_return_max']:.4f}]")
    print(f"  Positive preds: {val_metrics['n_positive_preds']}/{val_metrics['n_samples']}")
    print(f"  Class Balance: {val_metrics['class_balance']:.3f}")

    print(f"\nTest/Backtest (2025+, {test_metrics['n_samples']} samples):")
    print(f"  NLL: {test_metrics['nll']:.4f}")
    print(f"  AUC: {test_metrics['auc']:.4f}" if test_metrics['auc'] else "  AUC: N/A")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Forecast Range (norm): [{test_metrics['forecast_min']:.4f}, {test_metrics['forecast_max']:.4f}]")
    print(f"  Pred Return Range: [{test_metrics['pred_return_min']:.4f}, {test_metrics['pred_return_max']:.4f}]")
    print(f"  Positive preds: {test_metrics['n_positive_preds']}/{test_metrics['n_samples']}")
    print(f"  Class Balance: {test_metrics['class_balance']:.3f}")

    # Compare to baseline
    baseline_auc = 0.718  # PatchTST 200M H1
    fd01b_auc = 0.576  # Previous Lag-Llama classification
    if val_metrics['auc']:
        improvement_vs_patchtst = (val_metrics['auc'] - baseline_auc) / baseline_auc * 100
        improvement_vs_fd01b = (val_metrics['auc'] - fd01b_auc) / fd01b_auc * 100
        print(f"\nVs PatchTST baseline (0.718): {improvement_vs_patchtst:+.1f}%")
        print(f"Vs FD-01b classification (0.576): {improvement_vs_fd01b:+.1f}%")
        if val_metrics['auc'] >= 0.74:
            print("SUCCESS: Exceeds 5% improvement target!")
        else:
            print(f"Need {0.74 - val_metrics['auc']:.4f} more AUC for 5% improvement")

    # Check forecast variation
    forecast_spread = val_metrics['forecast_max'] - val_metrics['forecast_min']
    if forecast_spread < 0.05:
        print(f"\nWARNING: Forecast spread ({forecast_spread:.4f}) is very narrow!")
        print("Model may be producing near-constant predictions.")
    else:
        print(f"\nForecast spread: {forecast_spread:.4f} (good variation)")

    # Save results
    results = {
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "model": "Lag-Llama",
        "mode": "forecast",
        "loss": "NLL (StudentT)",
        "num_features": NUM_FEATURES,
        "horizon": HORIZON,
        "context_length": CONTEXT_LENGTH,
        "threshold": THRESHOLD,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "fine_tune_mode": "full",
        },
        "normalization": {
            "mean": float(norm_params["mean"]),
            "std": float(norm_params["std"]),
        },
        "splits": {
            "train": len(train_X),
            "val": len(val_X),
            "test": len(test_X),
        },
        "training": {
            "training_time_min": elapsed / 60,
            "best_val_nll": best_val_nll,
            "best_val_auc": best_val_auc,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_comparison": {
            "patchtst_200m_h1": 0.718,
            "fd01b_classification": 0.576,
            "improvement_vs_patchtst_pct": improvement_vs_patchtst if val_metrics['auc'] else None,
            "improvement_vs_fd01b_pct": improvement_vs_fd01b if val_metrics['auc'] else None,
        },
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return val_metrics["auc"]


if __name__ == "__main__":
    main()
