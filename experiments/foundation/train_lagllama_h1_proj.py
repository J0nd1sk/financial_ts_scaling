#!/usr/bin/env python3
"""
Foundation Investigation: Lag-Llama H1, Feature Projection (FD-01b)

Foundation model (decoder-only) fine-tuning for threshold classification.
Uses all 20 features with learnable projection to univariate.

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
from src.data.dataset import SimpleSplitter

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

EXPERIMENT_NAME = "lagllama_h1_proj"
HORIZON = 1

# Lag-Llama requires minimum context of 1124 (max_lag=1092 + 32)
CONTEXT_LENGTH = 1150
BATCH_SIZE = 32  # Smaller due to large context length
EPOCHS = 30
LEARNING_RATE = 1e-4

# Data
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
CHECKPOINT_PATH = PROJECT_ROOT / "models/pretrained/lag-llama.ckpt"
NUM_FEATURES = 20  # All features

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs/foundation" / EXPERIMENT_NAME


# ============================================================================
# DATA LOADING
# ============================================================================

def get_feature_columns(df):
    """Get the 20 feature columns (exclude Date, High for targets)."""
    exclude = {"Date", "High"}
    return [c for c in df.columns if c not in exclude][:NUM_FEATURES]


def create_multivariate_dataset(df, split_indices, context_length, horizon, threshold=0.01):
    """Create dataset using all features with projection."""
    feature_cols = get_feature_columns(df)
    features = df[feature_cols].values
    high_prices = df["High"].values
    close_prices = df["Close"].values

    def create_split(indices):
        X_list = []
        y_list = []

        for i in indices:
            end_idx = i + context_length
            if end_idx + horizon > len(features):
                continue

            # Input: All features, shape (context_length, num_features)
            x = features[i:end_idx]

            # Target: Did high price exceed threshold within horizon?
            future_highs = high_prices[end_idx:end_idx + horizon]
            current_close = close_prices[end_idx - 1]
            target = 1 if np.max(future_highs) >= current_close * (1 + threshold) else 0

            X_list.append(x)
            y_list.append(target)

        return np.array(X_list), np.array(y_list)

    train_X, train_y = create_split(split_indices.train_indices)
    val_X, val_y = create_split(split_indices.val_indices)

    return train_X, train_y, val_X, val_y


def create_dataloader(X, y, batch_size, shuffle=True):
    """Create PyTorch DataLoader."""
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = torch.nn.functional.binary_cross_entropy(preds, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate_model(model, dataloader, device):
    """Evaluate model with full metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_x)
            loss = torch.nn.functional.binary_cross_entropy(preds, batch_y)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
            total_loss += loss.item()
            n_batches += 1

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    binary_preds = (preds >= 0.5).astype(int)

    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = None

    return {
        "loss": total_loss / n_batches,
        "auc": auc,
        "accuracy": accuracy_score(labels, binary_preds),
        "precision": precision_score(labels, binary_preds, zero_division=0),
        "recall": recall_score(labels, binary_preds, zero_division=0),
        "f1": f1_score(labels, binary_preds, zero_division=0),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "n_positive_preds": int((preds >= 0.5).sum()),
        "n_samples": len(labels),
        "class_balance": float(labels.mean()),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"FOUNDATION INVESTIGATION: Lag-Llama / H{HORIZON} / Feature Projection (FD-01b)")
    print("=" * 70)

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
    print(f"Data: {len(df)} rows, Features: {get_feature_columns(df)[:5]}...")

    # Create splits
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    print(f"\nSplits: train={len(split_indices.train_indices)}, "
          f"val={len(split_indices.val_indices)}, test={len(split_indices.test_indices)}")

    # Create datasets
    print(f"\nCreating datasets ({NUM_FEATURES} features with projection)...")
    train_X, train_y, val_X, val_y = create_multivariate_dataset(
        df, split_indices, CONTEXT_LENGTH, HORIZON
    )
    print(f"Train: {train_X.shape}, Val: {val_X.shape}")
    print(f"Train positive rate: {train_y.mean():.3f}")
    print(f"Val positive rate: {val_y.mean():.3f}")

    train_loader = create_dataloader(train_X, train_y, BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_X, val_y, BATCH_SIZE, shuffle=False)

    # Initialize model with feature projection
    print(f"\nInitializing Lag-Llama wrapper with feature projection...")
    model = LagLlamaWrapper(
        context_length=CONTEXT_LENGTH,
        prediction_length=1,
        threshold=0.01,  # 1% threshold
        num_features=NUM_FEATURES,  # Enables learnable projection 20->1
        fine_tune_mode="full",  # Full fine-tuning
        dropout=0.1,
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
    print(f"\nTraining for {EPOCHS} epochs...")
    start_time = time.time()

    best_val_auc = 0
    patience = 10
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, device)

        val_auc = val_metrics["auc"] or 0
        print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
              f"val_loss={val_metrics['loss']:.4f}, val_auc={val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
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
    val_metrics = evaluate_model(model, val_loader, device)

    print(f"\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Training time: {elapsed/60:.1f} min")
    print(f"\nValidation (2023-2024, {val_metrics['n_samples']} samples):")
    print(f"  AUC: {val_metrics['auc']:.4f}" if val_metrics['auc'] else "  AUC: N/A")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1: {val_metrics['f1']:.4f}")
    print(f"  Pred Range: [{val_metrics['pred_min']:.4f}, {val_metrics['pred_max']:.4f}]")
    print(f"  Class Balance: {val_metrics['class_balance']:.3f}")

    # Compare to baseline
    baseline_auc = 0.718  # PatchTST 200M H1
    if val_metrics['auc']:
        improvement = (val_metrics['auc'] - baseline_auc) / baseline_auc * 100
        print(f"\nVs PatchTST baseline (0.718): {improvement:+.1f}%")
        if val_metrics['auc'] >= 0.74:
            print("SUCCESS: Exceeds 5% improvement target!")
        else:
            print(f"Need {0.74 - val_metrics['auc']:.4f} more AUC for 5% improvement")

    # Save results
    results = {
        "experiment": EXPERIMENT_NAME,
        "experiment_id": "FD-01b",
        "model": "Lag-Llama",
        "input_mode": "feature_projection",
        "num_features": NUM_FEATURES,
        "horizon": HORIZON,
        "context_length": CONTEXT_LENGTH,
        "hyperparameters": {
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "fine_tune_mode": "full",
        },
        "splits": {
            "train": len(train_X),
            "val": len(val_X),
        },
        "training": {
            "training_time_min": elapsed / 60,
            "best_val_auc": best_val_auc,
        },
        "val_metrics": val_metrics,
        "baseline_comparison": {
            "patchtst_200m_h1": 0.718,
            "improvement_pct": improvement if val_metrics['auc'] else None,
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
