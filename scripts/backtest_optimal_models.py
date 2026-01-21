#!/usr/bin/env python3
"""
Backtest Optimal Models with Full Classification Metrics

Tests n_heads variations (8, 4, 2, 1) on the proven 20M_wide architecture
(d=512, L=6, dropout=0.5) and evaluates on 2025 holdout data.

Reports: accuracy, precision, recall, F1, AUC-ROC for each configuration.

Usage:
    python scripts/backtest_optimal_models.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.data.dataset import SimpleSplitter, FinancialDataset
from src.models.patchtst import PatchTST, PatchTSTConfig, RevIN
from src.models.arch_grid import get_memory_safe_batch_config
from src.training.losses import FocalLoss
from torch.utils.data import DataLoader, Subset

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Fixed optimal parameters (proven from previous experiments)
D_MODEL = 512
N_LAYERS = 6
D_FF = 2048  # 4x d_model
DROPOUT = 0.5
LEARNING_RATE = 1e-4

# Data parameters
CONTEXT_LENGTH = 80
HORIZON = 1
PATCH_LENGTH = 16
STRIDE = 8
THRESHOLD = 0.01  # 1% threshold for binary classification
NUM_FEATURES = 25  # SPY_dataset_a20: 5 OHLCV + 20 indicators

# Training parameters
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Paths
DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs/backtest_optimal"

# Configurations to test - varying only n_heads
# Note: d_model=512 must be divisible by n_heads
# 512/8=64, 512/4=128, 512/2=256, 512/1=512 (all valid)
CONFIGS = [
    {"name": "20M_h8", "n_heads": 8},
    {"name": "20M_h4", "n_heads": 4},
    {"name": "20M_h2", "n_heads": 2},
    {"name": "20M_h1", "n_heads": 1},
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(n_heads: int, device: str) -> torch.nn.Module:
    """Create PatchTST model with specified n_heads."""
    config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=PATCH_LENGTH,
        stride=STRIDE,
        d_model=D_MODEL,
        n_heads=n_heads,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        head_dropout=0.0,
    )
    model = PatchTST(config)
    return model.to(device)


def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute full classification metrics.

    Args:
        predictions: Predicted probabilities (0-1)
        targets: Binary targets (0 or 1)
        threshold: Classification threshold for converting probs to binary

    Returns:
        Dict with accuracy, precision, recall, f1, auc_roc
    """
    pred_binary = (predictions >= threshold).astype(int)
    targets_int = targets.astype(int)

    metrics = {
        "accuracy": accuracy_score(targets_int, pred_binary),
        "precision": precision_score(targets_int, pred_binary, zero_division=0.0),
        "recall": recall_score(targets_int, pred_binary, zero_division=0.0),
        "f1": f1_score(targets_int, pred_binary, zero_division=0.0),
    }

    # AUC requires both classes
    unique_targets = np.unique(targets_int)
    if len(unique_targets) == 2:
        metrics["auc_roc"] = roc_auc_score(targets_int, predictions)
    else:
        metrics["auc_roc"] = float("nan")

    return metrics


def train_model(
    model: torch.nn.Module,
    revin: RevIN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_config: dict,
    device: str,
    checkpoint_dir: Path,
) -> dict:
    """Train model with checkpoint saving."""
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(revin.parameters()),
        lr=LEARNING_RATE,
    )

    accumulation_steps = batch_config["accumulation_steps"]

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    start_time = time.time()

    for epoch in range(EPOCHS):
        # Training
        model.train()
        revin.train()
        train_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            batch_X = revin.normalize(batch_X)
            preds = model(batch_X)

            if preds.dim() == 2:
                preds = preds.squeeze(-1)

            loss = criterion(preds, batch_y)
            loss = loss / accumulation_steps
            loss.backward()

            train_loss += loss.item() * accumulation_steps
            n_batches += 1

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        if n_batches % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= n_batches

        # Validation
        model.eval()
        revin.eval()
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_X = revin.normalize(batch_X)
                preds = model(batch_X)

                if preds.dim() == 2:
                    preds = preds.squeeze(-1)

                all_val_preds.append(preds.cpu().numpy())
                all_val_labels.append(batch_y.numpy())

        val_preds = np.concatenate(all_val_preds)
        val_labels = np.concatenate(all_val_labels)
        val_auc = roc_auc_score(val_labels, val_preds)

        # Save best checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0

            # Save checkpoint
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "revin_state_dict": revin.state_dict(),
                "val_auc": val_auc,
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1:3d}: val_auc={val_auc:.4f}, best={best_val_auc:.4f}")

    elapsed = time.time() - start_time

    return {
        "best_val_auc": float(best_val_auc),
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "training_time_min": elapsed / 60,
    }


def evaluate_on_test(
    model: torch.nn.Module,
    revin: RevIN,
    test_loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on test set."""
    model.eval()
    revin.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_X = revin.normalize(batch_X)
            preds = model(batch_X)

            if preds.dim() == 2:
                preds = preds.squeeze(-1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.numpy())

    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_labels)

    return predictions, targets


def run_single_config(
    config: dict,
    df: pd.DataFrame,
    split_indices,
    full_dataset,
    device: str,
) -> dict:
    """Run training and evaluation for a single config."""
    print(f"\n{'=' * 70}")
    print(f"CONFIG: {config['name']} (n_heads={config['n_heads']})")
    print(f"Architecture: d={D_MODEL}, L={N_LAYERS}, h={config['n_heads']}, dropout={DROPOUT}")
    print(f"{'=' * 70}")

    # Create model
    model = create_model(config["n_heads"], device)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    # Get batch config
    batch_config = get_memory_safe_batch_config(n_params, N_LAYERS)
    print(f"  Batch: micro={batch_config['micro_batch']}, accum={batch_config['accumulation_steps']}")

    # Create RevIN
    revin = RevIN(num_features=NUM_FEATURES).to(device)

    # Create data loaders
    train_dataset = Subset(full_dataset, split_indices.train_indices)
    val_dataset = Subset(full_dataset, split_indices.val_indices)
    test_dataset = Subset(full_dataset, split_indices.test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_config["micro_batch"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_config["micro_batch"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_config["micro_batch"], shuffle=False)

    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Train
    checkpoint_dir = OUTPUT_DIR / config["name"]
    train_metrics = train_model(
        model=model,
        revin=revin,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_config=batch_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    print(f"\n  Training complete:")
    print(f"    Best Val AUC: {train_metrics['best_val_auc']:.4f} (epoch {train_metrics['best_epoch']})")

    # Load best checkpoint for evaluation
    checkpoint_path = checkpoint_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    revin.load_state_dict(checkpoint["revin_state_dict"])

    # Evaluate on test set (2025 data)
    print(f"\n  Evaluating on 2025 test data ({len(test_dataset)} samples)...")
    predictions, targets = evaluate_on_test(model, revin, test_loader, device)

    # Compute classification metrics
    test_metrics = compute_classification_metrics(predictions, targets, threshold=0.5)

    print(f"\n  TEST RESULTS (2025 holdout):")
    print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"    Precision: {test_metrics['precision']:.4f}")
    print(f"    Recall:    {test_metrics['recall']:.4f}")
    print(f"    F1:        {test_metrics['f1']:.4f}")
    print(f"    AUC-ROC:   {test_metrics['auc_roc']:.4f}")

    # Prediction distribution
    pred_stats = {
        "pred_min": float(np.min(predictions)),
        "pred_max": float(np.max(predictions)),
        "pred_mean": float(np.mean(predictions)),
        "pred_std": float(np.std(predictions)),
    }
    print(f"    Pred range: [{pred_stats['pred_min']:.4f}, {pred_stats['pred_max']:.4f}]")
    print(f"    Pred std:   {pred_stats['pred_std']:.4f}")

    # Target distribution
    target_rate = float(np.mean(targets))
    print(f"    Target rate (% positive): {target_rate:.2%}")

    # Combine results
    result = {
        "config_name": config["name"],
        "n_heads": config["n_heads"],
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "dropout": DROPOUT,
        "n_params": n_params,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        **train_metrics,
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **pred_stats,
        "target_positive_rate": target_rate,
    }

    return result


def main():
    print("=" * 70)
    print("BACKTEST OPTIMAL MODELS")
    print("Testing n_heads variations on 20M_wide (d=512, L=6, dropout=0.5)")
    print("Evaluating on 2025 holdout data with full classification metrics")
    print("=" * 70)

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data: {len(df)} rows, date range: {df['Date'].min()} to {df['Date'].max()}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create splits (shared across all configs)
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    print(f"Splits: Train={len(split_indices.train_indices)}, "
          f"Val={len(split_indices.val_indices)}, Test={len(split_indices.test_indices)}")

    # Create full dataset (shared)
    # Use High prices for target calculation: max(High[future]) >= Close[t] * threshold
    close_prices = df["Close"].values
    high_prices = df["High"].values
    full_dataset = FinancialDataset(
        features_df=df,
        close_prices=close_prices,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        threshold=THRESHOLD,
        high_prices=high_prices,
    )

    # Run all configs
    results = []
    for config in CONFIGS:
        result = run_single_config(config, df, split_indices, full_dataset, device)
        results.append(result)

        # Save intermediate results
        csv_path = OUTPUT_DIR / "backtest_results.csv"
        pd.DataFrame(results).to_csv(csv_path, index=False)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: n_heads Ablation with Full Metrics")
    print(f"{'=' * 70}")
    print(f"{'Config':<12} {'h':<4} {'Val AUC':<10} {'Test Acc':<10} {'Test Prec':<10} {'Test Rec':<10} {'Test F1':<10} {'Test AUC':<10}")
    print("-" * 86)

    for r in results:
        print(f"{r['config_name']:<12} {r['n_heads']:<4} "
              f"{r['best_val_auc']:<10.4f} "
              f"{r['test_accuracy']:<10.4f} "
              f"{r['test_precision']:<10.4f} "
              f"{r['test_recall']:<10.4f} "
              f"{r['test_f1']:<10.4f} "
              f"{r['test_auc_roc']:<10.4f}")

    # Find best by each metric
    print(f"\n{'=' * 70}")
    print("BEST BY METRIC:")
    print(f"{'=' * 70}")

    metrics_to_check = ["test_accuracy", "test_precision", "test_f1", "test_auc_roc"]
    for metric in metrics_to_check:
        best = max(results, key=lambda x: x[metric])
        print(f"  Best {metric}: {best['config_name']} ({best[metric]:.4f})")

    # Save final results
    output = {
        "experiment": "backtest_optimal_models",
        "description": "n_heads ablation on 20M_wide with 2025 backtest",
        "fixed_params": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "d_ff": D_FF,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "context_length": CONTEXT_LENGTH,
            "threshold": THRESHOLD,
        },
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    json_path = OUTPUT_DIR / "backtest_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
