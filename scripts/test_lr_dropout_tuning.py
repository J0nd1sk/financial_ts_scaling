#!/usr/bin/env python3
"""
LR and Dropout Tuning Experiment
Tests modest LR reductions and slightly higher dropout on optimal L=6 architecture.

Previous finding: 20M_L6 (d=512, L=6) with LR=1e-4, dropout=0.5 achieved AUC 0.7342

This experiment tests:
1. Slower LRs (8e-5, 5e-5) with dropout=0.5
2. All three LRs (1e-4, 8e-5, 5e-5) with dropout=0.55

Configurations (5 total):
- lr8e5_d50:  LR=8e-5, dropout=0.50
- lr5e5_d50:  LR=5e-5, dropout=0.50
- lr1e4_d55:  LR=1e-4, dropout=0.55
- lr8e5_d55:  LR=8e-5, dropout=0.55
- lr5e5_d55:  LR=5e-5, dropout=0.55

Reference: LR=1e-4, dropout=0.50 achieved AUC 0.7342
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
from sklearn.metrics import roc_auc_score

from src.data.dataset import SimpleSplitter, FinancialDataset
from src.models.patchtst import PatchTST, PatchTSTConfig, RevIN
from src.models.arch_grid import get_memory_safe_batch_config
from src.training.losses import FocalLoss
from torch.utils.data import DataLoader, Subset

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Fixed parameters
CONTEXT_LENGTH = 80
HORIZON = 1
PATCH_LENGTH = 16
STRIDE = 8
EPOCHS = 50
NUM_FEATURES = 25
EARLY_STOPPING_PATIENCE = 10

DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs/lr_dropout_tuning"

THRESHOLD = 0.01

# Fixed architecture: L=6 (the optimal depth)
ARCHITECTURE = {
    "d_model": 512,
    "n_layers": 6,
    "n_heads": 8,
    "d_ff": 2048,
}

# Configurations to test
CONFIGS = [
    # Slower LRs with dropout=0.5
    {"name": "lr8e5_d50", "lr": 8e-5, "dropout": 0.50},
    {"name": "lr5e5_d50", "lr": 5e-5, "dropout": 0.50},
    # All LRs with dropout=0.55
    {"name": "lr1e4_d55", "lr": 1e-4, "dropout": 0.55},
    {"name": "lr8e5_d55", "lr": 8e-5, "dropout": 0.55},
    {"name": "lr5e5_d55", "lr": 5e-5, "dropout": 0.55},
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(arch: dict, dropout: float, device: str) -> torch.nn.Module:
    """Create PatchTST model with specified architecture and dropout."""
    config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=PATCH_LENGTH,
        stride=STRIDE,
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        d_ff=arch["d_ff"],
        dropout=dropout,
        head_dropout=0.0,
    )
    model = PatchTST(config)
    return model.to(device)


def train_model(
    model: torch.nn.Module,
    revin: RevIN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_config: dict,
    learning_rate: float,
    device: str,
    model_name: str,
) -> dict:
    """Train a model and return metrics."""
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(revin.parameters()),
        lr=learning_rate,
    )

    accumulation_steps = batch_config["accumulation_steps"]

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

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
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            train_loss += loss.item() * accumulation_steps
            n_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Handle remaining gradients
        if n_batches % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= n_batches

        # Validation
        model.eval()
        revin.eval()
        all_val_preds = []
        all_val_labels = []
        val_loss_sum = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_X = revin.normalize(batch_X)
                preds = model(batch_X)

                if preds.dim() == 2:
                    preds = preds.squeeze(-1)

                all_val_preds.append(preds.cpu().numpy())
                all_val_labels.append(batch_y.numpy())
                val_loss_sum += criterion(preds.cpu(), batch_y).item()
                val_batches += 1

        val_preds = np.concatenate(all_val_preds)
        val_labels = np.concatenate(all_val_labels)
        val_loss = val_loss_sum / val_batches
        val_auc = roc_auc_score(val_labels, val_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 5 == 0:
            print(
                f"    Epoch {epoch + 1:3d}: train_loss={train_loss:.4f}, "
                f"val_auc={val_auc:.4f}, best={best_val_auc:.4f}"
            )

    elapsed = time.time() - start_time

    # Final metrics
    final_auc = val_auc
    pred_min = float(np.min(val_preds))
    pred_max = float(np.max(val_preds))
    pred_std = float(np.std(val_preds))

    return {
        "best_val_auc": float(best_val_auc),
        "final_val_auc": float(final_auc),
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "pred_std": pred_std,
        "training_time_min": elapsed / 60,
        "history": history,
    }


def run_single_config(config: dict, df: pd.DataFrame, device: str) -> dict:
    """Run a single configuration."""
    print(f"\n{'=' * 70}")
    print(f"CONFIG: {config['name']}")
    print(f"  Architecture: d={ARCHITECTURE['d_model']}, L={ARCHITECTURE['n_layers']}, "
          f"h={ARCHITECTURE['n_heads']}, d_ff={ARCHITECTURE['d_ff']}")
    print(f"  LR={config['lr']:.0e}, Dropout={config['dropout']}")
    print(f"{'=' * 70}")

    # Create model
    model = create_model(ARCHITECTURE, config["dropout"], device)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    # Get memory-safe batch config
    batch_config = get_memory_safe_batch_config(n_params, ARCHITECTURE["n_layers"])
    print(f"  Batch config: micro={batch_config['micro_batch']}, "
          f"accum={batch_config['accumulation_steps']}, "
          f"effective={batch_config['effective_batch']}")

    # Create RevIN
    revin = RevIN(num_features=NUM_FEATURES).to(device)

    # Create splits
    splitter = SimpleSplitter(
        dates=df["Date"],
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        val_start="2023-01-01",
        test_start="2025-01-01",
    )
    split_indices = splitter.split()

    print(
        f"  Train: {len(split_indices.train_indices)}, "
        f"Val: {len(split_indices.val_indices)}, "
        f"Test: {len(split_indices.test_indices)}"
    )

    # Create dataset
    close_prices = df["Close"].values
    full_dataset = FinancialDataset(
        features_df=df,
        close_prices=close_prices,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        threshold=THRESHOLD,
    )

    train_dataset = Subset(full_dataset, split_indices.train_indices)
    val_dataset = Subset(full_dataset, split_indices.val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_config["micro_batch"],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_config["micro_batch"],
        shuffle=False
    )

    # Train
    metrics = train_model(
        model=model,
        revin=revin,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_config=batch_config,
        learning_rate=config["lr"],
        device=device,
        model_name=config["name"],
    )

    # Print results
    print(f"\n  Results:")
    print(f"    Best Val AUC: {metrics['best_val_auc']:.4f} (epoch {metrics['best_epoch']})")
    print(f"    Final Val AUC: {metrics['final_val_auc']:.4f}")
    print(f"    Pred Range: [{metrics['pred_min']:.4f}, {metrics['pred_max']:.4f}]")
    print(f"    Pred Std: {metrics['pred_std']:.4f}")
    print(f"    Training Time: {metrics['training_time_min']:.1f} min")

    # Remove history from CSV output (keep in JSON)
    history = metrics.pop("history")

    # Combine config with metrics
    result = {
        "config_name": config["name"],
        "learning_rate": config["lr"],
        "dropout": config["dropout"],
        "d_model": ARCHITECTURE["d_model"],
        "n_layers": ARCHITECTURE["n_layers"],
        "n_heads": ARCHITECTURE["n_heads"],
        "d_ff": ARCHITECTURE["d_ff"],
        "n_params": n_params,
        "micro_batch": batch_config["micro_batch"],
        "accumulation_steps": batch_config["accumulation_steps"],
        "train_samples": len(split_indices.train_indices),
        "val_samples": len(split_indices.val_indices),
        **metrics,
    }

    # Add history back for JSON
    result["history"] = history

    return result


def main():
    print("=" * 70)
    print("LR AND DROPOUT TUNING EXPERIMENT")
    print("Testing: Slower LRs and higher dropout on optimal L=6 architecture")
    print(f"Reference: LR=1e-4, dropout=0.50 achieved AUC 0.7342")
    print("=" * 70)

    # Check device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Data: {len(df)} rows")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run all configs
    results = []
    for config in CONFIGS:
        result = run_single_config(config, df, device)
        results.append(result)

        # Save intermediate results (in case of crash)
        csv_results = [{k: v for k, v in r.items() if k != "history"} for r in results]
        csv_path = OUTPUT_DIR / "lr_dropout_tuning_results.csv"
        pd.DataFrame(csv_results).to_csv(csv_path, index=False)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: LR and Dropout Tuning Experiment")
    print(f"{'=' * 70}")
    print(f"{'Config':<12} {'LR':<10} {'Dropout':<8} {'Best AUC':<10} {'vs ref':<10}")
    print("-" * 70)

    ref_auc = 0.7342  # Reference: LR=1e-4, dropout=0.50

    for r in results:
        diff = r["best_val_auc"] - ref_auc
        diff_str = f"{diff:+.4f}"
        print(
            f"{r['config_name']:<12} {r['learning_rate']:<10.0e} {r['dropout']:<8} "
            f"{r['best_val_auc']:.4f}     {diff_str}"
        )

    # Find best
    best = max(results, key=lambda x: x["best_val_auc"])
    print(f"\nBest: {best['config_name']} (LR={best['learning_rate']:.0e}, dropout={best['dropout']}) = {best['best_val_auc']:.4f}")

    if best["best_val_auc"] > ref_auc:
        print(f"  NEW BEST! +{best['best_val_auc'] - ref_auc:.4f} over reference")
    else:
        print(f"  Reference still best ({ref_auc:.4f})")

    # Group by dropout for comparison
    print(f"\n{'=' * 70}")
    print("COMPARISON BY DROPOUT")
    print(f"{'=' * 70}")

    for dropout in [0.50, 0.55]:
        dropout_results = [r for r in results if r["dropout"] == dropout]
        if dropout_results:
            print(f"\nDropout={dropout}:")
            for r in sorted(dropout_results, key=lambda x: x["learning_rate"], reverse=True):
                diff = r["best_val_auc"] - ref_auc
                print(f"  LR={r['learning_rate']:.0e}: AUC={r['best_val_auc']:.4f} ({diff:+.4f})")

    # Save final results with history
    output = {
        "experiment": "lr_dropout_tuning",
        "description": "Testing slower LRs and higher dropout on optimal L=6 architecture",
        "fixed_params": {
            "architecture": ARCHITECTURE,
            "context_length": CONTEXT_LENGTH,
            "epochs": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "threshold": THRESHOLD,
        },
        "reference": {
            "lr": 1e-4,
            "dropout": 0.50,
            "auc": ref_auc,
        },
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    json_path = OUTPUT_DIR / "lr_dropout_tuning_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    csv_path = OUTPUT_DIR / "lr_dropout_tuning_results.csv"
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
