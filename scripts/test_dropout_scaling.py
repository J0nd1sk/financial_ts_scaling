#!/usr/bin/env python3
"""
Dropout Scaling Experiment
Tests whether dropout=0.5 regularization finding generalizes to larger model scales.
Also explores width vs depth trade-offs at 20M and 200M parameter budgets.

At 2M scale (215K params, d=64, L=4), dropout=0.5 achieved AUC 0.7199 (only 0.4% below RF).
This experiment tests:
1. Does dropout=0.5 help at larger scales?
2. Do larger models prefer narrow-deep or wide-shallow architectures?

Configurations (6 architectures × dropout=0.5):
- 20M_narrow:   d=256, L=32 (25.4M) - deeper, tests if 2M narrow-deep finding scales
- 20M_balanced: d=384, L=12 (21.5M) - balanced scaling
- 20M_wide:     d=512, L=6  (19.1M) - wider, fewer layers
- 200M_narrow:  d=512, L=48 (151.5M) - deep architecture
- 200M_balanced: d=768, L=24 (170.4M) - balanced
- 200M_wide:    d=1024, L=12 (151.6M) - wide, moderate depth

Reference results (2M scale, d=64, L=4):
- PatchTST @ dropout=0.2: AUC 0.6945
- PatchTST @ dropout=0.5: AUC 0.7199 (+3.7%)
- RF target: AUC 0.716
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
LEARNING_RATE = 1e-4  # Lower LR for larger models
NUM_FEATURES = 25  # SPY_dataset_a20: 5 OHLCV + 20 indicators
EARLY_STOPPING_PATIENCE = 10
DROPOUT = 0.5  # Winning dropout from 2M experiments

DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs/dropout_scaling"

THRESHOLD = 0.01

# Architecture configurations - exploring width vs depth trade-offs
# Reference: 2M sweet spot was d=64, L=4 (d/L ratio = 16)
ARCHITECTURES = {
    # 20M variants (~20-25M params)
    "20M_narrow": {
        "d_model": 256,
        "n_layers": 32,
        "n_heads": 4,
        "d_ff": 1024,
    },
    "20M_balanced": {
        "d_model": 384,
        "n_layers": 12,
        "n_heads": 4,
        "d_ff": 1536,
    },
    "20M_wide": {
        "d_model": 512,
        "n_layers": 6,
        "n_heads": 8,
        "d_ff": 2048,
    },
    # 200M variants (~150-170M params)
    "200M_narrow": {
        "d_model": 512,
        "n_layers": 48,
        "n_heads": 8,
        "d_ff": 2048,
    },
    "200M_balanced": {
        "d_model": 768,
        "n_layers": 24,
        "n_heads": 8,
        "d_ff": 3072,
    },
    "200M_wide": {
        "d_model": 1024,
        "n_layers": 12,
        "n_heads": 16,
        "d_ff": 4096,
    },
}

# Configurations to test - all with dropout=0.5
CONFIGS = [
    {"name": "20M_narrow", "arch": "20M_narrow", "dropout": DROPOUT},
    {"name": "20M_balanced", "arch": "20M_balanced", "dropout": DROPOUT},
    {"name": "20M_wide", "arch": "20M_wide", "dropout": DROPOUT},
    {"name": "200M_narrow", "arch": "200M_narrow", "dropout": DROPOUT},
    {"name": "200M_balanced", "arch": "200M_balanced", "dropout": DROPOUT},
    {"name": "200M_wide", "arch": "200M_wide", "dropout": DROPOUT},
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
    device: str,
    model_name: str,
) -> dict:
    """Train a model and return metrics."""
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(revin.parameters()),
        lr=LEARNING_RATE,
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
    arch = ARCHITECTURES[config["arch"]]
    budget = "20M" if "20M" in config["arch"] else "200M"
    print(f"  Budget: {budget}, Dropout: {config['dropout']}")
    print(f"  Architecture: d={arch['d_model']}, L={arch['n_layers']}, h={arch['n_heads']}, d_ff={arch['d_ff']}")
    print(f"{'=' * 70}")

    # Create model
    model = create_model(arch, config["dropout"], device)
    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

    # Get memory-safe batch config
    batch_config = get_memory_safe_batch_config(n_params, arch["n_layers"])
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
        "arch_name": config["arch"],
        "budget": budget,
        "dropout": config["dropout"],
        "d_model": arch["d_model"],
        "n_layers": arch["n_layers"],
        "n_heads": arch["n_heads"],
        "d_ff": arch["d_ff"],
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
    print("DROPOUT SCALING EXPERIMENT")
    print("Testing: Width vs depth at 20M and 200M scales with dropout=0.5")
    print("Reference: 2M (d=64, L=4) @ dropout=0.5 achieved AUC 0.7199")
    print(f"Learning rate: {LEARNING_RATE}")
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
        # Remove history for CSV
        csv_results = [{k: v for k, v in r.items() if k != "history"} for r in results]
        csv_path = OUTPUT_DIR / "dropout_scaling_results.csv"
        pd.DataFrame(csv_results).to_csv(csv_path, index=False)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: Dropout Scaling Experiment")
    print(f"{'=' * 70}")
    print(f"{'Config':<16} {'d':<5} {'L':<4} {'Params':<12} {'Best AUC':<10} {'vs 2M ref':<10}")
    print("-" * 70)

    ref_2m_d05 = 0.7199  # 2M reference with dropout=0.5

    # Group by budget for comparison
    for budget in ["20M", "200M"]:
        print(f"\n{budget} Scale:")
        budget_results = [r for r in results if r["budget"] == budget]

        for r in budget_results:
            diff = r["best_val_auc"] - ref_2m_d05
            diff_str = f"{diff:+.4f}"

            print(
                f"  {r['config_name']:<14} {r['d_model']:<5} {r['n_layers']:<4} "
                f"{r['n_params']:>10,}  {r['best_val_auc']:.4f}     {diff_str}"
            )

    # Comparison to 2M reference and RF target
    print(f"\n{'=' * 70}")
    print("COMPARISON TO TARGETS")
    print(f"{'=' * 70}")
    print("Reference points:")
    print("  2M @ d=64, L=4, dropout=0.5: AUC 0.7199")
    print("  RF target:                   AUC 0.716")
    print()

    ref_rf = 0.716

    # Find best per budget
    for budget in ["20M", "200M"]:
        budget_results = [r for r in results if r["budget"] == budget]
        if budget_results:
            best = max(budget_results, key=lambda x: x["best_val_auc"])
            gap_to_rf = best["best_val_auc"] - ref_rf
            gap_to_2m = best["best_val_auc"] - ref_2m_d05
            print(f"Best {budget}: {best['config_name']} (d={best['d_model']}, L={best['n_layers']})")
            print(f"  AUC: {best['best_val_auc']:.4f} | vs RF: {gap_to_rf:+.4f} | vs 2M: {gap_to_2m:+.4f}")
            print()

    # Save final results with history
    output = {
        "experiment": "dropout_scaling",
        "description": "Testing if dropout=0.5 regularization generalizes to larger scales",
        "fixed_params": {
            "learning_rate": LEARNING_RATE,
            "context_length": CONTEXT_LENGTH,
            "epochs": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "threshold": THRESHOLD,
        },
        "reference_2m": {
            "dropout_0.5_auc": ref_2m_d05,
            "rf_target_auc": ref_rf,
        },
        "architectures": ARCHITECTURES,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    json_path = OUTPUT_DIR / "dropout_scaling_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    csv_path = OUTPUT_DIR / "dropout_scaling_results.csv"
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
