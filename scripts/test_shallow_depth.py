#!/usr/bin/env python3
"""
Shallow Depth Experiment
Tests whether depths shallower than L=6 improve performance at 20M parameter scale.

Previous finding: At 20M scale, L=6 (0.7342) > L=12 (0.7282) > L=32 (0.7253)
This suggests shallower might be better. Testing L=2, L=3, L=4, L=5.

All configs use same hyperparameters as best 20M result:
- LR: 1e-4
- Dropout: 0.5
- Context: 80 days
- Focal Loss + RevIN

Architectures (all ~20M params):
- 20M_L2: d=896, L=2, h=8, d_ff=3584 (19.7M)
- 20M_L3: d=720, L=3, h=8, d_ff=2880 (19.0M)
- 20M_L4: d=640, L=4, h=8, d_ff=2560 (20.0M)
- 20M_L5: d=560, L=5, h=8, d_ff=2240 (19.1M)

Reference: 20M_L6 (d=512, L=6) achieved AUC 0.7342
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

# Fixed parameters (same as 20M_wide best result)
CONTEXT_LENGTH = 80
HORIZON = 1
PATCH_LENGTH = 16
STRIDE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
DROPOUT = 0.5
NUM_FEATURES = 25
EARLY_STOPPING_PATIENCE = 10

DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs/shallow_depth"

THRESHOLD = 0.01

# Architecture configurations - shallow depth sweep at 20M
# All use h=8, d_ff=4*d_model
ARCHITECTURES = {
    "20M_L2": {
        "d_model": 896,
        "n_layers": 2,
        "n_heads": 8,
        "d_ff": 3584,
    },
    "20M_L3": {
        "d_model": 720,
        "n_layers": 3,
        "n_heads": 8,
        "d_ff": 2880,
    },
    "20M_L4": {
        "d_model": 640,
        "n_layers": 4,
        "n_heads": 8,
        "d_ff": 2560,
    },
    "20M_L5": {
        "d_model": 560,
        "n_layers": 5,
        "n_heads": 8,
        "d_ff": 2240,
    },
}

# Configurations to test
CONFIGS = [
    {"name": "20M_L2", "arch": "20M_L2"},
    {"name": "20M_L3", "arch": "20M_L3"},
    {"name": "20M_L4", "arch": "20M_L4"},
    {"name": "20M_L5", "arch": "20M_L5"},
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
    print(f"  Architecture: d={arch['d_model']}, L={arch['n_layers']}, h={arch['n_heads']}, d_ff={arch['d_ff']}")
    print(f"  LR={LEARNING_RATE}, Dropout={DROPOUT}")
    print(f"{'=' * 70}")

    # Create model
    model = create_model(arch, DROPOUT, device)
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
        "d_model": arch["d_model"],
        "n_layers": arch["n_layers"],
        "n_heads": arch["n_heads"],
        "d_ff": arch["d_ff"],
        "n_params": n_params,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
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
    print("SHALLOW DEPTH EXPERIMENT")
    print("Testing: L=2, L=3, L=4, L=5 at 20M parameter scale")
    print(f"Reference: 20M_L6 (d=512, L=6) achieved AUC 0.7342")
    print(f"Fixed: LR={LEARNING_RATE}, Dropout={DROPOUT}")
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
        csv_path = OUTPUT_DIR / "shallow_depth_results.csv"
        pd.DataFrame(csv_results).to_csv(csv_path, index=False)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: Shallow Depth Experiment")
    print(f"{'=' * 70}")
    print(f"{'Config':<12} {'d':<5} {'L':<4} {'Params':<12} {'Best AUC':<10} {'vs L=6':<10}")
    print("-" * 70)

    ref_l6 = 0.7342  # 20M_L6 reference

    for r in results:
        diff = r["best_val_auc"] - ref_l6
        diff_str = f"{diff:+.4f}"
        print(
            f"{r['config_name']:<12} {r['d_model']:<5} {r['n_layers']:<4} "
            f"{r['n_params']:>10,}  {r['best_val_auc']:.4f}     {diff_str}"
        )

    # Find best
    best = max(results, key=lambda x: x["best_val_auc"])
    print(f"\nBest: {best['config_name']} (d={best['d_model']}, L={best['n_layers']}) = {best['best_val_auc']:.4f}")

    if best["best_val_auc"] > ref_l6:
        print(f"  NEW BEST! +{best['best_val_auc'] - ref_l6:.4f} over L=6 reference")
    else:
        print(f"  L=6 still best ({ref_l6:.4f})")

    # Save final results with history
    output = {
        "experiment": "shallow_depth",
        "description": "Testing if depths L=2-5 improve over L=6 at 20M scale",
        "fixed_params": {
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "context_length": CONTEXT_LENGTH,
            "epochs": EPOCHS,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "threshold": THRESHOLD,
        },
        "reference": {
            "20M_L6_auc": ref_l6,
            "rf_target_auc": 0.716,
        },
        "architectures": ARCHITECTURES,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    json_path = OUTPUT_DIR / "shallow_depth_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    csv_path = OUTPUT_DIR / "shallow_depth_results.csv"
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
