#!/usr/bin/env python3
"""
Combined LR + Dropout Ablation Experiment
Tests combinations of lower LR and higher dropout on PatchTST and MLP models.
Run AFTER reviewing LR and Dropout ablation results.

Configurations:
- C1: PatchTST @ LR=1e-5, dropout=0.4
- C2: PatchTST @ LR=1e-5, dropout=0.5
- C3: PatchTST @ LR=1e-6, dropout=0.4
- C4: PatchTST @ LR=1e-6, dropout=0.5
- C5: MLP @ LR=1e-5, dropout=0.4
- C6: MLP @ LR=1e-5, dropout=0.5
- C7: MLP @ LR=1e-6, dropout=0.4
- C8: MLP @ LR=1e-6, dropout=0.5

Baselines:
- PatchTST @ LR=1e-3, dropout=0.2: AUC 0.6945
- MLP @ LR=1e-3, dropout=0.2: Best AUC 0.7077, Final AUC 0.6626
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data.dataset import SimpleSplitter, FinancialDataset
from src.models.patchtst import PatchTST, PatchTSTConfig, RevIN
from src.training.losses import FocalLoss
from torch.utils.data import DataLoader, Subset

# ============================================================================
# MLP MODEL (from test_mlp_only.py)
# ============================================================================


class PatchMLP(nn.Module):
    """MLP that processes patches independently without attention."""

    def __init__(
        self,
        num_features: int,
        context_length: int,
        patch_length: int = 16,
        stride: int = 8,
        hidden_dim: int = 256,
        output_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_features = num_features
        self.context_length = context_length
        self.patch_length = patch_length
        self.stride = stride

        self.n_patches = (context_length - patch_length) // stride + 1
        self.patch_input_dim = patch_length * num_features

        self.patch_mlp = nn.Sequential(
            nn.Linear(self.patch_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        self.head = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = start + self.patch_length
            patch = x[:, start:end, :]
            patches.append(patch)

        patches = torch.stack(patches, dim=1)
        patches = patches.view(batch_size, self.n_patches, -1)
        patch_outputs = self.patch_mlp(patches)
        pooled = patch_outputs.mean(dim=1)
        output = self.head(pooled).squeeze(-1)

        return output


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Fixed parameters
CONTEXT_LENGTH = 80
HORIZON = 1
PATCH_LENGTH = 16
STRIDE = 8
BATCH_SIZE = 128
EPOCHS = 100  # Extended for slow LRs
NUM_FEATURES = 25  # SPY_dataset_a20: 5 OHLCV + 20 indicators

# PatchTST architecture (baseline)
D_MODEL = 64
N_LAYERS = 4
N_HEADS = 4
D_FF = 256

# MLP architecture (baseline)
MLP_HIDDEN = 256
MLP_OUTPUT = 64

DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs/training_dynamics"

THRESHOLD = 0.01

# Configurations to test (2 models x 2 LRs x 2 dropouts = 8)
CONFIGS = [
    {"name": "PatchTST_lr1e-5_d0.4", "model": "PatchTST", "lr": 1e-5, "dropout": 0.4},
    {"name": "PatchTST_lr1e-5_d0.5", "model": "PatchTST", "lr": 1e-5, "dropout": 0.5},
    {"name": "PatchTST_lr1e-6_d0.4", "model": "PatchTST", "lr": 1e-6, "dropout": 0.4},
    {"name": "PatchTST_lr1e-6_d0.5", "model": "PatchTST", "lr": 1e-6, "dropout": 0.5},
    {"name": "MLP_lr1e-5_d0.4", "model": "MLP", "lr": 1e-5, "dropout": 0.4},
    {"name": "MLP_lr1e-5_d0.5", "model": "MLP", "lr": 1e-5, "dropout": 0.5},
    {"name": "MLP_lr1e-6_d0.4", "model": "MLP", "lr": 1e-6, "dropout": 0.4},
    {"name": "MLP_lr1e-6_d0.5", "model": "MLP", "lr": 1e-6, "dropout": 0.5},
]


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================


def create_patchtst_model(dropout: float, device: str) -> nn.Module:
    """Create PatchTST model."""
    config = PatchTSTConfig(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=PATCH_LENGTH,
        stride=STRIDE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        dropout=dropout,
        head_dropout=0.0,
    )
    model = PatchTST(config)
    return model.to(device)


def create_mlp_model(dropout: float, device: str) -> nn.Module:
    """Create MLP model."""
    model = PatchMLP(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=PATCH_LENGTH,
        stride=STRIDE,
        hidden_dim=MLP_HIDDEN,
        output_dim=MLP_OUTPUT,
        dropout=dropout,
    )
    return model.to(device)


def train_model(
    model: nn.Module,
    revin: RevIN,
    train_loader: DataLoader,
    val_loader: DataLoader,
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

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 10

    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    start_time = time.time()

    for epoch in range(EPOCHS):
        # Training
        model.train()
        revin.train()
        train_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            batch_X = revin.normalize(batch_X)

            optimizer.zero_grad()
            preds = model(batch_X)

            if preds.dim() == 2:
                preds = preds.squeeze(-1)

            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

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
            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1:3d}: train_loss={train_loss:.4f}, "
                  f"val_auc={val_auc:.4f}, best={best_val_auc:.4f}")

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
    }


def run_single_config(config: dict, df: pd.DataFrame, device: str) -> dict:
    """Run a single configuration."""
    print(f"\n{'=' * 70}")
    print(f"CONFIG: {config['name']}")
    print(f"  Model: {config['model']}, LR: {config['lr']}, Dropout: {config['dropout']}")
    print(f"{'=' * 70}")

    # Create model
    if config["model"] == "PatchTST":
        model = create_patchtst_model(config["dropout"], device)
    else:
        model = create_mlp_model(config["dropout"], device)

    n_params = count_parameters(model)
    print(f"  Parameters: {n_params:,}")

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

    print(f"  Train: {len(split_indices.train_indices)}, "
          f"Val: {len(split_indices.val_indices)}, "
          f"Test: {len(split_indices.test_indices)}")

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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Train
    metrics = train_model(
        model=model,
        revin=revin,
        train_loader=train_loader,
        val_loader=val_loader,
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

    # Combine config with metrics
    result = {
        "config_name": config["name"],
        "model_type": config["model"],
        "learning_rate": config["lr"],
        "dropout": config["dropout"],
        "n_params": n_params,
        "train_samples": len(split_indices.train_indices),
        "val_samples": len(split_indices.val_indices),
        **metrics,
    }

    return result


def main():
    print("=" * 70)
    print("COMBINED LR + DROPOUT ABLATION EXPERIMENT")
    print("Testing: All combinations of LR (1e-5, 1e-6) and Dropout (0.4, 0.5)")
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
        csv_path = OUTPUT_DIR / "combined_ablation_results.csv"
        pd.DataFrame(results).to_csv(csv_path, index=False)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY: Combined LR + Dropout Ablation")
    print(f"{'=' * 70}")
    print(f"{'Config':<25} {'Model':<10} {'LR':<10} {'Drop':<6} {'Best AUC':<10} {'Final AUC':<10}")
    print("-" * 76)

    for r in sorted(results, key=lambda x: x["best_val_auc"], reverse=True):
        print(f"{r['config_name']:<25} {r['model_type']:<10} {r['learning_rate']:<10.0e} "
              f"{r['dropout']:<6.1f} {r['best_val_auc']:.4f}     {r['final_val_auc']:.4f}")

    # Comparison to baselines
    print(f"\n{'=' * 70}")
    print("COMPARISON TO BASELINES")
    print(f"{'=' * 70}")
    print("Baselines (LR=1e-3, Dropout=0.2):")
    print("  PatchTST: AUC 0.6945 (stable)")
    print("  MLP:      Best AUC 0.7077, Final AUC 0.6626 (overfits)")
    print("  RF:       AUC 0.716 (TARGET)")
    print()

    patchtst_baseline = 0.6945
    mlp_best_baseline = 0.7077
    mlp_final_baseline = 0.6626
    rf_target = 0.716

    # Find best results
    best_patchtst = max([r for r in results if r["model_type"] == "PatchTST"],
                        key=lambda x: x["best_val_auc"])
    best_mlp = max([r for r in results if r["model_type"] == "MLP"],
                   key=lambda x: x["best_val_auc"])
    best_mlp_final = max([r for r in results if r["model_type"] == "MLP"],
                         key=lambda x: x["final_val_auc"])

    print(f"Best PatchTST: {best_patchtst['config_name']}")
    print(f"  Best AUC: {best_patchtst['best_val_auc']:.4f} "
          f"({best_patchtst['best_val_auc'] - patchtst_baseline:+.4f} vs baseline)")
    print()

    print(f"Best MLP (by best AUC): {best_mlp['config_name']}")
    print(f"  Best AUC: {best_mlp['best_val_auc']:.4f} "
          f"({best_mlp['best_val_auc'] - mlp_best_baseline:+.4f} vs baseline peak)")
    print(f"  Final AUC: {best_mlp['final_val_auc']:.4f} "
          f"({best_mlp['final_val_auc'] - mlp_final_baseline:+.4f} vs baseline final)")
    print()

    print(f"Best MLP (by final AUC): {best_mlp_final['config_name']}")
    print(f"  Final AUC: {best_mlp_final['final_val_auc']:.4f} "
          f"({best_mlp_final['final_val_auc'] - mlp_final_baseline:+.4f} vs baseline final)")
    print()

    # Gap to RF
    best_overall = max(results, key=lambda x: x["best_val_auc"])
    gap_to_rf = rf_target - best_overall["best_val_auc"]
    print(f"Gap to RF target (0.716): {gap_to_rf:.4f}")

    # Key findings
    print(f"\n{'=' * 70}")
    print("KEY FINDINGS")
    print(f"{'=' * 70}")

    # Check if MLP final improved (reduced overfitting)
    mlp_results = [r for r in results if r["model_type"] == "MLP"]
    best_mlp_final_auc = max(r["final_val_auc"] for r in mlp_results)
    if best_mlp_final_auc > mlp_final_baseline:
        print(f"[+] MLP overfitting REDUCED: Final AUC {best_mlp_final_auc:.4f} > baseline {mlp_final_baseline:.4f}")
    else:
        print(f"[-] MLP overfitting NOT reduced: Best final AUC {best_mlp_final_auc:.4f}")

    # Check if any config beats RF
    if best_overall["best_val_auc"] >= rf_target:
        print(f"[+] BREAKTHROUGH: {best_overall['config_name']} matches/beats RF!")
    elif best_overall["best_val_auc"] >= rf_target - 0.01:
        print(f"[~] CLOSE: {best_overall['config_name']} within 1% of RF")
    else:
        print(f"[-] Still {gap_to_rf:.4f} behind RF target")

    # Save final results
    output = {
        "experiment": "combined_ablation",
        "description": "Testing all combinations of LR (1e-5, 1e-6) and Dropout (0.4, 0.5)",
        "fixed_params": {
            "context_length": CONTEXT_LENGTH,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
        "baselines": {
            "patchtst_auc": patchtst_baseline,
            "mlp_best_auc": mlp_best_baseline,
            "mlp_final_auc": mlp_final_baseline,
            "rf_target": rf_target,
        },
        "results": results,
        "best_patchtst": best_patchtst["config_name"],
        "best_mlp_peak": best_mlp["config_name"],
        "best_mlp_final": best_mlp_final["config_name"],
        "timestamp": datetime.now().isoformat(),
    }

    json_path = OUTPUT_DIR / "combined_ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    csv_path = OUTPUT_DIR / "combined_ablation_results.csv"
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
