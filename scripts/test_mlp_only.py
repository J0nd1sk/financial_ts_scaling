#!/usr/bin/env python3
"""
MLP-Only Experiment
Hypothesis: Self-attention may be introducing noise by allowing patches to attend
to irrelevant historical patterns. A simple MLP processing patches independently
might perform equally well or better.

Architecture:
- Patch input sequence (same as PatchTST: patch_len=16, stride=8)
- Process each patch through MLP independently (no attention)
- Mean pool across patches
- Classification head

Baseline: PatchTST AUC 0.6945 (L=4, d=64, ctx=80, Focal Loss, RevIN)
Target: RF baseline AUC 0.716
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
from src.models.patchtst import RevIN
from src.training.losses import FocalLoss
from torch.utils.data import DataLoader, Subset

# ============================================================================
# MLP MODEL (NO ATTENTION)
# ============================================================================


class PatchMLP(nn.Module):
    """MLP that processes patches independently without attention.

    Architecture:
        1. Patch input sequence into overlapping patches
        2. Flatten each patch: (batch, n_patches, patch_len * num_features)
        3. MLP per patch: Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear
        4. Mean pool across patches
        5. Classification head
    """

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

        # Calculate number of patches
        self.n_patches = (context_length - patch_length) // stride + 1
        self.patch_input_dim = patch_length * num_features

        # MLP for each patch (shared weights across patches)
        self.patch_mlp = nn.Sequential(
            nn.Linear(self.patch_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(output_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            Predictions of shape (batch,)
        """
        batch_size = x.shape[0]

        # Create patches: (batch, n_patches, patch_len, num_features)
        patches = []
        for i in range(self.n_patches):
            start = i * self.stride
            end = start + self.patch_length
            patch = x[:, start:end, :]  # (batch, patch_len, num_features)
            patches.append(patch)

        # Stack patches: (batch, n_patches, patch_len, num_features)
        patches = torch.stack(patches, dim=1)

        # Flatten each patch: (batch, n_patches, patch_len * num_features)
        patches = patches.view(batch_size, self.n_patches, -1)

        # Process each patch through MLP: (batch, n_patches, output_dim)
        # Apply MLP to each patch independently
        patch_outputs = self.patch_mlp(patches)

        # Mean pool across patches: (batch, output_dim)
        pooled = patch_outputs.mean(dim=1)

        # Classification head: (batch, 1) -> (batch,)
        output = self.head(pooled).squeeze(-1)

        return output


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# CONFIGURATIONS TO TEST
# ============================================================================

CONFIGS = {
    # Match baseline ~215K params
    "MLP_h256_o64": {"hidden_dim": 256, "output_dim": 64},
    # Larger MLP to test if capacity helps
    "MLP_h512_o128": {"hidden_dim": 512, "output_dim": 128},
    # Smaller MLP for comparison
    "MLP_h128_o32": {"hidden_dim": 128, "output_dim": 32},
}

# ============================================================================
# FIXED PARAMETERS
# ============================================================================

CONTEXT_LENGTH = 80  # Optimal from ablation study
HORIZON = 1
PATCH_LENGTH = 16
STRIDE = 8
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 50
DROPOUT = 0.20
NUM_FEATURES = 25  # 5 OHLCV + 20 indicators

DATA_PATH = PROJECT_ROOT / "data/processed/v1/SPY_dataset_a20.parquet"
OUTPUT_DIR = PROJECT_ROOT / "outputs/mlp_only_experiment"

# Threshold for 1% task
THRESHOLD = 0.01


def run_single_config(config_name: str, mlp_config: dict, df: pd.DataFrame, device: str) -> dict:
    """Run a single MLP configuration."""
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*70}")

    # Create model
    model = PatchMLP(
        num_features=NUM_FEATURES,
        context_length=CONTEXT_LENGTH,
        patch_length=PATCH_LENGTH,
        stride=STRIDE,
        hidden_dim=mlp_config["hidden_dim"],
        output_dim=mlp_config["output_dim"],
        dropout=DROPOUT,
    ).to(device)

    n_params = count_parameters(model)
    print(f"Architecture: hidden={mlp_config['hidden_dim']}, output={mlp_config['output_dim']}")
    print(f"Parameters: {n_params:,}")
    print(f"Patches: {model.n_patches} (patch_len={PATCH_LENGTH}, stride={STRIDE})")

    # Create RevIN layer
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

    print(f"Train: {len(split_indices.train_indices)}, "
          f"Val: {len(split_indices.val_indices)}, "
          f"Test: {len(split_indices.test_indices)}")

    # Create FinancialDataset (handles target computation from Close prices)
    close_prices = df["Close"].values
    full_dataset = FinancialDataset(
        features_df=df,
        close_prices=close_prices,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        threshold=THRESHOLD,
    )

    # Create subset datasets for train/val
    train_dataset = Subset(full_dataset, split_indices.train_indices)
    val_dataset = Subset(full_dataset, split_indices.val_indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Training setup
    criterion = FocalLoss(gamma=2.0, alpha=0.25)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(revin.parameters()),
        lr=LEARNING_RATE,
    )

    # Training loop with early stopping
    best_val_auc = 0.0
    patience_counter = 0
    patience = 10

    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        revin.train()

        train_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Apply RevIN normalization
            batch_X = revin.normalize(batch_X)

            optimizer.zero_grad()
            preds = model(batch_X)
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
                all_val_preds.append(preds.cpu().numpy())
                all_val_labels.append(batch_y.numpy())
                val_loss_sum += criterion(preds.cpu(), batch_y).item()
                val_batches += 1

        val_preds = np.concatenate(all_val_preds)
        val_labels = np.concatenate(all_val_labels)
        val_loss = val_loss_sum / val_batches
        val_auc = roc_auc_score(val_labels, val_preds)

        # Early stopping check
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            pred_std = np.std(val_preds)
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}, "
                  f"pred_std={pred_std:.4f}")

    elapsed = time.time() - start_time

    # Final validation metrics (use last computed val_preds/val_labels)
    final_auc = val_auc
    final_preds = val_preds
    pred_min = float(np.min(final_preds))
    pred_max = float(np.max(final_preds))
    pred_std = float(np.std(final_preds))

    print(f"\nResults:")
    print(f"  Val AUC: {final_auc:.4f}")
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    print(f"  Pred Range: [{pred_min:.4f}, {pred_max:.4f}]")
    print(f"  Pred Std: {pred_std:.4f}")
    print(f"  Training Time: {elapsed/60:.1f} min")

    return {
        "config_name": config_name,
        "hidden_dim": mlp_config["hidden_dim"],
        "output_dim": mlp_config["output_dim"],
        "n_params": n_params,
        "val_auc": float(final_auc),
        "best_val_auc": float(best_val_auc),
        "pred_min": pred_min,
        "pred_max": pred_max,
        "pred_std": pred_std,
        "training_time_min": elapsed / 60,
        "train_samples": len(split_indices.train_indices),
        "val_samples": len(split_indices.val_indices),
        "n_patches": model.n_patches,
    }


def main():
    print("=" * 70)
    print("MLP-ONLY EXPERIMENT")
    print("Hypothesis: Self-attention may introduce noise; MLP might match/beat transformer")
    print("Baseline: PatchTST AUC 0.6945")
    print("Target: RF baseline AUC 0.716")
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
    for config_name, mlp_config in CONFIGS.items():
        result = run_single_config(config_name, mlp_config, df, device)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: MLP-Only Experiment")
    print(f"{'='*70}")
    print(f"{'Config':<18} {'Params':<10} {'Val AUC':<10} {'Pred Std':<10}")
    print("-" * 48)

    for r in sorted(results, key=lambda x: x["val_auc"], reverse=True):
        params_str = f"{r['n_params']/1000:.0f}K"
        print(f"{r['config_name']:<18} {params_str:<10} {r['val_auc']:.4f}     {r['pred_std']:.4f}")

    # Compare to baselines
    best_result = max(results, key=lambda x: x["val_auc"])
    patchtst_baseline = 0.6945
    rf_baseline = 0.716

    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print(f"Best MLP:        {best_result['config_name']} (AUC {best_result['val_auc']:.4f})")
    print(f"PatchTST:        L=4, d=64 (AUC {patchtst_baseline:.4f})")
    print(f"RF baseline:     AUC {rf_baseline:.4f}")

    mlp_vs_patchtst = best_result['val_auc'] - patchtst_baseline
    mlp_vs_rf = best_result['val_auc'] - rf_baseline

    print(f"\nMLP vs PatchTST: {mlp_vs_patchtst:+.4f} ({mlp_vs_patchtst/patchtst_baseline*100:+.1f}%)")
    print(f"MLP vs RF:       {mlp_vs_rf:+.4f} ({mlp_vs_rf/rf_baseline*100:+.1f}%)")

    if best_result['val_auc'] > patchtst_baseline:
        print("\n>>> FINDING: MLP OUTPERFORMS PatchTST! Attention may be hurting.")
    elif best_result['val_auc'] > patchtst_baseline - 0.01:
        print("\n>>> FINDING: MLP matches PatchTST. Attention provides minimal benefit.")
    else:
        print("\n>>> FINDING: PatchTST outperforms MLP. Attention is helping.")

    # Save results
    output = {
        "experiment": "mlp_only",
        "hypothesis": "Self-attention may introduce noise; MLP might match/beat transformer",
        "patchtst_baseline": patchtst_baseline,
        "rf_baseline": rf_baseline,
        "context_length": CONTEXT_LENGTH,
        "patch_length": PATCH_LENGTH,
        "stride": STRIDE,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Also save CSV for easy comparison
    csv_path = OUTPUT_DIR / "results.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")


if __name__ == "__main__":
    main()
