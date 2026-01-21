#!/usr/bin/env python3
"""
Test alternative loss functions to prevent probability collapse.

The diagnosis showed that PatchTST converges to predicting 0.5 for everything.
This script tests:
1. FocalLoss - focuses on hard examples, may encourage spread
2. SoftAUCLoss - directly optimizes ranking, should prevent collapse
3. BCE with class weights - may help with imbalance
4. BCE with increased pos_weight - stronger minority class signal
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.patchtst import PatchTST, PatchTSTConfig
from src.training.losses import FocalLoss, SoftAUCLoss


def create_target(df: pd.DataFrame, threshold: float = 0.01) -> pd.Series:
    """Create binary target: 1 if next-day return > threshold."""
    returns = df["Close"].pct_change().shift(-1)
    return (returns > threshold).astype(float)


class SimpleDataset(Dataset):
    """Simple sliding window dataset."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, context_length: int):
        self.context_length = context_length
        self.features = features
        self.labels = labels
        self.valid_indices = list(range(context_length, len(features) - 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        x = self.features[i - self.context_length : i]
        y = self.labels[i]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class LossExperimentTrainer:
    """Trainer that tests different loss functions."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        criterion_name: str,
        learning_rate: float = 3e-6,  # Very low LR based on diagnosis
        warmup_epochs: int = 10,
        total_epochs: int = 50,
        apply_sigmoid: bool = False,  # For SoftAUCLoss which needs sigmoid outputs
    ):
        self.model = model
        self.criterion = criterion
        self.criterion_name = criterion_name
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.apply_sigmoid = apply_sigmoid
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.history = []

    def get_lr(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.learning_rate * (epoch + 1) / self.warmup_epochs
        return self.learning_rate

    def train_epoch(self, dataloader, epoch: int) -> float:
        self.model.train()

        if epoch < self.warmup_epochs:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.get_lr(epoch)

        total_loss = 0
        n_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(batch_x)

            if output.dim() == 3:
                output = output[:, -1, 0]
            elif output.dim() == 2:
                output = output[:, -1]

            # For SoftAUCLoss, we need probabilities
            if self.apply_sigmoid:
                output = torch.sigmoid(output)

            loss = self.criterion(output, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        self.model.eval()

        all_probs = []
        all_labels = []

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            output = self.model(batch_x)

            if output.dim() == 3:
                output = output[:, -1, 0]
            elif output.dim() == 2:
                output = output[:, -1]

            probs = torch.sigmoid(output).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch_y.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5

        return {
            "auc": auc,
            "pred_mean": float(all_probs.mean()),
            "pred_std": float(all_probs.std()),
            "pred_min": float(all_probs.min()),
            "pred_max": float(all_probs.max()),
            "pos_mean": float(all_probs[all_labels == 1].mean()) if (all_labels == 1).sum() > 0 else 0,
            "neg_mean": float(all_probs[all_labels == 0].mean()) if (all_labels == 0).sum() > 0 else 0,
        }

    def train(self, train_loader, val_loader, epochs: int = None):
        epochs = epochs or self.total_epochs

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_stats = self.evaluate(val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "val_auc": val_stats["auc"],
                "val_pred_std": val_stats["pred_std"],
                "val_class_sep": val_stats["pos_mean"] - val_stats["neg_mean"],
            }
            self.history.append(record)

            if epoch % 10 == 0 or epoch < 5:
                print(
                    f"  Epoch {epoch:3d} | Loss {train_loss:.4f} | "
                    f"Val AUC {val_stats['auc']:.4f} | "
                    f"Pred [{val_stats['pred_min']:.3f}-{val_stats['pred_max']:.3f}] "
                    f"Std {val_stats['pred_std']:.4f}"
                )

        return self.history


def load_data():
    """Load and prepare data."""
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data/processed/v1/SPY_dataset_c.parquet"
    df = pd.read_parquet(data_path)

    df["target"] = create_target(df, threshold=0.01)
    df = df.dropna(subset=["target"])

    exclude = ["Date", "Open", "High", "Low", "Close", "Volume", "target"]
    feature_cols = [c for c in df.columns if c not in exclude]

    features = df[feature_cols].values.astype(np.float32)
    labels = df["target"].values.astype(np.float32)

    # Z-score normalize
    train_end_idx = (df["Date"] < "2023-01-01").sum()
    train_mean = features[:train_end_idx].mean(axis=0)
    train_std = features[:train_end_idx].std(axis=0)
    train_std[train_std == 0] = 1
    features = (features - train_mean) / train_std

    # Split
    val_start_idx = train_end_idx
    test_start_idx = (df["Date"] < "2025-01-01").sum()

    return {
        "train_features": features[:val_start_idx],
        "train_labels": labels[:val_start_idx],
        "val_features": features[val_start_idx:test_start_idx],
        "val_labels": labels[val_start_idx:test_start_idx],
        "n_features": len(feature_cols),
    }


def create_model(n_features: int):
    """Create a fresh model."""
    config = PatchTSTConfig(
        num_features=n_features,
        context_length=60,
        patch_length=8,
        stride=4,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        head_dropout=0.1,
    )
    return PatchTST(config, use_revin=True)


def run_loss_experiment(loss_name: str, criterion: nn.Module, data: dict, apply_sigmoid: bool = False):
    """Run experiment with a specific loss function."""
    print(f"\n{'='*60}")
    print(f"Loss: {loss_name}")
    print(f"{'='*60}")

    context_length = 60
    train_dataset = SimpleDataset(data["train_features"], data["train_labels"], context_length)
    val_dataset = SimpleDataset(data["val_features"], data["val_labels"], context_length)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = create_model(data["n_features"])

    trainer = LossExperimentTrainer(
        model=model,
        criterion=criterion,
        criterion_name=loss_name,
        learning_rate=3e-6,  # Very low LR from diagnosis
        warmup_epochs=10,
        total_epochs=50,
        apply_sigmoid=apply_sigmoid,
    )

    history = trainer.train(train_loader, val_loader)

    best_auc = max(h["val_auc"] for h in history)
    final_auc = history[-1]["val_auc"]
    final_std = history[-1]["val_pred_std"]

    return {
        "loss_name": loss_name,
        "best_auc": best_auc,
        "final_auc": final_auc,
        "final_pred_std": final_std,
        "best_epoch": max(range(len(history)), key=lambda i: history[i]["val_auc"]),
        "history": history,
    }


def main():
    print("Loading data...")
    data = load_data()

    print(f"Train samples: {len(data['train_features']) - 60}")
    print(f"Val samples: {len(data['val_features']) - 60}")
    print(f"Train positive rate: {data['train_labels'][60:].mean():.1%}")
    print(f"Val positive rate: {data['val_labels'][60:].mean():.1%}")

    # Compute pos_weight for weighted BCE
    pos_rate = data["train_labels"][60:].mean()
    pos_weight = (1 - pos_rate) / pos_rate
    print(f"Computed pos_weight: {pos_weight:.2f}")

    # Loss functions to test
    losses = [
        ("BCE", nn.BCEWithLogitsLoss(), False),
        ("BCE_weighted_2x", nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0)), False),
        ("BCE_weighted_5x", nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0)), False),
        ("BCE_weighted_auto", nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)), False),
        ("FocalLoss_g2", FocalLoss(gamma=2.0, alpha=0.25), False),
        ("FocalLoss_g3", FocalLoss(gamma=3.0, alpha=0.25), False),
        ("FocalLoss_g5", FocalLoss(gamma=5.0, alpha=0.25), False),
        ("SoftAUC_g1", SoftAUCLoss(gamma=1.0), True),
        ("SoftAUC_g2", SoftAUCLoss(gamma=2.0), True),
        ("SoftAUC_g5", SoftAUCLoss(gamma=5.0), True),
    ]

    results = []
    for name, criterion, apply_sigmoid in losses:
        result = run_loss_experiment(name, criterion, data, apply_sigmoid)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Loss':<20} {'Best AUC':<10} {'Final AUC':<10} {'Pred Std':<10}")
    print("-" * 50)

    for r in sorted(results, key=lambda x: x["best_auc"], reverse=True):
        print(
            f"{r['loss_name']:<20} "
            f"{r['best_auc']:<10.4f} "
            f"{r['final_auc']:<10.4f} "
            f"{r['final_pred_std']:<10.4f}"
        )

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs/loss_function_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = [{k: v for k, v in r.items() if k != "history"} for r in results]
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    best = max(results, key=lambda x: x["best_auc"])
    print(f"\n🏆 Best loss: {best['loss_name']} with AUC {best['best_auc']:.4f}")

    # Compare to tree models
    print(f"\n📊 Comparison to XGBoost:")
    print(f"   XGBoost 1.0% threshold: AUC 0.7555")
    print(f"   Best PatchTST ({best['loss_name']}): AUC {best['best_auc']:.4f}")
    gap = 0.7555 - best["best_auc"]
    print(f"   Gap: {gap:.4f} ({gap/0.7555*100:.1f}% relative)")


if __name__ == "__main__":
    main()
