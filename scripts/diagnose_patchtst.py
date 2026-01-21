#!/usr/bin/env python3
"""
Diagnose PatchTST convergence issues.

Key questions:
1. Does the model start with diverse predictions and collapse?
2. Or does it never learn to spread predictions?
3. What do the learning curves look like?
4. Can lower LR / longer warmup / scheduling help?
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

        # Valid indices (where we have full context + label)
        self.valid_indices = list(range(context_length, len(features) - 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        # Get context window
        x = self.features[i - self.context_length : i]
        y = self.labels[i]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class DiagnosticTrainer:
    """Trainer that logs detailed diagnostics at each epoch."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        warmup_epochs: int = 5,
        use_cosine: bool = False,
        total_epochs: int = 50,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.use_cosine = use_cosine
        self.total_epochs = total_epochs
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.criterion = nn.BCEWithLogitsLoss()

        # Learning rate scheduler
        if use_cosine:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=10, T_mult=2
            )
        else:
            self.scheduler = None

        self.history = []

    def get_lr(self, epoch: int) -> float:
        """Get learning rate with warmup."""
        if epoch < self.warmup_epochs:
            return self.learning_rate * (epoch + 1) / self.warmup_epochs
        return self.learning_rate

    def train_epoch(self, dataloader, epoch: int) -> float:
        """Train one epoch and return loss."""
        self.model.train()

        # Apply warmup
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

            # Handle output shape
            if output.dim() == 3:
                output = output[:, -1, 0]
            elif output.dim() == 2:
                output = output[:, -1]

            loss = self.criterion(output, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if self.scheduler and epoch >= self.warmup_epochs:
            self.scheduler.step()

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        """Evaluate and return detailed stats."""
        self.model.eval()

        all_probs = []
        all_labels = []
        total_loss = 0
        n_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            output = self.model(batch_x)

            if output.dim() == 3:
                output = output[:, -1, 0]
            elif output.dim() == 2:
                output = output[:, -1]

            loss = self.criterion(output, batch_y)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(output).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch_y.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Calculate AUC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5

        return {
            "loss": total_loss / n_batches,
            "auc": auc,
            "pred_mean": float(all_probs.mean()),
            "pred_std": float(all_probs.std()),
            "pred_min": float(all_probs.min()),
            "pred_max": float(all_probs.max()),
            "pos_mean": float(all_probs[all_labels == 1].mean()) if (all_labels == 1).sum() > 0 else 0,
            "neg_mean": float(all_probs[all_labels == 0].mean()) if (all_labels == 0).sum() > 0 else 0,
        }

    def train(self, train_loader, val_loader, epochs: int = None):
        """Train and collect diagnostics."""
        epochs = epochs or self.total_epochs

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_stats = self.evaluate(val_loader)
            train_stats = self.evaluate(train_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]

            record = {
                "epoch": epoch,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_auc": train_stats["auc"],
                "val_loss": val_stats["loss"],
                "val_auc": val_stats["auc"],
                "val_pred_mean": val_stats["pred_mean"],
                "val_pred_std": val_stats["pred_std"],
                "val_pred_range": val_stats["pred_max"] - val_stats["pred_min"],
                "val_pos_mean": val_stats["pos_mean"],
                "val_neg_mean": val_stats["neg_mean"],
                "val_class_sep": val_stats["pos_mean"] - val_stats["neg_mean"],
            }
            self.history.append(record)

            if epoch % 5 == 0 or epoch < 5:
                print(
                    f"Epoch {epoch:3d} | LR {current_lr:.2e} | "
                    f"Train Loss {train_loss:.4f} | Val AUC {val_stats['auc']:.4f} | "
                    f"Pred [{val_stats['pred_min']:.3f}-{val_stats['pred_max']:.3f}] "
                    f"Std {val_stats['pred_std']:.4f} | "
                    f"Sep {val_stats['pos_mean'] - val_stats['neg_mean']:.4f}"
                )

        return self.history


def run_experiment(config_name: str, learning_rate: float, warmup: int, use_cosine: bool,
                   epochs: int = 50, context_length: int = 60, use_revin: bool = True):
    """Run a single diagnostic experiment."""
    print(f"\n{'='*70}")
    print(f"Config: {config_name}")
    print(f"LR={learning_rate}, Warmup={warmup}, Cosine={use_cosine}, RevIN={use_revin}")
    print(f"{'='*70}")

    # Load data
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data/processed/v1/SPY_dataset_c.parquet"
    df = pd.read_parquet(data_path)

    # Create target
    df["target"] = create_target(df, threshold=0.01)
    df = df.dropna(subset=["target"])

    # Get feature columns
    exclude = ["Date", "Open", "High", "Low", "Close", "Volume", "target"]
    feature_cols = [c for c in df.columns if c not in exclude]

    # Convert to numpy
    features = df[feature_cols].values.astype(np.float32)
    labels = df["target"].values.astype(np.float32)

    # Manual Z-score normalization using train stats
    train_end_idx = (df["Date"] < "2023-01-01").sum()
    train_mean = features[:train_end_idx].mean(axis=0)
    train_std = features[:train_end_idx].std(axis=0)
    train_std[train_std == 0] = 1  # Avoid division by zero
    features = (features - train_mean) / train_std

    # Split by date
    val_start_idx = (df["Date"] < "2023-01-01").sum()
    test_start_idx = (df["Date"] < "2025-01-01").sum()

    train_features = features[:val_start_idx]
    train_labels = labels[:val_start_idx]
    val_features = features[val_start_idx:test_start_idx]
    val_labels = labels[val_start_idx:test_start_idx]

    # Create datasets
    train_dataset = SimpleDataset(train_features, train_labels, context_length)
    val_dataset = SimpleDataset(val_features, val_labels, context_length)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Train positive rate: {train_labels[context_length:].mean():.1%}")
    print(f"Val positive rate: {val_labels[context_length:].mean():.1%}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Create model - small 2M config
    model_config = PatchTSTConfig(
        num_features=len(feature_cols),
        context_length=context_length,
        patch_length=8,
        stride=4,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        head_dropout=0.1,
    )
    model = PatchTST(model_config, use_revin=use_revin)

    # Create trainer
    trainer = DiagnosticTrainer(
        model=model,
        learning_rate=learning_rate,
        warmup_epochs=warmup,
        use_cosine=use_cosine,
        total_epochs=epochs,
    )

    # Train
    history = trainer.train(train_loader, val_loader, epochs=epochs)

    return {
        "config_name": config_name,
        "learning_rate": learning_rate,
        "warmup": warmup,
        "use_cosine": use_cosine,
        "use_revin": use_revin,
        "final_val_auc": history[-1]["val_auc"],
        "final_pred_std": history[-1]["val_pred_std"],
        "final_class_sep": history[-1]["val_class_sep"],
        "best_val_auc": max(h["val_auc"] for h in history),
        "best_epoch": max(range(len(history)), key=lambda i: history[i]["val_auc"]),
        "history": history,
    }


def main():
    # Experiments to run - testing learning rate and schedule variations
    experiments = [
        # Baseline (typical config)
        ("baseline", 1e-4, 5, False, True),
        # Lower learning rate - slower convergence
        ("low_lr", 1e-5, 5, False, True),
        ("very_low_lr", 3e-6, 10, False, True),
        # With cosine annealing - helps escape local minima
        ("cosine", 1e-4, 5, True, True),
        ("low_lr_cosine", 1e-5, 5, True, True),
        # Longer warmup - gentler start
        ("long_warmup", 1e-4, 20, False, True),
        # Very slow start - maximum gentleness
        ("slow_start", 5e-6, 30, True, True),
        # No RevIN (for comparison)
        ("no_revin", 1e-4, 5, False, False),
    ]

    results = []
    for name, lr, warmup, cosine, use_revin in experiments:
        result = run_experiment(name, lr, warmup, cosine, epochs=50, use_revin=use_revin)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<15} {'Best AUC':<10} {'Final AUC':<10} {'Pred Std':<10} {'Class Sep':<10}")
    print("-" * 55)

    for r in results:
        print(
            f"{r['config_name']:<15} "
            f"{r['best_val_auc']:<10.4f} "
            f"{r['final_val_auc']:<10.4f} "
            f"{r['final_pred_std']:<10.4f} "
            f"{r['final_class_sep']:<10.4f}"
        )

    # Save results
    output_dir = Path(__file__).parent.parent / "outputs/patchtst_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary = [{k: v for k, v in r.items() if k != "history"} for r in results]
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save detailed history
    for r in results:
        hist_df = pd.DataFrame(r["history"])
        hist_df.to_csv(output_dir / f"{r['config_name']}_history.csv", index=False)

    print(f"\nResults saved to {output_dir}")

    # Key insight
    best = max(results, key=lambda x: x["best_val_auc"])
    print(f"\n🏆 Best config: {best['config_name']} with AUC {best['best_val_auc']:.4f}")

    # Compare to tree models
    print(f"\n📊 Comparison to tree models:")
    print(f"   XGBoost 1.0% threshold: AUC 0.7555")
    print(f"   Best PatchTST config:   AUC {best['best_val_auc']:.4f}")
    gap = 0.7555 - best['best_val_auc']
    print(f"   Gap: {gap:.4f} ({gap/0.7555*100:.1f}% relative)")


if __name__ == "__main__":
    main()
