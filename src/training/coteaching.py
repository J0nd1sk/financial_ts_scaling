"""Co-teaching trainer for noise-robust learning.

Implements the Co-teaching algorithm for training with noisy labels using
two neural networks that teach each other by selecting small-loss samples.

Reference: Han et al. "Co-teaching: Robust Training of Deep Neural Networks
           with Extremely Noisy Labels" (2018)

Key idea:
    - Train two networks simultaneously
    - Each network selects small-loss samples for its peer
    - Networks have different learning dynamics, so they disagree on
      different noisy samples
    - Cross-training helps both networks avoid memorizing noisy labels
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from src.models.patchtst import PatchTSTConfig


class CoTeachingTrainer:
    """Trainer using Co-teaching for noise-robust learning.

    Trains two networks simultaneously, with each network selecting
    small-loss samples for its peer to learn from. This helps both
    networks avoid memorizing noisy labels.

    Args:
        model_config: PatchTSTConfig for creating the two networks.
        forget_rate: Fraction of samples to discard each batch. Default 0.2.
            Higher values are more aggressive at filtering noise.
            Typical: 0.1 (low noise), 0.2 (moderate), 0.3 (high noise)
        exponent: Rate schedule exponent. Default 1.0.
            Controls how forget_rate increases during training.
        num_epochs: Total number of training epochs. Used for rate scheduling.
        device: Device to train on ("cpu", "mps", "cuda").

    Example:
        >>> trainer = CoTeachingTrainer(
        ...     model_config=config,
        ...     forget_rate=0.2,
        ...     num_epochs=50,
        ...     device="mps",
        ... )
        >>> result = trainer.train(train_loader, val_loader, epochs=50)
    """

    def __init__(
        self,
        model_config: "PatchTSTConfig",
        forget_rate: float = 0.2,
        exponent: float = 1.0,
        num_epochs: int = 50,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        use_revin: bool = True,
    ) -> None:
        from src.models.patchtst import PatchTST

        self.model_config = model_config
        self.forget_rate = forget_rate
        self.exponent = exponent
        self.num_epochs = num_epochs
        self.device = torch.device(device)
        self.learning_rate = learning_rate

        # Create two networks with different random initializations
        torch.manual_seed(42)
        self.model1 = PatchTST(model_config, use_revin=use_revin).to(self.device)

        torch.manual_seed(43)  # Different seed for different init
        self.model2 = PatchTST(model_config, use_revin=use_revin).to(self.device)

        # Separate optimizers
        self.optimizer1 = torch.optim.Adam(
            self.model1.parameters(), lr=learning_rate
        )
        self.optimizer2 = torch.optim.Adam(
            self.model2.parameters(), lr=learning_rate
        )

        # Loss function (per-sample, not reduced)
        self.criterion = nn.BCELoss(reduction="none")

    def _get_forget_rate(self, epoch: int) -> float:
        """Compute forget rate for current epoch with schedule.

        The forget rate increases during training following:
            rate = min(forget_rate, forget_rate * (epoch / num_epochs) ^ exponent)

        This allows the model to learn from all data early on, then
        progressively filter more aggressively as it becomes confident.

        Args:
            epoch: Current epoch number (0-indexed).

        Returns:
            Forget rate for this epoch.
        """
        # Linear warmup to forget_rate over first 10% of epochs
        warmup_epochs = max(1, self.num_epochs // 10)
        if epoch < warmup_epochs:
            return self.forget_rate * epoch / warmup_epochs

        # After warmup, use full forget_rate
        return self.forget_rate

    def _select_samples(
        self,
        losses: torch.Tensor,
        forget_rate: float,
    ) -> torch.Tensor:
        """Select samples with smallest losses.

        Returns a boolean mask indicating which samples to keep.

        Args:
            losses: Per-sample losses, shape (batch_size,).
            forget_rate: Fraction of samples to discard.

        Returns:
            Boolean mask of shape (batch_size,), True for samples to keep.
        """
        batch_size = losses.size(0)
        num_keep = max(1, int(batch_size * (1 - forget_rate)))

        # Get indices of samples with smallest losses
        _, indices = torch.topk(losses, num_keep, largest=False)

        # Create mask
        mask = torch.zeros(batch_size, dtype=torch.bool, device=losses.device)
        mask[indices] = True

        return mask

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> tuple[float, float]:
        """Train both networks for one epoch using co-teaching.

        Args:
            dataloader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Tuple of (loss1, loss2) average losses for each network.
        """
        self.model1.train()
        self.model2.train()

        forget_rate = self._get_forget_rate(epoch)

        total_loss1 = 0.0
        total_loss2 = 0.0
        num_batches = 0

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass through both networks
            pred1 = self.model1(batch_x)
            pred2 = self.model2(batch_x)

            # Compute per-sample losses
            loss1_per_sample = self.criterion(pred1, batch_y).view(-1)
            loss2_per_sample = self.criterion(pred2, batch_y).view(-1)

            # Each network selects samples for its peer
            # Model 1 selects samples with small loss for model 2 to learn
            mask1_for_2 = self._select_samples(loss1_per_sample.detach(), forget_rate)
            # Model 2 selects samples with small loss for model 1 to learn
            mask2_for_1 = self._select_samples(loss2_per_sample.detach(), forget_rate)

            # Compute losses on selected samples
            # Model 1 learns from samples selected by model 2
            if mask2_for_1.sum() > 0:
                loss1 = loss1_per_sample[mask2_for_1].mean()
            else:
                loss1 = loss1_per_sample.mean()

            # Model 2 learns from samples selected by model 1
            if mask1_for_2.sum() > 0:
                loss2 = loss2_per_sample[mask1_for_2].mean()
            else:
                loss2 = loss2_per_sample.mean()

            # Backward and optimize
            self.optimizer1.zero_grad()
            loss1.backward()
            self.optimizer1.step()

            self.optimizer2.zero_grad()
            loss2.backward()
            self.optimizer2.step()

            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            num_batches += 1

        return total_loss1 / max(num_batches, 1), total_loss2 / max(num_batches, 1)

    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate both models on validation set.

        Returns metrics for the ensemble (average of both models' predictions).

        Args:
            dataloader: Validation data loader.

        Returns:
            Dict with loss, accuracy, auc for the ensemble.
        """
        from sklearn.metrics import roc_auc_score

        self.model1.eval()
        self.model2.eval()

        all_preds = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Ensemble prediction (average of both models)
                pred1 = self.model1(batch_x)
                pred2 = self.model2(batch_x)
                pred_ensemble = (pred1 + pred2) / 2

                # Loss on ensemble prediction
                loss = nn.functional.binary_cross_entropy(pred_ensemble, batch_y)
                total_loss += loss.item()
                num_batches += 1

                all_preds.append(pred_ensemble.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        # Compute metrics
        preds = np.concatenate(all_preds).flatten()
        targets = np.concatenate(all_targets).flatten()

        pred_binary = (preds >= 0.5).astype(int)
        target_binary = targets.astype(int)

        accuracy = (pred_binary == target_binary).mean()

        # AUC (requires both classes)
        auc = None
        if len(np.unique(target_binary)) == 2:
            try:
                auc = float(roc_auc_score(target_binary, preds))
            except ValueError:
                pass

        return {
            "loss": total_loss / max(num_batches, 1),
            "accuracy": float(accuracy),
            "auc": auc,
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Train both networks using co-teaching.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of training epochs.
            verbose: If True, print progress.

        Returns:
            Dict with training results including final metrics for both models
            and the ensemble.
        """
        self.num_epochs = epochs  # Update for rate scheduling

        best_val_loss = float("inf")
        best_epoch = 0
        learning_curve = []

        for epoch in range(epochs):
            # Train one epoch
            train_loss1, train_loss2 = self.train_epoch(train_loader, epoch)

            # Evaluate
            val_metrics = self.evaluate(val_loader)

            if verbose:
                print(
                    f"Epoch {epoch}: "
                    f"train_loss1={train_loss1:.4f}, "
                    f"train_loss2={train_loss2:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_auc={val_metrics.get('auc', 'N/A')}"
                )

            learning_curve.append({
                "epoch": epoch,
                "train_loss1": train_loss1,
                "train_loss2": train_loss2,
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics.get("auc"),
            })

            # Track best
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch

        return {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "final_val_metrics": val_metrics,
            "learning_curve": learning_curve,
        }

    def get_ensemble_model(self) -> nn.Module:
        """Get an ensemble model that averages predictions from both networks.

        Returns:
            EnsembleModel that wraps both trained networks.
        """
        return EnsembleModel(self.model1, self.model2)


class EnsembleModel(nn.Module):
    """Wrapper that ensembles two models by averaging predictions."""

    def __init__(self, model1: nn.Module, model2: nn.Module) -> None:
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass averaging both models' predictions.

        Args:
            x: Input tensor.

        Returns:
            Averaged predictions.
        """
        pred1 = self.model1(x)
        pred2 = self.model2(x)
        return (pred1 + pred2) / 2
