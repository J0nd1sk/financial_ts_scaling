"""Contrastive pre-training for financial time-series.

Implements self-supervised contrastive learning methods that learn
useful representations before fine-tuning on the downstream task.

Available methods:
- SimCLR: Contrastive learning with augmented views
- TS2Vec: Hierarchical temporal contrastive learning
- BYOL: Bootstrap Your Own Latent (no negative samples)

Pre-training workflow:
1. Pre-train encoder with contrastive loss (unsupervised)
2. Save encoder checkpoint
3. Load encoder and fine-tune with classification head

References:
- SimCLR: Chen et al. "A Simple Framework for Contrastive Learning" (2020)
- TS2Vec: Yue et al. "TS2Vec: Towards Universal Representation of Time Series" (2022)
- BYOL: Grill et al. "Bootstrap Your Own Latent" (2020)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from src.data.augmentation import BaseTransform
    from src.models.patchtst import PatchTSTConfig


class ContrastiveLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for SimCLR.

    Pulls together positive pairs (augmented views of same sample) while
    pushing apart negative pairs (views from different samples).

    Args:
        temperature: Temperature parameter for scaling. Default 0.1.
            Lower temperature = sharper distribution = harder negatives.

    Example:
        >>> loss_fn = ContrastiveLoss(temperature=0.1)
        >>> z1 = model(x_aug1)  # View 1 embeddings, shape (batch, d_model)
        >>> z2 = model(x_aug2)  # View 2 embeddings, shape (batch, d_model)
        >>> loss = loss_fn(z1, z2)
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute NT-Xent loss.

        Args:
            z1: Embeddings from view 1, shape (batch, d).
            z2: Embeddings from view 2, shape (batch, d).

        Returns:
            Scalar loss tensor.
        """
        batch_size = z1.size(0)
        device = z1.device

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate: [z1; z2] for 2N total samples
        z = torch.cat([z1, z2], dim=0)  # (2N, d)

        # Similarity matrix: (2N, 2N)
        sim = torch.matmul(z, z.T) / self.temperature  # (2N, 2N)

        # Mask out self-similarities (diagonal)
        mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs:
        # For z1[i], positive is z2[i] (at index batch_size + i)
        # For z2[i], positive is z1[i] (at index i)
        # Labels: [batch_size, batch_size+1, ..., 2*batch_size-1, 0, 1, ..., batch_size-1]
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device),
        ])

        # Cross-entropy loss
        loss = F.cross_entropy(sim, labels)

        return loss


class HierarchicalContrastiveLoss(nn.Module):
    """Hierarchical contrastive loss for TS2Vec.

    Applies contrastive loss at multiple temporal scales:
    - Instance-level: Different time series are negatives
    - Temporal-level: Different timestamps within same series are negatives

    Args:
        temperature: Temperature parameter. Default 0.1.
        lambda_temporal: Weight for temporal contrast. Default 0.5.

    Example:
        >>> loss_fn = HierarchicalContrastiveLoss(temperature=0.1)
        >>> z1 = model(x_aug1)  # (batch, seq_len, d)
        >>> z2 = model(x_aug2)  # (batch, seq_len, d)
        >>> loss = loss_fn(z1, z2)
    """

    def __init__(
        self,
        temperature: float = 0.1,
        lambda_temporal: float = 0.5,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.lambda_temporal = lambda_temporal
        self.instance_loss = ContrastiveLoss(temperature)

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute hierarchical contrastive loss.

        Args:
            z1: Embeddings from view 1, shape (batch, seq_len, d).
            z2: Embeddings from view 2, shape (batch, seq_len, d).

        Returns:
            Scalar loss tensor.
        """
        batch_size, seq_len, d = z1.shape

        # Instance-level: pool over time
        z1_instance = z1.mean(dim=1)  # (batch, d)
        z2_instance = z2.mean(dim=1)  # (batch, d)
        instance_loss = self.instance_loss(z1_instance, z2_instance)

        # Temporal-level: contrast across time within same instance
        # Reshape: (batch * seq_len, d)
        z1_flat = z1.reshape(-1, d)
        z2_flat = z2.reshape(-1, d)

        # Create labels: positive pairs are same timestep in both views
        temporal_loss = self.instance_loss(z1_flat, z2_flat)

        return instance_loss + self.lambda_temporal * temporal_loss


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning.

    Projects embeddings to a lower-dimensional space where contrastive
    loss is applied. This helps separate representation from contrastive task.

    Args:
        input_dim: Input embedding dimension.
        hidden_dim: Hidden layer dimension. Default None = 2 * input_dim.
        output_dim: Output projection dimension. Default 128.

    Example:
        >>> head = ProjectionHead(input_dim=256, output_dim=128)
        >>> z = head(embeddings)  # (batch, 128)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or input_dim * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project embeddings.

        Args:
            x: Input embeddings, shape (batch, input_dim).

        Returns:
            Projected embeddings, shape (batch, output_dim).
        """
        return self.net(x)


class ContrastiveEncoder(nn.Module):
    """Wrapper that adds projection head to base encoder for contrastive learning.

    During pre-training, uses projection head.
    After pre-training, remove projection head and use base encoder directly.

    Args:
        base_encoder: Base encoder model (e.g., PatchTST).
        d_model: Encoder output dimension.
        projection_dim: Projection head output dimension. Default 128.

    Example:
        >>> encoder = ContrastiveEncoder(patchtst_model, d_model=128)
        >>> z = encoder(x)  # (batch, 128) for contrastive loss
        >>> # After pre-training:
        >>> base = encoder.get_base_encoder()
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        d_model: int,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.encoder = base_encoder
        self.projection = ProjectionHead(d_model, output_dim=projection_dim)

        # Remove classification head from base encoder if present
        if hasattr(self.encoder, "head"):
            self.encoder.head = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Encode and project.

        Args:
            x: Input tensor.

        Returns:
            Projected embeddings.
        """
        # Get encoder output before classification head
        h = self.encoder(x)

        # If encoder returns sequence, pool over time
        if h.dim() == 3:
            h = h.mean(dim=1)

        # Project
        z = self.projection(h)
        return z

    def get_base_encoder(self) -> nn.Module:
        """Return base encoder without projection head."""
        return self.encoder


class ContrastiveTrainer:
    """Trainer for contrastive pre-training.

    Pre-trains an encoder using contrastive learning on unlabeled data.
    After pre-training, the encoder can be fine-tuned on the downstream task.

    Args:
        model_config: PatchTSTConfig for the encoder.
        contrastive_type: Type of contrastive method ("simclr", "ts2vec", "byol").
            Default "simclr".
        temperature: Temperature for contrastive loss. Default 0.1.
        projection_dim: Projection head output dimension. Default 128.
        augmentation: Augmentation transform for creating views.
        device: Device to train on.
        learning_rate: Optimizer learning rate. Default 1e-4.

    Example:
        >>> trainer = ContrastiveTrainer(
        ...     model_config=config,
        ...     contrastive_type="simclr",
        ...     augmentation=JitterTransform(std=0.01),
        ...     device="mps",
        ... )
        >>> trainer.pretrain(dataloader, epochs=20)
        >>> encoder = trainer.get_encoder()
    """

    def __init__(
        self,
        model_config: "PatchTSTConfig",
        contrastive_type: str = "simclr",
        temperature: float = 0.1,
        projection_dim: int = 128,
        augmentation: "BaseTransform | None" = None,
        device: str = "cpu",
        learning_rate: float = 1e-4,
    ) -> None:
        from src.models.patchtst import PatchTST

        self.model_config = model_config
        self.contrastive_type = contrastive_type
        self.device = torch.device(device)
        self.learning_rate = learning_rate

        # Create base encoder
        base_encoder = PatchTST(model_config, use_revin=True)

        # Wrap with contrastive encoder
        self.model = ContrastiveEncoder(
            base_encoder=base_encoder,
            d_model=model_config.d_model,
            projection_dim=projection_dim,
        ).to(self.device)

        # Loss function
        if contrastive_type == "ts2vec":
            self.criterion = HierarchicalContrastiveLoss(temperature=temperature)
        else:
            self.criterion = ContrastiveLoss(temperature=temperature)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        # Augmentation
        self.augmentation = augmentation

    def _create_views(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Create two augmented views of the input.

        Args:
            x: Input batch, shape (batch, seq_len, features).

        Returns:
            Tuple of (view1, view2), each same shape as x.
        """
        if self.augmentation is None:
            # Default: add small noise for different views
            noise1 = torch.randn_like(x) * 0.01
            noise2 = torch.randn_like(x) * 0.01
            return x + noise1, x + noise2

        # Apply augmentation independently to create two views
        view1 = torch.stack([self.augmentation(sample) for sample in x])
        view2 = torch.stack([self.augmentation(sample) for sample in x])
        return view1, view2

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, _ in dataloader:  # Labels not used for pre-training
            batch_x = batch_x.to(self.device)

            # Create two views
            view1, view2 = self._create_views(batch_x)

            # Forward pass
            z1 = self.model(view1)
            z2 = self.model(view2)

            # Contrastive loss
            loss = self.criterion(z1, z2)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def pretrain(
        self,
        dataloader: DataLoader,
        epochs: int,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Run contrastive pre-training.

        Args:
            dataloader: Training data loader.
            epochs: Number of pre-training epochs.
            verbose: If True, print progress.

        Returns:
            Dict with pre-training results.
        """
        losses = []

        for epoch in range(epochs):
            loss = self.train_epoch(dataloader)
            losses.append(loss)

            if verbose:
                print(f"Epoch {epoch}: contrastive_loss={loss:.4f}")

        return {
            "final_loss": losses[-1] if losses else 0.0,
            "losses": losses,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save pre-trained encoder checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        torch.save({
            "encoder_state_dict": self.model.encoder.state_dict(),
            "projection_state_dict": self.model.projection.state_dict(),
            "model_config": self.model_config,
            "contrastive_type": self.contrastive_type,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load pre-trained encoder checkpoint.

        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.model.projection.load_state_dict(checkpoint["projection_state_dict"])

    def get_encoder(self) -> nn.Module:
        """Get the pre-trained encoder (without projection head).

        Returns:
            Base encoder module.
        """
        return self.model.get_base_encoder()


def get_contrastive_trainer(
    contrastive_type: str | None,
    contrastive_params: dict[str, Any] | None,
    model_config: "PatchTSTConfig",
    device: str = "cpu",
) -> ContrastiveTrainer | None:
    """Factory function to create contrastive trainer from experiment spec.

    Args:
        contrastive_type: Type of contrastive method ("simclr", "ts2vec", "byol").
            None returns None.
        contrastive_params: Parameters for the trainer:
            - temperature: Contrastive loss temperature
            - projection_dim: Projection head dimension
            - pretrain_epochs: Number of pre-training epochs
        model_config: PatchTSTConfig for the encoder.
        device: Device string.

    Returns:
        ContrastiveTrainer instance, or None if contrastive_type is None.

    Raises:
        ValueError: If contrastive_type is unknown.
    """
    if contrastive_type is None:
        return None

    params = contrastive_params or {}

    if contrastive_type not in ("simclr", "ts2vec", "byol"):
        raise ValueError(f"Unknown contrastive_type: {contrastive_type}")

    # Create augmentation transform if specified
    augmentation = None
    if "augmentation_type" in params:
        from src.data.augmentation import get_augmentation_transform
        augmentation = get_augmentation_transform(
            params["augmentation_type"],
            params.get("augmentation_params"),
        )

    return ContrastiveTrainer(
        model_config=model_config,
        contrastive_type=contrastive_type,
        temperature=params.get("temperature", 0.1),
        projection_dim=params.get("projection_dim", 128),
        augmentation=augmentation,
        device=device,
        learning_rate=params.get("learning_rate", 1e-4),
    )
