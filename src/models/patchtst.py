"""PatchTST model implementation from scratch using pure PyTorch.

Architecture:
1. PatchEmbedding: Split time series into patches, project to d_model
2. PositionalEncoding: Learnable position embeddings
3. TransformerEncoder: Stack of encoder layers (self-attention + FFN)
4. PredictionHead: Linear projection for binary classification output

Reference: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
https://arxiv.org/abs/2211.14730
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization for non-stationary time series.

    Normalizes input per-instance (per sample in batch) using instance statistics,
    then can denormalize output using the same statistics. This handles distribution
    shift elegantly by adapting to each input's scale.

    Reference: https://openreview.net/forum?id=cGDAkQo1C0p

    Args:
        num_features: Number of features in the input.
        eps: Small constant for numerical stability.
        affine: If True, adds learnable affine parameters (gamma, beta).
    """

    def __init__(
        self, num_features: int, eps: float = 1e-5, affine: bool = True
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            # Learnable affine parameters
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

        # Store statistics for denormalization (set during normalize)
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using per-instance statistics.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            Normalized tensor of same shape.
        """
        # Compute mean and std over sequence dimension (dim=1)
        self.mean = x.mean(dim=1, keepdim=True)  # (batch, 1, features)
        self.std = x.std(dim=1, keepdim=True) + self.eps  # (batch, 1, features)

        # Normalize
        x_norm = (x - self.mean) / self.std

        # Apply affine transformation if enabled
        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output using stored statistics from normalize().

        Args:
            x: Normalized tensor of shape (batch, seq_len, num_features)

        Returns:
            Denormalized tensor of same shape.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call normalize() before denormalize()")

        # Reverse affine transformation if enabled
        if self.affine:
            x = (x - self.beta) / self.gamma

        # Denormalize
        x_denorm = x * self.std + self.mean

        return x_denorm


@dataclass
class PatchTSTConfig:
    """Configuration for PatchTST model."""

    num_features: int  # Number of input features (e.g., 20)
    context_length: int  # Input sequence length (e.g., 60)
    patch_length: int  # Length of each patch (e.g., 16)
    stride: int  # Stride between patches (e.g., 8)
    d_model: int  # Model dimension (e.g., 128)
    n_heads: int  # Number of attention heads (e.g., 8)
    n_layers: int  # Number of transformer layers (e.g., 3)
    d_ff: int  # Feedforward dimension (e.g., 256)
    dropout: float  # Dropout rate (e.g., 0.1)
    head_dropout: float  # Dropout for prediction head (e.g., 0.0)
    num_classes: int = 1  # Output classes (1 for binary sigmoid)
    d_embed: int | None = None  # Feature embedding dimension. None = disabled.

    @property
    def num_patches(self) -> int:
        """Calculate number of patches from context length and patch params."""
        return (self.context_length - self.patch_length) // self.stride + 1

    @property
    def effective_features(self) -> int:
        """Feature dim for PatchEmbedding (d_embed if set, else num_features)."""
        return self.d_embed if self.d_embed is not None else self.num_features


class FeatureEmbedding(nn.Module):
    """Project raw features into a learned embedding space.

    This layer transforms the feature dimension, enabling principled scaling
    of model capacity with feature count. When d_embed < num_features, it
    compresses features before patching. When d_embed > num_features, it
    expands the representation.

    Args:
        num_features: Number of input features.
        d_embed: Output embedding dimension.
        dropout: Dropout rate (default: 0.0).
    """

    def __init__(
        self, num_features: int, d_embed: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(num_features, d_embed)
        self.norm = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to embedding space.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            Embedded tensor of shape (batch, seq_len, d_embed)
        """
        x = self.projection(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class PatchEmbedding(nn.Module):
    """Convert time series into patch embeddings.

    Takes input of shape (batch, seq_len, num_features) and produces
    patch embeddings of shape (batch, num_patches, d_model).

    Note: When feature embedding is used, num_features here refers to
    the effective feature dimension (d_embed), not the original feature count.
    """

    def __init__(self, config: PatchTSTConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_length = config.patch_length
        self.stride = config.stride
        # Use effective_features (d_embed if set, else num_features)
        self.num_features = config.effective_features

        # Linear projection from (patch_length * effective_features) to d_model
        patch_dim = config.patch_length * config.effective_features
        self.projection = nn.Linear(patch_dim, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and embed patches from input time series.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            Patch embeddings of shape (batch, num_patches, d_model)
        """
        batch_size, seq_len, num_features = x.shape

        # Extract patches using unfold
        # unfold(dimension, size, step) -> creates patches along dimension
        # x: (batch, seq_len, features) -> (batch, num_patches, patch_len, features)
        patches = x.unfold(dimension=1, size=self.patch_length, step=self.stride)
        # patches shape: (batch, num_patches, features, patch_length)
        # Need to transpose and flatten
        patches = patches.permute(0, 1, 3, 2)  # (batch, num_patches, patch_len, features)
        patches = patches.reshape(
            batch_size, -1, self.patch_length * self.num_features
        )  # (batch, num_patches, patch_len * features)

        # Project to d_model
        embeddings = self.projection(patches)  # (batch, num_patches, d_model)

        return embeddings


class PositionalEncoding(nn.Module):
    """Learnable positional encoding for patch sequences."""

    def __init__(self, d_model: int, max_patches: int = 100, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Learnable position embeddings
        self.position_embedding = nn.Parameter(torch.randn(1, max_patches, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to patch embeddings.

        Args:
            x: Patch embeddings of shape (batch, num_patches, d_model)

        Returns:
            Position-encoded embeddings of shape (batch, num_patches, d_model)
        """
        num_patches = x.shape[1]
        x = x + self.position_embedding[:, :num_patches, :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm architecture."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Pre-norm self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed)
        x = x + attn_out

        # Pre-norm FFN with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)  # Final layer norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all encoder layers.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class PredictionHead(nn.Module):
    """Classification head for binary prediction."""

    def __init__(
        self, d_model: int, num_patches: int, num_classes: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model * num_patches, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder output to class predictions.

        Args:
            x: Encoder output of shape (batch, num_patches, d_model)

        Returns:
            Predictions of shape (batch, num_classes) with sigmoid activation
        """
        x = self.flatten(x)  # (batch, num_patches * d_model)
        x = self.dropout(x)
        x = self.linear(x)  # (batch, num_classes)
        x = torch.sigmoid(x)  # Binary classification
        return x


class PatchTST(nn.Module):
    """PatchTST model for financial time-series classification.

    Architecture:
    1. PatchEmbedding: (batch, seq_len, features) -> (batch, num_patches, d_model)
    2. PositionalEncoding: Add learnable position embeddings
    3. TransformerEncoder: Stack of self-attention + FFN layers
    4. PredictionHead: (batch, num_patches, d_model) -> (batch, num_classes)
    """

    def __init__(self, config: PatchTSTConfig, use_revin: bool = False) -> None:
        super().__init__()
        self.config = config
        self.use_revin = use_revin

        # RevIN layer (optional)
        if self.use_revin:
            self.revin = RevIN(num_features=config.num_features)

        # Feature embedding layer (optional)
        if config.d_embed is not None:
            self.feature_embed = FeatureEmbedding(
                num_features=config.num_features,
                d_embed=config.d_embed,
                dropout=config.dropout,
            )

        # Model components
        self.patch_embed = PatchEmbedding(config)
        self.pos_encoding = PositionalEncoding(
            d_model=config.d_model,
            max_patches=config.num_patches + 10,  # Some buffer
            dropout=config.dropout,
        )
        self.encoder = TransformerEncoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )
        self.head = PredictionHead(
            d_model=config.d_model,
            num_patches=config.num_patches,
            num_classes=config.num_classes,
            dropout=config.head_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, context_length, num_features)

        Returns:
            Output tensor of shape (batch_size, num_classes) with sigmoid activation
        """
        # Apply RevIN normalization if enabled
        if self.use_revin:
            x = self.revin.normalize(x)

        # Apply feature embedding if enabled
        if hasattr(self, "feature_embed"):
            x = self.feature_embed(x)

        # Patch embedding: (batch, seq_len, features) -> (batch, num_patches, d_model)
        x = self.patch_embed(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoder
        x = self.encoder(x)

        # Classification head with sigmoid
        x = self.head(x)

        return x
