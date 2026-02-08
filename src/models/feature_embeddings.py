"""Advanced feature embedding architectures for PatchTST.

These modules provide drop-in replacements for the simple linear FeatureEmbedding
in patchtst.py. Each implements different strategies for compressing/transforming
high-dimensional feature spaces before patch embedding.

Design Principles:
1. Same interface: forward(x) -> embedded tensor
2. Support auxiliary losses (e.g., KL for VAE)
3. Parameter count estimation methods
4. Clear documentation of capacity and inductive biases

Input:  (batch, seq_len, num_features)
Output: (batch, seq_len, d_embed)

Reference: Feature embedding experiments for scaling law investigation.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingOutput(NamedTuple):
    """Output from embedding modules that may include auxiliary losses.

    Attributes:
        embedded: The embedded tensor of shape (batch, seq_len, d_embed)
        aux_loss: Optional auxiliary loss (e.g., KL divergence for VAE).
                  None if no auxiliary loss is computed.
    """
    embedded: torch.Tensor
    aux_loss: torch.Tensor | None = None


class BaseFeatureEmbedding(nn.Module, ABC):
    """Abstract base class for feature embedding modules.

    All feature embeddings must:
    1. Transform (batch, seq_len, num_features) -> (batch, seq_len, d_embed)
    2. Provide parameter count estimation
    3. Support returning auxiliary losses
    """

    def __init__(self, num_features: int, d_embed: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_embed = d_embed
        self.dropout_rate = dropout

    @abstractmethod
    def forward(self, x: torch.Tensor) -> EmbeddingOutput:
        """Transform input features to embedding space.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            EmbeddingOutput with embedded tensor and optional aux_loss
        """
        pass

    def count_parameters(self) -> int:
        """Count total trainable parameters in this module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_parameters(self) -> dict[str, int]:
        """Estimate parameter breakdown by component.

        Returns:
            Dictionary mapping component names to parameter counts.
        """
        # Default implementation - subclasses can override for more detail
        return {"total": self.count_parameters()}


# =============================================================================
# 1. Progressive Embedding - Multi-layer compression with nonlinearity
# =============================================================================

class ProgressiveEmbedding(BaseFeatureEmbedding):
    """Multi-layer progressive compression from num_features to d_embed.

    Architecture:
        num_features -> hidden_1 -> hidden_2 -> ... -> d_embed

    Each layer applies: Linear -> LayerNorm -> GELU -> Dropout

    Design Rationale:
    - Gradual dimensionality reduction may learn better intermediate representations
    - Nonlinearities allow learning complex feature interactions
    - Multiple layers increase capacity for feature compression

    The hidden dimensions are computed geometrically between num_features and d_embed,
    creating a smooth compression path.

    Args:
        num_features: Number of input features
        d_embed: Output embedding dimension
        num_layers: Number of compression layers (default: 3)
        dropout: Dropout rate applied after each layer
    """

    def __init__(
        self,
        num_features: int,
        d_embed: int,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(num_features, d_embed, dropout)
        self.num_layers = num_layers

        # Compute intermediate dimensions geometrically
        # For num_features=500, d_embed=64, 3 layers:
        # ratio = (64/500)^(1/3) ≈ 0.5
        # dims: 500 -> 250 -> 125 -> 64
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        dims = self._compute_layer_dims(num_features, d_embed, num_layers)

        # Build progressive layers
        layers = []
        for i in range(len(dims) - 1):
            in_dim, out_dim = dims[i], dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)
        self._layer_dims = dims

    @staticmethod
    def _compute_layer_dims(in_dim: int, out_dim: int, num_layers: int) -> list[int]:
        """Compute geometric progression of layer dimensions."""
        if num_layers == 1:
            return [in_dim, out_dim]

        # Geometric ratio for smooth compression
        ratio = (out_dim / in_dim) ** (1 / num_layers)
        dims = [in_dim]
        current = in_dim
        for i in range(num_layers - 1):
            current = int(current * ratio)
            # Ensure we don't go below output dim
            current = max(current, out_dim)
            dims.append(current)
        dims.append(out_dim)
        return dims

    def forward(self, x: torch.Tensor) -> EmbeddingOutput:
        """Progressive compression through multiple layers.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            EmbeddingOutput with embedded tensor of shape (batch, seq_len, d_embed)
        """
        embedded = self.layers(x)
        return EmbeddingOutput(embedded=embedded, aux_loss=None)

    def estimate_parameters(self) -> dict[str, int]:
        """Estimate parameter breakdown by layer."""
        params = {}
        for i in range(len(self._layer_dims) - 1):
            in_dim, out_dim = self._layer_dims[i], self._layer_dims[i + 1]
            # Linear: in_dim * out_dim + out_dim (bias)
            # LayerNorm: 2 * out_dim (gamma, beta)
            layer_params = in_dim * out_dim + out_dim + 2 * out_dim
            params[f"layer_{i}"] = layer_params
        params["total"] = sum(params.values())
        return params


# =============================================================================
# 2. Bottleneck Embedding - Compress then expand (deterministic)
# =============================================================================

class BottleneckEmbedding(BaseFeatureEmbedding):
    """Bottleneck architecture that compresses then expands features.

    Architecture:
        num_features -> bottleneck_dim -> d_embed

    Design Rationale:
    - Information bottleneck forces learning of essential feature representations
    - Compression ratio controls information flow constraint
    - Expansion after bottleneck allows recovery of useful variations
    - Skip connection provides gradient highway for deep models

    The bottleneck dimension is computed as:
        bottleneck_dim = min(num_features, d_embed) * compression_ratio

    Args:
        num_features: Number of input features
        d_embed: Output embedding dimension
        compression_ratio: Fraction of min(num_features, d_embed) for bottleneck
                          (default: 0.25, meaning 4x compression)
        use_skip: Whether to add skip connection around bottleneck
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_features: int,
        d_embed: int,
        compression_ratio: float = 0.25,
        use_skip: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(num_features, d_embed, dropout)

        if not 0 < compression_ratio < 1:
            raise ValueError("compression_ratio must be in (0, 1)")

        self.compression_ratio = compression_ratio
        self.use_skip = use_skip

        # Compute bottleneck dimension
        min_dim = min(num_features, d_embed)
        self.bottleneck_dim = max(4, int(min_dim * compression_ratio))

        # Encoder: compress to bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(num_features, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Decoder: expand from bottleneck
        self.decoder = nn.Sequential(
            nn.Linear(self.bottleneck_dim, d_embed),
            nn.LayerNorm(d_embed),
            nn.Dropout(dropout),
        )

        # Skip connection projection (if dimensions differ)
        if use_skip:
            if num_features != d_embed:
                self.skip_proj = nn.Linear(num_features, d_embed)
            else:
                self.skip_proj = nn.Identity()
            self.skip_norm = nn.LayerNorm(d_embed)

    def forward(self, x: torch.Tensor) -> EmbeddingOutput:
        """Compress through bottleneck then expand.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            EmbeddingOutput with embedded tensor of shape (batch, seq_len, d_embed)
        """
        # Bottleneck path
        bottleneck = self.encoder(x)
        embedded = self.decoder(bottleneck)

        # Add skip connection if enabled
        if self.use_skip:
            skip = self.skip_proj(x)
            embedded = self.skip_norm(embedded + skip)

        return EmbeddingOutput(embedded=embedded, aux_loss=None)

    def estimate_parameters(self) -> dict[str, int]:
        """Estimate parameter breakdown."""
        enc_params = (
            self.num_features * self.bottleneck_dim + self.bottleneck_dim  # Linear
            + 2 * self.bottleneck_dim  # LayerNorm
        )
        dec_params = (
            self.bottleneck_dim * self.d_embed + self.d_embed  # Linear
            + 2 * self.d_embed  # LayerNorm
        )
        skip_params = 0
        if self.use_skip and self.num_features != self.d_embed:
            skip_params = self.num_features * self.d_embed + self.d_embed
        skip_params += 2 * self.d_embed if self.use_skip else 0  # skip_norm

        return {
            "encoder": enc_params,
            "decoder": dec_params,
            "skip": skip_params,
            "total": enc_params + dec_params + skip_params,
        }


# =============================================================================
# 3. Variational Bottleneck Embedding - VAE-style with KL loss
# =============================================================================

class VariationalBottleneckEmbedding(BaseFeatureEmbedding):
    """Variational autoencoder style bottleneck with KL regularization.

    Architecture:
        num_features -> (mu, log_var) -> z (sampled) -> d_embed

    Design Rationale:
    - Probabilistic bottleneck forces learning of smooth latent manifold
    - KL loss encourages compact, well-organized feature space
    - Reparameterization trick enables gradient flow through sampling
    - Beta parameter controls KL weight (beta-VAE formulation)

    During training, samples z from N(mu, var).
    During inference, uses mu directly (deterministic).

    The KL loss is returned as aux_loss and should be added to the main loss
    with appropriate weighting (beta parameter).

    Args:
        num_features: Number of input features
        d_embed: Output embedding dimension
        latent_dim: Dimension of latent space (default: computed from compression_ratio)
        compression_ratio: If latent_dim not specified, compute as fraction of min dim
        beta: Weight for KL loss term (default: 0.1)
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_features: int,
        d_embed: int,
        latent_dim: int | None = None,
        compression_ratio: float = 0.25,
        beta: float = 0.1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(num_features, d_embed, dropout)

        self.beta = beta

        # Compute latent dimension
        if latent_dim is not None:
            self.latent_dim = latent_dim
        else:
            min_dim = min(num_features, d_embed)
            self.latent_dim = max(4, int(min_dim * compression_ratio))

        # Encoder: produces mean and log-variance
        self.encoder_shared = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.LayerNorm(num_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate heads for mu and log_var
        self.mu_head = nn.Linear(num_features // 2, self.latent_dim)
        self.logvar_head = nn.Linear(num_features // 2, self.latent_dim)

        # Decoder: expand from latent space
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, d_embed),
            nn.LayerNorm(d_embed),
            nn.Dropout(dropout),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            Tuple of (mu, log_var) each of shape (batch, seq_len, latent_dim)
        """
        h = self.encoder_shared(x)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick.

        z = mu + std * epsilon, where epsilon ~ N(0, 1)

        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution

        Returns:
            Sampled latent vector z
        """
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # During inference, use mean directly (deterministic)
            return mu

    def kl_divergence(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from N(mu, var) to N(0, 1).

        KL(q(z|x) || p(z)) = -0.5 * sum(1 + log_var - mu^2 - var)

        Args:
            mu: Mean of approximate posterior
            log_var: Log variance of approximate posterior

        Returns:
            KL divergence (scalar, averaged over batch and sequence)
        """
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return kl

    def forward(self, x: torch.Tensor) -> EmbeddingOutput:
        """Encode through variational bottleneck.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            EmbeddingOutput with:
                - embedded: tensor of shape (batch, seq_len, d_embed)
                - aux_loss: KL divergence loss (weighted by beta)
        """
        # Encode to latent distribution
        mu, log_var = self.encode(x)

        # Sample from latent space
        z = self.reparameterize(mu, log_var)

        # Decode to output dimension
        embedded = self.decoder(z)

        # Compute KL loss
        kl_loss = self.beta * self.kl_divergence(mu, log_var)

        return EmbeddingOutput(embedded=embedded, aux_loss=kl_loss)

    def estimate_parameters(self) -> dict[str, int]:
        """Estimate parameter breakdown."""
        hidden_dim = self.num_features // 2

        enc_shared = (
            self.num_features * hidden_dim + hidden_dim  # Linear
            + 2 * hidden_dim  # LayerNorm
        )
        mu_params = hidden_dim * self.latent_dim + self.latent_dim
        logvar_params = hidden_dim * self.latent_dim + self.latent_dim
        dec_params = (
            self.latent_dim * self.d_embed + self.d_embed  # Linear
            + 2 * self.d_embed  # LayerNorm
        )

        return {
            "encoder_shared": enc_shared,
            "mu_head": mu_params,
            "logvar_head": logvar_params,
            "decoder": dec_params,
            "total": enc_shared + mu_params + logvar_params + dec_params,
        }


# =============================================================================
# 4. Multi-Head Feature Embedding - Parallel projections with combination
# =============================================================================

class MultiHeadFeatureEmbedding(BaseFeatureEmbedding):
    """Project features through multiple parallel heads then combine.

    Architecture:
        input -> [head_1, head_2, ..., head_n] -> concat/sum -> output_proj -> d_embed

    Design Rationale:
    - Each head can specialize in different feature subsets or patterns
    - Analogous to multi-head attention but for feature transformation
    - Combination method affects capacity vs. complexity tradeoff
    - Feature grouping option enables domain-knowledge injection

    Two combination methods:
    - "concat": Concatenate heads then project (more parameters)
    - "sum": Sum heads with learned weights (fewer parameters)

    Args:
        num_features: Number of input features
        d_embed: Output embedding dimension
        num_heads: Number of parallel projection heads (default: 4)
        head_dim: Dimension per head (default: d_embed // num_heads)
        combine_method: How to combine heads - "concat" or "sum"
        feature_groups: Optional list of feature indices per head for grouped processing
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_features: int,
        d_embed: int,
        num_heads: int = 4,
        head_dim: int | None = None,
        combine_method: str = "concat",
        feature_groups: list[list[int]] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(num_features, d_embed, dropout)

        if combine_method not in ("concat", "sum"):
            raise ValueError("combine_method must be 'concat' or 'sum'")

        self.num_heads = num_heads
        self.combine_method = combine_method
        self.feature_groups = feature_groups

        # Compute head dimension
        if head_dim is not None:
            self.head_dim = head_dim
        else:
            self.head_dim = d_embed // num_heads

        # Create projection heads
        if feature_groups is not None:
            # Each head processes a subset of features
            if len(feature_groups) != num_heads:
                raise ValueError("feature_groups must have same length as num_heads")
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(len(group), self.head_dim),
                    nn.LayerNorm(self.head_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for group in feature_groups
            ])
            self._group_indices = [torch.tensor(g) for g in feature_groups]
        else:
            # Each head processes all features
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(num_features, self.head_dim),
                    nn.LayerNorm(self.head_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(num_heads)
            ])
            self._group_indices = None

        # Combination layers
        if combine_method == "concat":
            # Concatenate heads then project
            self.output_proj = nn.Sequential(
                nn.Linear(num_heads * self.head_dim, d_embed),
                nn.LayerNorm(d_embed),
                nn.Dropout(dropout),
            )
        else:
            # Weighted sum of heads
            self.head_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
            # Project each head to d_embed if head_dim != d_embed
            if self.head_dim != d_embed:
                self.head_proj = nn.Linear(self.head_dim, d_embed)
            else:
                self.head_proj = nn.Identity()
            self.output_norm = nn.LayerNorm(d_embed)
            self.output_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> EmbeddingOutput:
        """Process through parallel heads and combine.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            EmbeddingOutput with embedded tensor of shape (batch, seq_len, d_embed)
        """
        batch_size, seq_len, _ = x.shape

        # Process through each head
        head_outputs = []
        for i, head in enumerate(self.heads):
            if self._group_indices is not None:
                # Select feature subset for this head
                indices = self._group_indices[i].to(x.device)
                x_subset = x.index_select(dim=-1, index=indices)
                head_out = head(x_subset)
            else:
                head_out = head(x)
            head_outputs.append(head_out)

        # Combine heads
        if self.combine_method == "concat":
            # Concatenate along feature dimension
            combined = torch.cat(head_outputs, dim=-1)
            embedded = self.output_proj(combined)
        else:
            # Weighted sum
            weights = F.softmax(self.head_weights, dim=0)
            head_outputs = [self.head_proj(h) for h in head_outputs]
            combined = sum(w * h for w, h in zip(weights, head_outputs))
            embedded = self.output_drop(self.output_norm(combined))

        return EmbeddingOutput(embedded=embedded, aux_loss=None)

    def estimate_parameters(self) -> dict[str, int]:
        """Estimate parameter breakdown."""
        if self.feature_groups is not None:
            head_params = sum(
                len(g) * self.head_dim + self.head_dim + 2 * self.head_dim
                for g in self.feature_groups
            )
        else:
            per_head = (
                self.num_features * self.head_dim + self.head_dim  # Linear
                + 2 * self.head_dim  # LayerNorm
            )
            head_params = per_head * self.num_heads

        if self.combine_method == "concat":
            combine_params = (
                self.num_heads * self.head_dim * self.d_embed + self.d_embed
                + 2 * self.d_embed
            )
        else:
            combine_params = self.num_heads  # head_weights
            if self.head_dim != self.d_embed:
                combine_params += self.head_dim * self.d_embed + self.d_embed
            combine_params += 2 * self.d_embed  # output_norm

        return {
            "heads": head_params,
            "combine": combine_params,
            "total": head_params + combine_params,
        }


# =============================================================================
# 5. Gated Residual Embedding - GRN with GLU gating
# =============================================================================

class GatedResidualEmbedding(BaseFeatureEmbedding):
    """Gated Residual Network (GRN) style embedding with GLU gating.

    Architecture:
        input -> Linear -> ELU -> Linear -> GLU -> Dropout -> + skip -> LayerNorm

    Design Rationale:
    - GRN architecture from Temporal Fusion Transformers (TFT)
    - GLU gating provides adaptive feature suppression
    - Skip connection with projection handles dimension mismatch
    - Context input option enables conditional embedding

    The Gated Linear Unit (GLU) splits the projection into two halves:
        GLU(x) = x[:d] * sigmoid(x[d:])
    This allows the network to learn which features to pass through.

    Reference: "Temporal Fusion Transformers for Interpretable Multi-horizon
    Time Series Forecasting" (Lim et al., 2021)

    Args:
        num_features: Number of input features
        d_embed: Output embedding dimension
        hidden_dim: Hidden layer dimension (default: d_embed * 2)
        context_dim: Optional context input dimension for conditional embedding
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_features: int,
        d_embed: int,
        hidden_dim: int | None = None,
        context_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(num_features, d_embed, dropout)

        self.hidden_dim = hidden_dim if hidden_dim else d_embed * 2
        self.context_dim = context_dim

        # Input projection
        self.fc1 = nn.Linear(num_features, self.hidden_dim)

        # Context projection (optional)
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, self.hidden_dim, bias=False)

        # GLU projection (2x for split)
        self.fc2 = nn.Linear(self.hidden_dim, d_embed * 2)

        # Skip connection
        if num_features != d_embed:
            self.skip_proj = nn.Linear(num_features, d_embed)
        else:
            self.skip_proj = nn.Identity()

        # Output normalization and dropout
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_embed)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None
    ) -> EmbeddingOutput:
        """Apply gated residual transformation.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)
            context: Optional context tensor of shape (batch, seq_len, context_dim)

        Returns:
            EmbeddingOutput with embedded tensor of shape (batch, seq_len, d_embed)
        """
        # Skip connection
        skip = self.skip_proj(x)

        # Main path: Linear -> ELU
        hidden = F.elu(self.fc1(x))

        # Add context if provided
        if context is not None and self.context_dim is not None:
            hidden = hidden + self.context_proj(context)

        # GLU: Linear -> split -> sigmoid gate
        glu_input = self.fc2(hidden)
        value, gate = glu_input.chunk(2, dim=-1)
        gated = value * torch.sigmoid(gate)

        # Dropout and residual
        gated = self.dropout(gated)
        output = self.layer_norm(gated + skip)

        return EmbeddingOutput(embedded=output, aux_loss=None)

    def estimate_parameters(self) -> dict[str, int]:
        """Estimate parameter breakdown."""
        fc1_params = self.num_features * self.hidden_dim + self.hidden_dim
        context_params = 0
        if self.context_dim is not None:
            context_params = self.context_dim * self.hidden_dim
        fc2_params = self.hidden_dim * (self.d_embed * 2) + self.d_embed * 2
        skip_params = 0
        if self.num_features != self.d_embed:
            skip_params = self.num_features * self.d_embed + self.d_embed
        norm_params = 2 * self.d_embed

        return {
            "fc1": fc1_params,
            "context": context_params,
            "fc2": fc2_params,
            "skip": skip_params,
            "layer_norm": norm_params,
            "total": fc1_params + context_params + fc2_params + skip_params + norm_params,
        }


# =============================================================================
# 6. Attention Feature Embedding - Learned feature importance weights
# =============================================================================

class AttentionFeatureEmbedding(BaseFeatureEmbedding):
    """Self-attention based feature embedding with learned importance weights.

    Architecture:
        1. Project features to query/key/value
        2. Compute feature-wise attention scores (features attend to each other)
        3. Apply attention to values
        4. Project to output dimension

    Design Rationale:
    - Allows model to learn which features are most informative
    - Feature-to-feature attention captures cross-feature dependencies
    - Attention weights provide interpretability
    - Optional position encoding for ordered features

    Unlike standard sequence attention (token-to-token), this applies attention
    across the feature dimension, treating each feature as a "token" that can
    attend to other features.

    Args:
        num_features: Number of input features
        d_embed: Output embedding dimension
        num_heads: Number of attention heads (default: 4)
        use_position: Add learnable position encodings for features
        return_attention: Store attention weights for interpretability
        dropout: Dropout rate
    """

    def __init__(
        self,
        num_features: int,
        d_embed: int,
        num_heads: int = 4,
        use_position: bool = True,
        return_attention: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(num_features, d_embed, dropout)

        self.num_heads = num_heads
        self.use_position = use_position
        self.return_attention = return_attention

        # Each "token" is a scalar feature value, expanded to attention dim
        self.attn_dim = d_embed
        self.head_dim = d_embed // num_heads

        if d_embed % num_heads != 0:
            raise ValueError(f"d_embed ({d_embed}) must be divisible by num_heads ({num_heads})")

        # Feature value projection (scalar -> attn_dim per feature)
        self.value_expand = nn.Linear(1, self.attn_dim)

        # Position encodings for features
        if use_position:
            self.feature_pos = nn.Parameter(torch.randn(1, num_features, self.attn_dim))

        # Query, Key, Value projections
        self.query = nn.Linear(self.attn_dim, self.attn_dim)
        self.key = nn.Linear(self.attn_dim, self.attn_dim)
        self.value = nn.Linear(self.attn_dim, self.attn_dim)

        # Output projection (from attended features to d_embed per timestep)
        self.output_proj = nn.Linear(num_features * self.attn_dim, d_embed)
        self.output_norm = nn.LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)

        # Store attention weights if requested
        self.attention_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> EmbeddingOutput:
        """Apply feature-wise attention.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)

        Returns:
            EmbeddingOutput with embedded tensor of shape (batch, seq_len, d_embed)
        """
        batch_size, seq_len, num_features = x.shape

        # Expand each feature value to attention dimension
        # (batch, seq, features) -> (batch, seq, features, 1) -> (batch, seq, features, attn_dim)
        x_expanded = self.value_expand(x.unsqueeze(-1))

        # Add position encodings for features
        if self.use_position:
            x_expanded = x_expanded + self.feature_pos

        # Reshape for batch processing: (batch * seq, features, attn_dim)
        x_flat = x_expanded.view(batch_size * seq_len, num_features, self.attn_dim)

        # Compute Q, K, V
        q = self.query(x_flat)  # (batch * seq, features, attn_dim)
        k = self.key(x_flat)
        v = self.value(x_flat)

        # Reshape for multi-head attention
        # (batch * seq, features, num_heads, head_dim) -> (batch * seq, num_heads, features, head_dim)
        q = q.view(-1, num_features, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(-1, num_features, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(-1, num_features, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Store attention weights if requested
        if self.return_attention:
            self.attention_weights = attn_weights.view(
                batch_size, seq_len, self.num_heads, num_features, num_features
            )

        # Apply attention to values
        attended = torch.matmul(attn_weights, v)

        # Reshape back: (batch * seq, num_heads, features, head_dim) -> (batch * seq, features, attn_dim)
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(-1, num_features, self.attn_dim)

        # Flatten features and project to output
        # (batch * seq, features * attn_dim) -> (batch * seq, d_embed)
        attended_flat = attended.view(-1, num_features * self.attn_dim)
        output = self.output_proj(attended_flat)
        output = self.dropout(self.output_norm(output))

        # Reshape to (batch, seq, d_embed)
        output = output.view(batch_size, seq_len, self.d_embed)

        return EmbeddingOutput(embedded=output, aux_loss=None)

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return stored attention weights from last forward pass.

        Returns:
            Attention weights of shape (batch, seq_len, num_heads, num_features, num_features)
            or None if return_attention=False or forward hasn't been called.
        """
        return self.attention_weights

    def estimate_parameters(self) -> dict[str, int]:
        """Estimate parameter breakdown."""
        value_expand = 1 * self.attn_dim + self.attn_dim
        position = self.num_features * self.attn_dim if self.use_position else 0
        qkv = 3 * (self.attn_dim * self.attn_dim + self.attn_dim)
        output = (
            self.num_features * self.attn_dim * self.d_embed + self.d_embed
            + 2 * self.d_embed  # LayerNorm
        )

        return {
            "value_expand": value_expand,
            "position": position,
            "qkv": qkv,
            "output": output,
            "total": value_expand + position + qkv + output,
        }


# =============================================================================
# Factory function and registration
# =============================================================================

EMBEDDING_REGISTRY: dict[str, type[BaseFeatureEmbedding]] = {
    "progressive": ProgressiveEmbedding,
    "bottleneck": BottleneckEmbedding,
    "variational": VariationalBottleneckEmbedding,
    "multihead": MultiHeadFeatureEmbedding,
    "gated_residual": GatedResidualEmbedding,
    "attention": AttentionFeatureEmbedding,
}


def create_feature_embedding(
    embedding_type: str,
    num_features: int,
    d_embed: int,
    dropout: float = 0.0,
    **kwargs,
) -> BaseFeatureEmbedding:
    """Factory function to create feature embedding modules.

    Args:
        embedding_type: Type of embedding - one of:
            "progressive", "bottleneck", "variational",
            "multihead", "gated_residual", "attention"
        num_features: Number of input features
        d_embed: Output embedding dimension
        dropout: Dropout rate
        **kwargs: Additional arguments for specific embedding types

    Returns:
        Configured embedding module

    Raises:
        ValueError: If embedding_type is not recognized

    Example:
        >>> embed = create_feature_embedding(
        ...     "bottleneck",
        ...     num_features=500,
        ...     d_embed=64,
        ...     compression_ratio=0.25
        ... )
    """
    if embedding_type not in EMBEDDING_REGISTRY:
        valid_types = ", ".join(EMBEDDING_REGISTRY.keys())
        raise ValueError(
            f"Unknown embedding_type: {embedding_type}. Valid types: {valid_types}"
        )

    cls = EMBEDDING_REGISTRY[embedding_type]
    return cls(num_features=num_features, d_embed=d_embed, dropout=dropout, **kwargs)


def list_embedding_types() -> list[str]:
    """Return list of available embedding type names."""
    return list(EMBEDDING_REGISTRY.keys())
