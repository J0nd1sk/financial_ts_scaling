"""Multi-scale temporal modules for PatchTST.

Provides modules that capture patterns at different temporal scales,
which can help the model learn both short-term and long-term dependencies.

Available modules:
- HierarchicalTemporalPool: Pool patches at multiple scales and combine
- MultiScalePatchEmbedding: Parallel patch embeddings at different sizes
- DilatedTemporalConv: Dilated convolutions for multi-scale receptive field
- CrossScaleAttention: Attention between different temporal scales

These modules can be integrated with PatchTST by replacing or augmenting
the standard patch embedding.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HierarchicalTemporalPool(nn.Module):
    """Hierarchical pooling at multiple temporal scales.

    Takes patch embeddings and pools them at different scales (1x, 2x, 4x, etc.),
    then combines the multi-scale representations.

    Args:
        d_model: Model dimension (embedding size).
        scales: List of pooling scales. Default [1, 2, 4].
            Scale 1 = no pooling, 2 = pool pairs, 4 = pool quads.
        fusion: How to combine scales ("concat", "sum", "attention").
            Default "concat".

    Example:
        >>> pool = HierarchicalTemporalPool(d_model=128, scales=[1, 2, 4])
        >>> x = torch.randn(32, 10, 128)  # (batch, n_patches, d_model)
        >>> y = pool(x)  # (batch, n_patches, d_model) or more if concat
    """

    def __init__(
        self,
        d_model: int,
        scales: list[int] | None = None,
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.scales = scales or [1, 2, 4]
        self.fusion = fusion

        if fusion not in ("concat", "sum", "attention"):
            raise ValueError(f"fusion must be 'concat', 'sum', or 'attention', got {fusion}")

        # For concat fusion, we need a projection to reduce dimension back
        if fusion == "concat":
            self.proj = nn.Linear(d_model * len(self.scales), d_model)
        elif fusion == "attention":
            # Learnable weights for each scale
            self.scale_weights = nn.Parameter(torch.ones(len(self.scales)))

        # Pooling layers for each scale
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=s, stride=s) if s > 1 else nn.Identity()
            for s in self.scales
        ])

        # Upsampling to restore original sequence length
        self.upsample = nn.ModuleList([
            nn.Upsample(scale_factor=s, mode="nearest") if s > 1 else nn.Identity()
            for s in self.scales
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Apply hierarchical pooling.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Multi-scale pooled tensor of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, d_model = x.shape

        # Transpose for 1D pooling: (batch, d_model, seq_len)
        x_t = x.transpose(1, 2)

        # Pool at each scale and upsample back
        pooled = []
        for pool, upsample in zip(self.pools, self.upsample):
            p = pool(x_t)  # Downsample
            p = upsample(p)  # Upsample back to original size

            # Handle size mismatch from rounding
            if p.size(2) != seq_len:
                p = F.pad(p, (0, seq_len - p.size(2)))

            pooled.append(p.transpose(1, 2))  # Back to (batch, seq_len, d_model)

        # Combine multi-scale representations
        if self.fusion == "concat":
            combined = torch.cat(pooled, dim=-1)  # (batch, seq_len, d_model * n_scales)
            return self.proj(combined)

        elif self.fusion == "sum":
            return sum(pooled)

        else:  # attention
            weights = F.softmax(self.scale_weights, dim=0)
            combined = sum(w * p for w, p in zip(weights, pooled))
            return combined


class MultiScalePatchEmbedding(nn.Module):
    """Parallel patch embeddings at different patch sizes.

    Creates multiple patch embeddings using different patch sizes,
    capturing patterns at different temporal granularities.

    Args:
        num_features: Number of input features.
        d_model: Output embedding dimension per head.
        patch_sizes: List of patch sizes. Default [8, 16, 32].
        context_length: Input sequence length.
        stride_ratio: Stride as fraction of patch size. Default 0.5.
        fusion: How to combine patches ("concat", "sum", "interleave").
            Default "concat".

    Example:
        >>> embed = MultiScalePatchEmbedding(
        ...     num_features=100,
        ...     d_model=128,
        ...     patch_sizes=[8, 16, 32],
        ...     context_length=80,
        ... )
        >>> x = torch.randn(32, 80, 100)  # (batch, seq_len, features)
        >>> patches = embed(x)  # (batch, n_patches_total, d_model)
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        patch_sizes: list[int] | None = None,
        context_length: int = 80,
        stride_ratio: float = 0.5,
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.patch_sizes = patch_sizes or [8, 16, 32]
        self.context_length = context_length
        self.stride_ratio = stride_ratio
        self.fusion = fusion

        # Create embedding for each patch size
        self.patch_embeds = nn.ModuleList()
        self.n_patches_per_scale = []

        for patch_size in self.patch_sizes:
            stride = max(1, int(patch_size * stride_ratio))
            n_patches = (context_length - patch_size) // stride + 1
            self.n_patches_per_scale.append(n_patches)

            # Linear projection from (patch_size * num_features) to d_model
            self.patch_embeds.append(
                nn.Linear(patch_size * num_features, d_model)
            )

        # Positional embeddings per scale
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
            for n_patches in self.n_patches_per_scale
        ])

        # Scale indicator embedding
        self.scale_embed = nn.Embedding(len(self.patch_sizes), d_model)

        # For interleave fusion, project back to single stream
        if fusion == "interleave":
            # Keep as is, patches will be interleaved in sequence
            pass

    def forward(self, x: Tensor) -> Tensor:
        """Extract multi-scale patch embeddings.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features).

        Returns:
            Patch embeddings of shape (batch, n_patches_total, d_model).
        """
        batch_size = x.size(0)
        all_patches = []

        for i, (patch_size, embed, pos_embed) in enumerate(
            zip(self.patch_sizes, self.patch_embeds, self.pos_embeds)
        ):
            stride = max(1, int(patch_size * self.stride_ratio))

            # Extract patches using unfold
            # x: (batch, seq_len, features) -> patches: (batch, n_patches, patch_size, features)
            patches = x.unfold(1, patch_size, stride)  # (batch, n_patches, features, patch_size)
            patches = patches.transpose(2, 3)  # (batch, n_patches, patch_size, features)

            n_patches = patches.size(1)

            # Flatten and project
            patches = patches.reshape(batch_size, n_patches, -1)  # (batch, n_patches, patch_size*features)
            patches = embed(patches)  # (batch, n_patches, d_model)

            # Add positional embedding (handle size mismatch)
            if n_patches <= pos_embed.size(1):
                patches = patches + pos_embed[:, :n_patches, :]
            else:
                # Extend positional embedding by repeating
                repeats = (n_patches // pos_embed.size(1)) + 1
                extended = pos_embed.repeat(1, repeats, 1)[:, :n_patches, :]
                patches = patches + extended

            # Add scale indicator
            scale_indicator = self.scale_embed(
                torch.tensor([i], device=x.device)
            ).unsqueeze(0).expand(batch_size, n_patches, -1)
            patches = patches + scale_indicator

            all_patches.append(patches)

        # Combine patches from all scales
        if self.fusion == "concat":
            return torch.cat(all_patches, dim=1)

        elif self.fusion == "sum":
            # Pad to same length and sum
            max_patches = max(p.size(1) for p in all_patches)
            padded = []
            for p in all_patches:
                if p.size(1) < max_patches:
                    pad = torch.zeros(
                        batch_size, max_patches - p.size(1), self.d_model,
                        device=p.device
                    )
                    p = torch.cat([p, pad], dim=1)
                padded.append(p)
            return sum(padded)

        else:  # interleave
            # Interleave patches from different scales
            return torch.cat(all_patches, dim=1)  # Simple concat for now

    @property
    def total_n_patches(self) -> int:
        """Return total number of patches across all scales."""
        return sum(self.n_patches_per_scale)


class DilatedTemporalConv(nn.Module):
    """Dilated temporal convolutions for multi-scale receptive field.

    Uses dilated 1D convolutions with exponentially increasing dilation rates
    to capture patterns at different temporal scales without pooling.

    Args:
        d_model: Model dimension.
        kernel_size: Base kernel size. Default 3.
        dilation_rates: List of dilation rates. Default [1, 2, 4, 8].
        dropout: Dropout rate. Default 0.1.

    Example:
        >>> conv = DilatedTemporalConv(d_model=128, dilation_rates=[1, 2, 4, 8])
        >>> x = torch.randn(32, 10, 128)  # (batch, seq_len, d_model)
        >>> y = conv(x)  # (batch, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        dilation_rates: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.dilation_rates = dilation_rates or [1, 2, 4, 8]

        # Create dilated conv layers
        self.convs = nn.ModuleList()
        for dilation in self.dilation_rates:
            padding = (kernel_size - 1) * dilation // 2
            self.convs.append(
                nn.Conv1d(
                    d_model, d_model,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
            )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Projection to combine multi-scale features
        self.proj = nn.Linear(d_model * len(self.dilation_rates), d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Apply dilated convolutions.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tensor of shape (batch, seq_len, d_model).
        """
        # Transpose for conv1d: (batch, d_model, seq_len)
        x_t = x.transpose(1, 2)

        # Apply each dilated conv
        outputs = []
        for conv in self.convs:
            out = conv(x_t)
            out = F.gelu(out)
            out = self.dropout(out)
            outputs.append(out.transpose(1, 2))  # Back to (batch, seq_len, d_model)

        # Concatenate and project
        combined = torch.cat(outputs, dim=-1)
        combined = self.proj(combined)

        # Residual and norm
        combined = self.norm(combined + x)

        return combined


class CrossScaleAttention(nn.Module):
    """Attention between different temporal scales.

    Allows patches from different scales to attend to each other,
    enabling the model to combine local and global information.

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads. Default 4.
        dropout: Dropout rate. Default 0.1.

    Example:
        >>> attn = CrossScaleAttention(d_model=128, n_heads=4)
        >>> fine_scale = torch.randn(32, 20, 128)  # Fine-grained patches
        >>> coarse_scale = torch.randn(32, 5, 128)  # Coarse patches
        >>> fine_attended = attn(fine_scale, coarse_scale)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
    ) -> Tensor:
        """Cross-scale attention.

        Args:
            query: Query tensor (e.g., fine scale), shape (batch, n_q, d_model).
            key_value: Key/Value tensor (e.g., coarse scale), shape (batch, n_kv, d_model).

        Returns:
            Attended query tensor, shape (batch, n_q, d_model).
        """
        batch_size = query.size(0)

        # Project
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        out = self.o_proj(out)

        # Residual and norm
        out = self.norm(out + query)

        return out


def get_multiscale_module(
    multiscale_type: str | None,
    multiscale_params: dict[str, Any] | None = None,
    d_model: int = 128,
    num_features: int = 100,
    context_length: int = 80,
) -> nn.Module | None:
    """Factory function to create multi-scale module from experiment spec.

    Args:
        multiscale_type: Type of module ("hierarchical_pool", "multi_patch",
            "dilated_conv", "cross_attention"). None returns None.
        multiscale_params: Parameters for the module:
            - hierarchical_pool: {"scales": [1, 2, 4], "fusion": "concat"}
            - multi_patch: {"patch_sizes": [8, 16], "fusion": "concat"}
            - dilated_conv: {"dilation_rates": [1, 2, 4, 8]}
            - cross_attention: {"n_heads": 4}
        d_model: Model dimension. Default 128.
        num_features: Number of input features. Default 100.
        context_length: Input sequence length. Default 80.

    Returns:
        nn.Module instance, or None if multiscale_type is None.

    Raises:
        ValueError: If multiscale_type is unknown.
    """
    if multiscale_type is None:
        return None

    params = multiscale_params or {}

    if multiscale_type == "hierarchical_pool":
        return HierarchicalTemporalPool(
            d_model=d_model,
            scales=params.get("scales", [1, 2, 4]),
            fusion=params.get("fusion", "concat"),
        )

    elif multiscale_type == "multi_patch":
        return MultiScalePatchEmbedding(
            num_features=num_features,
            d_model=d_model,
            patch_sizes=params.get("patch_sizes", [8, 16]),
            context_length=context_length,
            stride_ratio=params.get("stride_ratio", 0.5),
            fusion=params.get("fusion", "concat"),
        )

    elif multiscale_type == "dilated_conv":
        return DilatedTemporalConv(
            d_model=d_model,
            kernel_size=params.get("kernel_size", 3),
            dilation_rates=params.get("dilation_rates", [1, 2, 4, 8]),
            dropout=params.get("dropout", 0.1),
        )

    elif multiscale_type == "cross_attention":
        return CrossScaleAttention(
            d_model=d_model,
            n_heads=params.get("n_heads", 4),
            dropout=params.get("dropout", 0.1),
        )

    else:
        raise ValueError(f"Unknown multiscale_type: {multiscale_type}")
