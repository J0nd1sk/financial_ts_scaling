"""Architecture grid generation for PatchTST HPO.

This module pre-computes valid architecture combinations for each parameter budget,
enabling architectural search during HPO rather than just training parameter search.

Design doc: docs/architectural_hpo_design.md
"""

from __future__ import annotations

from itertools import product
from typing import TypedDict


class ArchitectureConfig(TypedDict):
    """Type definition for architecture configuration dict."""

    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    param_count: int


# Definitive search space from design doc
# Extended n_layers to explore very deep architectures (especially for larger budgets)
ARCH_SEARCH_SPACE: dict[str, list[int]] = {
    "d_model": [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
    "n_layers": [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 180, 192, 256],
    "n_heads": [2, 4, 8, 16, 32],
    "d_ff_ratio": [2, 4],
}

# Budget targets and tolerances (±25%)
BUDGET_TARGETS: dict[str, int] = {
    "2m": 2_000_000,
    "20m": 20_000_000,
    "200m": 200_000_000,
    "2b": 2_000_000_000,
}


def estimate_param_count(
    d_model: int,
    n_layers: int,
    n_heads: int,  # Used for validation, doesn't affect count directly
    d_ff: int,
    num_features: int,
    context_length: int = 60,
    patch_len: int = 10,
    stride: int = 5,
    num_classes: int = 1,
) -> int:
    """Estimate total parameter count for a PatchTST configuration.

    This formula must match the actual model.parameters().numel() count.
    See tests/test_arch_grid.py for validation against real models.

    Args:
        d_model: Model embedding dimension.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads (for validation only).
        d_ff: Feedforward dimension.
        num_features: Number of input features.
        context_length: Input sequence length (default 60).
        patch_len: Length of each patch (default 10).
        stride: Stride between patches (default 5).
        num_classes: Output classes (default 1 for binary).

    Returns:
        Estimated total parameter count as integer.
    """
    # Validate n_heads divides d_model
    if d_model % n_heads != 0:
        raise ValueError(f"n_heads={n_heads} must divide d_model={d_model}")

    num_patches = (context_length - patch_len) // stride + 1

    # 1. PatchEmbedding.projection (nn.Linear)
    #    in: patch_len * num_features, out: d_model
    patch_embed = (patch_len * num_features) * d_model + d_model

    # 2. PositionalEncoding.position_embedding (nn.Parameter)
    #    shape: (1, max_patches, d_model) where max_patches = num_patches + 10
    pos_embed = (num_patches + 10) * d_model

    # 3. TransformerEncoderLayer (per layer):
    #    - norm1 (LayerNorm): 2 * d_model (weight + bias)
    #    - norm2 (LayerNorm): 2 * d_model
    #    - self_attn (MultiheadAttention): 4 * d_model² + 4 * d_model
    #    - ffn.0 (Linear d_model→d_ff): d_model * d_ff + d_ff
    #    - ffn.3 (Linear d_ff→d_model): d_ff * d_model + d_model
    mha_params = 4 * d_model * d_model + 4 * d_model
    norm_params = 4 * d_model  # Two LayerNorms
    ffn_params = 2 * d_model * d_ff + d_ff + d_model
    per_layer = mha_params + norm_params + ffn_params
    encoder_layers = n_layers * per_layer

    # 4. TransformerEncoder.norm (final LayerNorm)
    encoder_final_norm = 2 * d_model

    # 5. PredictionHead.linear (nn.Linear)
    #    in: d_model * num_patches, out: num_classes
    pred_head = (d_model * num_patches) * num_classes + num_classes

    total = patch_embed + pos_embed + encoder_layers + encoder_final_norm + pred_head
    return int(total)


def estimate_param_count_with_embedding(
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    num_features: int,
    d_embed: int | None = None,
    context_length: int = 60,
    patch_len: int = 10,
    stride: int = 5,
    num_classes: int = 1,
) -> int:
    """Estimate parameter count for PatchTST with optional Feature Embedding.

    When d_embed is set, adds a FeatureEmbedding layer that projects features
    before patching. This changes the patch dimension from patch_len * num_features
    to patch_len * d_embed.

    Args:
        d_model: Model embedding dimension.
        n_layers: Number of transformer encoder layers.
        n_heads: Number of attention heads (for validation).
        d_ff: Feedforward dimension.
        num_features: Number of input features.
        d_embed: Feature embedding dimension. None = no feature embedding.
        context_length: Input sequence length (default 60).
        patch_len: Length of each patch (default 10).
        stride: Stride between patches (default 5).
        num_classes: Output classes (default 1 for binary).

    Returns:
        Estimated total parameter count as integer.
    """
    # Validate n_heads divides d_model
    if d_model % n_heads != 0:
        raise ValueError(f"n_heads={n_heads} must divide d_model={d_model}")

    num_patches = (context_length - patch_len) // stride + 1

    # Feature embedding parameters (optional)
    if d_embed is not None:
        # FeatureEmbedding:
        #   - projection (nn.Linear): num_features * d_embed + d_embed
        #   - norm (LayerNorm): d_embed + d_embed
        feature_embed = (num_features * d_embed + d_embed) + (d_embed + d_embed)
        effective_features = d_embed
    else:
        feature_embed = 0
        effective_features = num_features

    # 1. PatchEmbedding.projection (nn.Linear)
    #    Uses effective_features (d_embed if set, else num_features)
    patch_embed = (patch_len * effective_features) * d_model + d_model

    # 2. PositionalEncoding.position_embedding (nn.Parameter)
    pos_embed = (num_patches + 10) * d_model

    # 3. TransformerEncoderLayer (per layer):
    #    - norm1 (LayerNorm): 2 * d_model
    #    - norm2 (LayerNorm): 2 * d_model
    #    - self_attn (MultiheadAttention): 4 * d_model² + 4 * d_model
    #    - ffn.0 (Linear d_model→d_ff): d_model * d_ff + d_ff
    #    - ffn.3 (Linear d_ff→d_model): d_ff * d_model + d_model
    mha_params = 4 * d_model * d_model + 4 * d_model
    norm_params = 4 * d_model  # Two LayerNorms
    ffn_params = 2 * d_model * d_ff + d_ff + d_model
    per_layer = mha_params + norm_params + ffn_params
    encoder_layers = n_layers * per_layer

    # 4. TransformerEncoder.norm (final LayerNorm)
    encoder_final_norm = 2 * d_model

    # 5. PredictionHead.linear (nn.Linear)
    pred_head = (d_model * num_patches) * num_classes + num_classes

    total = feature_embed + patch_embed + pos_embed + encoder_layers + encoder_final_norm + pred_head
    return int(total)


def generate_architecture_grid(
    num_features: int,
    context_length: int = 60,
    patch_len: int = 10,
    stride: int = 5,
    num_classes: int = 1,
) -> list[ArchitectureConfig]:
    """Generate all valid architecture combinations.

    Filters out invalid combinations where n_heads doesn't divide d_model.
    Computes d_ff as d_model * ratio for each d_ff_ratio value.

    Args:
        num_features: Number of input features.
        context_length: Input sequence length.
        patch_len: Length of each patch.
        stride: Stride between patches.
        num_classes: Output classes.

    Returns:
        List of architecture config dicts with param_count included.
    """
    grid: list[ArchitectureConfig] = []

    for d_model, n_layers, n_heads, d_ff_ratio in product(
        ARCH_SEARCH_SPACE["d_model"],
        ARCH_SEARCH_SPACE["n_layers"],
        ARCH_SEARCH_SPACE["n_heads"],
        ARCH_SEARCH_SPACE["d_ff_ratio"],
    ):
        # Filter: n_heads must divide d_model
        if d_model % n_heads != 0:
            continue

        d_ff = d_model * d_ff_ratio

        param_count = estimate_param_count(
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            num_features=num_features,
            context_length=context_length,
            patch_len=patch_len,
            stride=stride,
            num_classes=num_classes,
        )

        grid.append(
            {
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "d_ff": d_ff,
                "param_count": param_count,
            }
        )

    return grid


def filter_by_budget(
    grid: list[ArchitectureConfig],
    budget: int,
    tolerance: float = 0.25,
) -> list[ArchitectureConfig]:
    """Filter architecture grid to those within budget tolerance.

    Args:
        grid: List of architecture configs with param_count.
        budget: Target parameter budget.
        tolerance: Acceptable deviation from budget (default ±25%).

    Returns:
        Filtered list of architectures within tolerance, sorted by param_count.
    """
    min_allowed = int(budget * (1 - tolerance))
    max_allowed = int(budget * (1 + tolerance))

    filtered = [
        arch for arch in grid
        if min_allowed <= arch["param_count"] <= max_allowed
    ]

    # Sort by param_count ascending
    filtered.sort(key=lambda x: x["param_count"])

    return filtered


def get_memory_safe_batch_config(
    d_model: int,
    n_layers: int,
    target_effective_batch: int = 256,
) -> dict[str, int]:
    """Return memory-safe batch configuration based on architecture size.

    Uses a heuristic based on model memory footprint to determine safe batch sizes.
    Memory pressure scales approximately with d_model² × n_layers (attention + FFN).

    Args:
        d_model: Model embedding dimension.
        n_layers: Number of transformer encoder layers.
        target_effective_batch: Desired effective batch size (default 256).

    Returns:
        dict with keys:
        - 'micro_batch': Physical batch size per forward pass
        - 'accumulation_steps': Number of gradient accumulation steps
        - 'effective_batch': micro_batch × accumulation_steps
    """
    # Memory ∝ d_model² × n_layers (attention + FFN activations)
    # Normalize to ~1.0 for d=1024, L=256 (known problematic config)
    memory_score = (d_model ** 2) * n_layers / 1e9

    if memory_score <= 0.1:  # Small models (d≤512 or shallow)
        micro_batch = 256
    elif memory_score <= 0.5:  # Medium models
        micro_batch = 128
    elif memory_score <= 1.5:  # Large models
        micro_batch = 64
    elif memory_score <= 3.0:  # XLarge models
        micro_batch = 32
    else:  # Massive models (2B+)
        micro_batch = 16

    accumulation_steps = max(1, target_effective_batch // micro_batch)

    return {
        "micro_batch": micro_batch,
        "accumulation_steps": accumulation_steps,
        "effective_batch": micro_batch * accumulation_steps,
    }


def get_architectures_for_budget(
    budget: str,
    num_features: int,
    context_length: int = 60,
    patch_len: int = 10,
    stride: int = 5,
    num_classes: int = 1,
    tolerance: float = 0.25,
) -> list[ArchitectureConfig]:
    """Get valid architectures for a parameter budget.

    Main entry point for architectural HPO. Returns pre-computed list of
    valid architectures within the specified budget tolerance.

    Args:
        budget: Budget string ("2M", "20M", "200M", "2B"), case-insensitive.
        num_features: Number of input features.
        context_length: Input sequence length.
        patch_len: Length of each patch.
        stride: Stride between patches.
        num_classes: Output classes.
        tolerance: Acceptable deviation from budget (default ±25%).

    Returns:
        List of architecture configs sorted by param_count.

    Raises:
        ValueError: If budget string is not recognized.
    """
    budget_lower = budget.lower()
    if budget_lower not in BUDGET_TARGETS:
        valid = list(BUDGET_TARGETS.keys())
        raise ValueError(f"Invalid budget '{budget}'. Must be one of: {valid}")

    target = BUDGET_TARGETS[budget_lower]

    # Generate full grid
    grid = generate_architecture_grid(
        num_features=num_features,
        context_length=context_length,
        patch_len=patch_len,
        stride=stride,
        num_classes=num_classes,
    )

    # Filter by budget
    filtered = filter_by_budget(grid, target, tolerance)

    return filtered
