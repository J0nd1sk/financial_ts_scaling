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
    "n_layers": [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
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
