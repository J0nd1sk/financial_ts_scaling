"""Budget-aware HPO with forced extremes for transformer architectures.

This module implements a two-phase HPO strategy with budget-aware forced extremes
to systematically explore transformer architectures across parameter scales:
- 750k (small baseline)
- 2M (standard baseline from original experiments)
- 20M (10x scale)
- 200M (100x scale)

Phase 1: 18 forced extreme trials + ~50 TPE trials (60-70 total) with early stopping
Phase 2: Supplementary trials on top 2 performing budgets

Key concepts:
- Shallow-wide: Larger d_model, fewer layers (better for simple patterns)
- Deep-narrow: Smaller d_model, more layers (better for complex hierarchical patterns)
- Parameter formula: params ≈ 12 × n_layers × d_model²
"""

from typing import Any


def estimate_params(d_model: int, n_layers: int) -> int:
    """Estimate parameter count for a transformer model.

    Uses the approximation: params ≈ 12 × n_layers × d_model²

    This covers:
    - Self-attention: 4 × d_model² per layer (Q, K, V, O projections)
    - FFN: 8 × d_model² per layer (assuming ff_dim = 4 × d_model)

    Args:
        d_model: Model dimension / hidden size
        n_layers: Number of transformer layers

    Returns:
        Estimated parameter count
    """
    return 12 * n_layers * d_model * d_model


# ============================================================================
# BUDGET CONFIGURATIONS
# ============================================================================

# Architecture configs for each parameter budget
# Each budget has shallow-wide and deep-narrow variants
BUDGET_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "750k": {
        "shallow": {"d_model": 192, "n_layers": 2, "n_heads": 4},  # ~884k params
        "deep": {"d_model": 128, "n_layers": 4, "n_heads": 4},    # ~786k params
    },
    "2M": {
        "shallow": {"d_model": 320, "n_layers": 2, "n_heads": 4},  # ~2.46M params
        "deep": {"d_model": 192, "n_layers": 5, "n_heads": 4},    # ~2.21M params
    },
    "20M": {
        "shallow": {"d_model": 768, "n_layers": 3, "n_heads": 8},  # ~21.2M params
        "deep": {"d_model": 384, "n_layers": 12, "n_heads": 8},   # ~21.2M params
    },
    "200M": {
        "shallow": {"d_model": 1536, "n_layers": 8, "n_heads": 16},  # ~226M params
        "deep": {"d_model": 768, "n_layers": 28, "n_heads": 16},     # ~197M params
    },
}

# Default regularization values (from ablation studies)
DEFAULT_REGULARIZATION: dict[str, float] = {
    "dropout": 0.5,
    "learning_rate": 1e-4,
    "weight_decay": 1e-3,
}

# Extreme regularization values to test
REGULARIZATION_EXTREMES: dict[str, list[float]] = {
    "dropout": [0.1, 0.3, 0.5, 0.7],  # Low to high
    "learning_rate": [1e-5, 1e-4, 1e-3],  # Low, default, high
    "weight_decay": [0.0, 1e-3, 1e-2],  # None, default, high
}


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def compute_budget_aware_extremes(
    budget: str,
    arch_style: str,
    regularization: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute architecture config for a given budget and style.

    Args:
        budget: Parameter budget ("750k", "2M", "20M", "200M")
        arch_style: Architecture style ("shallow" or "deep")
        regularization: Override regularization params (defaults to DEFAULT_REGULARIZATION)

    Returns:
        Complete config dict with d_model, n_layers, n_heads, dropout, learning_rate, weight_decay
    """
    if budget not in BUDGET_CONFIGS:
        raise ValueError(f"Unknown budget: {budget}. Must be one of {list(BUDGET_CONFIGS.keys())}")

    if arch_style not in BUDGET_CONFIGS[budget]:
        raise ValueError(f"Unknown arch_style: {arch_style}. Must be 'shallow' or 'deep'")

    # Get base architecture config
    arch_config = BUDGET_CONFIGS[budget][arch_style].copy()

    # Use default regularization if not provided
    if regularization is None:
        regularization = DEFAULT_REGULARIZATION.copy()

    # Merge regularization into config
    config = {
        **arch_config,
        **regularization,
    }

    return config


def generate_forced_configs(
    budgets: list[str] | None = None,
    include_reg_extremes: bool = True,
) -> list[dict[str, Any]]:
    """Generate all 18 forced extreme configurations.

    The configs are organized into 4 groups:
    - Group 1: Budget × Architecture (8 configs) - all 4 budgets × 2 arch styles
    - Group 2: Dropout extremes (4 configs) - on 2M and 200M budgets
    - Group 3: Learning rate extremes (3 configs) - on 2M and 20M budgets
    - Group 4: Weight decay extremes (3 configs) - on 2M and 200M budgets

    Args:
        budgets: Subset of budgets to include (default: all 4)
        include_reg_extremes: Whether to include regularization extreme configs

    Returns:
        List of 18 config dicts, each with name, budget, group, and all hyperparameters
    """
    if budgets is None:
        budgets = ["750k", "2M", "20M", "200M"]

    configs = []

    # Group 1: Budget × Architecture (8 configs with default regularization)
    for budget in budgets:
        for style in ["shallow", "deep"]:
            config = compute_budget_aware_extremes(budget, style)
            config["name"] = f"{budget}_{style}"
            config["budget"] = budget
            config["group"] = "budget_architecture"
            config["arch_style"] = style
            configs.append(config)

    if not include_reg_extremes:
        return configs

    # Group 2: Dropout extremes (4 configs on 2M and 200M)
    # Config 9-12 from the plan
    dropout_configs = [
        {"budget": "2M", "dropout": 0.1, "name": "2M_dropout_low"},
        {"budget": "2M", "dropout": 0.3, "name": "2M_dropout_midlow"},
        {"budget": "2M", "dropout": 0.7, "name": "2M_dropout_high"},
        {"budget": "200M", "dropout": 0.1, "name": "200M_dropout_low"},  # Cross-budget
    ]

    for dc in dropout_configs:
        if dc["budget"] in budgets:
            base = compute_budget_aware_extremes(dc["budget"], "shallow")
            base["dropout"] = dc["dropout"]
            base["learning_rate"] = 1e-4
            base["weight_decay"] = 1e-3
            base["name"] = dc["name"]
            base["budget"] = dc["budget"]
            base["group"] = "dropout_extremes"
            configs.append(base)

    # Group 3: Learning rate extremes (3 configs)
    # Config 13-15 from the plan
    lr_configs = [
        {"budget": "2M", "learning_rate": 1e-5, "name": "2M_lr_low"},
        {"budget": "2M", "learning_rate": 1e-3, "name": "2M_lr_high"},
        {"budget": "20M", "learning_rate": 1e-5, "name": "20M_lr_low"},  # Cross-budget
    ]

    for lc in lr_configs:
        if lc["budget"] in budgets:
            base = compute_budget_aware_extremes(lc["budget"], "shallow")
            base["learning_rate"] = lc["learning_rate"]
            base["dropout"] = 0.5
            base["weight_decay"] = 1e-3
            base["name"] = lc["name"]
            base["budget"] = lc["budget"]
            base["group"] = "lr_extremes"
            configs.append(base)

    # Group 4: Weight decay extremes (3 configs)
    # Config 16-18 from the plan
    wd_configs = [
        {"budget": "2M", "weight_decay": 0.0, "name": "2M_wd_none"},
        {"budget": "2M", "weight_decay": 1e-2, "name": "2M_wd_high"},
        {"budget": "200M", "weight_decay": 0.0, "name": "200M_wd_none"},  # Cross-budget
    ]

    for wc in wd_configs:
        if wc["budget"] in budgets:
            base = compute_budget_aware_extremes(wc["budget"], "shallow")
            base["weight_decay"] = wc["weight_decay"]
            base["dropout"] = 0.5
            base["learning_rate"] = 1e-4
            base["name"] = wc["name"]
            base["budget"] = wc["budget"]
            base["group"] = "wd_extremes"
            configs.append(base)

    return configs


def check_early_stopping_convergence(
    study: Any,  # optuna.Study
    top_n: int = 5,
    threshold: float = 0.02,
    min_trials: int = 20,
) -> bool:
    """Check if top-N trials have converged within threshold.

    Early stopping triggers when:
    1. At least min_trials have completed
    2. The range of top-N best values is <= threshold

    Args:
        study: Optuna study object
        top_n: Number of top trials to consider for convergence
        threshold: Maximum allowed range for convergence (in AUC units)
        min_trials: Minimum number of completed trials before checking

    Returns:
        True if top-N trials have converged, False otherwise
    """
    # Get completed trials
    completed_trials = [
        t for t in study.trials
        if t.state.name == "COMPLETE"
    ]

    # Need minimum trials before checking
    if len(completed_trials) < min_trials:
        return False

    # Need at least top_n trials
    if len(completed_trials) < top_n:
        return False

    # Get values and sort (lower is better since we minimize negative AUC)
    values = sorted([t.value for t in completed_trials if t.value is not None])

    # Get top-N (lowest values = highest AUC)
    top_values = values[:top_n]

    if len(top_values) < top_n:
        return False

    # Check if range is within threshold
    value_range = max(top_values) - min(top_values)

    # Note: values are negative AUC, so range of -0.70 to -0.68 = 0.02
    return value_range <= threshold
