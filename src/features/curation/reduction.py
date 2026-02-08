"""Iterative feature reduction module.

Implements the reduction algorithm:
1. BASELINE: Train model on full tier, record metrics
2. RANK: Compute multi-method importance scores
3. REMOVE: Remove bottom 10% (with category preservation)
4. VALIDATE: Retrain and check thresholds
5. Repeat until convergence

Category Preservation Minimums:
| Category          | Min Keep |
|-------------------|----------|
| Moving Averages   | 5        |
| Oscillators       | 4        |
| Momentum          | 3        |
| Volatility        | 3        |
| Volume            | 3        |
| Trend             | 2        |
| Risk Metrics      | 2        |
| Support/Resistance| 2        |
| Entropy/Regime    | 1        |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# =============================================================================
# Category Minimums (Features to preserve in each category)
# =============================================================================

CATEGORY_MINIMUMS = {
    "moving_averages": 5,
    "oscillators": 4,
    "momentum": 3,
    "volatility": 3,
    "volume": 3,
    "trend": 2,
    "risk_metrics": 2,
    "support_resistance": 2,
    "entropy_regime": 1,
    "calendar": 1,
    "candle": 1,
    "other": 1,
}


# =============================================================================
# Feature Categorization
# =============================================================================

# Keywords for category detection
CATEGORY_KEYWORDS = {
    "moving_averages": ["sma", "ema", "wma", "tema", "kama", "hma", "vwma", "ma_"],
    "oscillators": ["rsi", "stoch", "cci", "mfi", "williams", "cmo", "roc"],
    "momentum": ["macd", "roc_", "momentum", "ppo", "trix"],
    "volatility": ["atr", "bb_", "volatility", "keltner", "vol_", "std_"],
    "volume": ["volume", "obv", "ad_", "cmf", "vwap"],
    "trend": ["adx", "aroon", "trend", "psar"],
    "risk_metrics": ["sharpe", "sortino", "max_drawdown", "var_"],
    "support_resistance": ["pivot", "support", "resistance", "fib", "donchian"],
    "entropy_regime": ["entropy", "regime", "permutation"],
    "calendar": ["day_of_week", "is_monday", "is_friday", "month", "quarter"],
    "candle": ["candle", "body", "wick", "doji", "range_vs"],
}


def get_category_for_feature(feature_name: str) -> str:
    """Determine category for a single feature based on name patterns.

    Args:
        feature_name: Name of the feature.

    Returns:
        Category string (e.g., "moving_averages", "oscillators").
    """
    feature_lower = feature_name.lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in feature_lower:
                return category

    return "other"


def categorize_features(features: list[str]) -> dict[str, str]:
    """Categorize all features.

    Args:
        features: List of feature names.

    Returns:
        Dict mapping feature name to category.
    """
    return {f: get_category_for_feature(f) for f in features}


def count_features_per_category(features: list[str]) -> dict[str, int]:
    """Count features in each category.

    Args:
        features: List of feature names.

    Returns:
        Dict mapping category to count.
    """
    categories = categorize_features(features)
    counts: dict[str, int] = {}

    for category in categories.values():
        counts[category] = counts.get(category, 0) + 1

    return counts


# =============================================================================
# Feature Selection for Removal
# =============================================================================


def select_features_for_removal(
    features: list[str],
    importance_scores: pd.Series,
    removal_fraction: float = 0.10,
) -> list[str]:
    """Select features for removal based on importance scores.

    Respects category minimums - will not remove features that would
    drop a category below its minimum.

    Args:
        features: Current list of features.
        importance_scores: Series with importance score per feature.
        removal_fraction: Fraction of features to attempt to remove.

    Returns:
        List of features to remove.
    """
    n_to_remove = max(1, int(len(features) * removal_fraction))

    # Get current category counts
    category_counts = count_features_per_category(features)

    # Sort features by importance (lowest first = candidates for removal)
    sorted_features = importance_scores.loc[features].sort_values().index.tolist()

    to_remove = []
    feature_categories = categorize_features(features)

    for feature in sorted_features:
        if len(to_remove) >= n_to_remove:
            break

        category = feature_categories[feature]
        min_count = CATEGORY_MINIMUMS.get(category, 1)

        # Check if we can remove from this category
        if category_counts.get(category, 0) > min_count:
            to_remove.append(feature)
            category_counts[category] -= 1

    return to_remove


# =============================================================================
# Reduction Round Result
# =============================================================================


@dataclass
class ReductionRoundResult:
    """Result of a single reduction round."""

    removed_features: list[str]
    remaining_features: list[str]
    removal_fraction: float


def run_reduction_round(
    features: list[str],
    importance_scores: pd.Series,
    removal_fraction: float = 0.10,
) -> ReductionRoundResult:
    """Execute a single reduction round.

    Args:
        features: Current list of features.
        importance_scores: Series with importance score per feature.
        removal_fraction: Fraction of features to attempt to remove.

    Returns:
        ReductionRoundResult with removed and remaining features.
    """
    removed = select_features_for_removal(
        features=features,
        importance_scores=importance_scores,
        removal_fraction=removal_fraction,
    )

    remaining = [f for f in features if f not in removed]

    return ReductionRoundResult(
        removed_features=removed,
        remaining_features=remaining,
        removal_fraction=removal_fraction,
    )


# =============================================================================
# Reduction History
# =============================================================================


@dataclass
class ReductionHistory:
    """Tracks reduction history across multiple rounds."""

    rounds: list[ReductionRoundResult] = field(default_factory=list)

    def add_round(self, result: ReductionRoundResult) -> None:
        """Add a round result to history."""
        self.rounds.append(result)

    def get_all_removed(self) -> list[str]:
        """Get all features removed across all rounds."""
        removed = []
        for round_result in self.rounds:
            removed.extend(round_result.removed_features)
        return removed

    def get_current_features(self) -> list[str]:
        """Get the current remaining features after all rounds."""
        if not self.rounds:
            return []
        return self.rounds[-1].remaining_features


# =============================================================================
# Reduction Configuration
# =============================================================================


@dataclass
class ReductionConfig:
    """Configuration for iterative reduction."""

    initial_removal_fraction: float = 0.10
    fallback_removal_fraction: float = 0.05
    max_rounds: int = 20


# =============================================================================
# Output Formatting
# =============================================================================


def format_curated_features(
    features: list[str],
    tier: str,
    rounds: int,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Format curated features for JSON output.

    Args:
        features: List of curated feature names.
        tier: Original tier name (e.g., "a200").
        rounds: Number of reduction rounds performed.
        metrics: Optional final metrics dict.

    Returns:
        Dict suitable for JSON serialization.
    """
    result: dict[str, Any] = {
        "tier": tier,
        "curated_features": features,
        "feature_count": len(features),
        "reduction_rounds": rounds,
        "categories": count_features_per_category(features),
    }

    if metrics:
        result["final_metrics"] = metrics

    return result
