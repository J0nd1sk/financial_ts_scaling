"""Feature quality assessment module.

Provides pass/fail/marginal thresholds for feature quality metrics:
- Target Correlation: >= 0.02 (PASS), 0.01-0.02 (MARGINAL), < 0.01 (FAIL)
- Mutual Information: >= 0.001 (PASS), 0.0005-0.001 (MARGINAL), < 0.0005 (FAIL)
- Max Redundancy: < 0.90 (PASS), 0.90-0.95 (MARGINAL), >= 0.95 (FAIL)
- Temporal Stability CV: < 0.5 (PASS), 0.5-0.8 (MARGINAL), > 0.8 (FAIL)
- Permutation Precision Drop: >= 0.002 (PASS), 0.001-0.002 (MARGINAL), < 0.001 (FAIL)
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd


# =============================================================================
# Quality Thresholds
# =============================================================================

# Target Correlation (absolute value) - higher is better
TARGET_CORRELATION_PASS = 0.02
TARGET_CORRELATION_MARGINAL = 0.01

# Mutual Information - higher is better
MUTUAL_INFORMATION_PASS = 0.001
MUTUAL_INFORMATION_MARGINAL = 0.0005

# Max Redundancy - lower is better (highly redundant = bad)
MAX_REDUNDANCY_PASS = 0.90
MAX_REDUNDANCY_MARGINAL = 0.95

# Temporal Stability CV (coefficient of variation) - lower is better
TEMPORAL_STABILITY_PASS = 0.5
TEMPORAL_STABILITY_MARGINAL = 0.8

# Permutation Precision Drop - higher is better (removing feature hurts precision)
PRECISION_DROP_PASS = 0.002
PRECISION_DROP_MARGINAL = 0.001


class QualityGrade(Enum):
    """Quality grade for a feature metric."""

    PASS = "pass"
    MARGINAL = "marginal"
    FAIL = "fail"

    def __lt__(self, other: "QualityGrade") -> bool:
        """Allow comparison for sorting: FAIL < MARGINAL < PASS."""
        order = {QualityGrade.FAIL: 0, QualityGrade.MARGINAL: 1, QualityGrade.PASS: 2}
        return order[self] < order[other]

    def __le__(self, other: "QualityGrade") -> bool:
        """Allow comparison for filtering."""
        return self < other or self == other


# =============================================================================
# Individual Metric Grading Functions
# =============================================================================


def grade_target_correlation(correlation: float) -> QualityGrade:
    """Grade target correlation value.

    Uses absolute value since negative correlation can be just as useful.

    Args:
        correlation: Correlation value (or absolute correlation).

    Returns:
        QualityGrade.PASS if >= 0.02
        QualityGrade.MARGINAL if 0.01 <= x < 0.02
        QualityGrade.FAIL if < 0.01
    """
    abs_corr = abs(correlation)

    if abs_corr >= TARGET_CORRELATION_PASS:
        return QualityGrade.PASS
    elif abs_corr >= TARGET_CORRELATION_MARGINAL:
        return QualityGrade.MARGINAL
    else:
        return QualityGrade.FAIL


def grade_mutual_information(mi: float) -> QualityGrade:
    """Grade mutual information value.

    Args:
        mi: Mutual information value.

    Returns:
        QualityGrade.PASS if >= 0.001
        QualityGrade.MARGINAL if 0.0005 <= x < 0.001
        QualityGrade.FAIL if < 0.0005
    """
    if mi >= MUTUAL_INFORMATION_PASS:
        return QualityGrade.PASS
    elif mi >= MUTUAL_INFORMATION_MARGINAL:
        return QualityGrade.MARGINAL
    else:
        return QualityGrade.FAIL


def grade_max_redundancy(redundancy: float) -> QualityGrade:
    """Grade max redundancy value.

    Lower is better - high redundancy means the feature is nearly a duplicate.

    Args:
        redundancy: Max correlation with any other feature.

    Returns:
        QualityGrade.PASS if < 0.90
        QualityGrade.MARGINAL if 0.90 <= x < 0.95
        QualityGrade.FAIL if >= 0.95
    """
    if redundancy < MAX_REDUNDANCY_PASS:
        return QualityGrade.PASS
    elif redundancy < MAX_REDUNDANCY_MARGINAL:
        return QualityGrade.MARGINAL
    else:
        return QualityGrade.FAIL


def grade_temporal_stability(cv: float) -> QualityGrade:
    """Grade temporal stability (coefficient of variation).

    Lower CV is better - stable importance across time.

    Args:
        cv: Coefficient of variation of importance across time periods.

    Returns:
        QualityGrade.PASS if < 0.5
        QualityGrade.MARGINAL if 0.5 <= x < 0.8
        QualityGrade.FAIL if >= 0.8
    """
    if cv < TEMPORAL_STABILITY_PASS:
        return QualityGrade.PASS
    elif cv < TEMPORAL_STABILITY_MARGINAL:
        return QualityGrade.MARGINAL
    else:
        return QualityGrade.FAIL


def grade_precision_drop(drop: float) -> QualityGrade:
    """Grade permutation importance (precision drop).

    Higher drop is better - removing the feature hurts model precision.

    Args:
        drop: Precision drop when feature is permuted.

    Returns:
        QualityGrade.PASS if >= 0.002
        QualityGrade.MARGINAL if 0.001 <= x < 0.002
        QualityGrade.FAIL if < 0.001
    """
    if drop >= PRECISION_DROP_PASS:
        return QualityGrade.PASS
    elif drop >= PRECISION_DROP_MARGINAL:
        return QualityGrade.MARGINAL
    else:
        return QualityGrade.FAIL


# =============================================================================
# Overall Quality Assessment
# =============================================================================


def assess_feature_quality(
    target_correlation: float,
    mutual_information: float,
    max_redundancy: float,
    temporal_stability_cv: float | None = None,
    precision_drop: float | None = None,
) -> dict[str, Any]:
    """Assess overall quality of a single feature.

    The overall verdict is:
    - FAIL if any metric is FAIL
    - MARGINAL if no FAIL but any metric is MARGINAL
    - PASS if all metrics are PASS

    Args:
        target_correlation: Correlation with target.
        mutual_information: MI with target.
        max_redundancy: Max correlation with any other feature.
        temporal_stability_cv: Optional CV of importance over time.
        precision_drop: Optional precision drop from permutation importance.

    Returns:
        Dict with individual grades and overall verdict.
    """
    grades = {
        "target_correlation_grade": grade_target_correlation(target_correlation),
        "mutual_information_grade": grade_mutual_information(mutual_information),
        "max_redundancy_grade": grade_max_redundancy(max_redundancy),
    }

    if temporal_stability_cv is not None:
        grades["temporal_stability_grade"] = grade_temporal_stability(temporal_stability_cv)

    if precision_drop is not None:
        grades["precision_drop_grade"] = grade_precision_drop(precision_drop)

    # Determine overall verdict
    all_grades = list(grades.values())

    if any(g == QualityGrade.FAIL for g in all_grades):
        overall = QualityGrade.FAIL
    elif any(g == QualityGrade.MARGINAL for g in all_grades):
        overall = QualityGrade.MARGINAL
    else:
        overall = QualityGrade.PASS

    grades["overall_verdict"] = overall
    return grades


def assess_all_features(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Assess quality of all features in a DataFrame.

    Args:
        metrics_df: DataFrame with columns:
            - target_correlation
            - mutual_information
            - max_redundancy
            - temporal_stability_cv (optional)
            - precision_drop (optional)
            Index should be feature names.

    Returns:
        DataFrame with original metrics plus grade columns and overall_verdict.
    """
    results = []

    for feature_name in metrics_df.index:
        row = metrics_df.loc[feature_name]

        assessment = assess_feature_quality(
            target_correlation=row["target_correlation"],
            mutual_information=row["mutual_information"],
            max_redundancy=row["max_redundancy"],
            temporal_stability_cv=row.get("temporal_stability_cv"),
            precision_drop=row.get("precision_drop"),
        )

        assessment["feature"] = feature_name
        results.append(assessment)

    result_df = pd.DataFrame(results).set_index("feature")
    return result_df


def filter_features(
    metrics_df: pd.DataFrame,
    min_grade: QualityGrade = QualityGrade.PASS,
) -> list[str]:
    """Filter features by minimum quality grade.

    Args:
        metrics_df: DataFrame with quality metrics (see assess_all_features).
        min_grade: Minimum acceptable grade (PASS, MARGINAL, or FAIL).

    Returns:
        List of feature names meeting the quality threshold.
    """
    assessed = assess_all_features(metrics_df)

    # Filter based on min_grade
    passing_features = []
    for feature_name in assessed.index:
        verdict = assessed.loc[feature_name, "overall_verdict"]
        if verdict >= min_grade:
            passing_features.append(feature_name)

    return passing_features


def generate_quality_report(metrics_df: pd.DataFrame) -> dict[str, Any]:
    """Generate summary quality report for all features.

    Args:
        metrics_df: DataFrame with quality metrics.

    Returns:
        Dict with summary statistics and feature lists by grade.
    """
    assessed = assess_all_features(metrics_df)

    verdicts = assessed["overall_verdict"]

    pass_features = [f for f in assessed.index if verdicts[f] == QualityGrade.PASS]
    marginal_features = [f for f in assessed.index if verdicts[f] == QualityGrade.MARGINAL]
    fail_features = [f for f in assessed.index if verdicts[f] == QualityGrade.FAIL]

    total = len(assessed)
    pass_count = len(pass_features)

    return {
        "total_features": total,
        "pass_count": pass_count,
        "marginal_count": len(marginal_features),
        "fail_count": len(fail_features),
        "pass_rate": pass_count / total if total > 0 else 0.0,
        "passing_features": pass_features,
        "marginal_features": marginal_features,
        "failing_features": fail_features,
    }
