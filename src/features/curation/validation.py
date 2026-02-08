"""Reduction validation module.

Validates model performance during iterative feature reduction.
Compares current metrics to baseline and determines if reduction should continue.

Thresholds (vs baseline):
| Metric     | Maintain | Warning | Abort  |
|------------|----------|---------|--------|
| Precision  | >= 95%   | 90-95%  | < 90%  |
| Recall     | >= 90%   | 80-90%  | < 80%  |
| pred_range | >= 0.10  | 0.05-0.10| < 0.05|
| AUC        | >= 95%   | 90-95%  | < 90%  |
"""

from __future__ import annotations

from enum import Enum
from typing import Any


# =============================================================================
# Validation Thresholds (as ratios of baseline)
# =============================================================================

# Precision: primary metric, must maintain >= 95% of baseline
PRECISION_MAINTAIN = 0.95
PRECISION_WARNING = 0.90
PRECISION_ABORT = 0.90  # Abort threshold (below WARNING)

# Recall: must maintain >= 90% of baseline
RECALL_MAINTAIN = 0.90
RECALL_WARNING = 0.80
RECALL_ABORT = 0.80  # Abort threshold

# Prediction range: absolute threshold (detect probability collapse)
PRED_RANGE_MAINTAIN = 0.10
PRED_RANGE_WARNING = 0.05
PRED_RANGE_ABORT = 0.05  # Abort threshold

# AUC: must maintain >= 95% of baseline
AUC_MAINTAIN = 0.95
AUC_WARNING = 0.90
AUC_ABORT = 0.90  # Abort threshold


class ValidationResult(Enum):
    """Result of validation check."""

    MAINTAIN = "maintain"  # Continue reduction at full step
    WARNING = "warning"    # Continue with smaller step
    ABORT = "abort"        # Stop reduction, restore previous


# =============================================================================
# Individual Metric Validation
# =============================================================================


def validate_precision(current: float, baseline: float) -> ValidationResult:
    """Validate precision against baseline.

    Args:
        current: Current precision after reduction.
        baseline: Baseline precision before reduction.

    Returns:
        ValidationResult based on ratio to baseline.
    """
    if baseline == 0:
        # Edge case: if baseline is 0, any non-negative result is acceptable
        return ValidationResult.MAINTAIN if current >= 0 else ValidationResult.WARNING

    ratio = current / baseline

    if ratio >= PRECISION_MAINTAIN:
        return ValidationResult.MAINTAIN
    elif ratio >= PRECISION_WARNING:
        return ValidationResult.WARNING
    else:
        return ValidationResult.ABORT


def validate_recall(current: float, baseline: float) -> ValidationResult:
    """Validate recall against baseline.

    Args:
        current: Current recall after reduction.
        baseline: Baseline recall before reduction.

    Returns:
        ValidationResult based on ratio to baseline.
    """
    if baseline == 0:
        return ValidationResult.MAINTAIN if current >= 0 else ValidationResult.WARNING

    ratio = current / baseline

    if ratio >= RECALL_MAINTAIN:
        return ValidationResult.MAINTAIN
    elif ratio >= RECALL_WARNING:
        return ValidationResult.WARNING
    else:
        return ValidationResult.ABORT


def validate_pred_range(pred_range: float) -> ValidationResult:
    """Validate prediction range (detect probability collapse).

    Unlike other metrics, this is an absolute threshold, not relative to baseline.
    A collapsed prediction range (all predictions ~0.5) indicates model failure.

    Args:
        pred_range: Current prediction range (max - min of predictions).

    Returns:
        ValidationResult based on absolute threshold.
    """
    if pred_range >= PRED_RANGE_MAINTAIN:
        return ValidationResult.MAINTAIN
    elif pred_range >= PRED_RANGE_WARNING:
        return ValidationResult.WARNING
    else:
        return ValidationResult.ABORT


def validate_auc(current: float, baseline: float) -> ValidationResult:
    """Validate AUC against baseline.

    Args:
        current: Current AUC after reduction.
        baseline: Baseline AUC before reduction.

    Returns:
        ValidationResult based on ratio to baseline.
    """
    if baseline == 0:
        return ValidationResult.MAINTAIN if current >= 0 else ValidationResult.WARNING

    ratio = current / baseline

    if ratio >= AUC_MAINTAIN:
        return ValidationResult.MAINTAIN
    elif ratio >= AUC_WARNING:
        return ValidationResult.WARNING
    else:
        return ValidationResult.ABORT


# =============================================================================
# Overall Validation
# =============================================================================


def validate_reduction(
    current_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> dict[str, Any]:
    """Validate feature reduction by comparing current to baseline metrics.

    The overall result is:
    - ABORT if any individual metric is ABORT
    - WARNING if no ABORT but any metric is WARNING
    - MAINTAIN if all metrics are MAINTAIN

    Args:
        current_metrics: Dict with precision, recall, pred_range, auc after reduction.
        baseline_metrics: Dict with same metrics before reduction.

    Returns:
        Dict with individual results, ratios, and overall result.
    """
    # Validate each metric
    precision_result = validate_precision(
        current_metrics["precision"],
        baseline_metrics["precision"]
    )
    recall_result = validate_recall(
        current_metrics["recall"],
        baseline_metrics["recall"]
    )
    pred_range_result = validate_pred_range(current_metrics["pred_range"])
    auc_result = validate_auc(
        current_metrics["auc"],
        baseline_metrics["auc"]
    )

    # Compute ratios
    precision_ratio = (
        current_metrics["precision"] / baseline_metrics["precision"]
        if baseline_metrics["precision"] > 0 else 1.0
    )
    recall_ratio = (
        current_metrics["recall"] / baseline_metrics["recall"]
        if baseline_metrics["recall"] > 0 else 1.0
    )
    auc_ratio = (
        current_metrics["auc"] / baseline_metrics["auc"]
        if baseline_metrics["auc"] > 0 else 1.0
    )

    # Determine overall result
    all_results = [precision_result, recall_result, pred_range_result, auc_result]

    if any(r == ValidationResult.ABORT for r in all_results):
        overall = ValidationResult.ABORT
    elif any(r == ValidationResult.WARNING for r in all_results):
        overall = ValidationResult.WARNING
    else:
        overall = ValidationResult.MAINTAIN

    return {
        "precision_result": precision_result,
        "recall_result": recall_result,
        "pred_range_result": pred_range_result,
        "auc_result": auc_result,
        "precision_ratio": precision_ratio,
        "recall_ratio": recall_ratio,
        "auc_ratio": auc_ratio,
        "overall_result": overall,
    }


def should_continue_reduction(validation_result: dict[str, Any]) -> bool:
    """Determine if reduction should continue based on validation result.

    Args:
        validation_result: Result from validate_reduction.

    Returns:
        True if reduction should continue, False if it should stop.
    """
    overall = validation_result["overall_result"]
    return overall != ValidationResult.ABORT


def get_recommended_reduction_step(validation_result: dict[str, Any]) -> float:
    """Get recommended feature reduction step size based on validation.

    Args:
        validation_result: Result from validate_reduction.

    Returns:
        Recommended reduction step: 0.10 (10%), 0.05 (5%), or 0.0 (stop).
    """
    overall = validation_result["overall_result"]

    if overall == ValidationResult.MAINTAIN:
        return 0.10  # Full 10% reduction
    elif overall == ValidationResult.WARNING:
        return 0.05  # Smaller 5% reduction
    else:  # ABORT
        return 0.0   # Stop reduction
