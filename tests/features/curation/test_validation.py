"""Tests for reduction validation module.

Tests model performance validation during feature reduction:
- Precision maintenance (vs baseline)
- Recall maintenance (vs baseline)
- Prediction range maintenance
- AUC maintenance (vs baseline)
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.curation import validation


class TestValidationThresholds:
    """Test threshold constants."""

    def test_precision_thresholds_exist(self) -> None:
        """Precision thresholds are defined."""
        assert hasattr(validation, "PRECISION_MAINTAIN")
        assert hasattr(validation, "PRECISION_WARNING")
        assert hasattr(validation, "PRECISION_ABORT")

    def test_recall_thresholds_exist(self) -> None:
        """Recall thresholds are defined."""
        assert hasattr(validation, "RECALL_MAINTAIN")
        assert hasattr(validation, "RECALL_WARNING")
        assert hasattr(validation, "RECALL_ABORT")

    def test_pred_range_thresholds_exist(self) -> None:
        """Prediction range thresholds are defined."""
        assert hasattr(validation, "PRED_RANGE_MAINTAIN")
        assert hasattr(validation, "PRED_RANGE_WARNING")
        assert hasattr(validation, "PRED_RANGE_ABORT")

    def test_auc_thresholds_exist(self) -> None:
        """AUC thresholds are defined."""
        assert hasattr(validation, "AUC_MAINTAIN")
        assert hasattr(validation, "AUC_WARNING")
        assert hasattr(validation, "AUC_ABORT")

    def test_threshold_ordering(self) -> None:
        """Thresholds should be ordered: MAINTAIN > WARNING >= ABORT.

        Note: ABORT equals WARNING because anything below WARNING triggers ABORT.
        """
        # For all metrics, higher is better, so MAINTAIN > WARNING
        assert validation.PRECISION_MAINTAIN > validation.PRECISION_WARNING
        assert validation.PRECISION_WARNING >= validation.PRECISION_ABORT
        assert validation.RECALL_MAINTAIN > validation.RECALL_WARNING
        assert validation.RECALL_WARNING >= validation.RECALL_ABORT
        assert validation.PRED_RANGE_MAINTAIN > validation.PRED_RANGE_WARNING
        assert validation.PRED_RANGE_WARNING >= validation.PRED_RANGE_ABORT
        assert validation.AUC_MAINTAIN > validation.AUC_WARNING
        assert validation.AUC_WARNING >= validation.AUC_ABORT


class TestValidationResult:
    """Test ValidationResult enum."""

    def test_validation_result_values_exist(self) -> None:
        """ValidationResult enum has expected values."""
        assert hasattr(validation, "ValidationResult")
        assert hasattr(validation.ValidationResult, "MAINTAIN")
        assert hasattr(validation.ValidationResult, "WARNING")
        assert hasattr(validation.ValidationResult, "ABORT")


class TestPrecisionValidation:
    """Test precision validation."""

    def test_high_precision_ratio_maintains(self) -> None:
        """Precision >= 95% of baseline should MAINTAIN."""
        result = validation.validate_precision(0.60, 0.60)  # 100%
        assert result == validation.ValidationResult.MAINTAIN

        result = validation.validate_precision(0.57, 0.60)  # 95%
        assert result == validation.ValidationResult.MAINTAIN

    def test_medium_precision_ratio_warns(self) -> None:
        """Precision 90-95% of baseline should WARNING."""
        result = validation.validate_precision(0.54, 0.60)  # 90%
        assert result == validation.ValidationResult.WARNING

    def test_low_precision_ratio_aborts(self) -> None:
        """Precision < 90% of baseline should ABORT."""
        result = validation.validate_precision(0.50, 0.60)  # ~83%
        assert result == validation.ValidationResult.ABORT

    def test_handles_zero_baseline(self) -> None:
        """Should handle zero baseline gracefully."""
        result = validation.validate_precision(0.0, 0.0)
        # When baseline is 0, any result is acceptable
        assert result in [
            validation.ValidationResult.MAINTAIN,
            validation.ValidationResult.WARNING,
        ]


class TestRecallValidation:
    """Test recall validation."""

    def test_high_recall_ratio_maintains(self) -> None:
        """Recall >= 90% of baseline should MAINTAIN."""
        result = validation.validate_recall(0.90, 1.0)  # 90%
        assert result == validation.ValidationResult.MAINTAIN

    def test_medium_recall_ratio_warns(self) -> None:
        """Recall 80-90% of baseline should WARNING."""
        result = validation.validate_recall(0.85, 1.0)  # 85%
        assert result == validation.ValidationResult.WARNING

    def test_low_recall_ratio_aborts(self) -> None:
        """Recall < 80% of baseline should ABORT."""
        result = validation.validate_recall(0.70, 1.0)  # 70%
        assert result == validation.ValidationResult.ABORT


class TestPredRangeValidation:
    """Test prediction range validation."""

    def test_good_pred_range_maintains(self) -> None:
        """pred_range >= 0.10 should MAINTAIN."""
        result = validation.validate_pred_range(0.15)
        assert result == validation.ValidationResult.MAINTAIN

    def test_medium_pred_range_warns(self) -> None:
        """pred_range 0.05-0.10 should WARNING."""
        result = validation.validate_pred_range(0.07)
        assert result == validation.ValidationResult.WARNING

    def test_collapsed_pred_range_aborts(self) -> None:
        """pred_range < 0.05 should ABORT (probability collapse)."""
        result = validation.validate_pred_range(0.03)
        assert result == validation.ValidationResult.ABORT


class TestAUCValidation:
    """Test AUC validation."""

    def test_high_auc_ratio_maintains(self) -> None:
        """AUC >= 95% of baseline should MAINTAIN."""
        result = validation.validate_auc(0.70, 0.73)  # ~96%
        assert result == validation.ValidationResult.MAINTAIN

    def test_medium_auc_ratio_warns(self) -> None:
        """AUC 90-95% of baseline should WARNING."""
        result = validation.validate_auc(0.67, 0.73)  # ~92%
        assert result == validation.ValidationResult.WARNING

    def test_low_auc_ratio_aborts(self) -> None:
        """AUC < 90% of baseline should ABORT."""
        result = validation.validate_auc(0.60, 0.73)  # ~82%
        assert result == validation.ValidationResult.ABORT


class TestValidateReduction:
    """Test overall reduction validation."""

    def test_validate_reduction_returns_dict(self) -> None:
        """validate_reduction returns a dict with results."""
        baseline = {
            "precision": 0.60,
            "recall": 0.70,
            "pred_range": 0.15,
            "auc": 0.75,
        }
        current = {
            "precision": 0.58,
            "recall": 0.68,
            "pred_range": 0.14,
            "auc": 0.73,
        }

        result = validation.validate_reduction(current, baseline)

        assert isinstance(result, dict)
        assert "precision_result" in result
        assert "recall_result" in result
        assert "pred_range_result" in result
        assert "auc_result" in result
        assert "overall_result" in result

    def test_all_maintain_gives_overall_maintain(self) -> None:
        """All MAINTAIN results should give MAINTAIN overall."""
        baseline = {
            "precision": 0.60,
            "recall": 0.70,
            "pred_range": 0.15,
            "auc": 0.75,
        }
        # All metrics at >= threshold of baseline
        current = {
            "precision": 0.60,
            "recall": 0.70,
            "pred_range": 0.15,
            "auc": 0.75,
        }

        result = validation.validate_reduction(current, baseline)

        assert result["overall_result"] == validation.ValidationResult.MAINTAIN

    def test_any_abort_gives_overall_abort(self) -> None:
        """Any ABORT result should give ABORT overall."""
        baseline = {
            "precision": 0.60,
            "recall": 0.70,
            "pred_range": 0.15,
            "auc": 0.75,
        }
        # Precision is way below threshold
        current = {
            "precision": 0.40,  # < 90% of 0.60
            "recall": 0.70,
            "pred_range": 0.15,
            "auc": 0.75,
        }

        result = validation.validate_reduction(current, baseline)

        assert result["overall_result"] == validation.ValidationResult.ABORT

    def test_no_abort_some_warning_gives_overall_warning(self) -> None:
        """No ABORT but some WARNING should give WARNING overall."""
        baseline = {
            "precision": 0.60,
            "recall": 0.70,
            "pred_range": 0.15,
            "auc": 0.75,
        }
        # Precision slightly below threshold but not ABORT
        current = {
            "precision": 0.55,  # ~92% of 0.60, WARNING
            "recall": 0.70,     # 100%, MAINTAIN
            "pred_range": 0.15, # MAINTAIN
            "auc": 0.75,        # 100%, MAINTAIN
        }

        result = validation.validate_reduction(current, baseline)

        assert result["overall_result"] == validation.ValidationResult.WARNING

    def test_result_includes_ratios(self) -> None:
        """Result should include ratio comparisons."""
        baseline = {
            "precision": 0.60,
            "recall": 0.70,
            "pred_range": 0.15,
            "auc": 0.75,
        }
        current = {
            "precision": 0.54,
            "recall": 0.63,
            "pred_range": 0.12,
            "auc": 0.68,
        }

        result = validation.validate_reduction(current, baseline)

        assert "precision_ratio" in result
        assert "recall_ratio" in result
        assert "auc_ratio" in result
        assert abs(result["precision_ratio"] - 0.90) < 0.01


class TestShouldContinueReduction:
    """Test decision logic for continuing reduction."""

    def test_maintain_allows_continue(self) -> None:
        """MAINTAIN result should allow continuing reduction."""
        result = {
            "overall_result": validation.ValidationResult.MAINTAIN,
        }
        assert validation.should_continue_reduction(result) is True

    def test_warning_allows_continue_with_smaller_step(self) -> None:
        """WARNING result should suggest smaller step but allow continue."""
        result = {
            "overall_result": validation.ValidationResult.WARNING,
        }
        # WARNING still allows continuation
        assert validation.should_continue_reduction(result) is True

    def test_abort_stops_reduction(self) -> None:
        """ABORT result should stop reduction."""
        result = {
            "overall_result": validation.ValidationResult.ABORT,
        }
        assert validation.should_continue_reduction(result) is False


class TestGetRecommendedReductionStep:
    """Test recommended reduction step size."""

    def test_maintain_recommends_full_step(self) -> None:
        """MAINTAIN should recommend full 10% reduction."""
        result = {
            "overall_result": validation.ValidationResult.MAINTAIN,
        }
        step = validation.get_recommended_reduction_step(result)
        assert step == 0.10

    def test_warning_recommends_smaller_step(self) -> None:
        """WARNING should recommend smaller 5% reduction."""
        result = {
            "overall_result": validation.ValidationResult.WARNING,
        }
        step = validation.get_recommended_reduction_step(result)
        assert step == 0.05

    def test_abort_recommends_no_step(self) -> None:
        """ABORT should recommend 0% (stop) reduction."""
        result = {
            "overall_result": validation.ValidationResult.ABORT,
        }
        step = validation.get_recommended_reduction_step(result)
        assert step == 0.0
