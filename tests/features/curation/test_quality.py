"""Tests for feature quality assessment module.

Tests pass/fail/marginal thresholds for:
- Target Correlation
- Mutual Information
- Max Redundancy
- Temporal Stability CV
- Permutation Precision Drop
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.curation import quality


class TestQualityThresholds:
    """Test threshold constants."""

    def test_target_correlation_thresholds_exist(self) -> None:
        """Target correlation thresholds are defined."""
        assert hasattr(quality, "TARGET_CORRELATION_PASS")
        assert hasattr(quality, "TARGET_CORRELATION_MARGINAL")

    def test_mutual_information_thresholds_exist(self) -> None:
        """Mutual information thresholds are defined."""
        assert hasattr(quality, "MUTUAL_INFORMATION_PASS")
        assert hasattr(quality, "MUTUAL_INFORMATION_MARGINAL")

    def test_max_redundancy_thresholds_exist(self) -> None:
        """Max redundancy thresholds are defined."""
        assert hasattr(quality, "MAX_REDUNDANCY_PASS")
        assert hasattr(quality, "MAX_REDUNDANCY_MARGINAL")

    def test_temporal_stability_thresholds_exist(self) -> None:
        """Temporal stability thresholds are defined."""
        assert hasattr(quality, "TEMPORAL_STABILITY_PASS")
        assert hasattr(quality, "TEMPORAL_STABILITY_MARGINAL")

    def test_precision_drop_thresholds_exist(self) -> None:
        """Precision drop thresholds are defined."""
        assert hasattr(quality, "PRECISION_DROP_PASS")
        assert hasattr(quality, "PRECISION_DROP_MARGINAL")

    def test_threshold_ordering(self) -> None:
        """Thresholds should be ordered correctly (PASS > MARGINAL)."""
        # For metrics where higher is better
        assert quality.TARGET_CORRELATION_PASS >= quality.TARGET_CORRELATION_MARGINAL
        assert quality.MUTUAL_INFORMATION_PASS >= quality.MUTUAL_INFORMATION_MARGINAL
        assert quality.PRECISION_DROP_PASS >= quality.PRECISION_DROP_MARGINAL

        # For max_redundancy, lower is better (PASS < MARGINAL)
        assert quality.MAX_REDUNDANCY_PASS <= quality.MAX_REDUNDANCY_MARGINAL

        # For temporal stability CV, lower is better (PASS < MARGINAL)
        assert quality.TEMPORAL_STABILITY_PASS <= quality.TEMPORAL_STABILITY_MARGINAL


class TestQualityGrade:
    """Test quality grade enum."""

    def test_quality_grades_exist(self) -> None:
        """QualityGrade enum has PASS, MARGINAL, FAIL values."""
        assert hasattr(quality, "QualityGrade")
        assert hasattr(quality.QualityGrade, "PASS")
        assert hasattr(quality.QualityGrade, "MARGINAL")
        assert hasattr(quality.QualityGrade, "FAIL")


class TestGradeCorrelation:
    """Test correlation grading."""

    def test_high_correlation_passes(self) -> None:
        """Correlation >= 0.02 should PASS."""
        grade = quality.grade_target_correlation(0.05)
        assert grade == quality.QualityGrade.PASS

    def test_medium_correlation_marginal(self) -> None:
        """Correlation between 0.01 and 0.02 should be MARGINAL."""
        grade = quality.grade_target_correlation(0.015)
        assert grade == quality.QualityGrade.MARGINAL

    def test_low_correlation_fails(self) -> None:
        """Correlation < 0.01 should FAIL."""
        grade = quality.grade_target_correlation(0.005)
        assert grade == quality.QualityGrade.FAIL

    def test_uses_absolute_value(self) -> None:
        """Negative correlation should use absolute value."""
        grade = quality.grade_target_correlation(-0.05)
        assert grade == quality.QualityGrade.PASS


class TestGradeMutualInformation:
    """Test mutual information grading."""

    def test_high_mi_passes(self) -> None:
        """MI >= 0.001 should PASS."""
        grade = quality.grade_mutual_information(0.01)
        assert grade == quality.QualityGrade.PASS

    def test_medium_mi_marginal(self) -> None:
        """MI between 0.0005 and 0.001 should be MARGINAL."""
        grade = quality.grade_mutual_information(0.0007)
        assert grade == quality.QualityGrade.MARGINAL

    def test_low_mi_fails(self) -> None:
        """MI < 0.0005 should FAIL."""
        grade = quality.grade_mutual_information(0.0001)
        assert grade == quality.QualityGrade.FAIL


class TestGradeRedundancy:
    """Test redundancy grading."""

    def test_low_redundancy_passes(self) -> None:
        """Redundancy < 0.90 should PASS."""
        grade = quality.grade_max_redundancy(0.7)
        assert grade == quality.QualityGrade.PASS

    def test_medium_redundancy_marginal(self) -> None:
        """Redundancy between 0.90 and 0.95 should be MARGINAL."""
        grade = quality.grade_max_redundancy(0.92)
        assert grade == quality.QualityGrade.MARGINAL

    def test_high_redundancy_fails(self) -> None:
        """Redundancy >= 0.95 should FAIL."""
        grade = quality.grade_max_redundancy(0.98)
        assert grade == quality.QualityGrade.FAIL


class TestGradeTemporalStability:
    """Test temporal stability grading."""

    def test_stable_passes(self) -> None:
        """CV < 0.5 should PASS."""
        grade = quality.grade_temporal_stability(0.3)
        assert grade == quality.QualityGrade.PASS

    def test_medium_stability_marginal(self) -> None:
        """CV between 0.5 and 0.8 should be MARGINAL."""
        grade = quality.grade_temporal_stability(0.6)
        assert grade == quality.QualityGrade.MARGINAL

    def test_unstable_fails(self) -> None:
        """CV > 0.8 should FAIL."""
        grade = quality.grade_temporal_stability(1.0)
        assert grade == quality.QualityGrade.FAIL


class TestGradePrecisionDrop:
    """Test precision drop grading."""

    def test_high_importance_passes(self) -> None:
        """Precision drop >= 0.002 should PASS."""
        grade = quality.grade_precision_drop(0.01)
        assert grade == quality.QualityGrade.PASS

    def test_medium_importance_marginal(self) -> None:
        """Precision drop between 0.001 and 0.002 should be MARGINAL."""
        grade = quality.grade_precision_drop(0.0015)
        assert grade == quality.QualityGrade.MARGINAL

    def test_low_importance_fails(self) -> None:
        """Precision drop < 0.001 should FAIL."""
        grade = quality.grade_precision_drop(0.0005)
        assert grade == quality.QualityGrade.FAIL


@pytest.fixture
def sample_quality_metrics() -> pd.DataFrame:
    """Create sample quality metrics for multiple features."""
    return pd.DataFrame({
        "target_correlation": [0.05, 0.015, 0.005, 0.03],
        "mutual_information": [0.01, 0.0007, 0.0001, 0.005],
        "max_redundancy": [0.7, 0.92, 0.98, 0.5],
    }, index=["good_feature", "marginal_feature", "bad_feature", "great_feature"])


class TestAssessFeatureQuality:
    """Test overall feature quality assessment."""

    def test_assess_single_feature_returns_dict(
        self, sample_quality_metrics: pd.DataFrame
    ) -> None:
        """assess_feature_quality returns a dict with grades and verdict."""
        result = quality.assess_feature_quality(
            target_correlation=0.05,
            mutual_information=0.01,
            max_redundancy=0.7,
        )

        assert isinstance(result, dict)
        assert "target_correlation_grade" in result
        assert "mutual_information_grade" in result
        assert "max_redundancy_grade" in result
        assert "overall_verdict" in result

    def test_all_pass_gives_pass_verdict(self) -> None:
        """All PASS grades should give PASS verdict."""
        result = quality.assess_feature_quality(
            target_correlation=0.05,
            mutual_information=0.01,
            max_redundancy=0.5,
        )

        assert result["overall_verdict"] == quality.QualityGrade.PASS

    def test_any_fail_gives_fail_verdict(self) -> None:
        """Any FAIL grade should give FAIL verdict."""
        result = quality.assess_feature_quality(
            target_correlation=0.001,  # FAIL
            mutual_information=0.01,    # PASS
            max_redundancy=0.5,         # PASS
        )

        assert result["overall_verdict"] == quality.QualityGrade.FAIL

    def test_no_fail_some_marginal_gives_marginal_verdict(self) -> None:
        """No FAIL but some MARGINAL should give MARGINAL verdict."""
        result = quality.assess_feature_quality(
            target_correlation=0.015,   # MARGINAL
            mutual_information=0.01,    # PASS
            max_redundancy=0.5,         # PASS
        )

        assert result["overall_verdict"] == quality.QualityGrade.MARGINAL


class TestAssessMultipleFeatures:
    """Test batch feature quality assessment."""

    def test_assess_all_features_returns_dataframe(
        self, sample_quality_metrics: pd.DataFrame
    ) -> None:
        """assess_all_features returns DataFrame with grades."""
        result = quality.assess_all_features(sample_quality_metrics)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_quality_metrics)

    def test_assess_all_features_has_grade_columns(
        self, sample_quality_metrics: pd.DataFrame
    ) -> None:
        """Result should have grade columns for each metric."""
        result = quality.assess_all_features(sample_quality_metrics)

        assert "target_correlation_grade" in result.columns
        assert "mutual_information_grade" in result.columns
        assert "max_redundancy_grade" in result.columns
        assert "overall_verdict" in result.columns

    def test_assess_all_features_correct_verdicts(
        self, sample_quality_metrics: pd.DataFrame
    ) -> None:
        """Verdicts should be correct for each feature."""
        result = quality.assess_all_features(sample_quality_metrics)

        # good_feature: all PASS
        assert result.loc["good_feature", "overall_verdict"] == quality.QualityGrade.PASS

        # marginal_feature: some MARGINAL
        assert result.loc["marginal_feature", "overall_verdict"] == quality.QualityGrade.MARGINAL

        # bad_feature: has FAIL
        assert result.loc["bad_feature", "overall_verdict"] == quality.QualityGrade.FAIL


class TestFilterByQuality:
    """Test filtering features by quality."""

    def test_filter_passing_features(
        self, sample_quality_metrics: pd.DataFrame
    ) -> None:
        """filter_features should return only PASS features."""
        result = quality.filter_features(
            sample_quality_metrics,
            min_grade=quality.QualityGrade.PASS
        )

        assert "good_feature" in result
        assert "great_feature" in result
        assert "marginal_feature" not in result
        assert "bad_feature" not in result

    def test_filter_marginal_and_above(
        self, sample_quality_metrics: pd.DataFrame
    ) -> None:
        """filter_features with MARGINAL min should include PASS and MARGINAL."""
        result = quality.filter_features(
            sample_quality_metrics,
            min_grade=quality.QualityGrade.MARGINAL
        )

        assert "good_feature" in result
        assert "great_feature" in result
        assert "marginal_feature" in result
        assert "bad_feature" not in result

    def test_filter_returns_list(
        self, sample_quality_metrics: pd.DataFrame
    ) -> None:
        """filter_features returns a list of feature names."""
        result = quality.filter_features(
            sample_quality_metrics,
            min_grade=quality.QualityGrade.PASS
        )

        assert isinstance(result, list)


class TestQualityReport:
    """Test quality report generation."""

    def test_generate_quality_report_returns_dict(
        self, sample_quality_metrics: pd.DataFrame
    ) -> None:
        """generate_quality_report returns a summary dict."""
        result = quality.generate_quality_report(sample_quality_metrics)

        assert isinstance(result, dict)
        assert "total_features" in result
        assert "pass_count" in result
        assert "marginal_count" in result
        assert "fail_count" in result
        assert "pass_rate" in result

    def test_generate_quality_report_correct_counts(
        self, sample_quality_metrics: pd.DataFrame
    ) -> None:
        """Report should have correct counts."""
        result = quality.generate_quality_report(sample_quality_metrics)

        assert result["total_features"] == 4
        assert result["pass_count"] == 2  # good_feature, great_feature
        assert result["marginal_count"] == 1  # marginal_feature
        assert result["fail_count"] == 1  # bad_feature

    def test_generate_quality_report_includes_feature_lists(
        self, sample_quality_metrics: pd.DataFrame
    ) -> None:
        """Report should include lists of features by grade."""
        result = quality.generate_quality_report(sample_quality_metrics)

        assert "passing_features" in result
        assert "marginal_features" in result
        assert "failing_features" in result

        assert "good_feature" in result["passing_features"]
        assert "marginal_feature" in result["marginal_features"]
        assert "bad_feature" in result["failing_features"]
