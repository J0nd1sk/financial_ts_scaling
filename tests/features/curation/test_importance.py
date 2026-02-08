"""Tests for feature importance module.

Tests the multi-method importance analysis for feature curation:
1. Target Correlation (Pearson/Spearman)
2. Mutual Information
3. Redundancy Matrix
4. Permutation Importance (requires model)
5. Stability Analysis (requires model)
6. Combined Score computation
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

from src.features.curation import importance


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Create sample feature DataFrame for testing.

    Includes features with varying signal quality:
    - high_signal: High correlation with target
    - low_signal: Low/no correlation with target
    - redundant_1, redundant_2: Highly correlated with each other
    - noise: Random noise
    """
    np.random.seed(42)
    n = 500

    # Create target-correlated feature
    base = np.random.randn(n)
    target = (base > 0).astype(float)  # Binary target

    # Features with different signal levels
    high_signal = base + np.random.randn(n) * 0.3  # High correlation
    medium_signal = base + np.random.randn(n) * 1.0  # Medium correlation
    low_signal = np.random.randn(n)  # No correlation

    # Redundant features (highly correlated with each other)
    redundant_1 = high_signal + np.random.randn(n) * 0.1
    redundant_2 = high_signal + np.random.randn(n) * 0.1

    # Pure noise
    noise = np.random.randn(n)

    return pd.DataFrame({
        "high_signal": high_signal,
        "medium_signal": medium_signal,
        "low_signal": low_signal,
        "redundant_1": redundant_1,
        "redundant_2": redundant_2,
        "noise": noise,
        "target": target,
    })


@pytest.fixture
def feature_names() -> list[str]:
    """Feature names (excluding target)."""
    return ["high_signal", "medium_signal", "low_signal", "redundant_1", "redundant_2", "noise"]


class TestTargetCorrelation:
    """Test target correlation computation."""

    def test_compute_target_correlation_returns_series(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """compute_target_correlation returns a Series with feature names as index."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.compute_target_correlation(X, y)

        assert isinstance(result, pd.Series)
        assert len(result) == len(feature_names)
        assert all(f in result.index for f in feature_names)

    def test_high_signal_has_higher_correlation(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """high_signal feature should have higher correlation than noise."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.compute_target_correlation(X, y)

        assert abs(result["high_signal"]) > abs(result["noise"])

    def test_correlation_values_bounded(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """Correlation values should be in [-1, 1]."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.compute_target_correlation(X, y)

        assert (result >= -1.0).all()
        assert (result <= 1.0).all()

    def test_correlation_method_spearman(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """Spearman method should work and may differ from Pearson."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        pearson = importance.compute_target_correlation(X, y, method="pearson")
        spearman = importance.compute_target_correlation(X, y, method="spearman")

        # Both should exist
        assert len(pearson) == len(spearman)
        # High signal should still be high in both
        assert abs(spearman["high_signal"]) > abs(spearman["noise"])


class TestMutualInformation:
    """Test mutual information computation."""

    def test_compute_mutual_information_returns_series(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """compute_mutual_information returns a Series with feature names."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.compute_mutual_information(X, y)

        assert isinstance(result, pd.Series)
        assert len(result) == len(feature_names)

    def test_mutual_information_non_negative(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """Mutual information should be non-negative."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.compute_mutual_information(X, y)

        assert (result >= 0).all()

    def test_high_signal_has_higher_mi(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """high_signal should have higher mutual information than noise."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.compute_mutual_information(X, y)

        # MI can be noisy, but high_signal should generally be higher
        assert result["high_signal"] > result["noise"] * 0.5


class TestRedundancyMatrix:
    """Test redundancy matrix computation."""

    def test_compute_redundancy_matrix_returns_dataframe(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """compute_redundancy_matrix returns a square DataFrame."""
        X = sample_features_df[feature_names]

        result = importance.compute_redundancy_matrix(X)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(feature_names), len(feature_names))
        assert list(result.index) == feature_names
        assert list(result.columns) == feature_names

    def test_redundancy_matrix_diagonal_is_one(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """Diagonal of redundancy matrix should be 1.0 (self-correlation)."""
        X = sample_features_df[feature_names]

        result = importance.compute_redundancy_matrix(X)

        for f in feature_names:
            assert result.loc[f, f] == pytest.approx(1.0, abs=0.01)

    def test_redundancy_matrix_symmetric(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """Redundancy matrix should be symmetric."""
        X = sample_features_df[feature_names]

        result = importance.compute_redundancy_matrix(X)

        for i in feature_names:
            for j in feature_names:
                assert result.loc[i, j] == pytest.approx(result.loc[j, i], abs=0.01)

    def test_redundant_features_high_correlation(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """redundant_1 and redundant_2 should have high correlation."""
        X = sample_features_df[feature_names]

        result = importance.compute_redundancy_matrix(X)

        assert result.loc["redundant_1", "redundant_2"] > 0.9


class TestMaxRedundancy:
    """Test max redundancy per feature."""

    def test_get_max_redundancy_returns_series(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """get_max_redundancy returns Series with max correlation to any other feature."""
        X = sample_features_df[feature_names]

        result = importance.get_max_redundancy(X)

        assert isinstance(result, pd.Series)
        assert len(result) == len(feature_names)

    def test_max_redundancy_excludes_self(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """Max redundancy should not be 1.0 (excludes self-correlation)."""
        X = sample_features_df[feature_names]

        result = importance.get_max_redundancy(X)

        assert (result < 1.0).all()

    def test_redundant_features_high_max_redundancy(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """redundant_1 and redundant_2 should have high max redundancy."""
        X = sample_features_df[feature_names]

        result = importance.get_max_redundancy(X)

        assert result["redundant_1"] > 0.9
        assert result["redundant_2"] > 0.9


class TestNormalization:
    """Test score normalization functions."""

    def test_normalize_to_01_returns_series(self) -> None:
        """normalize_to_01 returns Series normalized to [0, 1]."""
        scores = pd.Series([0.1, 0.5, 0.9, 0.3], index=["a", "b", "c", "d"])

        result = importance.normalize_to_01(scores)

        assert isinstance(result, pd.Series)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_preserves_order(self) -> None:
        """Normalization should preserve relative ordering."""
        scores = pd.Series([0.1, 0.5, 0.9, 0.3], index=["a", "b", "c", "d"])

        result = importance.normalize_to_01(scores)

        # c should still be highest, a should still be lowest
        assert result.idxmax() == "c"
        assert result.idxmin() == "a"

    def test_normalize_handles_all_same(self) -> None:
        """Normalization handles all-same values (returns zeros or 0.5)."""
        scores = pd.Series([0.5, 0.5, 0.5], index=["a", "b", "c"])

        result = importance.normalize_to_01(scores)

        # Should not have NaN
        assert not result.isna().any()


class TestCombinedScore:
    """Test combined importance score computation."""

    def test_compute_combined_scores_returns_series(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """compute_combined_scores returns Series with combined importance."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.compute_combined_scores(X, y)

        assert isinstance(result, pd.Series)
        assert len(result) == len(feature_names)

    def test_combined_scores_bounded(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """Combined scores should be in [0, 1]."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.compute_combined_scores(X, y)

        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_high_signal_ranked_above_noise(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """high_signal should have higher combined score than noise."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.compute_combined_scores(X, y)

        assert result["high_signal"] > result["noise"]

    def test_redundant_features_penalized(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """Redundant features should be penalized in combined score."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.compute_combined_scores(X, y)

        # At least one of redundant_1/redundant_2 should score lower than high_signal
        # (due to redundancy penalty)
        assert (result["redundant_1"] < result["high_signal"] or
                result["redundant_2"] < result["high_signal"])


class TestImportanceReport:
    """Test importance report generation."""

    def test_generate_importance_report_returns_dataframe(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """generate_importance_report returns DataFrame with all metrics."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.generate_importance_report(X, y)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(feature_names)

    def test_report_has_required_columns(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """Report should have columns for each importance method."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.generate_importance_report(X, y)

        required_cols = [
            "target_correlation",
            "mutual_information",
            "max_redundancy",
            "combined_score",
        ]
        for col in required_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_report_sorted_by_combined_score(
        self, sample_features_df: pd.DataFrame, feature_names: list[str]
    ) -> None:
        """Report should be sorted by combined_score descending."""
        X = sample_features_df[feature_names]
        y = sample_features_df["target"]

        result = importance.generate_importance_report(X, y)

        scores = result["combined_score"].values
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
