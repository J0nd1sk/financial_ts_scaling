"""Tests for VIX feature engineering (tier c)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import tier_c_vix


@pytest.fixture()
def sample_vix_df() -> pd.DataFrame:
    """Create synthetic VIX data with realistic patterns."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-02", periods=200, freq="B")
    # Simulate VIX with mean ~18, std ~5, occasional spikes
    base_vix = 18 + 5 * np.random.randn(len(dates))
    # Add some spikes (simulate market stress)
    base_vix[50:55] = 35  # Spike to high regime
    base_vix[100:105] = 12  # Drop to low regime
    base_vix = np.clip(base_vix, 9, 80)  # VIX realistic bounds

    data = {
        "Date": dates,
        "Open": base_vix * 0.99,
        "High": base_vix * 1.02,
        "Low": base_vix * 0.98,
        "Close": base_vix,
        "Volume": np.nan,  # VIX has no volume
    }
    return pd.DataFrame(data)


class TestVIXFeaturesShape:
    """Test output shape and structure."""

    def test_vix_features_shape(self, sample_vix_df: pd.DataFrame) -> None:
        """Output has 9 columns: Date + 8 features."""
        result = tier_c_vix.build_vix_features(sample_vix_df)
        assert result.shape[1] == 9, f"Expected 9 columns, got {result.shape[1]}"

    def test_vix_features_columns(self, sample_vix_df: pd.DataFrame) -> None:
        """All 8 expected feature names present."""
        result = tier_c_vix.build_vix_features(sample_vix_df)
        expected_cols = ["Date"] + tier_c_vix.VIX_FEATURE_LIST
        assert result.columns.tolist() == expected_cols

    def test_vix_features_no_nan(self, sample_vix_df: pd.DataFrame) -> None:
        """No NaN in output after warmup rows dropped."""
        result = tier_c_vix.build_vix_features(sample_vix_df)
        nan_count = result.drop(columns=["vix_regime"]).isnull().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in numeric columns"
        # Regime should also have no NaN
        assert result["vix_regime"].isnull().sum() == 0

    def test_vix_features_row_count(self, sample_vix_df: pd.DataFrame) -> None:
        """Output has fewer rows than input due to warmup."""
        result = tier_c_vix.build_vix_features(sample_vix_df)
        assert len(result) < len(sample_vix_df), "Warmup rows should be dropped"
        # With 60-day lookback, expect ~140 rows from 200 input
        assert len(result) >= 100, f"Too few rows: {len(result)}"


class TestVIXFeatureRanges:
    """Test feature value ranges."""

    def test_vix_percentile_range(self, sample_vix_df: pd.DataFrame) -> None:
        """Percentile values in [0, 100]."""
        result = tier_c_vix.build_vix_features(sample_vix_df)
        percentile = result["vix_percentile_60d"]
        assert percentile.min() >= 0, f"Min percentile {percentile.min()} < 0"
        assert percentile.max() <= 100, f"Max percentile {percentile.max()} > 100"

    def test_vix_zscore_reasonable(self, sample_vix_df: pd.DataFrame) -> None:
        """95% of z-scores in [-4, 4] range."""
        result = tier_c_vix.build_vix_features(sample_vix_df)
        zscore = result["vix_zscore_20d"]
        within_range = ((zscore >= -4) & (zscore <= 4)).mean()
        assert within_range >= 0.95, f"Only {within_range:.1%} of z-scores in [-4, 4]"

    def test_vix_sma_values_bounded(self, sample_vix_df: pd.DataFrame) -> None:
        """SMA values between min and max of Close."""
        result = tier_c_vix.build_vix_features(sample_vix_df)
        close_min = sample_vix_df["Close"].min()
        close_max = sample_vix_df["Close"].max()

        for col in ["vix_sma_10", "vix_sma_20"]:
            assert result[col].min() >= close_min * 0.9, f"{col} min too low"
            assert result[col].max() <= close_max * 1.1, f"{col} max too high"


class TestVIXRegime:
    """Test VIX regime classification.

    Regime encoding: 0=low (<15), 1=normal (15-25), 2=high (>=25)
    """

    def test_vix_regime_categories(self, sample_vix_df: pd.DataFrame) -> None:
        """Only 0, 1, 2 values (low, normal, high)."""
        result = tier_c_vix.build_vix_features(sample_vix_df)
        unique_regimes = set(result["vix_regime"].unique())
        expected = {0, 1, 2}  # low=0, normal=1, high=2
        assert unique_regimes.issubset(expected), f"Unexpected regimes: {unique_regimes - expected}"

    def test_vix_regime_boundary_25(self) -> None:
        """VIX=25 should be classified as high (2)."""
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-02", periods=100, freq="B"),
            "Open": [25.0] * 100,
            "High": [26.0] * 100,
            "Low": [24.0] * 100,
            "Close": [25.0] * 100,
            "Volume": [np.nan] * 100,
        })
        result = tier_c_vix.build_vix_features(df)
        # All regime values should be 2 (high) when VIX >= 25
        assert (result["vix_regime"] == 2).all()

    def test_vix_regime_boundary_15(self) -> None:
        """VIX=15 should be classified as normal (1)."""
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-02", periods=100, freq="B"),
            "Open": [15.0] * 100,
            "High": [16.0] * 100,
            "Low": [14.0] * 100,
            "Close": [15.0] * 100,
            "Volume": [np.nan] * 100,
        })
        result = tier_c_vix.build_vix_features(df)
        # VIX=15 is at boundary, should be 1 (normal) (15 <= x < 25)
        assert (result["vix_regime"] == 1).all()

    def test_vix_regime_low(self) -> None:
        """VIX=12 should be classified as low (0)."""
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-02", periods=100, freq="B"),
            "Open": [12.0] * 100,
            "High": [13.0] * 100,
            "Low": [11.0] * 100,
            "Close": [12.0] * 100,
            "Volume": [np.nan] * 100,
        })
        result = tier_c_vix.build_vix_features(df)
        # VIX < 15 should be 0 (low)
        assert (result["vix_regime"] == 0).all()


class TestVIXChangeCalculations:
    """Test change/momentum calculations."""

    def test_vix_change_1d_calculation(self) -> None:
        """Verify 1-day change formula: (close - lag1) / lag1 * 100."""
        # Use 150 rows with transition at index 80 (after 60-day warmup)
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-02", periods=150, freq="B"),
            "Open": [20.0] * 150,
            "High": [21.0] * 150,
            "Low": [19.0] * 150,
            "Close": [20.0] * 80 + [22.0] * 70,  # Jump from 20 to 22 at index 80
            "Volume": [np.nan] * 150,
        })
        result = tier_c_vix.build_vix_features(df)

        # At the jump point (index 80 in input), change should be (22-20)/20 * 100 = 10%
        # After 60-row warmup drop, this appears at output index ~20
        change_values = result["vix_change_1d"].values
        # The 10% change should appear somewhere in the output
        assert any(abs(v - 10.0) < 0.1 for v in change_values), "Expected 10% change not found"

    def test_vix_change_5d_calculation(self) -> None:
        """Verify 5-day change is computed correctly."""
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-02", periods=100, freq="B"),
            "Open": [20.0] * 100,
            "High": [21.0] * 100,
            "Low": [19.0] * 100,
            "Close": list(range(20, 120)),  # Linear increase
            "Volume": [np.nan] * 100,
        })
        result = tier_c_vix.build_vix_features(df)
        # 5-day change should be positive throughout (prices increasing)
        assert (result["vix_change_5d"] > 0).all()


class TestVIXEdgeCases:
    """Test edge cases and error handling."""

    def test_vix_zscore_handles_zero_std(self) -> None:
        """No crash when std=0 (constant values)."""
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-02", periods=100, freq="B"),
            "Open": [20.0] * 100,
            "High": [21.0] * 100,
            "Low": [19.0] * 100,
            "Close": [20.0] * 100,  # Constant - std = 0
            "Volume": [np.nan] * 100,
        })
        # Should not raise, should return valid result
        result = tier_c_vix.build_vix_features(df)
        assert not result["vix_zscore_20d"].isnull().any()
        # With constant values, z-score should be 0 (or handled gracefully)
        assert (result["vix_zscore_20d"] == 0).all()

    def test_vix_close_preserved(self, sample_vix_df: pd.DataFrame) -> None:
        """vix_close should match input Close values (for overlapping dates)."""
        result = tier_c_vix.build_vix_features(sample_vix_df)
        # Get dates that appear in both
        result_dates = set(result["Date"])
        input_subset = sample_vix_df[sample_vix_df["Date"].isin(result_dates)]

        # Merge and compare
        merged = result.merge(input_subset[["Date", "Close"]], on="Date", suffixes=("_feat", "_raw"))
        assert np.allclose(merged["vix_close"], merged["Close"])
