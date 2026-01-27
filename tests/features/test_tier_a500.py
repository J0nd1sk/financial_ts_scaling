"""Tests for tier_a500 indicator module (indicators 207-500).

Sub-Chunk 6a (rank 207-230): MA Extended Part 1 (~24 features)
- sma_5, sma_14, sma_21, sma_63 - New SMA periods
- ema_5, ema_9, ema_50, ema_100, ema_200 - New EMA periods
- sma_5_slope, sma_21_slope, sma_63_slope - SMA slopes (5-day change)
- ema_9_slope, ema_50_slope, ema_100_slope - EMA slopes (5-day change)
- price_pct_from_sma_5, price_pct_from_sma_21 - Price distance from SMA
- price_pct_from_ema_9, price_pct_from_ema_50, price_pct_from_ema_100 - Price distance from EMA
- sma_5_21_proximity, sma_21_50_proximity, sma_63_200_proximity - SMA proximity
- ema_9_50_proximity - EMA proximity
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import tier_a500


@pytest.fixture()
def sample_daily_df() -> pd.DataFrame:
    """Create 400 days of synthetic OHLCV data for indicator testing.

    Uses 400 rows to ensure sufficient warmup for 252-day indicators
    plus meaningful output rows (400 - ~297 warmup ≈ 100+ rows).
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-02", periods=400, freq="B")
    n = len(dates)

    # Create realistic price series with trend and noise
    base = 100 + np.cumsum(np.random.randn(n) * 0.5)
    noise = np.random.randn(n) * 0.3

    close = base + noise
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    open_price = close + np.random.randn(n) * 0.3

    data = {
        "Date": dates,
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }
    return pd.DataFrame(data)


@pytest.fixture()
def sample_vix_df() -> pd.DataFrame:
    """Create synthetic VIX data aligned with sample_daily_df.

    VIX is required for indicators inherited from lower tiers.
    Uses 400 days to match sample_daily_df.
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-02", periods=400, freq="B")
    n = len(dates)

    # Simulate VIX with mean ~18, std ~5
    vix_close = 18 + 5 * np.random.randn(n)
    vix_close = np.clip(vix_close, 9, 80)

    data = {
        "Date": dates,
        "Open": vix_close * 0.99,
        "High": vix_close * 1.02,
        "Low": vix_close * 0.98,
        "Close": vix_close,
        "Volume": np.nan,  # VIX has no volume
    }
    return pd.DataFrame(data)


# =============================================================================
# Feature List Structure Tests
# =============================================================================


class TestA500FeatureListStructure:
    """Test feature list structure and counts."""

    def test_a500_addition_list_exists(self) -> None:
        """A500_ADDITION_LIST constant exists."""
        assert hasattr(tier_a500, "A500_ADDITION_LIST")

    def test_chunk_6a_features_exists(self) -> None:
        """CHUNK_6A_FEATURES constant exists."""
        assert hasattr(tier_a500, "CHUNK_6A_FEATURES")

    def test_chunk_6a_count_is_24(self) -> None:
        """CHUNK_6A_FEATURES has exactly 24 features."""
        assert len(tier_a500.CHUNK_6A_FEATURES) == 24

    def test_a500_feature_list_extends_a200(self) -> None:
        """FEATURE_LIST includes all a200 features plus new additions."""
        from src.features import tier_a200

        for feature in tier_a200.FEATURE_LIST:
            assert feature in tier_a500.FEATURE_LIST, f"Missing a200 feature: {feature}"

    def test_feature_list_includes_chunk_6a(self) -> None:
        """FEATURE_LIST includes all Chunk 6a features."""
        for feature in tier_a500.CHUNK_6A_FEATURES:
            assert feature in tier_a500.FEATURE_LIST, f"Missing chunk 6a feature: {feature}"

    def test_no_duplicate_features(self) -> None:
        """FEATURE_LIST has no duplicate feature names."""
        assert len(tier_a500.FEATURE_LIST) == len(set(tier_a500.FEATURE_LIST))

    def test_chunk_6a_no_duplicates_with_a200(self) -> None:
        """CHUNK_6A_FEATURES has no overlap with a200 features."""
        from src.features import tier_a200

        overlap = set(tier_a500.CHUNK_6A_FEATURES) & set(tier_a200.FEATURE_LIST)
        assert len(overlap) == 0, f"Overlapping features: {overlap}"


class TestChunk6aFeatureListContents:
    """Test Chunk 6a feature list contents."""

    def test_chunk6a_sma_periods_in_list(self) -> None:
        """Chunk 6a new SMA period indicators are in the list."""
        sma_indicators = ["sma_5", "sma_14", "sma_21", "sma_63"]
        for indicator in sma_indicators:
            assert indicator in tier_a500.CHUNK_6A_FEATURES, f"Missing: {indicator}"

    def test_chunk6a_ema_periods_in_list(self) -> None:
        """Chunk 6a new EMA period indicators are in the list."""
        ema_indicators = ["ema_5", "ema_9", "ema_50", "ema_100", "ema_200"]
        for indicator in ema_indicators:
            assert indicator in tier_a500.CHUNK_6A_FEATURES, f"Missing: {indicator}"

    def test_chunk6a_sma_slopes_in_list(self) -> None:
        """Chunk 6a SMA slope indicators are in the list."""
        slope_indicators = ["sma_5_slope", "sma_21_slope", "sma_63_slope"]
        for indicator in slope_indicators:
            assert indicator in tier_a500.CHUNK_6A_FEATURES, f"Missing: {indicator}"

    def test_chunk6a_ema_slopes_in_list(self) -> None:
        """Chunk 6a EMA slope indicators are in the list."""
        slope_indicators = ["ema_9_slope", "ema_50_slope", "ema_100_slope"]
        for indicator in slope_indicators:
            assert indicator in tier_a500.CHUNK_6A_FEATURES, f"Missing: {indicator}"

    def test_chunk6a_sma_price_distance_in_list(self) -> None:
        """Chunk 6a price-to-SMA distance indicators are in the list."""
        distance_indicators = ["price_pct_from_sma_5", "price_pct_from_sma_21"]
        for indicator in distance_indicators:
            assert indicator in tier_a500.CHUNK_6A_FEATURES, f"Missing: {indicator}"

    def test_chunk6a_ema_price_distance_in_list(self) -> None:
        """Chunk 6a price-to-EMA distance indicators are in the list."""
        distance_indicators = [
            "price_pct_from_ema_9",
            "price_pct_from_ema_50",
            "price_pct_from_ema_100",
        ]
        for indicator in distance_indicators:
            assert indicator in tier_a500.CHUNK_6A_FEATURES, f"Missing: {indicator}"

    def test_chunk6a_sma_proximity_in_list(self) -> None:
        """Chunk 6a SMA-to-SMA proximity indicators are in the list."""
        proximity_indicators = [
            "sma_5_21_proximity",
            "sma_21_50_proximity",
            "sma_63_200_proximity",
        ]
        for indicator in proximity_indicators:
            assert indicator in tier_a500.CHUNK_6A_FEATURES, f"Missing: {indicator}"

    def test_chunk6a_ema_proximity_in_list(self) -> None:
        """Chunk 6a EMA-to-EMA proximity indicators are in the list."""
        proximity_indicators = ["ema_9_50_proximity"]
        for indicator in proximity_indicators:
            assert indicator in tier_a500.CHUNK_6A_FEATURES, f"Missing: {indicator}"


# =============================================================================
# Chunk 6a Indicator Computation Tests
# =============================================================================


class TestChunk6aSmaIndicators:
    """Test new SMA period indicators (sma_5, sma_14, sma_21, sma_63)."""

    # --- Existence tests ---

    def test_sma_5_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_5 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_5" in result.columns

    def test_sma_14_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_14 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_14" in result.columns

    def test_sma_21_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_21 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_21" in result.columns

    def test_sma_63_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_63 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_63" in result.columns

    # --- Range tests ---

    def test_sma_values_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """SMA values should be positive (price-based MA)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["sma_5", "sma_14", "sma_21", "sma_63"]:
            assert (result[col] > 0).all(), f"{col} has non-positive values"

    def test_sma_values_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """SMA values should be within reasonable price range.

        Test data has prices around 100, so SMA should be similar.
        """
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["sma_5", "sma_14", "sma_21", "sma_63"]:
            assert result[col].min() > 50, f"{col} too low: {result[col].min()}"
            assert result[col].max() < 200, f"{col} too high: {result[col].max()}"

    # --- No-NaN test ---

    def test_sma_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in SMA columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["sma_5", "sma_14", "sma_21", "sma_63"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk6aEmaIndicators:
    """Test new EMA period indicators (ema_5, ema_9, ema_50, ema_100, ema_200)."""

    # --- Existence tests ---

    def test_ema_5_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_5 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_5" in result.columns

    def test_ema_9_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_9 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_9" in result.columns

    def test_ema_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_50 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_50" in result.columns

    def test_ema_100_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_100 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_100" in result.columns

    def test_ema_200_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_200 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_200" in result.columns

    # --- Range tests ---

    def test_ema_values_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """EMA values should be positive (price-based MA)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["ema_5", "ema_9", "ema_50", "ema_100", "ema_200"]:
            assert (result[col] > 0).all(), f"{col} has non-positive values"

    def test_ema_values_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """EMA values should be within reasonable price range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["ema_5", "ema_9", "ema_50", "ema_100", "ema_200"]:
            assert result[col].min() > 50, f"{col} too low: {result[col].min()}"
            assert result[col].max() < 200, f"{col} too high: {result[col].max()}"

    # --- No-NaN test ---

    def test_ema_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in EMA columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["ema_5", "ema_9", "ema_50", "ema_100", "ema_200"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk6aSlopeIndicators:
    """Test MA slope indicators (5-day change in MAs)."""

    # --- Existence tests ---

    def test_sma_5_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_5_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_5_slope" in result.columns

    def test_sma_21_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_21_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_21_slope" in result.columns

    def test_sma_63_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_63_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_63_slope" in result.columns

    def test_ema_9_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_9_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_9_slope" in result.columns

    def test_ema_50_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_50_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_50_slope" in result.columns

    def test_ema_100_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_100_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_100_slope" in result.columns

    # --- No-NaN test ---

    def test_slope_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in slope columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        slope_cols = [
            "sma_5_slope",
            "sma_21_slope",
            "sma_63_slope",
            "ema_9_slope",
            "ema_50_slope",
            "ema_100_slope",
        ]
        for col in slope_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk6aPriceDistanceIndicators:
    """Test price-to-MA distance indicators (% distance from MAs)."""

    # --- Existence tests ---

    def test_price_pct_from_sma_5_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_sma_5 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_pct_from_sma_5" in result.columns

    def test_price_pct_from_sma_21_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_sma_21 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_pct_from_sma_21" in result.columns

    def test_price_pct_from_ema_9_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_ema_9 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_pct_from_ema_9" in result.columns

    def test_price_pct_from_ema_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_ema_50 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_pct_from_ema_50" in result.columns

    def test_price_pct_from_ema_100_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_ema_100 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_pct_from_ema_100" in result.columns

    # --- Range tests ---

    def test_price_distance_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Price distance should be within reasonable % range (e.g., -50% to +50%)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        distance_cols = [
            "price_pct_from_sma_5",
            "price_pct_from_sma_21",
            "price_pct_from_ema_9",
            "price_pct_from_ema_50",
            "price_pct_from_ema_100",
        ]
        for col in distance_cols:
            assert result[col].min() > -50, f"{col} too low: {result[col].min()}"
            assert result[col].max() < 50, f"{col} too high: {result[col].max()}"

    # --- No-NaN test ---

    def test_price_distance_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in price distance columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        distance_cols = [
            "price_pct_from_sma_5",
            "price_pct_from_sma_21",
            "price_pct_from_ema_9",
            "price_pct_from_ema_50",
            "price_pct_from_ema_100",
        ]
        for col in distance_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk6aProximityIndicators:
    """Test MA-to-MA proximity indicators."""

    # --- Existence tests ---

    def test_sma_5_21_proximity_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_5_21_proximity column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_5_21_proximity" in result.columns

    def test_sma_21_50_proximity_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_21_50_proximity column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_21_50_proximity" in result.columns

    def test_sma_63_200_proximity_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_63_200_proximity column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_63_200_proximity" in result.columns

    def test_ema_9_50_proximity_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_9_50_proximity column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_9_50_proximity" in result.columns

    # --- Range tests ---

    def test_proximity_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Proximity should be within reasonable % range (e.g., -30% to +30%)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        proximity_cols = [
            "sma_5_21_proximity",
            "sma_21_50_proximity",
            "sma_63_200_proximity",
            "ema_9_50_proximity",
        ]
        for col in proximity_cols:
            assert result[col].min() > -30, f"{col} too low: {result[col].min()}"
            assert result[col].max() < 30, f"{col} too high: {result[col].max()}"

    # --- No-NaN test ---

    def test_proximity_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in proximity columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        proximity_cols = [
            "sma_5_21_proximity",
            "sma_21_50_proximity",
            "sma_63_200_proximity",
            "ema_9_50_proximity",
        ]
        for col in proximity_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk6aIntegration:
    """Integration tests for Chunk 6a."""

    def test_build_feature_dataframe_returns_dataframe(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """build_feature_dataframe returns a DataFrame."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_includes_date_column(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output includes Date column."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "Date" in result.columns

    def test_output_includes_all_chunk_6a_features(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output includes all Chunk 6a features."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a500.CHUNK_6A_FEATURES:
            assert feature in result.columns, f"Missing: {feature}"

    def test_output_includes_all_a200_features(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output includes all a200 features (inherited)."""
        from src.features import tier_a200

        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a200.FEATURE_LIST:
            assert feature in result.columns, f"Missing a200 feature: {feature}"

    def test_no_nan_values_after_warmup(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in any column after warmup period."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        nan_cols = result.columns[result.isna().any()].tolist()
        assert len(nan_cols) == 0, f"Columns with NaN: {nan_cols}"

    def test_output_row_count_reasonable(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output should have reasonable row count after warmup.

        With 400 input rows and ~297 day warmup (252 + stacking), expect ~100 output rows.
        """
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert len(result) >= 50, f"Too few rows: {len(result)}"
        assert len(result) <= 200, f"Too many rows (warmup not applied?): {len(result)}"
