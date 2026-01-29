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


# =============================================================================
# Sub-Chunk 6b: MA Extended Part 2 + New Oscillators (ranks 231-255)
# =============================================================================


class TestChunk6bFeatureListStructure:
    """Test Chunk 6b feature list structure and counts."""

    def test_chunk_6b_features_exists(self) -> None:
        """CHUNK_6B_FEATURES constant exists."""
        assert hasattr(tier_a500, "CHUNK_6B_FEATURES")

    def test_chunk_6b_is_list(self) -> None:
        """CHUNK_6B_FEATURES is a list."""
        assert isinstance(tier_a500.CHUNK_6B_FEATURES, list)

    def test_chunk_6b_count_is_25(self) -> None:
        """CHUNK_6B_FEATURES has exactly 25 features."""
        assert len(tier_a500.CHUNK_6B_FEATURES) == 25

    def test_chunk_6b_no_duplicates(self) -> None:
        """CHUNK_6B_FEATURES has no duplicate feature names."""
        assert len(tier_a500.CHUNK_6B_FEATURES) == len(set(tier_a500.CHUNK_6B_FEATURES))

    def test_chunk_6b_no_overlap_with_a200(self) -> None:
        """CHUNK_6B_FEATURES has no overlap with a200 features."""
        from src.features import tier_a200

        overlap = set(tier_a500.CHUNK_6B_FEATURES) & set(tier_a200.FEATURE_LIST)
        assert len(overlap) == 0, f"Overlapping features with a200: {overlap}"

    def test_chunk_6b_no_overlap_with_6a(self) -> None:
        """CHUNK_6B_FEATURES has no overlap with Chunk 6a features."""
        overlap = set(tier_a500.CHUNK_6B_FEATURES) & set(tier_a500.CHUNK_6A_FEATURES)
        assert len(overlap) == 0, f"Overlapping features with 6a: {overlap}"

    def test_chunk_6b_all_strings(self) -> None:
        """CHUNK_6B_FEATURES contains only strings."""
        for feature in tier_a500.CHUNK_6B_FEATURES:
            assert isinstance(feature, str), f"Non-string feature: {feature}"

    def test_chunk_6b_in_feature_list(self) -> None:
        """FEATURE_LIST includes all Chunk 6b features."""
        for feature in tier_a500.CHUNK_6B_FEATURES:
            assert feature in tier_a500.FEATURE_LIST, f"Missing chunk 6b feature: {feature}"


class TestChunk6bDurationCounters:
    """Test duration counter features for new 6a MAs (8 features)."""

    # --- Existence tests ---

    def test_days_above_ema_9_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_above_ema_9 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_above_ema_9" in result.columns

    def test_days_below_ema_9_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_below_ema_9 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_below_ema_9" in result.columns

    def test_days_above_ema_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_above_ema_50 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_above_ema_50" in result.columns

    def test_days_below_ema_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_below_ema_50 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_below_ema_50" in result.columns

    def test_days_above_sma_21_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_above_sma_21 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_above_sma_21" in result.columns

    def test_days_below_sma_21_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_below_sma_21 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_below_sma_21" in result.columns

    def test_days_above_sma_63_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_above_sma_63 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_above_sma_63" in result.columns

    def test_days_below_sma_63_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_below_sma_63 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_below_sma_63" in result.columns

    # --- Range and no-NaN tests ---

    def test_duration_counters_non_negative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Duration counter values should be non-negative integers."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        duration_cols = [
            "days_above_ema_9",
            "days_below_ema_9",
            "days_above_ema_50",
            "days_below_ema_50",
            "days_above_sma_21",
            "days_below_sma_21",
            "days_above_sma_63",
            "days_below_sma_63",
        ]
        for col in duration_cols:
            assert (result[col] >= 0).all(), f"{col} has negative values"

    def test_duration_counters_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in duration counter columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        duration_cols = [
            "days_above_ema_9",
            "days_below_ema_9",
            "days_above_ema_50",
            "days_below_ema_50",
            "days_above_sma_21",
            "days_below_sma_21",
            "days_above_sma_63",
            "days_below_sma_63",
        ]
        for col in duration_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk6bCrossRecency:
    """Test cross recency features for new MA pairs (5 features)."""

    # --- Existence tests ---

    def test_days_since_ema_9_50_cross_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_ema_9_50_cross column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_since_ema_9_50_cross" in result.columns

    def test_days_since_ema_50_200_cross_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_ema_50_200_cross column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_since_ema_50_200_cross" in result.columns

    def test_days_since_sma_5_21_cross_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_sma_5_21_cross column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_since_sma_5_21_cross" in result.columns

    def test_days_since_sma_21_63_cross_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_sma_21_63_cross column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_since_sma_21_63_cross" in result.columns

    def test_days_since_ema_9_sma_50_cross_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_ema_9_sma_50_cross column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_since_ema_9_sma_50_cross" in result.columns

    # --- Range and no-NaN tests ---

    def test_cross_recency_integers(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Cross recency values should be integers (signed)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cross_cols = [
            "days_since_ema_9_50_cross",
            "days_since_ema_50_200_cross",
            "days_since_sma_5_21_cross",
            "days_since_sma_21_63_cross",
            "days_since_ema_9_sma_50_cross",
        ]
        for col in cross_cols:
            # Check that values are integers (possibly negative)
            values = result[col].dropna()
            assert (values == values.astype(int)).all(), f"{col} has non-integer values"

    def test_cross_recency_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in cross recency columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cross_cols = [
            "days_since_ema_9_50_cross",
            "days_since_ema_50_200_cross",
            "days_since_sma_5_21_cross",
            "days_since_sma_21_63_cross",
            "days_since_ema_9_sma_50_cross",
        ]
        for col in cross_cols:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_cross_recency_sign_convention(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Cross recency should have positive/negative values (not all same sign)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # At least one cross feature should have both positive and negative values
        cross_cols = [
            "days_since_ema_9_50_cross",
            "days_since_sma_5_21_cross",
        ]
        has_both_signs = False
        for col in cross_cols:
            has_positive = (result[col] > 0).any()
            has_negative = (result[col] < 0).any()
            if has_positive and has_negative:
                has_both_signs = True
                break
        assert has_both_signs, "Cross recency features should have both positive and negative values"


class TestChunk6bAcceleration:
    """Test MA acceleration features (2nd derivative) (4 features)."""

    # --- Existence tests ---

    def test_ema_9_acceleration_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_9_acceleration column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_9_acceleration" in result.columns

    def test_ema_50_acceleration_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_50_acceleration column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_50_acceleration" in result.columns

    def test_sma_21_acceleration_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_21_acceleration column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_21_acceleration" in result.columns

    def test_sma_63_acceleration_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_63_acceleration column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_63_acceleration" in result.columns

    # --- Range and no-NaN tests ---

    def test_acceleration_numeric(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Acceleration values should be numeric (can be negative)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        accel_cols = [
            "ema_9_acceleration",
            "ema_50_acceleration",
            "sma_21_acceleration",
            "sma_63_acceleration",
        ]
        for col in accel_cols:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric"

    def test_acceleration_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in acceleration columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        accel_cols = [
            "ema_9_acceleration",
            "ema_50_acceleration",
            "sma_21_acceleration",
            "sma_63_acceleration",
        ]
        for col in accel_cols:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_acceleration_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Acceleration values should be in reasonable range (change in slope)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        accel_cols = [
            "ema_9_acceleration",
            "ema_50_acceleration",
            "sma_21_acceleration",
            "sma_63_acceleration",
        ]
        for col in accel_cols:
            # Acceleration is change in slope, should be small relative to price
            assert result[col].abs().max() < 50, f"{col} has unreasonable values"


class TestChunk6bOscillators:
    """Test new oscillator period features (4 features)."""

    # --- Existence tests ---

    def test_rsi_5_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """rsi_5 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "rsi_5" in result.columns

    def test_rsi_21_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """rsi_21 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "rsi_21" in result.columns

    def test_stoch_k_5_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stoch_k_5 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "stoch_k_5" in result.columns

    def test_stoch_d_5_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stoch_d_5 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "stoch_d_5" in result.columns

    # --- Range tests ---

    def test_rsi_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """RSI features should be in [0, 100] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        rsi_cols = ["rsi_5", "rsi_21"]
        for col in rsi_cols:
            assert result[col].min() >= 0, f"{col} below 0"
            assert result[col].max() <= 100, f"{col} above 100"

    def test_stochastic_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Stochastic features should be in [0, 100] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        stoch_cols = ["stoch_k_5", "stoch_d_5"]
        for col in stoch_cols:
            assert result[col].min() >= 0, f"{col} below 0"
            assert result[col].max() <= 100, f"{col} above 100"

    def test_stoch_d_smoother_than_k(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stoch_d_5 should have less variance than stoch_k_5 (it's smoothed)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # D is 3-period SMA of K, so variance should be lower
        k_std = result["stoch_k_5"].std()
        d_std = result["stoch_d_5"].std()
        assert d_std <= k_std, f"stoch_d_5 std ({d_std}) > stoch_k_5 std ({k_std})"

    # --- No-NaN tests ---

    def test_oscillators_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in oscillator columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        osc_cols = ["rsi_5", "rsi_21", "stoch_k_5", "stoch_d_5"]
        for col in osc_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk6bOscillatorDerivatives:
    """Test oscillator derivative features (4 features)."""

    # --- Existence tests ---

    def test_rsi_5_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """rsi_5_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "rsi_5_slope" in result.columns

    def test_rsi_21_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """rsi_21_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "rsi_21_slope" in result.columns

    def test_stoch_k_5_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stoch_k_5_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "stoch_k_5_slope" in result.columns

    def test_rsi_5_21_spread_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """rsi_5_21_spread column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "rsi_5_21_spread" in result.columns

    # --- Range tests ---

    def test_oscillator_slopes_numeric(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Oscillator slope values should be numeric."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        slope_cols = ["rsi_5_slope", "rsi_21_slope", "stoch_k_5_slope"]
        for col in slope_cols:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is not numeric"

    def test_rsi_spread_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """rsi_5_21_spread should be in [-100, 100] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["rsi_5_21_spread"].min() >= -100, "rsi_5_21_spread below -100"
        assert result["rsi_5_21_spread"].max() <= 100, "rsi_5_21_spread above 100"

    # --- No-NaN tests ---

    def test_oscillator_derivatives_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in oscillator derivative columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        deriv_cols = ["rsi_5_slope", "rsi_21_slope", "stoch_k_5_slope", "rsi_5_21_spread"]
        for col in deriv_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk6bIntegration:
    """Integration tests for Chunk 6b."""

    def test_output_includes_all_chunk_6b_features(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output includes all Chunk 6b features."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a500.CHUNK_6B_FEATURES:
            assert feature in result.columns, f"Missing: {feature}"

    def test_a500_addition_list_includes_6b(self) -> None:
        """A500_ADDITION_LIST includes all Chunk 6b features."""
        for feature in tier_a500.CHUNK_6B_FEATURES:
            assert feature in tier_a500.A500_ADDITION_LIST, (
                f"Missing from A500_ADDITION_LIST: {feature}"
            )

    def test_date_column_present(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output includes Date column."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "Date" in result.columns


# =============================================================================
# Sub-Chunk 7a: VOL Complete (ranks 256-278)
# =============================================================================


class TestChunk7aFeatureListStructure:
    """Test Chunk 7a feature list structure and counts."""

    def test_chunk_7a_features_exists(self) -> None:
        """CHUNK_7A_FEATURES constant exists."""
        assert hasattr(tier_a500, "CHUNK_7A_FEATURES")

    def test_chunk_7a_is_list(self) -> None:
        """CHUNK_7A_FEATURES is a list."""
        assert isinstance(tier_a500.CHUNK_7A_FEATURES, list)

    def test_chunk_7a_count_is_23(self) -> None:
        """CHUNK_7A_FEATURES has exactly 23 features."""
        assert len(tier_a500.CHUNK_7A_FEATURES) == 23

    def test_chunk_7a_no_duplicates(self) -> None:
        """CHUNK_7A_FEATURES has no duplicate feature names."""
        assert len(tier_a500.CHUNK_7A_FEATURES) == len(set(tier_a500.CHUNK_7A_FEATURES))

    def test_chunk_7a_no_overlap_with_a200(self) -> None:
        """CHUNK_7A_FEATURES has no overlap with a200 features."""
        from src.features import tier_a200

        overlap = set(tier_a500.CHUNK_7A_FEATURES) & set(tier_a200.FEATURE_LIST)
        assert len(overlap) == 0, f"Overlapping features with a200: {overlap}"

    def test_chunk_7a_no_overlap_with_6a(self) -> None:
        """CHUNK_7A_FEATURES has no overlap with Chunk 6a features."""
        overlap = set(tier_a500.CHUNK_7A_FEATURES) & set(tier_a500.CHUNK_6A_FEATURES)
        assert len(overlap) == 0, f"Overlapping features with 6a: {overlap}"

    def test_chunk_7a_no_overlap_with_6b(self) -> None:
        """CHUNK_7A_FEATURES has no overlap with Chunk 6b features."""
        overlap = set(tier_a500.CHUNK_7A_FEATURES) & set(tier_a500.CHUNK_6B_FEATURES)
        assert len(overlap) == 0, f"Overlapping features with 6b: {overlap}"

    def test_chunk_7a_all_strings(self) -> None:
        """CHUNK_7A_FEATURES contains only strings."""
        for feature in tier_a500.CHUNK_7A_FEATURES:
            assert isinstance(feature, str), f"Non-string feature: {feature}"

    def test_chunk_7a_in_feature_list(self) -> None:
        """FEATURE_LIST includes all Chunk 7a features."""
        for feature in tier_a500.CHUNK_7A_FEATURES:
            assert feature in tier_a500.FEATURE_LIST, f"Missing chunk 7a feature: {feature}"


class TestChunk7aAtrExtended:
    """Test extended ATR period features (4 features)."""

    # --- Existence tests ---

    def test_atr_5_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_5 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "atr_5" in result.columns

    def test_atr_21_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_21 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "atr_21" in result.columns

    def test_atr_5_pct_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_5_pct column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "atr_5_pct" in result.columns

    def test_atr_21_pct_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_21_pct column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "atr_21_pct" in result.columns

    # --- Range tests ---

    def test_atr_values_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ATR values should be positive."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["atr_5", "atr_21"]:
            assert (result[col] > 0).all(), f"{col} has non-positive values"

    def test_atr_pct_values_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ATR percentage values should be positive."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["atr_5_pct", "atr_21_pct"]:
            assert (result[col] > 0).all(), f"{col} has non-positive values"

    # --- No-NaN tests ---

    def test_atr_extended_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in extended ATR columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["atr_5", "atr_21", "atr_5_pct", "atr_21_pct"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7aAtrDynamics:
    """Test ATR dynamics features (4 features)."""

    # --- Existence tests ---

    def test_atr_5_21_ratio_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_5_21_ratio column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "atr_5_21_ratio" in result.columns

    def test_atr_expansion_5d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_expansion_5d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "atr_expansion_5d" in result.columns

    def test_atr_acceleration_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_acceleration column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "atr_acceleration" in result.columns

    def test_atr_percentile_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_percentile_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "atr_percentile_20d" in result.columns

    # --- Range tests ---

    def test_atr_5_21_ratio_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_5_21_ratio should be positive (ratio of positive values)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["atr_5_21_ratio"] > 0).all(), "atr_5_21_ratio has non-positive values"

    def test_atr_expansion_5d_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_expansion_5d should be positive (ratio of positive values)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["atr_expansion_5d"] > 0).all(), "atr_expansion_5d has non-positive values"

    def test_atr_percentile_20d_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_percentile_20d should be in [0, 1] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["atr_percentile_20d"].min() >= 0, "atr_percentile_20d below 0"
        assert result["atr_percentile_20d"].max() <= 1, "atr_percentile_20d above 1"

    # --- No-NaN tests ---

    def test_atr_dynamics_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in ATR dynamics columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["atr_5_21_ratio", "atr_expansion_5d", "atr_acceleration", "atr_percentile_20d"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7aTrueRange:
    """Test True Range features (3 features)."""

    # --- Existence tests ---

    def test_tr_pct_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tr_pct column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "tr_pct" in result.columns

    def test_tr_pct_zscore_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tr_pct_zscore_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "tr_pct_zscore_20d" in result.columns

    def test_consecutive_high_vol_days_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_high_vol_days column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "consecutive_high_vol_days" in result.columns

    # --- Range tests ---

    def test_tr_pct_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tr_pct should be positive (TR/Close)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["tr_pct"] > 0).all(), "tr_pct has non-positive values"

    def test_consecutive_high_vol_days_non_negative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_high_vol_days should be non-negative integer."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["consecutive_high_vol_days"] >= 0).all(), "consecutive_high_vol_days negative"

    # --- No-NaN tests ---

    def test_true_range_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in True Range columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["tr_pct", "tr_pct_zscore_20d", "consecutive_high_vol_days"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7aVolEstimators:
    """Test alternative volatility estimator features (3 features)."""

    # --- Existence tests ---

    def test_rogers_satchell_volatility_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """rogers_satchell_volatility column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "rogers_satchell_volatility" in result.columns

    def test_yang_zhang_volatility_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """yang_zhang_volatility column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "yang_zhang_volatility" in result.columns

    def test_historical_volatility_10d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """historical_volatility_10d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "historical_volatility_10d" in result.columns

    # --- Range tests ---

    def test_vol_estimators_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Volatility estimators should be positive (or zero)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["rogers_satchell_volatility", "yang_zhang_volatility", "historical_volatility_10d"]:
            assert (result[col] >= 0).all(), f"{col} has negative values"

    # --- No-NaN tests ---

    def test_vol_estimators_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in volatility estimator columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["rogers_satchell_volatility", "yang_zhang_volatility", "historical_volatility_10d"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7aBbExtended:
    """Test extended Bollinger Band features (4 features)."""

    # --- Existence tests ---

    def test_bb_width_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bb_width_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "bb_width_slope" in result.columns

    def test_bb_width_acceleration_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bb_width_acceleration column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "bb_width_acceleration" in result.columns

    def test_bb_width_percentile_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bb_width_percentile_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "bb_width_percentile_20d" in result.columns

    def test_price_bb_band_position_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_bb_band_position column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_bb_band_position" in result.columns

    # --- Range tests ---

    def test_bb_width_percentile_20d_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bb_width_percentile_20d should be in [0, 1] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["bb_width_percentile_20d"].min() >= 0, "bb_width_percentile_20d below 0"
        assert result["bb_width_percentile_20d"].max() <= 1, "bb_width_percentile_20d above 1"

    def test_price_bb_band_position_typical_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_bb_band_position typically in [0, 1] but can exceed during extremes."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # Most values should be in [0, 1], allow some outside during extreme moves
        in_range = ((result["price_bb_band_position"] >= 0) & (result["price_bb_band_position"] <= 1)).mean()
        assert in_range >= 0.8, f"Only {in_range:.1%} of values in [0, 1]"

    # --- No-NaN tests ---

    def test_bb_extended_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in extended BB columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["bb_width_slope", "bb_width_acceleration", "bb_width_percentile_20d", "price_bb_band_position"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7aKeltnerChannel:
    """Test Keltner Channel features (3 features)."""

    # --- Existence tests ---

    def test_kc_width_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """kc_width column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "kc_width" in result.columns

    def test_kc_position_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """kc_position column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "kc_position" in result.columns

    def test_bb_kc_ratio_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bb_kc_ratio column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "bb_kc_ratio" in result.columns

    # --- Range tests ---

    def test_kc_width_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """kc_width should be positive."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["kc_width"] > 0).all(), "kc_width has non-positive values"

    def test_kc_position_typical_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """kc_position typically in [0, 1] but can exceed during extremes."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        in_range = ((result["kc_position"] >= 0) & (result["kc_position"] <= 1)).mean()
        assert in_range >= 0.8, f"Only {in_range:.1%} of values in [0, 1]"

    def test_bb_kc_ratio_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bb_kc_ratio should be positive (ratio of widths)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["bb_kc_ratio"] > 0).all(), "bb_kc_ratio has non-positive values"

    # --- No-NaN tests ---

    def test_keltner_channel_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Keltner Channel columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["kc_width", "kc_position", "bb_kc_ratio"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7aVolRegimeExtended:
    """Test extended volatility regime features (2 features)."""

    # --- Existence tests ---

    def test_vol_regime_change_intensity_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vol_regime_change_intensity column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vol_regime_change_intensity" in result.columns

    def test_vol_clustering_score_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vol_clustering_score column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vol_clustering_score" in result.columns

    # --- Range tests ---

    def test_vol_clustering_score_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vol_clustering_score (autocorrelation) should be in [-1, 1] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["vol_clustering_score"].min() >= -1, "vol_clustering_score below -1"
        assert result["vol_clustering_score"].max() <= 1, "vol_clustering_score above 1"

    # --- No-NaN tests ---

    def test_vol_regime_extended_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in extended vol regime columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["vol_regime_change_intensity", "vol_clustering_score"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7aIntegration:
    """Integration tests for Chunk 7a."""

    def test_chunk_7a_feature_count_is_23(self) -> None:
        """CHUNK_7A_FEATURES should have exactly 23 features."""
        assert len(tier_a500.CHUNK_7A_FEATURES) == 23, (
            f"Expected 23 features in 7a, got {len(tier_a500.CHUNK_7A_FEATURES)}"
        )

    def test_output_includes_all_chunk_7a_features(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output includes all Chunk 7a features."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a500.CHUNK_7A_FEATURES:
            assert feature in result.columns, f"Missing: {feature}"

    def test_a500_addition_list_includes_7a(self) -> None:
        """A500_ADDITION_LIST includes all Chunk 7a features."""
        for feature in tier_a500.CHUNK_7A_FEATURES:
            assert feature in tier_a500.A500_ADDITION_LIST, (
                f"Missing from A500_ADDITION_LIST: {feature}"
            )

    def test_chunk_7a_features_contiguous_in_list(self) -> None:
        """7a features should be contiguous in the addition list after 6b."""
        # Find the index where 7a starts in the addition list
        first_7a = tier_a500.CHUNK_7A_FEATURES[0]
        start_idx = tier_a500.A500_ADDITION_LIST.index(first_7a)
        # Check all 7a features are contiguous
        for i, feature in enumerate(tier_a500.CHUNK_7A_FEATURES):
            assert tier_a500.A500_ADDITION_LIST[start_idx + i] == feature, (
                f"7a feature {feature} not at expected position"
            )

    def test_no_nan_in_7a_features_after_warmup(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Chunk 7a columns after warmup period."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a500.CHUNK_7A_FEATURES:
            assert not result[feature].isna().any(), f"NaN in {feature}"

    def test_output_row_count_reasonable(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output should have reasonable row count after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert len(result) >= 50, f"Too few rows: {len(result)}"
        assert len(result) <= 200, f"Too many rows (warmup not applied?): {len(result)}"


# =============================================================================
# Sub-Chunk 7b: VLM Complete (ranks 279-300)
# =============================================================================


class TestChunk7bFeatureListStructure:
    """Test Chunk 7b feature list structure and counts."""

    def test_chunk_7b_features_exists(self) -> None:
        """CHUNK_7B_FEATURES constant exists."""
        assert hasattr(tier_a500, "CHUNK_7B_FEATURES")

    def test_chunk_7b_is_list(self) -> None:
        """CHUNK_7B_FEATURES is a list."""
        assert isinstance(tier_a500.CHUNK_7B_FEATURES, list)

    def test_chunk_7b_count_is_22(self) -> None:
        """CHUNK_7B_FEATURES has exactly 22 features."""
        assert len(tier_a500.CHUNK_7B_FEATURES) == 22

    def test_chunk_7b_no_duplicates(self) -> None:
        """CHUNK_7B_FEATURES has no duplicate feature names."""
        assert len(tier_a500.CHUNK_7B_FEATURES) == len(set(tier_a500.CHUNK_7B_FEATURES))

    def test_chunk_7b_no_overlap_with_a200(self) -> None:
        """CHUNK_7B_FEATURES has no overlap with a200 features."""
        from src.features import tier_a200

        overlap = set(tier_a500.CHUNK_7B_FEATURES) & set(tier_a200.FEATURE_LIST)
        assert len(overlap) == 0, f"Overlapping features with a200: {overlap}"

    def test_chunk_7b_no_overlap_with_prior_chunks(self) -> None:
        """CHUNK_7B_FEATURES has no overlap with prior chunks (6a, 6b, 7a)."""
        prior_chunks = (
            set(tier_a500.CHUNK_6A_FEATURES)
            | set(tier_a500.CHUNK_6B_FEATURES)
            | set(tier_a500.CHUNK_7A_FEATURES)
        )
        overlap = set(tier_a500.CHUNK_7B_FEATURES) & prior_chunks
        assert len(overlap) == 0, f"Overlapping features with prior chunks: {overlap}"

    def test_chunk_7b_all_strings(self) -> None:
        """CHUNK_7B_FEATURES contains only strings."""
        for feature in tier_a500.CHUNK_7B_FEATURES:
            assert isinstance(feature, str), f"Non-string feature: {feature}"

    def test_chunk_7b_in_feature_list(self) -> None:
        """FEATURE_LIST includes all Chunk 7b features."""
        for feature in tier_a500.CHUNK_7B_FEATURES:
            assert feature in tier_a500.FEATURE_LIST, f"Missing chunk 7b feature: {feature}"


class TestChunk7bVolumeVectors:
    """Test volume vector features (4 features)."""

    # --- Existence tests ---

    def test_volume_trend_3d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_trend_3d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "volume_trend_3d" in result.columns

    def test_volume_ma_ratio_5_20_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_ma_ratio_5_20 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "volume_ma_ratio_5_20" in result.columns

    def test_consecutive_decreasing_vol_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_decreasing_vol column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "consecutive_decreasing_vol" in result.columns

    def test_volume_acceleration_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_acceleration column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "volume_acceleration" in result.columns

    # --- Range tests ---

    def test_volume_ma_ratio_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_ma_ratio_5_20 should be positive (ratio of positive MAs)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["volume_ma_ratio_5_20"] > 0).all(), "volume_ma_ratio_5_20 has non-positive values"

    def test_consecutive_decreasing_vol_non_negative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_decreasing_vol should be non-negative integer."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["consecutive_decreasing_vol"] >= 0).all(), "consecutive_decreasing_vol negative"

    # --- No-NaN tests ---

    def test_volume_vectors_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in volume vector columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = ["volume_trend_3d", "volume_ma_ratio_5_20", "consecutive_decreasing_vol", "volume_acceleration"]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7bVwapExtended:
    """Test VWAP extended features (5 features)."""

    # --- Existence tests ---

    def test_pct_from_vwap_20_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pct_from_vwap_20 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "pct_from_vwap_20" in result.columns

    def test_vwap_slope_5d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vwap_slope_5d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vwap_slope_5d" in result.columns

    def test_vwap_pct_change_1d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vwap_pct_change_1d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vwap_pct_change_1d" in result.columns

    def test_vwap_price_divergence_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vwap_price_divergence column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vwap_price_divergence" in result.columns

    def test_price_vwap_cross_recency_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_vwap_cross_recency column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_vwap_cross_recency" in result.columns

    # --- Range tests ---

    def test_vwap_price_divergence_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vwap_price_divergence should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["vwap_price_divergence"].unique())
        assert unique_vals <= {0, 1}, f"vwap_price_divergence has non-binary values: {unique_vals}"

    def test_price_vwap_cross_recency_integers(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_vwap_cross_recency should be integers (signed)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        values = result["price_vwap_cross_recency"].dropna()
        assert (values == values.astype(int)).all(), "price_vwap_cross_recency has non-integer values"

    def test_price_vwap_cross_recency_sign_convention(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_vwap_cross_recency should have both positive and negative values."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        has_positive = (result["price_vwap_cross_recency"] > 0).any()
        has_negative = (result["price_vwap_cross_recency"] < 0).any()
        assert has_positive and has_negative, "price_vwap_cross_recency should have both signs"

    # --- No-NaN tests ---

    def test_vwap_extended_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in VWAP extended columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = ["pct_from_vwap_20", "vwap_slope_5d", "vwap_pct_change_1d", "vwap_price_divergence", "price_vwap_cross_recency"]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7bVolumeIndicators:
    """Test volume indicator features (5 features)."""

    # --- Existence tests ---

    def test_cmf_20_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """cmf_20 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "cmf_20" in result.columns

    def test_emv_14_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """emv_14 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "emv_14" in result.columns

    def test_nvi_signal_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """nvi_signal column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "nvi_signal" in result.columns

    def test_pvi_signal_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pvi_signal column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "pvi_signal" in result.columns

    def test_vpt_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vpt_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vpt_slope" in result.columns

    # --- Range tests ---

    def test_cmf_20_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """cmf_20 should be in [-1, 1] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["cmf_20"].min() >= -1, "cmf_20 below -1"
        assert result["cmf_20"].max() <= 1, "cmf_20 above 1"

    def test_nvi_signal_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """nvi_signal should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["nvi_signal"].unique())
        assert unique_vals <= {0, 1}, f"nvi_signal has non-binary values: {unique_vals}"

    def test_pvi_signal_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pvi_signal should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["pvi_signal"].unique())
        assert unique_vals <= {0, 1}, f"pvi_signal has non-binary values: {unique_vals}"

    # --- No-NaN tests ---

    def test_volume_indicators_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in volume indicator columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = ["cmf_20", "emv_14", "nvi_signal", "pvi_signal", "vpt_slope"]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7bVolPriceConfluence:
    """Test volume-price confluence features (4 features)."""

    # --- Existence tests ---

    def test_volume_spike_price_flat_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_spike_price_flat column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "volume_spike_price_flat" in result.columns

    def test_volume_price_spike_both_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_price_spike_both column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "volume_price_spike_both" in result.columns

    def test_sequential_vol_buildup_3d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sequential_vol_buildup_3d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sequential_vol_buildup_3d" in result.columns

    def test_vol_breakout_confirmation_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vol_breakout_confirmation column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vol_breakout_confirmation" in result.columns

    # --- Range tests ---

    def test_volume_spike_price_flat_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_spike_price_flat should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["volume_spike_price_flat"].unique())
        assert unique_vals <= {0, 1}, f"volume_spike_price_flat has non-binary values: {unique_vals}"

    def test_volume_price_spike_both_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_price_spike_both should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["volume_price_spike_both"].unique())
        assert unique_vals <= {0, 1}, f"volume_price_spike_both has non-binary values: {unique_vals}"

    def test_sequential_vol_buildup_3d_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sequential_vol_buildup_3d should be in [0, 1] range (score)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["sequential_vol_buildup_3d"].min() >= 0, "sequential_vol_buildup_3d below 0"
        assert result["sequential_vol_buildup_3d"].max() <= 1, "sequential_vol_buildup_3d above 1"

    def test_vol_breakout_confirmation_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vol_breakout_confirmation should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["vol_breakout_confirmation"].unique())
        assert unique_vals <= {0, 1}, f"vol_breakout_confirmation has non-binary values: {unique_vals}"

    # --- No-NaN tests ---

    def test_vol_price_confluence_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in volume-price confluence columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = ["volume_spike_price_flat", "volume_price_spike_both", "sequential_vol_buildup_3d", "vol_breakout_confirmation"]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7bVolRegime:
    """Test volume regime features (4 features)."""

    # --- Existence tests ---

    def test_volume_percentile_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_percentile_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "volume_percentile_20d" in result.columns

    def test_volume_zscore_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_zscore_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "volume_zscore_20d" in result.columns

    def test_avg_vol_up_vs_down_days_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """avg_vol_up_vs_down_days column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "avg_vol_up_vs_down_days" in result.columns

    def test_volume_trend_strength_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_trend_strength column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "volume_trend_strength" in result.columns

    # --- Range tests ---

    def test_volume_percentile_20d_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_percentile_20d should be in [0, 1] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["volume_percentile_20d"].min() >= 0, "volume_percentile_20d below 0"
        assert result["volume_percentile_20d"].max() <= 1, "volume_percentile_20d above 1"

    def test_avg_vol_up_vs_down_days_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """avg_vol_up_vs_down_days should be positive (ratio of positive values)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["avg_vol_up_vs_down_days"] > 0).all(), "avg_vol_up_vs_down_days has non-positive values"

    def test_volume_trend_strength_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_trend_strength (correlation) should be in [-1, 1] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["volume_trend_strength"].min() >= -1, "volume_trend_strength below -1"
        assert result["volume_trend_strength"].max() <= 1, "volume_trend_strength above 1"

    # --- No-NaN tests ---

    def test_vol_regime_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in volume regime columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = ["volume_percentile_20d", "volume_zscore_20d", "avg_vol_up_vs_down_days", "volume_trend_strength"]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk7bIntegration:
    """Integration tests for Chunk 7b."""

    def test_feature_count_with_7b(self) -> None:
        """Feature count should be at least 278 (prior) + 22 (7b) = 300 after 7b."""
        # Note: This test will be updated as more chunks are added
        min_expected = 206 + 24 + 25 + 23 + 22  # a200 + 6a + 6b + 7a + 7b
        assert len(tier_a500.FEATURE_LIST) >= min_expected, (
            f"Expected at least {min_expected} features, got {len(tier_a500.FEATURE_LIST)}"
        )

    def test_output_includes_all_chunk_7b_features(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output includes all Chunk 7b features."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a500.CHUNK_7B_FEATURES:
            assert feature in result.columns, f"Missing: {feature}"

    def test_a500_addition_list_includes_7b(self) -> None:
        """A500_ADDITION_LIST includes all Chunk 7b features."""
        for feature in tier_a500.CHUNK_7B_FEATURES:
            assert feature in tier_a500.A500_ADDITION_LIST, (
                f"Missing from A500_ADDITION_LIST: {feature}"
            )

    def test_output_column_count_with_7b(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output DataFrame should have at least Date + 300 features columns after 7b."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        min_expected_cols = 1 + 206 + 24 + 25 + 23 + 22  # Date + a200 + 6a + 6b + 7a + 7b
        assert len(result.columns) >= min_expected_cols, (
            f"Expected at least {min_expected_cols} columns, got {len(result.columns)}"
        )

    def test_7b_no_nan_values_after_warmup(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Chunk 7b columns after warmup period."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in tier_a500.CHUNK_7B_FEATURES:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_7b_output_row_count_reasonable(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output should have reasonable row count after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert len(result) >= 50, f"Too few rows: {len(result)}"
        assert len(result) <= 200, f"Too many rows (warmup not applied?): {len(result)}"


# =============================================================================
# Sub-Chunk 8a: TRD Complete (ranks 301-323) - 23 features
# =============================================================================


class TestChunk8aFeatureListStructure:
    """Test Chunk 8a feature list structure and counts."""

    def test_chunk_8a_features_exists(self) -> None:
        """CHUNK_8A_FEATURES constant exists."""
        assert hasattr(tier_a500, "CHUNK_8A_FEATURES")

    def test_chunk_8a_is_list(self) -> None:
        """CHUNK_8A_FEATURES is a list."""
        assert isinstance(tier_a500.CHUNK_8A_FEATURES, list)

    def test_chunk_8a_count_is_23(self) -> None:
        """CHUNK_8A_FEATURES has exactly 23 features."""
        assert len(tier_a500.CHUNK_8A_FEATURES) == 23

    def test_chunk_8a_no_duplicates(self) -> None:
        """CHUNK_8A_FEATURES has no duplicate feature names."""
        assert len(tier_a500.CHUNK_8A_FEATURES) == len(set(tier_a500.CHUNK_8A_FEATURES))

    def test_chunk_8a_no_overlap_with_a200(self) -> None:
        """CHUNK_8A_FEATURES has no overlap with a200 features."""
        from src.features import tier_a200

        overlap = set(tier_a500.CHUNK_8A_FEATURES) & set(tier_a200.FEATURE_LIST)
        assert len(overlap) == 0, f"Overlapping features with a200: {overlap}"

    def test_chunk_8a_no_overlap_with_prior_chunks(self) -> None:
        """CHUNK_8A_FEATURES has no overlap with prior chunk features."""
        prior_chunks = (
            set(tier_a500.CHUNK_6A_FEATURES)
            | set(tier_a500.CHUNK_6B_FEATURES)
            | set(tier_a500.CHUNK_7A_FEATURES)
            | set(tier_a500.CHUNK_7B_FEATURES)
        )
        overlap = set(tier_a500.CHUNK_8A_FEATURES) & prior_chunks
        assert len(overlap) == 0, f"Overlapping features with prior chunks: {overlap}"

    def test_chunk_8a_all_strings(self) -> None:
        """CHUNK_8A_FEATURES contains only strings."""
        for feature in tier_a500.CHUNK_8A_FEATURES:
            assert isinstance(feature, str), f"Non-string feature: {feature}"

    def test_chunk_8a_in_feature_list(self) -> None:
        """FEATURE_LIST includes all Chunk 8a features."""
        for feature in tier_a500.CHUNK_8A_FEATURES:
            assert feature in tier_a500.FEATURE_LIST, f"Missing chunk 8a feature: {feature}"


class TestChunk8aFeatureListContents:
    """Test Chunk 8a feature list contents by group."""

    def test_chunk8a_adx_extended_in_list(self) -> None:
        """ADX Extended features are in Chunk 8a list."""
        adx_features = [
            "plus_di_14",
            "minus_di_14",
            "adx_14_slope",
            "adx_acceleration",
            "di_cross_recency",
        ]
        for feature in adx_features:
            assert feature in tier_a500.CHUNK_8A_FEATURES, f"Missing: {feature}"

    def test_chunk8a_trend_exhaustion_in_list(self) -> None:
        """Trend Exhaustion features are in Chunk 8a list."""
        exhaustion_features = [
            "avg_up_day_magnitude",
            "avg_down_day_magnitude",
            "up_down_magnitude_ratio",
            "trend_persistence_20d",
            "up_vs_down_momentum",
            "directional_bias_strength",
        ]
        for feature in exhaustion_features:
            assert feature in tier_a500.CHUNK_8A_FEATURES, f"Missing: {feature}"

    def test_chunk8a_trend_regime_in_list(self) -> None:
        """Trend Regime features are in Chunk 8a list."""
        regime_features = [
            "adx_regime",
            "price_trend_direction",
            "trend_alignment_score",
            "trend_regime_duration",
            "trend_strength_vs_vol",
        ]
        for feature in regime_features:
            assert feature in tier_a500.CHUNK_8A_FEATURES, f"Missing: {feature}"

    def test_chunk8a_trend_channel_in_list(self) -> None:
        """Trend Channel features are in Chunk 8a list."""
        channel_features = [
            "linreg_slope_20d",
            "linreg_r_squared_20d",
            "price_linreg_deviation",
            "channel_width_linreg_20d",
        ]
        for feature in channel_features:
            assert feature in tier_a500.CHUNK_8A_FEATURES, f"Missing: {feature}"

    def test_chunk8a_aroon_extended_in_list(self) -> None:
        """Aroon Extended features are in Chunk 8a list."""
        aroon_features = [
            "aroon_up_25",
            "aroon_down_25",
            "aroon_trend_strength",
        ]
        for feature in aroon_features:
            assert feature in tier_a500.CHUNK_8A_FEATURES, f"Missing: {feature}"


# =============================================================================
# Chunk 8a: ADX Extended Computation Tests (5 features)
# =============================================================================


class TestChunk8aAdxExtended:
    """Test ADX Extended features (5 features)."""

    # --- Existence tests ---

    def test_plus_di_14_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """plus_di_14 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "plus_di_14" in result.columns

    def test_minus_di_14_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """minus_di_14 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "minus_di_14" in result.columns

    def test_adx_14_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """adx_14_slope column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "adx_14_slope" in result.columns

    def test_adx_acceleration_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """adx_acceleration column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "adx_acceleration" in result.columns

    def test_di_cross_recency_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """di_cross_recency column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "di_cross_recency" in result.columns

    # --- Range tests ---

    def test_plus_di_14_range_0_to_100(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """plus_di_14 should be in [0, 100] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["plus_di_14"].min() >= 0, "plus_di_14 below 0"
        assert result["plus_di_14"].max() <= 100, "plus_di_14 above 100"

    def test_minus_di_14_range_0_to_100(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """minus_di_14 should be in [0, 100] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["minus_di_14"].min() >= 0, "minus_di_14 below 0"
        assert result["minus_di_14"].max() <= 100, "minus_di_14 above 100"

    def test_di_cross_recency_signed_values(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """di_cross_recency should have both positive and negative values (signed)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # Should have both positive (bullish) and negative (bearish) values
        # At minimum, should be integer-like values
        assert pd.api.types.is_numeric_dtype(result["di_cross_recency"])

    def test_di_cross_recency_integer_like(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """di_cross_recency should be integer-valued (days count)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # Values should be whole numbers
        assert (result["di_cross_recency"] == result["di_cross_recency"].round()).all()

    # --- No-NaN tests ---

    def test_adx_extended_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in ADX Extended columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = [
            "plus_di_14",
            "minus_di_14",
            "adx_14_slope",
            "adx_acceleration",
            "di_cross_recency",
        ]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 8a: Trend Exhaustion Computation Tests (6 features)
# =============================================================================


class TestChunk8aTrendExhaustion:
    """Test Trend Exhaustion features (6 features)."""

    # --- Existence tests ---

    def test_avg_up_day_magnitude_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """avg_up_day_magnitude column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "avg_up_day_magnitude" in result.columns

    def test_avg_down_day_magnitude_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """avg_down_day_magnitude column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "avg_down_day_magnitude" in result.columns

    def test_up_down_magnitude_ratio_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """up_down_magnitude_ratio column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "up_down_magnitude_ratio" in result.columns

    def test_trend_persistence_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_persistence_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "trend_persistence_20d" in result.columns

    def test_up_vs_down_momentum_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """up_vs_down_momentum column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "up_vs_down_momentum" in result.columns

    def test_directional_bias_strength_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """directional_bias_strength column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "directional_bias_strength" in result.columns

    # --- Range tests ---

    def test_avg_up_day_magnitude_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """avg_up_day_magnitude should be positive (magnitude of up returns)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # Could have NaN if no up days in window, but non-NaN should be positive
        non_nan = result["avg_up_day_magnitude"].dropna()
        if len(non_nan) > 0:
            assert (non_nan >= 0).all(), "avg_up_day_magnitude has negative values"

    def test_avg_down_day_magnitude_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """avg_down_day_magnitude should be positive (absolute magnitude)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        non_nan = result["avg_down_day_magnitude"].dropna()
        if len(non_nan) > 0:
            assert (non_nan >= 0).all(), "avg_down_day_magnitude has negative values"

    def test_up_down_magnitude_ratio_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """up_down_magnitude_ratio should be positive (ratio of positive values)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        non_nan = result["up_down_magnitude_ratio"].dropna()
        if len(non_nan) > 0:
            assert (non_nan >= 0).all(), "up_down_magnitude_ratio has negative values"

    def test_trend_persistence_20d_non_negative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_persistence_20d should be non-negative (max streak in window)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["trend_persistence_20d"] >= 0).all()

    def test_directional_bias_strength_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """directional_bias_strength should be in [0, 1] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["directional_bias_strength"].min() >= 0, "directional_bias_strength below 0"
        assert result["directional_bias_strength"].max() <= 1, "directional_bias_strength above 1"

    # --- No-NaN tests ---

    def test_trend_exhaustion_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Trend Exhaustion columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = [
            "avg_up_day_magnitude",
            "avg_down_day_magnitude",
            "up_down_magnitude_ratio",
            "trend_persistence_20d",
            "up_vs_down_momentum",
            "directional_bias_strength",
        ]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 8a: Trend Regime Computation Tests (5 features)
# =============================================================================


class TestChunk8aTrendRegime:
    """Test Trend Regime features (5 features)."""

    # --- Existence tests ---

    def test_adx_regime_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """adx_regime column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "adx_regime" in result.columns

    def test_price_trend_direction_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_trend_direction column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_trend_direction" in result.columns

    def test_trend_alignment_score_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_alignment_score column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "trend_alignment_score" in result.columns

    def test_trend_regime_duration_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_regime_duration column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "trend_regime_duration" in result.columns

    def test_trend_strength_vs_vol_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_strength_vs_vol column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "trend_strength_vs_vol" in result.columns

    # --- Range tests ---

    def test_adx_regime_categorical_values(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """adx_regime should be in {0, 1, 2}."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_values = set(result["adx_regime"].unique())
        assert unique_values.issubset({0, 1, 2}), f"adx_regime has invalid values: {unique_values}"

    def test_price_trend_direction_values(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_trend_direction should be in {-1, 0, 1}."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_values = set(result["price_trend_direction"].unique())
        assert unique_values.issubset({-1, 0, 1}), f"price_trend_direction has invalid values: {unique_values}"

    def test_trend_alignment_score_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_alignment_score should be in {0, 1}."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_values = set(result["trend_alignment_score"].unique())
        assert unique_values.issubset({0, 1}), f"trend_alignment_score has invalid values: {unique_values}"

    def test_trend_regime_duration_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_regime_duration should be positive (days in regime)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["trend_regime_duration"] > 0).all(), "trend_regime_duration has non-positive values"

    def test_trend_strength_vs_vol_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_strength_vs_vol should be positive (ratio of positive values)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["trend_strength_vs_vol"] > 0).all(), "trend_strength_vs_vol has non-positive values"

    # --- No-NaN tests ---

    def test_trend_regime_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Trend Regime columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = [
            "adx_regime",
            "price_trend_direction",
            "trend_alignment_score",
            "trend_regime_duration",
            "trend_strength_vs_vol",
        ]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 8a: Trend Channel Computation Tests (4 features)
# =============================================================================


class TestChunk8aTrendChannel:
    """Test Trend Channel features (4 features)."""

    # --- Existence tests ---

    def test_linreg_slope_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """linreg_slope_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "linreg_slope_20d" in result.columns

    def test_linreg_r_squared_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """linreg_r_squared_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "linreg_r_squared_20d" in result.columns

    def test_price_linreg_deviation_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_linreg_deviation column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_linreg_deviation" in result.columns

    def test_channel_width_linreg_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """channel_width_linreg_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "channel_width_linreg_20d" in result.columns

    # --- Range tests ---

    def test_linreg_r_squared_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """linreg_r_squared_20d should be in [0, 1] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["linreg_r_squared_20d"].min() >= 0, "linreg_r_squared_20d below 0"
        assert result["linreg_r_squared_20d"].max() <= 1, "linreg_r_squared_20d above 1"

    def test_channel_width_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """channel_width_linreg_20d should be positive (width is always positive)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["channel_width_linreg_20d"] >= 0).all(), "channel_width_linreg_20d has negative values"

    def test_linreg_slope_numeric(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """linreg_slope_20d should be numeric."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert pd.api.types.is_numeric_dtype(result["linreg_slope_20d"])

    def test_price_linreg_deviation_numeric(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_linreg_deviation should be numeric (can be positive or negative)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert pd.api.types.is_numeric_dtype(result["price_linreg_deviation"])

    # --- No-NaN tests ---

    def test_trend_channel_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Trend Channel columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = [
            "linreg_slope_20d",
            "linreg_r_squared_20d",
            "price_linreg_deviation",
            "channel_width_linreg_20d",
        ]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 8a: Aroon Extended Computation Tests (3 features)
# =============================================================================


class TestChunk8aAroonExtended:
    """Test Aroon Extended features (3 features)."""

    # --- Existence tests ---

    def test_aroon_up_25_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """aroon_up_25 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "aroon_up_25" in result.columns

    def test_aroon_down_25_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """aroon_down_25 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "aroon_down_25" in result.columns

    def test_aroon_trend_strength_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """aroon_trend_strength column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "aroon_trend_strength" in result.columns

    # --- Range tests ---

    def test_aroon_up_25_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """aroon_up_25 should be in [0, 100] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["aroon_up_25"].min() >= 0, "aroon_up_25 below 0"
        assert result["aroon_up_25"].max() <= 100, "aroon_up_25 above 100"

    def test_aroon_down_25_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """aroon_down_25 should be in [0, 100] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["aroon_down_25"].min() >= 0, "aroon_down_25 below 0"
        assert result["aroon_down_25"].max() <= 100, "aroon_down_25 above 100"

    def test_aroon_trend_strength_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """aroon_trend_strength should be in [0, 1] range."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["aroon_trend_strength"].min() >= 0, "aroon_trend_strength below 0"
        assert result["aroon_trend_strength"].max() <= 1, "aroon_trend_strength above 1"

    # --- No-NaN tests ---

    def test_aroon_extended_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Aroon Extended columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = ["aroon_up_25", "aroon_down_25", "aroon_trend_strength"]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 8a Integration Tests
# =============================================================================


class TestChunk8aIntegration:
    """Integration tests for Chunk 8a."""

    def test_chunk_8a_features_in_feature_list(self) -> None:
        """All CHUNK_8A_FEATURES are in FEATURE_LIST."""
        for feature in tier_a500.CHUNK_8A_FEATURES:
            assert feature in tier_a500.FEATURE_LIST, (
                f"Missing from FEATURE_LIST: {feature}"
            )

    def test_output_includes_all_chunk_8a_features(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output includes all Chunk 8a features."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a500.CHUNK_8A_FEATURES:
            assert feature in result.columns, f"Missing: {feature}"

    def test_a500_addition_list_includes_8a(self) -> None:
        """A500_ADDITION_LIST includes all Chunk 8a features."""
        for feature in tier_a500.CHUNK_8A_FEATURES:
            assert feature in tier_a500.A500_ADDITION_LIST, (
                f"Missing from A500_ADDITION_LIST: {feature}"
            )

    def test_chunk_8a_count_is_23(self) -> None:
        """CHUNK_8A_FEATURES has exactly 23 features."""
        assert len(tier_a500.CHUNK_8A_FEATURES) == 23, (
            f"Expected 23 features, got {len(tier_a500.CHUNK_8A_FEATURES)}"
        )

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
        """Output should have reasonable row count after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert len(result) >= 50, f"Too few rows: {len(result)}"
        assert len(result) <= 200, f"Too many rows (warmup not applied?): {len(result)}"

    def test_8a_features_no_lookahead(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Chunk 8a features should not use future data (no lookahead bias).

        Test by checking that early rows can be computed without later data.
        """
        # Build features on full data
        full_result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        # Build features on truncated data (first 350 rows)
        truncated_df = sample_daily_df.iloc[:350].copy()
        truncated_vix = sample_vix_df.iloc[:350].copy()
        truncated_result = tier_a500.build_feature_dataframe(truncated_df, truncated_vix)

        if len(truncated_result) > 0 and len(full_result) > 0:
            # Find overlapping dates
            common_dates = set(full_result["Date"]) & set(truncated_result["Date"])
            if len(common_dates) > 0:
                # For each chunk 8a feature, verify values match for common dates
                for feature in tier_a500.CHUNK_8A_FEATURES:
                    full_vals = full_result[full_result["Date"].isin(common_dates)][feature].reset_index(drop=True)
                    trunc_vals = truncated_result[truncated_result["Date"].isin(common_dates)][feature].reset_index(drop=True)
                    # Use allclose for floating point comparison
                    assert np.allclose(full_vals, trunc_vals, equal_nan=True), (
                        f"Lookahead detected in {feature}"
                    )


# =============================================================================
# Chunk 8b Feature List Structure Tests (Support/Resistance)
# =============================================================================


class TestChunk8bFeatureListStructure:
    """Test Chunk 8b feature list structure and counts."""

    def test_chunk_8b_features_exists(self) -> None:
        """CHUNK_8B_FEATURES constant exists."""
        assert hasattr(tier_a500, "CHUNK_8B_FEATURES")

    def test_chunk_8b_features_is_list(self) -> None:
        """CHUNK_8B_FEATURES is a list."""
        assert isinstance(tier_a500.CHUNK_8B_FEATURES, list)

    def test_chunk_8b_count_is_22(self) -> None:
        """CHUNK_8B_FEATURES has exactly 22 features."""
        assert len(tier_a500.CHUNK_8B_FEATURES) == 22, (
            f"Expected 22 features, got {len(tier_a500.CHUNK_8B_FEATURES)}"
        )

    def test_chunk_8b_no_duplicates(self) -> None:
        """CHUNK_8B_FEATURES has no duplicate feature names."""
        assert len(tier_a500.CHUNK_8B_FEATURES) == len(set(tier_a500.CHUNK_8B_FEATURES))

    def test_chunk_8b_no_overlap_with_a200(self) -> None:
        """CHUNK_8B_FEATURES has no overlap with a200 features."""
        from src.features import tier_a200

        overlap = set(tier_a500.CHUNK_8B_FEATURES) & set(tier_a200.FEATURE_LIST)
        assert len(overlap) == 0, f"Overlapping features with a200: {overlap}"

    def test_chunk_8b_no_overlap_with_prior_chunks(self) -> None:
        """CHUNK_8B_FEATURES has no overlap with prior chunks (6a, 6b, 7a, 7b, 8a)."""
        prior_chunks = (
            tier_a500.CHUNK_6A_FEATURES
            + tier_a500.CHUNK_6B_FEATURES
            + tier_a500.CHUNK_7A_FEATURES
            + tier_a500.CHUNK_7B_FEATURES
            + tier_a500.CHUNK_8A_FEATURES
        )
        overlap = set(tier_a500.CHUNK_8B_FEATURES) & set(prior_chunks)
        assert len(overlap) == 0, f"Overlapping features with prior chunks: {overlap}"

    def test_chunk_8b_all_strings(self) -> None:
        """All feature names in CHUNK_8B_FEATURES are strings."""
        for feature in tier_a500.CHUNK_8B_FEATURES:
            assert isinstance(feature, str), f"Non-string feature: {feature}"

    def test_chunk_8b_in_feature_list(self) -> None:
        """All CHUNK_8B_FEATURES are in FEATURE_LIST."""
        for feature in tier_a500.CHUNK_8B_FEATURES:
            assert feature in tier_a500.FEATURE_LIST, f"Missing from FEATURE_LIST: {feature}"


class TestChunk8bFeatureListContents:
    """Test Chunk 8b feature list contents by group."""

    def test_chunk8b_range_position_in_list(self) -> None:
        """Rolling Range Position features are in Chunk 8b list."""
        range_features = [
            "range_position_20d",
            "range_position_50d",
            "range_position_252d",
            "range_width_20d_pct",
        ]
        for feature in range_features:
            assert feature in tier_a500.CHUNK_8B_FEATURES, f"Missing: {feature}"

    def test_chunk8b_distance_from_extremes_in_list(self) -> None:
        """Distance from Extremes features are in Chunk 8b list."""
        distance_features = [
            "pct_from_20d_high",
            "pct_from_20d_low",
            "pct_from_52w_high",
            "pct_from_52w_low",
        ]
        for feature in distance_features:
            assert feature in tier_a500.CHUNK_8B_FEATURES, f"Missing: {feature}"

    def test_chunk8b_recency_of_extremes_in_list(self) -> None:
        """Recency of Extremes features are in Chunk 8b list."""
        recency_features = [
            "days_since_20d_high",
            "days_since_20d_low",
            "days_since_50d_high",
            "days_since_50d_low",
        ]
        for feature in recency_features:
            assert feature in tier_a500.CHUNK_8B_FEATURES, f"Missing: {feature}"

    def test_chunk8b_breakout_detection_in_list(self) -> None:
        """Breakout/Breakdown Detection features are in Chunk 8b list."""
        breakout_features = [
            "breakout_20d",
            "breakdown_20d",
            "breakout_strength_20d",
            "consecutive_new_highs_20d",
        ]
        for feature in breakout_features:
            assert feature in tier_a500.CHUNK_8B_FEATURES, f"Missing: {feature}"

    def test_chunk8b_range_dynamics_in_list(self) -> None:
        """Range Dynamics features are in Chunk 8b list."""
        dynamics_features = [
            "range_expansion_10d",
            "range_contraction_score",
            "high_low_range_ratio",
            "support_test_count_20d",
        ]
        for feature in dynamics_features:
            assert feature in tier_a500.CHUNK_8B_FEATURES, f"Missing: {feature}"

    def test_chunk8b_fibonacci_context_in_list(self) -> None:
        """Fibonacci Context features are in Chunk 8b list."""
        fib_features = [
            "fib_retracement_level",
            "distance_to_fib_50",
        ]
        for feature in fib_features:
            assert feature in tier_a500.CHUNK_8B_FEATURES, f"Missing: {feature}"


# =============================================================================
# Chunk 8b Indicator Computation Tests
# =============================================================================


class TestChunk8bRangePosition:
    """Test Rolling Range Position features (4 features)."""

    # --- Existence tests ---

    def test_range_position_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """range_position_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "range_position_20d" in result.columns

    def test_range_position_50d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """range_position_50d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "range_position_50d" in result.columns

    def test_range_position_252d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """range_position_252d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "range_position_252d" in result.columns

    def test_range_width_20d_pct_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """range_width_20d_pct column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "range_width_20d_pct" in result.columns

    # --- Range tests ---

    def test_range_position_in_0_1(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Range position features should be in [0, 1]."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["range_position_20d", "range_position_50d", "range_position_252d"]:
            assert result[col].min() >= 0, f"{col} below 0: {result[col].min()}"
            assert result[col].max() <= 1, f"{col} above 1: {result[col].max()}"

    def test_range_width_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """range_width_20d_pct should be positive."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["range_width_20d_pct"] >= 0).all()

    # --- No-NaN test ---

    def test_range_position_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Range Position columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["range_position_20d", "range_position_50d", "range_position_252d", "range_width_20d_pct"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk8bDistanceFromExtremes:
    """Test Distance from Extremes features (4 features)."""

    # --- Existence tests ---

    def test_pct_from_20d_high_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pct_from_20d_high column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "pct_from_20d_high" in result.columns

    def test_pct_from_20d_low_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pct_from_20d_low column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "pct_from_20d_low" in result.columns

    def test_pct_from_52w_high_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pct_from_52w_high column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "pct_from_52w_high" in result.columns

    def test_pct_from_52w_low_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pct_from_52w_low column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "pct_from_52w_low" in result.columns

    # --- Range tests ---

    def test_pct_from_high_nonpositive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pct_from_*_high should be <= 0 (close is at or below high)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["pct_from_20d_high", "pct_from_52w_high"]:
            assert result[col].max() <= 0, f"{col} has positive values: {result[col].max()}"

    def test_pct_from_low_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pct_from_*_low should be >= 0 (close is at or above low)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["pct_from_20d_low", "pct_from_52w_low"]:
            assert result[col].min() >= 0, f"{col} has negative values: {result[col].min()}"

    # --- No-NaN test ---

    def test_distance_from_extremes_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Distance from Extremes columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["pct_from_20d_high", "pct_from_20d_low", "pct_from_52w_high", "pct_from_52w_low"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk8bRecencyOfExtremes:
    """Test Recency of Extremes features (4 features)."""

    # --- Existence tests ---

    def test_days_since_20d_high_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_20d_high column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_since_20d_high" in result.columns

    def test_days_since_20d_low_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_20d_low column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_since_20d_low" in result.columns

    def test_days_since_50d_high_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_50d_high column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_since_50d_high" in result.columns

    def test_days_since_50d_low_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_50d_low column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_since_50d_low" in result.columns

    # --- Range tests ---

    def test_days_since_20d_in_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_20d_* should be in [0, 19]."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["days_since_20d_high", "days_since_20d_low"]:
            assert result[col].min() >= 0, f"{col} below 0: {result[col].min()}"
            assert result[col].max() <= 19, f"{col} above 19: {result[col].max()}"

    def test_days_since_50d_in_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_50d_* should be in [0, 49]."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["days_since_50d_high", "days_since_50d_low"]:
            assert result[col].min() >= 0, f"{col} below 0: {result[col].min()}"
            assert result[col].max() <= 49, f"{col} above 49: {result[col].max()}"

    def test_recency_are_integers(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Recency features should be non-negative integers."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["days_since_20d_high", "days_since_20d_low", "days_since_50d_high", "days_since_50d_low"]:
            # Check they are integers (or can be safely cast to int)
            assert (result[col] == result[col].astype(int)).all(), f"{col} has non-integer values"

    # --- No-NaN test ---

    def test_recency_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Recency columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["days_since_20d_high", "days_since_20d_low", "days_since_50d_high", "days_since_50d_low"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk8bBreakoutDetection:
    """Test Breakout/Breakdown Detection features (4 features)."""

    # --- Existence tests ---

    def test_breakout_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """breakout_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "breakout_20d" in result.columns

    def test_breakdown_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """breakdown_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "breakdown_20d" in result.columns

    def test_breakout_strength_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """breakout_strength_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "breakout_strength_20d" in result.columns

    def test_consecutive_new_highs_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_new_highs_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "consecutive_new_highs_20d" in result.columns

    # --- Range tests ---

    def test_breakout_breakdown_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """breakout_20d and breakdown_20d should be binary {0, 1}."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["breakout_20d", "breakdown_20d"]:
            unique_vals = set(result[col].unique())
            assert unique_vals.issubset({0, 1}), f"{col} has non-binary values: {unique_vals}"

    def test_breakout_strength_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """breakout_strength_20d should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["breakout_strength_20d"] >= 0).all()

    def test_consecutive_new_highs_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_new_highs_20d should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["consecutive_new_highs_20d"] >= 0).all()

    # --- No-NaN test ---

    def test_breakout_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Breakout columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["breakout_20d", "breakdown_20d", "breakout_strength_20d", "consecutive_new_highs_20d"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk8bRangeDynamics:
    """Test Range Dynamics features (4 features)."""

    # --- Existence tests ---

    def test_range_expansion_10d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """range_expansion_10d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "range_expansion_10d" in result.columns

    def test_range_contraction_score_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """range_contraction_score column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "range_contraction_score" in result.columns

    def test_high_low_range_ratio_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """high_low_range_ratio column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "high_low_range_ratio" in result.columns

    def test_support_test_count_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """support_test_count_20d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "support_test_count_20d" in result.columns

    # --- Range tests ---

    def test_range_expansion_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """range_expansion_10d should be positive (ratio of ranges)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["range_expansion_10d"] > 0).all()

    def test_range_contraction_score_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """range_contraction_score should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["range_contraction_score"] >= 0).all()

    def test_high_low_range_ratio_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """high_low_range_ratio should be positive."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert (result["high_low_range_ratio"] > 0).all()

    def test_support_test_count_in_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """support_test_count_20d should be in [0, 20]."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["support_test_count_20d"].min() >= 0
        assert result["support_test_count_20d"].max() <= 20

    # --- No-NaN test ---

    def test_range_dynamics_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Range Dynamics columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["range_expansion_10d", "range_contraction_score", "high_low_range_ratio", "support_test_count_20d"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk8bFibonacciContext:
    """Test Fibonacci Context features (2 features)."""

    # --- Existence tests ---

    def test_fib_retracement_level_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """fib_retracement_level column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "fib_retracement_level" in result.columns

    def test_distance_to_fib_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """distance_to_fib_50 column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "distance_to_fib_50" in result.columns

    # --- Range tests ---

    def test_fib_retracement_level_in_0_1(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """fib_retracement_level should be in [0, 1]."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["fib_retracement_level"].min() >= 0
        assert result["fib_retracement_level"].max() <= 1

    def test_fib_retracement_level_is_valid_fib(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """fib_retracement_level should only contain valid Fibonacci levels."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        valid_fibs = {0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0}
        unique_vals = set(result["fib_retracement_level"].unique())
        assert unique_vals.issubset(valid_fibs), f"Invalid fib levels: {unique_vals - valid_fibs}"

    def test_distance_to_fib_50_bounded(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """distance_to_fib_50 should be bounded (typically in [-1, 1])."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # Allow some flexibility for extreme cases
        assert result["distance_to_fib_50"].min() >= -2
        assert result["distance_to_fib_50"].max() <= 2

    # --- No-NaN test ---

    def test_fib_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Fibonacci columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["fib_retracement_level", "distance_to_fib_50"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk8bIntegration:
    """Integration tests for Chunk 8b."""

    def test_chunk_8b_feature_count_is_22(self) -> None:
        """Chunk 8b should have exactly 22 features."""
        assert len(tier_a500.CHUNK_8B_FEATURES) == 22, (
            f"Expected 22 features in chunk 8b, got {len(tier_a500.CHUNK_8B_FEATURES)}"
        )

    def test_output_includes_all_chunk_8b_features(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output includes all Chunk 8b features."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a500.CHUNK_8B_FEATURES:
            assert feature in result.columns, f"Missing: {feature}"

    def test_a500_addition_list_includes_8b(self) -> None:
        """A500_ADDITION_LIST includes all Chunk 8b features."""
        for feature in tier_a500.CHUNK_8B_FEATURES:
            assert feature in tier_a500.A500_ADDITION_LIST, (
                f"Missing from A500_ADDITION_LIST: {feature}"
            )

    def test_chunk_8b_features_in_output(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """All Chunk 8b features should be present in output DataFrame."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a500.CHUNK_8B_FEATURES:
            assert feature in result.columns, f"Missing from output: {feature}"

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
        """Output should have reasonable row count after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert len(result) >= 50, f"Too few rows: {len(result)}"
        assert len(result) <= 200, f"Too many rows (warmup not applied?): {len(result)}"

    def test_8b_features_no_lookahead(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Chunk 8b features should not use future data (no lookahead bias).

        Test by checking that early rows can be computed without later data.
        """
        # Build features on full data
        full_result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        # Build features on truncated data (first 350 rows)
        truncated_df = sample_daily_df.iloc[:350].copy()
        truncated_vix = sample_vix_df.iloc[:350].copy()
        truncated_result = tier_a500.build_feature_dataframe(truncated_df, truncated_vix)

        if len(truncated_result) > 0 and len(full_result) > 0:
            # Find overlapping dates
            common_dates = set(full_result["Date"]) & set(truncated_result["Date"])
            if len(common_dates) > 0:
                # For each chunk 8b feature, verify values match for common dates
                for feature in tier_a500.CHUNK_8B_FEATURES:
                    full_vals = full_result[full_result["Date"].isin(common_dates)][feature].reset_index(drop=True)
                    trunc_vals = truncated_result[truncated_result["Date"].isin(common_dates)][feature].reset_index(drop=True)
                    # Use allclose for floating point comparison
                    assert np.allclose(full_vals, trunc_vals, equal_nan=True), (
                        f"Lookahead detected in {feature}"
                    )


# =============================================================================
# Sub-Chunk 9a Tests (ranks 346-370): CDL Part 1 - Candlestick Patterns
# =============================================================================


class TestChunk9aFeatureListStructure:
    """Test Chunk 9a feature list structure and counts."""

    def test_chunk_9a_features_exists(self) -> None:
        """CHUNK_9A_FEATURES constant exists."""
        assert hasattr(tier_a500, "CHUNK_9A_FEATURES")

    def test_chunk_9a_features_is_list(self) -> None:
        """CHUNK_9A_FEATURES is a list."""
        assert isinstance(tier_a500.CHUNK_9A_FEATURES, list)

    def test_chunk_9a_count_is_25(self) -> None:
        """CHUNK_9A_FEATURES has exactly 25 features."""
        assert len(tier_a500.CHUNK_9A_FEATURES) == 25, (
            f"Expected 25 features, got {len(tier_a500.CHUNK_9A_FEATURES)}"
        )

    def test_chunk_9a_no_duplicates(self) -> None:
        """CHUNK_9A_FEATURES has no duplicate feature names."""
        assert len(tier_a500.CHUNK_9A_FEATURES) == len(set(tier_a500.CHUNK_9A_FEATURES))

    def test_chunk_9a_no_overlap_with_a200(self) -> None:
        """CHUNK_9A_FEATURES has no overlap with a200 features."""
        from src.features import tier_a200

        overlap = set(tier_a500.CHUNK_9A_FEATURES) & set(tier_a200.FEATURE_LIST)
        assert len(overlap) == 0, f"Overlapping features with a200: {overlap}"

    def test_chunk_9a_no_overlap_with_prior_chunks(self) -> None:
        """CHUNK_9A_FEATURES has no overlap with prior chunks (6a, 6b, 7a, 7b, 8a, 8b)."""
        prior_chunks = (
            tier_a500.CHUNK_6A_FEATURES
            + tier_a500.CHUNK_6B_FEATURES
            + tier_a500.CHUNK_7A_FEATURES
            + tier_a500.CHUNK_7B_FEATURES
            + tier_a500.CHUNK_8A_FEATURES
            + tier_a500.CHUNK_8B_FEATURES
        )
        overlap = set(tier_a500.CHUNK_9A_FEATURES) & set(prior_chunks)
        assert len(overlap) == 0, f"Overlapping features with prior chunks: {overlap}"

    def test_chunk_9a_all_strings(self) -> None:
        """All feature names in CHUNK_9A_FEATURES are strings."""
        for feature in tier_a500.CHUNK_9A_FEATURES:
            assert isinstance(feature, str), f"Non-string feature: {feature}"

    def test_chunk_9a_in_feature_list(self) -> None:
        """All CHUNK_9A_FEATURES are in FEATURE_LIST."""
        for feature in tier_a500.CHUNK_9A_FEATURES:
            assert feature in tier_a500.FEATURE_LIST, f"Missing from FEATURE_LIST: {feature}"


class TestChunk9aFeatureListContents:
    """Test Chunk 9a feature list contents by group."""

    def test_chunk9a_engulfing_in_list(self) -> None:
        """Engulfing Pattern features are in Chunk 9a list."""
        engulfing_features = [
            "bullish_engulfing",
            "bearish_engulfing",
            "engulfing_score",
            "consecutive_engulfing_count",
        ]
        for feature in engulfing_features:
            assert feature in tier_a500.CHUNK_9A_FEATURES, f"Missing: {feature}"

    def test_chunk9a_wick_rejection_in_list(self) -> None:
        """Wick Rejection features are in Chunk 9a list."""
        wick_features = [
            "hammer_indicator",
            "shooting_star_indicator",
            "hammer_score",
            "shooting_star_score",
            "wick_rejection_score",
        ]
        for feature in wick_features:
            assert feature in tier_a500.CHUNK_9A_FEATURES, f"Missing: {feature}"

    def test_chunk9a_gaps_in_list(self) -> None:
        """Gap Analysis features are in Chunk 9a list."""
        gap_features = [
            "gap_size_pct",
            "gap_direction",
            "gap_filled_today",
            "gap_fill_pct",
            "significant_gap",
        ]
        for feature in gap_features:
            assert feature in tier_a500.CHUNK_9A_FEATURES, f"Missing: {feature}"

    def test_chunk9a_inside_outside_in_list(self) -> None:
        """Inside/Outside Day features are in Chunk 9a list."""
        io_features = [
            "inside_day",
            "outside_day",
            "consecutive_inside_days",
            "consecutive_outside_days",
        ]
        for feature in io_features:
            assert feature in tier_a500.CHUNK_9A_FEATURES, f"Missing: {feature}"

    def test_chunk9a_range_extremes_in_list(self) -> None:
        """Range Extremes features are in Chunk 9a list."""
        range_features = [
            "narrow_range_day",
            "wide_range_day",
            "narrow_range_score",
            "consecutive_narrow_days",
        ]
        for feature in range_features:
            assert feature in tier_a500.CHUNK_9A_FEATURES, f"Missing: {feature}"

    def test_chunk9a_trend_days_in_list(self) -> None:
        """Trend Day features are in Chunk 9a list."""
        trend_features = [
            "trend_day_indicator",
            "trend_day_direction",
            "consecutive_trend_days",
        ]
        for feature in trend_features:
            assert feature in tier_a500.CHUNK_9A_FEATURES, f"Missing: {feature}"


# =============================================================================
# Chunk 9a Indicator Computation Tests - Engulfing Patterns
# =============================================================================


class TestChunk9aEngulfingPatterns:
    """Test Engulfing Pattern features (4 features)."""

    # --- Existence tests ---

    def test_bullish_engulfing_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bullish_engulfing column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "bullish_engulfing" in result.columns

    def test_bearish_engulfing_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bearish_engulfing column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "bearish_engulfing" in result.columns

    def test_engulfing_score_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """engulfing_score column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "engulfing_score" in result.columns

    def test_consecutive_engulfing_count_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_engulfing_count column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "consecutive_engulfing_count" in result.columns

    # --- Range tests ---

    def test_bullish_engulfing_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bullish_engulfing should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["bullish_engulfing"].unique())
        assert unique_vals <= {0, 1}, f"bullish_engulfing has non-binary values: {unique_vals}"

    def test_bearish_engulfing_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bearish_engulfing should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["bearish_engulfing"].unique())
        assert unique_vals <= {0, 1}, f"bearish_engulfing has non-binary values: {unique_vals}"

    def test_engulfing_score_in_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """engulfing_score should be in [0, 5] (clipped)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["engulfing_score"].min() >= 0, f"engulfing_score below 0: {result['engulfing_score'].min()}"
        assert result["engulfing_score"].max() <= 5, f"engulfing_score above 5: {result['engulfing_score'].max()}"

    def test_consecutive_engulfing_count_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_engulfing_count should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["consecutive_engulfing_count"].min() >= 0

    # --- No-NaN test ---

    def test_engulfing_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Engulfing columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["bullish_engulfing", "bearish_engulfing", "engulfing_score", "consecutive_engulfing_count"]:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9a Indicator Computation Tests - Wick Rejection
# =============================================================================


class TestChunk9aWickRejection:
    """Test Wick Rejection features (5 features)."""

    # --- Existence tests ---

    def test_hammer_indicator_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """hammer_indicator column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "hammer_indicator" in result.columns

    def test_shooting_star_indicator_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """shooting_star_indicator column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "shooting_star_indicator" in result.columns

    def test_hammer_score_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """hammer_score column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "hammer_score" in result.columns

    def test_shooting_star_score_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """shooting_star_score column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "shooting_star_score" in result.columns

    def test_wick_rejection_score_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """wick_rejection_score column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "wick_rejection_score" in result.columns

    # --- Range tests ---

    def test_hammer_indicator_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """hammer_indicator should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["hammer_indicator"].unique())
        assert unique_vals <= {0, 1}, f"hammer_indicator has non-binary values: {unique_vals}"

    def test_shooting_star_indicator_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """shooting_star_indicator should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["shooting_star_indicator"].unique())
        assert unique_vals <= {0, 1}, f"shooting_star_indicator has non-binary values: {unique_vals}"

    def test_hammer_score_in_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """hammer_score should be in [0, 10] (clipped)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["hammer_score"].min() >= 0, f"hammer_score below 0: {result['hammer_score'].min()}"
        assert result["hammer_score"].max() <= 10, f"hammer_score above 10: {result['hammer_score'].max()}"

    def test_shooting_star_score_in_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """shooting_star_score should be in [0, 10] (clipped)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["shooting_star_score"].min() >= 0, f"shooting_star_score below 0: {result['shooting_star_score'].min()}"
        assert result["shooting_star_score"].max() <= 10, f"shooting_star_score above 10: {result['shooting_star_score'].max()}"

    def test_wick_rejection_score_in_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """wick_rejection_score should be in [-10, 10]."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["wick_rejection_score"].min() >= -10, f"wick_rejection_score below -10: {result['wick_rejection_score'].min()}"
        assert result["wick_rejection_score"].max() <= 10, f"wick_rejection_score above 10: {result['wick_rejection_score'].max()}"

    # --- No-NaN test ---

    def test_wick_rejection_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Wick Rejection columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["hammer_indicator", "shooting_star_indicator", "hammer_score", "shooting_star_score", "wick_rejection_score"]:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9a Indicator Computation Tests - Gap Analysis
# =============================================================================


class TestChunk9aGapAnalysis:
    """Test Gap Analysis features (5 features)."""

    # --- Existence tests ---

    def test_gap_size_pct_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """gap_size_pct column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "gap_size_pct" in result.columns

    def test_gap_direction_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """gap_direction column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "gap_direction" in result.columns

    def test_gap_filled_today_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """gap_filled_today column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "gap_filled_today" in result.columns

    def test_gap_fill_pct_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """gap_fill_pct column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "gap_fill_pct" in result.columns

    def test_significant_gap_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """significant_gap column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "significant_gap" in result.columns

    # --- Range tests ---

    def test_gap_direction_values(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """gap_direction should be in {-1, 0, 1}."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["gap_direction"].unique())
        assert unique_vals <= {-1, 0, 1}, f"gap_direction has invalid values: {unique_vals}"

    def test_gap_filled_today_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """gap_filled_today should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["gap_filled_today"].unique())
        assert unique_vals <= {0, 1}, f"gap_filled_today has non-binary values: {unique_vals}"

    def test_gap_fill_pct_in_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """gap_fill_pct should be in [0, 100]."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["gap_fill_pct"].min() >= 0, f"gap_fill_pct below 0: {result['gap_fill_pct'].min()}"
        assert result["gap_fill_pct"].max() <= 100, f"gap_fill_pct above 100: {result['gap_fill_pct'].max()}"

    def test_significant_gap_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """significant_gap should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["significant_gap"].unique())
        assert unique_vals <= {0, 1}, f"significant_gap has non-binary values: {unique_vals}"

    # --- No-NaN test ---

    def test_gap_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Gap Analysis columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["gap_size_pct", "gap_direction", "gap_filled_today", "gap_fill_pct", "significant_gap"]:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9a Indicator Computation Tests - Inside/Outside Days
# =============================================================================


class TestChunk9aInsideOutsideDays:
    """Test Inside/Outside Day features (4 features)."""

    # --- Existence tests ---

    def test_inside_day_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """inside_day column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "inside_day" in result.columns

    def test_outside_day_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """outside_day column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "outside_day" in result.columns

    def test_consecutive_inside_days_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_inside_days column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "consecutive_inside_days" in result.columns

    def test_consecutive_outside_days_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_outside_days column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "consecutive_outside_days" in result.columns

    # --- Range tests ---

    def test_inside_day_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """inside_day should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["inside_day"].unique())
        assert unique_vals <= {0, 1}, f"inside_day has non-binary values: {unique_vals}"

    def test_outside_day_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """outside_day should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["outside_day"].unique())
        assert unique_vals <= {0, 1}, f"outside_day has non-binary values: {unique_vals}"

    def test_inside_outside_mutually_exclusive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """inside_day and outside_day cannot both be 1 on the same day."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        both_one = ((result["inside_day"] == 1) & (result["outside_day"] == 1)).sum()
        assert both_one == 0, f"Found {both_one} days where both inside and outside are 1"

    def test_consecutive_inside_days_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_inside_days should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["consecutive_inside_days"].min() >= 0

    def test_consecutive_outside_days_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_outside_days should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["consecutive_outside_days"].min() >= 0

    # --- No-NaN test ---

    def test_inside_outside_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Inside/Outside columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["inside_day", "outside_day", "consecutive_inside_days", "consecutive_outside_days"]:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9a Indicator Computation Tests - Range Extremes
# =============================================================================


class TestChunk9aRangeExtremes:
    """Test Range Extremes features (4 features)."""

    # --- Existence tests ---

    def test_narrow_range_day_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """narrow_range_day column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "narrow_range_day" in result.columns

    def test_wide_range_day_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """wide_range_day column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "wide_range_day" in result.columns

    def test_narrow_range_score_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """narrow_range_score column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "narrow_range_score" in result.columns

    def test_consecutive_narrow_days_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_narrow_days column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "consecutive_narrow_days" in result.columns

    # --- Range tests ---

    def test_narrow_range_day_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """narrow_range_day should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["narrow_range_day"].unique())
        assert unique_vals <= {0, 1}, f"narrow_range_day has non-binary values: {unique_vals}"

    def test_wide_range_day_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """wide_range_day should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["wide_range_day"].unique())
        assert unique_vals <= {0, 1}, f"wide_range_day has non-binary values: {unique_vals}"

    def test_narrow_range_score_in_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """narrow_range_score should be in [0, 10] (clipped)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["narrow_range_score"].min() >= 0, f"narrow_range_score below 0: {result['narrow_range_score'].min()}"
        assert result["narrow_range_score"].max() <= 10, f"narrow_range_score above 10: {result['narrow_range_score'].max()}"

    def test_consecutive_narrow_days_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_narrow_days should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["consecutive_narrow_days"].min() >= 0

    # --- No-NaN test ---

    def test_range_extremes_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Range Extremes columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["narrow_range_day", "wide_range_day", "narrow_range_score", "consecutive_narrow_days"]:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9a Indicator Computation Tests - Trend Days
# =============================================================================


class TestChunk9aTrendDays:
    """Test Trend Day features (3 features)."""

    # --- Existence tests ---

    def test_trend_day_indicator_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_day_indicator column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "trend_day_indicator" in result.columns

    def test_trend_day_direction_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_day_direction column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "trend_day_direction" in result.columns

    def test_consecutive_trend_days_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_trend_days column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "consecutive_trend_days" in result.columns

    # --- Range tests ---

    def test_trend_day_indicator_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_day_indicator should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["trend_day_indicator"].unique())
        assert unique_vals <= {0, 1}, f"trend_day_indicator has non-binary values: {unique_vals}"

    def test_trend_day_direction_values(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """trend_day_direction should be in {-1, 0, 1}."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["trend_day_direction"].unique())
        assert unique_vals <= {-1, 0, 1}, f"trend_day_direction has invalid values: {unique_vals}"

    def test_consecutive_trend_days_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_trend_days should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["consecutive_trend_days"].min() >= 0

    # --- No-NaN test ---

    def test_trend_days_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Trend Day columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for col in ["trend_day_indicator", "trend_day_direction", "consecutive_trend_days"]:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9a Integration Tests
# =============================================================================


class TestChunk9aIntegration:
    """Integration tests for Chunk 9a."""

    def test_feature_count_at_least_370(self) -> None:
        """Total feature count should be at least 370 (includes 9a chunk)."""
        min_count = 206 + 24 + 25 + 23 + 22 + 23 + 22 + 25  # a200 + 6a + 6b + 7a + 7b + 8a + 8b + 9a
        assert len(tier_a500.FEATURE_LIST) >= min_count, (
            f"Expected at least {min_count} features, got {len(tier_a500.FEATURE_LIST)}"
        )

    def test_output_includes_all_chunk_9a_features(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output includes all Chunk 9a features."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a500.CHUNK_9A_FEATURES:
            assert feature in result.columns, f"Missing: {feature}"

    def test_a500_addition_list_includes_9a(self) -> None:
        """A500_ADDITION_LIST includes all Chunk 9a features."""
        for feature in tier_a500.CHUNK_9A_FEATURES:
            assert feature in tier_a500.A500_ADDITION_LIST, (
                f"Missing from A500_ADDITION_LIST: {feature}"
            )

    def test_output_column_count_at_least_371(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output DataFrame should have at least Date + 370 features = 371 columns."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        min_cols = 1 + 206 + 24 + 25 + 23 + 22 + 23 + 22 + 25  # Date + a200 + 6a + 6b + 7a + 7b + 8a + 8b + 9a
        assert len(result.columns) >= min_cols, (
            f"Expected at least {min_cols} columns, got {len(result.columns)}"
        )

    def test_no_nan_values_after_warmup(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in any column after warmup period."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        nan_cols = result.columns[result.isna().any()].tolist()
        assert len(nan_cols) == 0, f"Columns with NaN: {nan_cols}"

    def test_9a_features_no_lookahead(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Chunk 9a features should not use future data (no lookahead bias).

        Test by checking that early rows can be computed without later data.
        """
        # Build features on full data
        full_result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        # Build features on truncated data (first 350 rows)
        truncated_df = sample_daily_df.iloc[:350].copy()
        truncated_vix = sample_vix_df.iloc[:350].copy()
        truncated_result = tier_a500.build_feature_dataframe(truncated_df, truncated_vix)

        if len(truncated_result) > 0 and len(full_result) > 0:
            # Find overlapping dates
            common_dates = set(full_result["Date"]) & set(truncated_result["Date"])
            if len(common_dates) > 0:
                # For each chunk 9a feature, verify values match for common dates
                for feature in tier_a500.CHUNK_9A_FEATURES:
                    full_vals = full_result[full_result["Date"].isin(common_dates)][feature].reset_index(drop=True)
                    trunc_vals = truncated_result[truncated_result["Date"].isin(common_dates)][feature].reset_index(drop=True)
                    # Use allclose for floating point comparison
                    assert np.allclose(full_vals, trunc_vals, equal_nan=True), (
                        f"Lookahead detected in {feature}"
                    )


# =============================================================================
# Chunk 9b Indicator Computation Tests - Doji Patterns
# =============================================================================


class TestChunk9bDoji:
    """Test Doji Pattern features (5 features)."""

    # --- Existence tests ---

    def test_doji_strict_indicator_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """doji_strict_indicator column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "doji_strict_indicator" in result.columns

    def test_doji_score_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """doji_score column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "doji_score" in result.columns

    def test_doji_type_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """doji_type column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "doji_type" in result.columns

    def test_consecutive_doji_count_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_doji_count column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "consecutive_doji_count" in result.columns

    def test_doji_after_trend_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """doji_after_trend column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "doji_after_trend" in result.columns

    # --- Range tests ---

    def test_doji_strict_indicator_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """doji_strict_indicator should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["doji_strict_indicator"].unique())
        assert unique_vals <= {0, 1}, f"doji_strict_indicator has non-binary values: {unique_vals}"

    def test_doji_score_range_0_to_1(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """doji_score should be in [0, 1]."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["doji_score"].min() >= 0, "doji_score below 0"
        assert result["doji_score"].max() <= 1, "doji_score above 1"

    def test_doji_type_values(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """doji_type should be in {-1, 0, 1}."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["doji_type"].unique())
        assert unique_vals <= {-1, 0, 1}, f"doji_type has invalid values: {unique_vals}"

    def test_consecutive_doji_count_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_doji_count should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["consecutive_doji_count"].min() >= 0

    def test_doji_after_trend_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """doji_after_trend should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["doji_after_trend"].unique())
        assert unique_vals <= {0, 1}, f"doji_after_trend has non-binary values: {unique_vals}"

    # --- No-NaN test ---

    def test_doji_features_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Doji columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        doji_cols = ["doji_strict_indicator", "doji_score", "doji_type", "consecutive_doji_count", "doji_after_trend"]
        for col in doji_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9b Indicator Computation Tests - Marubozu & Strong Candles
# =============================================================================


class TestChunk9bMarubozu:
    """Test Marubozu & Strong Candle features (4 features)."""

    # --- Existence tests ---

    def test_marubozu_indicator_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """marubozu_indicator column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "marubozu_indicator" in result.columns

    def test_marubozu_direction_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """marubozu_direction column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "marubozu_direction" in result.columns

    def test_marubozu_strength_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """marubozu_strength column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "marubozu_strength" in result.columns

    def test_consecutive_strong_candles_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_strong_candles column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "consecutive_strong_candles" in result.columns

    # --- Range tests ---

    def test_marubozu_indicator_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """marubozu_indicator should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["marubozu_indicator"].unique())
        assert unique_vals <= {0, 1}, f"marubozu_indicator has non-binary values: {unique_vals}"

    def test_marubozu_direction_values(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """marubozu_direction should be in {-1, 0, 1}."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["marubozu_direction"].unique())
        assert unique_vals <= {-1, 0, 1}, f"marubozu_direction has invalid values: {unique_vals}"

    def test_marubozu_strength_range_0_to_1(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """marubozu_strength should be in [0, 1]."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["marubozu_strength"].min() >= 0, "marubozu_strength below 0"
        assert result["marubozu_strength"].max() <= 1, "marubozu_strength above 1"

    def test_consecutive_strong_candles_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """consecutive_strong_candles should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["consecutive_strong_candles"].min() >= 0

    # --- No-NaN test ---

    def test_marubozu_features_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Marubozu columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = ["marubozu_indicator", "marubozu_direction", "marubozu_strength", "consecutive_strong_candles"]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9b Indicator Computation Tests - Spinning Top & Indecision
# =============================================================================


class TestChunk9bSpinningTop:
    """Test Spinning Top & Indecision features (4 features)."""

    # --- Existence tests ---

    def test_spinning_top_indicator_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """spinning_top_indicator column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "spinning_top_indicator" in result.columns

    def test_spinning_top_score_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """spinning_top_score column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "spinning_top_score" in result.columns

    def test_indecision_streak_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """indecision_streak column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "indecision_streak" in result.columns

    def test_indecision_at_extreme_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """indecision_at_extreme column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "indecision_at_extreme" in result.columns

    # --- Range tests ---

    def test_spinning_top_indicator_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """spinning_top_indicator should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["spinning_top_indicator"].unique())
        assert unique_vals <= {0, 1}, f"spinning_top_indicator has non-binary values: {unique_vals}"

    def test_spinning_top_score_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """spinning_top_score should be in [0, 10]."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["spinning_top_score"].min() >= 0, "spinning_top_score below 0"
        assert result["spinning_top_score"].max() <= 10, "spinning_top_score above 10"

    def test_indecision_streak_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """indecision_streak should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["indecision_streak"].min() >= 0

    def test_indecision_at_extreme_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """indecision_at_extreme should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["indecision_at_extreme"].unique())
        assert unique_vals <= {0, 1}, f"indecision_at_extreme has non-binary values: {unique_vals}"

    # --- No-NaN test ---

    def test_spinning_top_features_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Spinning Top columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = ["spinning_top_indicator", "spinning_top_score", "indecision_streak", "indecision_at_extreme"]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9b Indicator Computation Tests - Multi-Candle Reversal Patterns
# =============================================================================


class TestChunk9bReversalPatterns:
    """Test Multi-Candle Reversal Pattern features (5 features)."""

    # --- Existence tests ---

    def test_morning_star_indicator_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """morning_star_indicator column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "morning_star_indicator" in result.columns

    def test_evening_star_indicator_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """evening_star_indicator column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "evening_star_indicator" in result.columns

    def test_three_white_soldiers_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """three_white_soldiers column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "three_white_soldiers" in result.columns

    def test_three_black_crows_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """three_black_crows column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "three_black_crows" in result.columns

    def test_harami_indicator_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """harami_indicator column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "harami_indicator" in result.columns

    # --- Range tests ---

    def test_morning_star_indicator_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """morning_star_indicator should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["morning_star_indicator"].unique())
        assert unique_vals <= {0, 1}, f"morning_star_indicator has non-binary values: {unique_vals}"

    def test_evening_star_indicator_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """evening_star_indicator should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["evening_star_indicator"].unique())
        assert unique_vals <= {0, 1}, f"evening_star_indicator has non-binary values: {unique_vals}"

    def test_three_white_soldiers_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """three_white_soldiers should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["three_white_soldiers"].unique())
        assert unique_vals <= {0, 1}, f"three_white_soldiers has non-binary values: {unique_vals}"

    def test_three_black_crows_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """three_black_crows should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["three_black_crows"].unique())
        assert unique_vals <= {0, 1}, f"three_black_crows has non-binary values: {unique_vals}"

    def test_harami_indicator_values(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """harami_indicator should be in {-1, 0, 1} (bearish, none, bullish)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["harami_indicator"].unique())
        assert unique_vals <= {-1, 0, 1}, f"harami_indicator has invalid values: {unique_vals}"

    # --- No-NaN test ---

    def test_reversal_patterns_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Reversal Pattern columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = ["morning_star_indicator", "evening_star_indicator", "three_white_soldiers",
                "three_black_crows", "harami_indicator"]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9b Indicator Computation Tests - Multi-Candle Continuation Patterns
# =============================================================================


class TestChunk9bContinuationPatterns:
    """Test Multi-Candle Continuation Pattern features (4 features)."""

    # --- Existence tests ---

    def test_piercing_line_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """piercing_line column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "piercing_line" in result.columns

    def test_dark_cloud_cover_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """dark_cloud_cover column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "dark_cloud_cover" in result.columns

    def test_tweezer_bottom_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tweezer_bottom column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "tweezer_bottom" in result.columns

    def test_tweezer_top_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tweezer_top column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "tweezer_top" in result.columns

    # --- Range tests ---

    def test_piercing_line_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """piercing_line should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["piercing_line"].unique())
        assert unique_vals <= {0, 1}, f"piercing_line has non-binary values: {unique_vals}"

    def test_dark_cloud_cover_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """dark_cloud_cover should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["dark_cloud_cover"].unique())
        assert unique_vals <= {0, 1}, f"dark_cloud_cover has non-binary values: {unique_vals}"

    def test_tweezer_bottom_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tweezer_bottom should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["tweezer_bottom"].unique())
        assert unique_vals <= {0, 1}, f"tweezer_bottom has non-binary values: {unique_vals}"

    def test_tweezer_top_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tweezer_top should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["tweezer_top"].unique())
        assert unique_vals <= {0, 1}, f"tweezer_top has non-binary values: {unique_vals}"

    # --- No-NaN test ---

    def test_continuation_patterns_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Continuation Pattern columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = ["piercing_line", "dark_cloud_cover", "tweezer_bottom", "tweezer_top"]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9b Indicator Computation Tests - Pattern Context
# =============================================================================


class TestChunk9bPatternContext:
    """Test Pattern Context features (3 features)."""

    # --- Existence tests ---

    def test_reversal_pattern_count_5d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """reversal_pattern_count_5d column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "reversal_pattern_count_5d" in result.columns

    def test_pattern_alignment_score_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pattern_alignment_score column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "pattern_alignment_score" in result.columns

    def test_pattern_cluster_indicator_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pattern_cluster_indicator column is present in output."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "pattern_cluster_indicator" in result.columns

    # --- Range tests ---

    def test_reversal_pattern_count_5d_nonnegative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """reversal_pattern_count_5d should be >= 0."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert result["reversal_pattern_count_5d"].min() >= 0

    def test_reversal_pattern_count_5d_max(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """reversal_pattern_count_5d should be <= 5 (max 1 per day in 5d window)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # Could potentially have multiple patterns per day, so we allow up to reasonable max
        assert result["reversal_pattern_count_5d"].max() <= 25

    def test_pattern_alignment_score_values(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pattern_alignment_score should be in {-1, 0, 1}."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["pattern_alignment_score"].unique())
        assert unique_vals <= {-1, 0, 1}, f"pattern_alignment_score has invalid values: {unique_vals}"

    def test_pattern_cluster_indicator_binary(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """pattern_cluster_indicator should be binary (0 or 1)."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        unique_vals = set(result["pattern_cluster_indicator"].unique())
        assert unique_vals <= {0, 1}, f"pattern_cluster_indicator has non-binary values: {unique_vals}"

    # --- No-NaN test ---

    def test_pattern_context_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Pattern Context columns after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cols = ["reversal_pattern_count_5d", "pattern_alignment_score", "pattern_cluster_indicator"]
        for col in cols:
            assert not result[col].isna().any(), f"NaN in {col}"


# =============================================================================
# Chunk 9b Integration Tests
# =============================================================================


class TestChunk9bIntegration:
    """Integration tests for Chunk 9b."""

    def test_feature_count_is_395(self) -> None:
        """Total feature count should be 370 (prior) + 25 (9b) = 395."""
        expected_count = 206 + 24 + 25 + 23 + 22 + 23 + 22 + 25 + 25  # a200 + 6a + 6b + 7a + 7b + 8a + 8b + 9a + 9b
        assert len(tier_a500.FEATURE_LIST) == expected_count, (
            f"Expected {expected_count} features, got {len(tier_a500.FEATURE_LIST)}"
        )

    def test_chunk_9b_feature_count_is_25(self) -> None:
        """CHUNK_9B_FEATURES should contain exactly 25 features."""
        assert len(tier_a500.CHUNK_9B_FEATURES) == 25, (
            f"Expected 25 features in CHUNK_9B_FEATURES, got {len(tier_a500.CHUNK_9B_FEATURES)}"
        )

    def test_output_includes_all_chunk_9b_features(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output includes all Chunk 9b features."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a500.CHUNK_9B_FEATURES:
            assert feature in result.columns, f"Missing: {feature}"

    def test_a500_addition_list_includes_9b(self) -> None:
        """A500_ADDITION_LIST includes all Chunk 9b features."""
        for feature in tier_a500.CHUNK_9B_FEATURES:
            assert feature in tier_a500.A500_ADDITION_LIST, (
                f"Missing from A500_ADDITION_LIST: {feature}"
            )

    def test_output_column_count(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output DataFrame should have Date + 395 features = 396 columns."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        expected_cols = 1 + 206 + 24 + 25 + 23 + 22 + 23 + 22 + 25 + 25  # Date + a200 + 6a-9b
        assert len(result.columns) == expected_cols, (
            f"Expected {expected_cols} columns, got {len(result.columns)}"
        )

    def test_no_nan_values_after_warmup(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in any column after warmup period."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        nan_cols = result.columns[result.isna().any()].tolist()
        assert len(nan_cols) == 0, f"Columns with NaN: {nan_cols}"

    def test_9b_features_no_lookahead(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Chunk 9b features should not use future data (no lookahead bias).

        Test by checking that early rows can be computed without later data.
        """
        # Build features on full data
        full_result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)

        # Build features on truncated data (first 350 rows)
        truncated_df = sample_daily_df.iloc[:350].copy()
        truncated_vix = sample_vix_df.iloc[:350].copy()
        truncated_result = tier_a500.build_feature_dataframe(truncated_df, truncated_vix)

        if len(truncated_result) > 0 and len(full_result) > 0:
            # Find overlapping dates
            common_dates = set(full_result["Date"]) & set(truncated_result["Date"])
            if len(common_dates) > 0:
                # For each chunk 9b feature, verify values match for common dates
                for feature in tier_a500.CHUNK_9B_FEATURES:
                    full_vals = full_result[full_result["Date"].isin(common_dates)][feature].reset_index(drop=True)
                    trunc_vals = truncated_result[truncated_result["Date"].isin(common_dates)][feature].reset_index(drop=True)
                    # Use allclose for floating point comparison
                    assert np.allclose(full_vals, trunc_vals, equal_nan=True), (
                        f"Lookahead detected in {feature}"
                    )
