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

    def test_feature_count_is_300(self) -> None:
        """Total feature count should be 278 (prior) + 22 (7b) = 300."""
        expected_count = 206 + 24 + 25 + 23 + 22  # a200 + 6a + 6b + 7a + 7b
        assert len(tier_a500.FEATURE_LIST) == expected_count, (
            f"Expected {expected_count} features, got {len(tier_a500.FEATURE_LIST)}"
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

    def test_output_column_count(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output DataFrame should have Date + 300 features = 301 columns."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        expected_cols = 1 + 206 + 24 + 25 + 23 + 22  # Date + a200 + 6a + 6b + 7a + 7b
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

    def test_output_row_count_reasonable(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output should have reasonable row count after warmup."""
        result = tier_a500.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert len(result) >= 50, f"Too few rows: {len(result)}"
        assert len(result) <= 200, f"Too many rows (warmup not applied?): {len(result)}"
