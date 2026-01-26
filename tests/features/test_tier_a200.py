"""Tests for tier_a200 indicator module (indicators 101-200).

Chunk 1 (rank 101-120): Extended MA Types
- tema_9, tema_20, tema_50, tema_100 - Triple EMA at various periods
- wma_10, wma_20, wma_50, wma_200 - Weighted MA at various periods
- kama_10, kama_20, kama_50 - Kaufman Adaptive MA at various periods
- hma_9, hma_21, hma_50 - Hull MA at various periods
- vwma_10, vwma_20, vwma_50 - Volume-Weighted MA at various periods
- tema_20_slope - 5-day change in TEMA_20
- price_pct_from_tema_50 - % distance from TEMA_50
- price_pct_from_kama_20 - % distance from KAMA_20
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

from src.features import tier_a200


@pytest.fixture()
def sample_daily_df() -> pd.DataFrame:
    """Create 300 days of synthetic OHLCV data for indicator testing.

    Uses 300 rows to ensure sufficient warmup for 200+ day indicators.
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-02", periods=300, freq="B")
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
    """
    np.random.seed(42)
    dates = pd.date_range("2023-01-02", periods=300, freq="B")
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


class TestA200FeatureListStructure:
    """Test feature list structure and counts."""

    def test_a200_addition_list_exists(self) -> None:
        """A200_ADDITION_LIST constant exists."""
        assert hasattr(tier_a200, "A200_ADDITION_LIST")

    def test_a200_chunk1_count_is_20(self) -> None:
        """A200_ADDITION_LIST has exactly 20 indicators for Chunk 1."""
        assert len(tier_a200.A200_ADDITION_LIST) == 20

    def test_a200_feature_list_extends_a100(self) -> None:
        """FEATURE_LIST includes all a100 features plus new additions."""
        from src.features import tier_a100

        for feature in tier_a100.FEATURE_LIST:
            assert feature in tier_a200.FEATURE_LIST, f"Missing a100 feature: {feature}"

    def test_feature_list_total_120(self) -> None:
        """FEATURE_LIST has exactly 120 features (100 a100 + 20 a200 Chunk 1)."""
        assert len(tier_a200.FEATURE_LIST) == 120

    def test_chunk1_tema_indicators_in_list(self) -> None:
        """Chunk 1 TEMA indicators are in the list."""
        tema_indicators = ["tema_9", "tema_20", "tema_50", "tema_100"]
        for indicator in tema_indicators:
            assert indicator in tier_a200.A200_ADDITION_LIST, f"Missing: {indicator}"

    def test_chunk1_wma_indicators_in_list(self) -> None:
        """Chunk 1 WMA indicators are in the list."""
        wma_indicators = ["wma_10", "wma_20", "wma_50", "wma_200"]
        for indicator in wma_indicators:
            assert indicator in tier_a200.A200_ADDITION_LIST, f"Missing: {indicator}"

    def test_chunk1_kama_indicators_in_list(self) -> None:
        """Chunk 1 KAMA indicators are in the list."""
        kama_indicators = ["kama_10", "kama_20", "kama_50"]
        for indicator in kama_indicators:
            assert indicator in tier_a200.A200_ADDITION_LIST, f"Missing: {indicator}"

    def test_chunk1_hma_indicators_in_list(self) -> None:
        """Chunk 1 HMA indicators are in the list."""
        hma_indicators = ["hma_9", "hma_21", "hma_50"]
        for indicator in hma_indicators:
            assert indicator in tier_a200.A200_ADDITION_LIST, f"Missing: {indicator}"

    def test_chunk1_vwma_indicators_in_list(self) -> None:
        """Chunk 1 VWMA indicators are in the list."""
        vwma_indicators = ["vwma_10", "vwma_20", "vwma_50"]
        for indicator in vwma_indicators:
            assert indicator in tier_a200.A200_ADDITION_LIST, f"Missing: {indicator}"

    def test_chunk1_derived_indicators_in_list(self) -> None:
        """Chunk 1 derived indicators (slope, pct_from) are in the list."""
        derived_indicators = [
            "tema_20_slope",
            "price_pct_from_tema_50",
            "price_pct_from_kama_20",
        ]
        for indicator in derived_indicators:
            assert indicator in tier_a200.A200_ADDITION_LIST, f"Missing: {indicator}"


class TestChunk1TemaIndicators:
    """Test TEMA (Triple EMA) indicators (rank 101-104)."""

    # --- Existence tests ---

    def test_tema_9_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tema_9 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "tema_9" in result.columns

    def test_tema_20_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tema_20 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "tema_20" in result.columns

    def test_tema_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tema_50 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "tema_50" in result.columns

    def test_tema_100_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tema_100 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "tema_100" in result.columns

    # --- Range tests ---

    def test_tema_values_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """TEMA values should be positive (price-based MA)."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["tema_9", "tema_20", "tema_50", "tema_100"]:
            assert (result[col] > 0).all(), f"{col} has non-positive values"

    def test_tema_values_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """TEMA values should be within reasonable price range.

        Test data has prices around 100, so TEMA should be similar.
        """
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["tema_9", "tema_20", "tema_50", "tema_100"]:
            assert result[col].min() > 50, f"{col} too low: {result[col].min()}"
            assert result[col].max() < 200, f"{col} too high: {result[col].max()}"

    # --- No-NaN test ---

    def test_tema_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in TEMA columns after warmup."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["tema_9", "tema_20", "tema_50", "tema_100"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk1WmaIndicators:
    """Test WMA (Weighted MA) indicators (rank 105-108)."""

    # --- Existence tests ---

    def test_wma_10_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """wma_10 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "wma_10" in result.columns

    def test_wma_20_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """wma_20 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "wma_20" in result.columns

    def test_wma_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """wma_50 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "wma_50" in result.columns

    def test_wma_200_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """wma_200 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "wma_200" in result.columns

    # --- Range tests ---

    def test_wma_values_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """WMA values should be positive (price-based MA)."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["wma_10", "wma_20", "wma_50", "wma_200"]:
            assert (result[col] > 0).all(), f"{col} has non-positive values"

    def test_wma_values_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """WMA values should be within reasonable price range."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["wma_10", "wma_20", "wma_50", "wma_200"]:
            assert result[col].min() > 50, f"{col} too low: {result[col].min()}"
            assert result[col].max() < 200, f"{col} too high: {result[col].max()}"

    # --- No-NaN test ---

    def test_wma_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in WMA columns after warmup."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["wma_10", "wma_20", "wma_50", "wma_200"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk1KamaIndicators:
    """Test KAMA (Kaufman Adaptive MA) indicators (rank 109-111)."""

    # --- Existence tests ---

    def test_kama_10_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """kama_10 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "kama_10" in result.columns

    def test_kama_20_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """kama_20 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "kama_20" in result.columns

    def test_kama_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """kama_50 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "kama_50" in result.columns

    # --- Range tests ---

    def test_kama_values_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """KAMA values should be positive (price-based MA)."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["kama_10", "kama_20", "kama_50"]:
            assert (result[col] > 0).all(), f"{col} has non-positive values"

    def test_kama_values_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """KAMA values should be within reasonable price range."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["kama_10", "kama_20", "kama_50"]:
            assert result[col].min() > 50, f"{col} too low: {result[col].min()}"
            assert result[col].max() < 200, f"{col} too high: {result[col].max()}"

    # --- No-NaN test ---

    def test_kama_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in KAMA columns after warmup."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["kama_10", "kama_20", "kama_50"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk1HmaIndicators:
    """Test HMA (Hull MA) indicators (rank 112-114)."""

    # --- Existence tests ---

    def test_hma_9_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """hma_9 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "hma_9" in result.columns

    def test_hma_21_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """hma_21 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "hma_21" in result.columns

    def test_hma_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """hma_50 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "hma_50" in result.columns

    # --- Range tests ---

    def test_hma_values_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """HMA values should be positive (price-based MA)."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["hma_9", "hma_21", "hma_50"]:
            assert (result[col] > 0).all(), f"{col} has non-positive values"

    def test_hma_values_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """HMA values should be within reasonable price range."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["hma_9", "hma_21", "hma_50"]:
            assert result[col].min() > 50, f"{col} too low: {result[col].min()}"
            assert result[col].max() < 200, f"{col} too high: {result[col].max()}"

    # --- No-NaN test ---

    def test_hma_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in HMA columns after warmup."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["hma_9", "hma_21", "hma_50"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk1VwmaIndicators:
    """Test VWMA (Volume-Weighted MA) indicators (rank 115-117)."""

    # --- Existence tests ---

    def test_vwma_10_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vwma_10 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vwma_10" in result.columns

    def test_vwma_20_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vwma_20 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vwma_20" in result.columns

    def test_vwma_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vwma_50 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vwma_50" in result.columns

    # --- Range tests ---

    def test_vwma_values_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """VWMA values should be positive (price-based MA)."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["vwma_10", "vwma_20", "vwma_50"]:
            assert (result[col] > 0).all(), f"{col} has non-positive values"

    def test_vwma_values_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """VWMA values should be within reasonable price range."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["vwma_10", "vwma_20", "vwma_50"]:
            assert result[col].min() > 50, f"{col} too low: {result[col].min()}"
            assert result[col].max() < 200, f"{col} too high: {result[col].max()}"

    # --- No-NaN test ---

    def test_vwma_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in VWMA columns after warmup."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        for col in ["vwma_10", "vwma_20", "vwma_50"]:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestChunk1DerivedIndicators:
    """Test derived MA indicators (rank 118-120)."""

    # --- Existence tests ---

    def test_tema_20_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tema_20_slope column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "tema_20_slope" in result.columns

    def test_price_pct_from_tema_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_tema_50 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_pct_from_tema_50" in result.columns

    def test_price_pct_from_kama_20_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_kama_20 column is present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_pct_from_kama_20" in result.columns

    # --- Range tests ---

    def test_tema_20_slope_centered_near_zero(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tema_20_slope mean should be near zero (derivative of oscillating MA)."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        slope = result["tema_20_slope"]

        assert abs(slope.mean()) < 1, f"tema_20_slope mean too far from zero: {slope.mean()}"

    def test_tema_20_slope_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """tema_20_slope is in reasonable range [-10, +10].

        5-day change in TEMA should be bounded for normal price movements.
        """
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        slope = result["tema_20_slope"]

        assert slope.min() >= -10, f"tema_20_slope too negative: {slope.min()}"
        assert slope.max() <= 10, f"tema_20_slope too positive: {slope.max()}"

    def test_price_pct_from_tema_50_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_tema_50 is in reasonable range [-20, +20].

        Price typically stays within 20% of its 50-period moving average.
        """
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        pct = result["price_pct_from_tema_50"]

        assert pct.min() >= -20, f"price_pct_from_tema_50 too negative: {pct.min()}"
        assert pct.max() <= 20, f"price_pct_from_tema_50 too positive: {pct.max()}"

    def test_price_pct_from_kama_20_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_kama_20 is in reasonable range [-15, +15].

        KAMA is adaptive, so price should stay relatively close to it.
        """
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        pct = result["price_pct_from_kama_20"]

        assert pct.min() >= -15, f"price_pct_from_kama_20 too negative: {pct.min()}"
        assert pct.max() <= 15, f"price_pct_from_kama_20 too positive: {pct.max()}"

    # --- No-NaN test ---

    def test_derived_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in derived indicator columns after warmup."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        derived_cols = [
            "tema_20_slope",
            "price_pct_from_tema_50",
            "price_pct_from_kama_20",
        ]
        for col in derived_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestA200OutputShape:
    """Test output shape and structure for Chunk 1."""

    def test_output_has_date_column(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output DataFrame has Date column."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "Date" in result.columns

    def test_output_column_count(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output has 121 columns (Date + 120 features)."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert len(result.columns) == 121, f"Expected 121 columns, got {len(result.columns)}"

    def test_output_fewer_rows_than_input(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output has fewer rows due to warmup period."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert len(result) < len(sample_daily_df), "Warmup rows should be dropped"

    def test_no_nan_in_output(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in any column after warmup rows dropped."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)
        nan_count = result.drop(columns=["Date"]).isnull().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"

    def test_chunk1_all_features_present(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """All 20 Chunk 1 features are present in output."""
        result = tier_a200.build_feature_dataframe(sample_daily_df, sample_vix_df)

        chunk1_features = [
            "tema_9", "tema_20", "tema_50", "tema_100",
            "wma_10", "wma_20", "wma_50", "wma_200",
            "kama_10", "kama_20", "kama_50",
            "hma_9", "hma_21", "hma_50",
            "vwma_10", "vwma_20", "vwma_50",
            "tema_20_slope", "price_pct_from_tema_50", "price_pct_from_kama_20",
        ]

        for feature in chunk1_features:
            assert feature in result.columns, f"Missing feature: {feature}"
