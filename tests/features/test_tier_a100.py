"""Tests for tier_a100 indicator module (indicators 51-100).

Chunk 1: Momentum derivatives (rank 51-52)
- return_1d_acceleration
- return_5d_acceleration

Chunk 2: QQE/STC derivatives (rank 53-56)
- qqe_slope
- qqe_extreme_dist
- stc_slope
- stc_extreme_dist

Chunk 3: Standard oscillators (rank 57-64)
- demarker_value
- demarker_from_half
- stoch_k_14
- stoch_d_14
- stoch_extreme_dist
- cci_14
- mfi_14
- williams_r_14

Chunk 4: VRP + Risk metrics (rank 65-73)
- vrp_5d
- vrp_slope
- sharpe_252d
- sortino_252d
- sharpe_slope_20d
- sortino_slope_20d
- var_95
- var_99
- cvar_95

Chunk 5: MA extensions (rank 74-80)
- sma_9_50_proximity
- sma_50_slope
- sma_200_slope
- days_since_sma_50_cross
- days_since_sma_200_cross
- ema_12
- ema_26

Chunk 6: Advanced volatility (rank 81-85)
- atr_pct_percentile_60d
- bb_width_percentile_60d
- parkinson_volatility
- garman_klass_volatility
- vol_of_vol

Chunk 7: Trend indicators (rank 86-90)
- adx_slope
- di_spread
- aroon_oscillator
- price_pct_from_supertrend
- supertrend_direction
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

from src.features import tier_a100


@pytest.fixture()
def sample_daily_df() -> pd.DataFrame:
    """Create 300 days of synthetic OHLCV data for indicator testing.

    Uses 300 rows to ensure sufficient warmup for 252-day indicators.
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

    VIX is required for VRP indicators inherited from a50.
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


class TestA100FeatureListStructure:
    """Test feature list structure and counts."""

    def test_a100_addition_list_exists(self) -> None:
        """A100_ADDITION_LIST constant exists."""
        assert hasattr(tier_a100, "A100_ADDITION_LIST")

    def test_a100_addition_list_chunk2_count(self) -> None:
        """A100_ADDITION_LIST has at least 6 indicators (Chunk 1 + Chunk 2)."""
        # Chunk 1: 2 indicators, Chunk 2: 4 indicators = 6 total
        assert len(tier_a100.A100_ADDITION_LIST) >= 6

    def test_a100_feature_list_extends_a50(self) -> None:
        """FEATURE_LIST includes all a50 features plus new additions."""
        from src.features import tier_a50

        for feature in tier_a50.FEATURE_LIST:
            assert feature in tier_a100.FEATURE_LIST, f"Missing a50 feature: {feature}"

    def test_chunk1_indicators_in_list(self) -> None:
        """Chunk 1 momentum derivative indicators are in the list."""
        assert "return_1d_acceleration" in tier_a100.A100_ADDITION_LIST
        assert "return_5d_acceleration" in tier_a100.A100_ADDITION_LIST

    def test_chunk2_indicators_in_list(self) -> None:
        """Chunk 2 QQE/STC derivative indicators are in the list."""
        assert "qqe_slope" in tier_a100.A100_ADDITION_LIST
        assert "qqe_extreme_dist" in tier_a100.A100_ADDITION_LIST
        assert "stc_slope" in tier_a100.A100_ADDITION_LIST
        assert "stc_extreme_dist" in tier_a100.A100_ADDITION_LIST

    def test_a100_addition_list_chunk3_count(self) -> None:
        """A100_ADDITION_LIST has at least 14 indicators (Chunks 1-3)."""
        # Chunk 1: 2, Chunk 2: 4, Chunk 3: 8 = 14 total
        assert len(tier_a100.A100_ADDITION_LIST) >= 14

    def test_chunk3_indicators_in_list(self) -> None:
        """Chunk 3 standard oscillator indicators are in the list."""
        chunk3_indicators = [
            "demarker_value",
            "demarker_from_half",
            "stoch_k_14",
            "stoch_d_14",
            "stoch_extreme_dist",
            "cci_14",
            "mfi_14",
            "williams_r_14",
        ]
        for indicator in chunk3_indicators:
            assert indicator in tier_a100.A100_ADDITION_LIST, f"Missing: {indicator}"

    def test_a100_addition_list_chunk4_count(self) -> None:
        """A100_ADDITION_LIST has at least 23 indicators (Chunks 1-4)."""
        # Chunk 1: 2, Chunk 2: 4, Chunk 3: 8, Chunk 4: 9 = 23 total
        assert len(tier_a100.A100_ADDITION_LIST) >= 23

    def test_chunk4_indicators_in_list(self) -> None:
        """Chunk 4 VRP + risk metric indicators are in the list."""
        chunk4_indicators = [
            "vrp_5d",
            "vrp_slope",
            "sharpe_252d",
            "sortino_252d",
            "sharpe_slope_20d",
            "sortino_slope_20d",
            "var_95",
            "var_99",
            "cvar_95",
        ]
        for indicator in chunk4_indicators:
            assert indicator in tier_a100.A100_ADDITION_LIST, f"Missing: {indicator}"

    def test_a100_addition_list_chunk5_count(self) -> None:
        """A100_ADDITION_LIST has at least 30 indicators (Chunks 1-5)."""
        # Chunk 1: 2, Chunk 2: 4, Chunk 3: 8, Chunk 4: 9, Chunk 5: 7 = 30 total
        assert len(tier_a100.A100_ADDITION_LIST) >= 30

    def test_chunk5_indicators_in_list(self) -> None:
        """Chunk 5 MA extension indicators are in the list."""
        chunk5_indicators = [
            "sma_9_50_proximity",
            "sma_50_slope",
            "sma_200_slope",
            "days_since_sma_50_cross",
            "days_since_sma_200_cross",
            "ema_12",
            "ema_26",
        ]
        for indicator in chunk5_indicators:
            assert indicator in tier_a100.A100_ADDITION_LIST, f"Missing: {indicator}"

    def test_a100_addition_list_chunk6_count(self) -> None:
        """A100_ADDITION_LIST has at least 35 indicators (Chunks 1-6)."""
        # Chunk 1: 2, Chunk 2: 4, Chunk 3: 8, Chunk 4: 9, Chunk 5: 7, Chunk 6: 5 = 35 total
        assert len(tier_a100.A100_ADDITION_LIST) >= 35

    def test_chunk6_indicators_in_list(self) -> None:
        """Chunk 6 advanced volatility indicators are in the list."""
        chunk6_indicators = [
            "atr_pct_percentile_60d",
            "bb_width_percentile_60d",
            "parkinson_volatility",
            "garman_klass_volatility",
            "vol_of_vol",
        ]
        for indicator in chunk6_indicators:
            assert indicator in tier_a100.A100_ADDITION_LIST, f"Missing: {indicator}"

    def test_a100_addition_list_chunk7_count(self) -> None:
        """A100_ADDITION_LIST has at least 40 indicators (Chunks 1-7)."""
        # Chunk 1: 2, Chunk 2: 4, Chunk 3: 8, Chunk 4: 9, Chunk 5: 7,
        # Chunk 6: 5, Chunk 7: 5 = 40 total
        assert len(tier_a100.A100_ADDITION_LIST) >= 40

    def test_chunk7_indicators_in_list(self) -> None:
        """Chunk 7 trend indicators are in the list."""
        chunk7_indicators = [
            "adx_slope",
            "di_spread",
            "aroon_oscillator",
            "price_pct_from_supertrend",
            "supertrend_direction",
        ]
        for indicator in chunk7_indicators:
            assert indicator in tier_a100.A100_ADDITION_LIST, f"Missing: {indicator}"

    def test_a100_addition_list_chunk8_count(self) -> None:
        """A100_ADDITION_LIST has exactly 50 indicators (Chunks 1-8 complete)."""
        assert len(tier_a100.A100_ADDITION_LIST) == 50

    def test_chunk8_indicators_in_list(self) -> None:
        """Chunk 8 Volume + Momentum + S/R indicators are in the list."""
        chunk8_indicators = [
            "obv_slope",
            "volume_price_trend",
            "kvo_histogram",
            "accumulation_dist",
            "expectancy_20d",
            "win_rate_20d",
            "buying_pressure_ratio",
            "fib_range_position",
            "prior_high_20d_dist",
            "prior_low_20d_dist",
        ]
        for indicator in chunk8_indicators:
            assert indicator in tier_a100.A100_ADDITION_LIST, f"Missing: {indicator}"

    def test_feature_list_total_100(self) -> None:
        """FEATURE_LIST has exactly 100 features (50 a50 + 50 a100 additions)."""
        assert len(tier_a100.FEATURE_LIST) == 100


class TestMomentumDerivatives:
    """Test momentum derivative indicators (rank 51-52)."""

    def test_return_1d_acceleration_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """return_1d_acceleration column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "return_1d_acceleration" in result.columns

    def test_return_5d_acceleration_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """return_5d_acceleration column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "return_5d_acceleration" in result.columns

    def test_return_1d_acceleration_formula(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """return_1d_acceleration = return_1d - return_1d.shift(1).

        This measures the change in daily return, i.e., momentum shift.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        accel = result["return_1d_acceleration"]

        # Acceleration is the derivative of returns, should be centered around 0
        assert abs(accel.mean()) < 1, f"Mean acceleration too large: {accel.mean()}"

    def test_return_5d_acceleration_formula(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """return_5d_acceleration = return_5d - return_5d.shift(1).

        This measures the change in weekly return.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        accel = result["return_5d_acceleration"]

        # Should have non-zero variance
        assert accel.std() > 0, "Acceleration should have variance"

    def test_accelerations_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Acceleration values should be in reasonable range.

        Since returns are typically -10% to +10%, acceleration (change in return)
        should typically be in similar range.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        accel_1d = result["return_1d_acceleration"]
        accel_5d = result["return_5d_acceleration"]

        # 1-day acceleration (change in daily return)
        assert accel_1d.min() > -20, f"1d acceleration too negative: {accel_1d.min()}"
        assert accel_1d.max() < 20, f"1d acceleration too positive: {accel_1d.max()}"

        # 5-day acceleration (change in weekly return)
        assert accel_5d.min() > -30, f"5d acceleration too negative: {accel_5d.min()}"
        assert accel_5d.max() < 30, f"5d acceleration too positive: {accel_5d.max()}"

    def test_accelerations_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in acceleration columns after warmup."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        assert not result["return_1d_acceleration"].isna().any(), "NaN in return_1d_acceleration"
        assert not result["return_5d_acceleration"].isna().any(), "NaN in return_5d_acceleration"


class TestQQESTCDerivatives:
    """Test QQE/STC derivative indicators (rank 53-56)."""

    def test_qqe_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """qqe_slope column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "qqe_slope" in result.columns

    def test_qqe_slope_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """qqe_slope values in [-30, +30] range.

        QQE is 0-100, so 5-day change should be bounded.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        qqe_slope = result["qqe_slope"]

        assert qqe_slope.min() >= -30, f"qqe_slope too negative: {qqe_slope.min()}"
        assert qqe_slope.max() <= 30, f"qqe_slope too positive: {qqe_slope.max()}"

    def test_qqe_slope_centered_around_zero(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """qqe_slope mean should be near zero (oscillator derivative)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        qqe_slope = result["qqe_slope"]

        assert abs(qqe_slope.mean()) < 5, f"qqe_slope mean too far from zero: {qqe_slope.mean()}"

    def test_qqe_extreme_dist_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """qqe_extreme_dist column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "qqe_extreme_dist" in result.columns

    def test_qqe_extreme_dist_non_negative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """qqe_extreme_dist is always >= 0 (it's a distance)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        qqe_extreme_dist = result["qqe_extreme_dist"]

        assert (qqe_extreme_dist >= 0).all(), "qqe_extreme_dist has negative values"

    def test_qqe_extreme_dist_max_30(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """qqe_extreme_dist max is 30 (QQE at 50 is equidistant from 20 and 80)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        qqe_extreme_dist = result["qqe_extreme_dist"]

        assert qqe_extreme_dist.max() <= 30, f"qqe_extreme_dist too large: {qqe_extreme_dist.max()}"

    def test_stc_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stc_slope column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "stc_slope" in result.columns

    def test_stc_slope_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stc_slope values in [-100, +100] range.

        STC is 0-100 and can move rapidly (double stochastic on MACD),
        so 5-day change could span full range.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        stc_slope = result["stc_slope"]

        assert stc_slope.min() >= -100, f"stc_slope too negative: {stc_slope.min()}"
        assert stc_slope.max() <= 100, f"stc_slope too positive: {stc_slope.max()}"

    def test_stc_extreme_dist_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stc_extreme_dist column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "stc_extreme_dist" in result.columns

    def test_stc_extreme_dist_non_negative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stc_extreme_dist is always >= 0 (it's a distance)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        stc_extreme_dist = result["stc_extreme_dist"]

        assert (stc_extreme_dist >= 0).all(), "stc_extreme_dist has negative values"

    def test_stc_extreme_dist_max_25(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stc_extreme_dist max is 25 (STC at 50 is equidistant from 25 and 75)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        stc_extreme_dist = result["stc_extreme_dist"]

        assert stc_extreme_dist.max() <= 25, f"stc_extreme_dist too large: {stc_extreme_dist.max()}"

    def test_chunk2_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Chunk 2 indicator columns after warmup."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        chunk2_cols = ["qqe_slope", "qqe_extreme_dist", "stc_slope", "stc_extreme_dist"]
        for col in chunk2_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestStandardOscillators:
    """Test standard oscillator indicators (rank 57-64)."""

    # --- Existence tests ---

    def test_demarker_value_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """demarker_value column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "demarker_value" in result.columns

    def test_demarker_from_half_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """demarker_from_half column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "demarker_from_half" in result.columns

    def test_stoch_k_14_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stoch_k_14 column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "stoch_k_14" in result.columns

    def test_stoch_d_14_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stoch_d_14 column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "stoch_d_14" in result.columns

    def test_stoch_extreme_dist_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stoch_extreme_dist column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "stoch_extreme_dist" in result.columns

    def test_cci_14_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """cci_14 column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "cci_14" in result.columns

    def test_mfi_14_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """mfi_14 column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "mfi_14" in result.columns

    def test_williams_r_14_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """williams_r_14 column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "williams_r_14" in result.columns

    # --- Range tests ---

    def test_demarker_value_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """demarker_value is in [0, 1] range."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        demarker = result["demarker_value"]

        assert demarker.min() >= 0, f"demarker_value below 0: {demarker.min()}"
        assert demarker.max() <= 1, f"demarker_value above 1: {demarker.max()}"

    def test_demarker_from_half_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """demarker_from_half is in [-0.5, +0.5] range."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        demarker_half = result["demarker_from_half"]

        assert demarker_half.min() >= -0.5, f"demarker_from_half below -0.5: {demarker_half.min()}"
        assert demarker_half.max() <= 0.5, f"demarker_from_half above 0.5: {demarker_half.max()}"

    def test_stoch_k_14_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stoch_k_14 is in [0, 100] range."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        stoch_k = result["stoch_k_14"]

        assert stoch_k.min() >= 0, f"stoch_k_14 below 0: {stoch_k.min()}"
        assert stoch_k.max() <= 100, f"stoch_k_14 above 100: {stoch_k.max()}"

    def test_stoch_d_14_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stoch_d_14 is in [0, 100] range."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        stoch_d = result["stoch_d_14"]

        assert stoch_d.min() >= 0, f"stoch_d_14 below 0: {stoch_d.min()}"
        assert stoch_d.max() <= 100, f"stoch_d_14 above 100: {stoch_d.max()}"

    def test_stoch_extreme_dist_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stoch_extreme_dist is in [0, 30] range (max at stoch_k=50)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        stoch_dist = result["stoch_extreme_dist"]

        assert stoch_dist.min() >= 0, f"stoch_extreme_dist below 0: {stoch_dist.min()}"
        assert stoch_dist.max() <= 30, f"stoch_extreme_dist above 30: {stoch_dist.max()}"

    def test_cci_14_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """cci_14 is in reasonable range (typically -200 to +200)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        cci = result["cci_14"]

        # CCI is unbounded but rarely exceeds these in normal data
        assert cci.min() > -500, f"cci_14 extremely negative: {cci.min()}"
        assert cci.max() < 500, f"cci_14 extremely positive: {cci.max()}"

    def test_mfi_14_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """mfi_14 is in [0, 100] range."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        mfi = result["mfi_14"]

        assert mfi.min() >= 0, f"mfi_14 below 0: {mfi.min()}"
        assert mfi.max() <= 100, f"mfi_14 above 100: {mfi.max()}"

    def test_williams_r_14_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """williams_r_14 is in [-100, 0] range."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        williams_r = result["williams_r_14"]

        assert williams_r.min() >= -100, f"williams_r_14 below -100: {williams_r.min()}"
        assert williams_r.max() <= 0, f"williams_r_14 above 0: {williams_r.max()}"

    # --- No-NaN test ---

    def test_chunk3_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Chunk 3 indicator columns after warmup."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        chunk3_cols = [
            "demarker_value",
            "demarker_from_half",
            "stoch_k_14",
            "stoch_d_14",
            "stoch_extreme_dist",
            "cci_14",
            "mfi_14",
            "williams_r_14",
        ]
        for col in chunk3_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestVRPExtensions:
    """Test VRP extension indicators (rank 65-66)."""

    def test_vrp_5d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vrp_5d column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vrp_5d" in result.columns

    def test_vrp_5d_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vrp_5d is in reasonable range [-50, +80].

        VRP = VIX - realized_vol. VIX ~9-80, realized_vol ~5-100.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        vrp_5d = result["vrp_5d"]

        assert vrp_5d.min() >= -50, f"vrp_5d too negative: {vrp_5d.min()}"
        assert vrp_5d.max() <= 80, f"vrp_5d too positive: {vrp_5d.max()}"

    def test_vrp_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vrp_slope column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vrp_slope" in result.columns

    def test_vrp_slope_centered_near_zero(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vrp_slope mean should be near zero (derivative of mean-reverting VRP)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        vrp_slope = result["vrp_slope"]

        assert abs(vrp_slope.mean()) < 5, f"vrp_slope mean too far from zero: {vrp_slope.mean()}"

    def test_vrp_slope_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vrp_slope is in reasonable range [-30, +30]."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        vrp_slope = result["vrp_slope"]

        assert vrp_slope.min() >= -30, f"vrp_slope too negative: {vrp_slope.min()}"
        assert vrp_slope.max() <= 30, f"vrp_slope too positive: {vrp_slope.max()}"


class TestExtendedRiskMetrics:
    """Test extended risk metric indicators (rank 67-70)."""

    def test_sharpe_252d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sharpe_252d column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sharpe_252d" in result.columns

    def test_sortino_252d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sortino_252d column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sortino_252d" in result.columns

    def test_sharpe_slope_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sharpe_slope_20d column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sharpe_slope_20d" in result.columns

    def test_sortino_slope_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sortino_slope_20d column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sortino_slope_20d" in result.columns

    def test_sharpe_252d_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sharpe_252d is in reasonable range [-10, +10].

        Annualized Sharpe rarely exceeds +/- 3 for real data,
        but synthetic data may produce more extreme values.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        sharpe = result["sharpe_252d"]

        assert sharpe.min() >= -10, f"sharpe_252d too negative: {sharpe.min()}"
        assert sharpe.max() <= 10, f"sharpe_252d too positive: {sharpe.max()}"

    def test_sortino_252d_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sortino_252d is in reasonable range [-15, +15].

        Sortino can be more extreme than Sharpe since denominator is smaller.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        sortino = result["sortino_252d"]

        assert sortino.min() >= -15, f"sortino_252d too negative: {sortino.min()}"
        assert sortino.max() <= 15, f"sortino_252d too positive: {sortino.max()}"

    def test_sharpe_slope_20d_centered_near_zero(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sharpe_slope_20d mean should be near zero (derivative of oscillating metric)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        slope = result["sharpe_slope_20d"]

        assert abs(slope.mean()) < 0.5, f"sharpe_slope_20d mean too far from zero: {slope.mean()}"

    def test_sortino_slope_20d_centered_near_zero(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sortino_slope_20d mean should be near zero (derivative of oscillating metric)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        slope = result["sortino_slope_20d"]

        assert abs(slope.mean()) < 0.5, f"sortino_slope_20d mean too far from zero: {slope.mean()}"


class TestVaRCVaR:
    """Test VaR and CVaR risk indicators (rank 71-73)."""

    def test_var_95_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """var_95 column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "var_95" in result.columns

    def test_var_99_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """var_99 column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "var_99" in result.columns

    def test_cvar_95_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """cvar_95 column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "cvar_95" in result.columns

    def test_var_95_is_negative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """var_95 median should be negative (5th percentile of returns)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        var_95 = result["var_95"]

        assert var_95.median() < 0, f"var_95 median should be negative: {var_95.median()}"

    def test_var_99_more_extreme_than_var_95(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """var_99 should be more extreme (more negative) than var_95."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        var_95 = result["var_95"]
        var_99 = result["var_99"]

        # var_99 (1st percentile) should be more negative than var_95 (5th percentile)
        assert var_99.median() <= var_95.median(), (
            f"var_99 ({var_99.median()}) should be <= var_95 ({var_95.median()})"
        )

    def test_cvar_95_more_extreme_than_var_95(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """cvar_95 should be more extreme (more negative) than var_95.

        CVaR is the mean of returns below VaR threshold.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        var_95 = result["var_95"]
        cvar_95 = result["cvar_95"]

        # CVaR (expected shortfall) should be more negative than VaR
        assert cvar_95.median() <= var_95.median(), (
            f"cvar_95 ({cvar_95.median()}) should be <= var_95 ({var_95.median()})"
        )

    def test_var_95_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """var_95 is in reasonable range [-20, +5].

        5th percentile of returns in percent. Usually negative for risk measure.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        var_95 = result["var_95"]

        assert var_95.min() >= -20, f"var_95 too negative: {var_95.min()}"
        assert var_95.max() <= 5, f"var_95 too positive: {var_95.max()}"

    def test_chunk4_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Chunk 4 indicator columns after warmup."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        chunk4_cols = [
            "vrp_5d",
            "vrp_slope",
            "sharpe_252d",
            "sortino_252d",
            "sharpe_slope_20d",
            "sortino_slope_20d",
            "var_95",
            "var_99",
            "cvar_95",
        ]
        for col in chunk4_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestMAProximitySlope:
    """Test MA proximity and slope indicators (rank 74-78, 79-80)."""

    def test_sma_9_50_proximity_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_9_50_proximity column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_9_50_proximity" in result.columns

    def test_sma_9_50_proximity_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_9_50_proximity is in reasonable range [-50, +50].

        Measures % difference between SMA_9 and SMA_50.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        proximity = result["sma_9_50_proximity"]

        assert proximity.min() >= -50, f"sma_9_50_proximity too negative: {proximity.min()}"
        assert proximity.max() <= 50, f"sma_9_50_proximity too positive: {proximity.max()}"

    def test_sma_50_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_50_slope column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_50_slope" in result.columns

    def test_sma_200_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_200_slope column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sma_200_slope" in result.columns

    def test_sma_50_slope_centered_near_zero(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_50_slope mean should be near zero (5-day change in SMA)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        slope = result["sma_50_slope"]

        assert abs(slope.mean()) < 1, f"sma_50_slope mean too far from zero: {slope.mean()}"

    def test_sma_200_slope_centered_near_zero(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_200_slope mean should be near zero (5-day change in SMA)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        slope = result["sma_200_slope"]

        assert abs(slope.mean()) < 0.5, f"sma_200_slope mean too far from zero: {slope.mean()}"

    def test_ema_12_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_12 column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_12" in result.columns

    def test_ema_26_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """ema_26 column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "ema_26" in result.columns


class TestDaysSinceCross:
    """Test days since MA cross indicators (rank 77-78)."""

    def test_days_since_sma_50_cross_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_sma_50_cross column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_since_sma_50_cross" in result.columns

    def test_days_since_sma_200_cross_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_sma_200_cross column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "days_since_sma_200_cross" in result.columns

    def test_days_since_sma_50_cross_non_negative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_sma_50_cross is always >= 0."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        days_since = result["days_since_sma_50_cross"]

        assert days_since.min() >= 0, f"days_since_sma_50_cross has negative: {days_since.min()}"

    def test_days_since_sma_200_cross_non_negative(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_sma_200_cross is always >= 0."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        days_since = result["days_since_sma_200_cross"]

        assert days_since.min() >= 0, f"days_since_sma_200_cross has negative: {days_since.min()}"

    def test_days_since_cross_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """days_since_cross values should be less than fixture length (300 days)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        days_50 = result["days_since_sma_50_cross"]
        days_200 = result["days_since_sma_200_cross"]

        assert days_50.max() < 300, f"days_since_sma_50_cross too large: {days_50.max()}"
        assert days_200.max() < 300, f"days_since_sma_200_cross too large: {days_200.max()}"

    def test_chunk5_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Chunk 5 indicator columns after warmup."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        chunk5_cols = [
            "sma_9_50_proximity",
            "sma_50_slope",
            "sma_200_slope",
            "days_since_sma_50_cross",
            "days_since_sma_200_cross",
            "ema_12",
            "ema_26",
        ]
        for col in chunk5_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestAdvancedVolatility:
    """Test advanced volatility indicators (rank 81-85)."""

    # --- Existence tests ---

    def test_atr_pct_percentile_60d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_pct_percentile_60d column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "atr_pct_percentile_60d" in result.columns

    def test_bb_width_percentile_60d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bb_width_percentile_60d column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "bb_width_percentile_60d" in result.columns

    def test_parkinson_volatility_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """parkinson_volatility column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "parkinson_volatility" in result.columns

    def test_garman_klass_volatility_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """garman_klass_volatility column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "garman_klass_volatility" in result.columns

    def test_vol_of_vol_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vol_of_vol column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vol_of_vol" in result.columns

    # --- Range tests ---

    def test_atr_pct_percentile_60d_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_pct_percentile_60d is in [0, 100] range (percentile)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        pctl = result["atr_pct_percentile_60d"]

        assert pctl.min() >= 0, f"atr_pct_percentile_60d below 0: {pctl.min()}"
        assert pctl.max() <= 100, f"atr_pct_percentile_60d above 100: {pctl.max()}"

    def test_bb_width_percentile_60d_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bb_width_percentile_60d is in [0, 100] range (percentile)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        pctl = result["bb_width_percentile_60d"]

        assert pctl.min() >= 0, f"bb_width_percentile_60d below 0: {pctl.min()}"
        assert pctl.max() <= 100, f"bb_width_percentile_60d above 100: {pctl.max()}"

    def test_parkinson_volatility_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """parkinson_volatility is always >= 0 (volatility cannot be negative)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        parkinson = result["parkinson_volatility"]

        assert (parkinson >= 0).all(), f"parkinson_volatility has negative: {parkinson.min()}"

    def test_garman_klass_volatility_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """garman_klass_volatility is always >= 0 (volatility cannot be negative)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        gk = result["garman_klass_volatility"]

        assert (gk >= 0).all(), f"garman_klass_volatility has negative: {gk.min()}"

    def test_vol_of_vol_positive(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vol_of_vol is always >= 0 (std dev cannot be negative)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        vov = result["vol_of_vol"]

        assert (vov >= 0).all(), f"vol_of_vol has negative: {vov.min()}"

    # --- No-NaN test ---

    def test_chunk6_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Chunk 6 indicator columns after warmup."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        chunk6_cols = [
            "atr_pct_percentile_60d",
            "bb_width_percentile_60d",
            "parkinson_volatility",
            "garman_klass_volatility",
            "vol_of_vol",
        ]
        for col in chunk6_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestTrendIndicators:
    """Test trend strength and direction indicators (rank 86-90)."""

    # --- Existence tests ---

    def test_adx_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """adx_slope column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "adx_slope" in result.columns

    def test_di_spread_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """di_spread column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "di_spread" in result.columns

    def test_aroon_oscillator_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """aroon_oscillator column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "aroon_oscillator" in result.columns

    def test_price_pct_from_supertrend_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_supertrend column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "price_pct_from_supertrend" in result.columns

    def test_supertrend_direction_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """supertrend_direction column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "supertrend_direction" in result.columns

    # --- Range tests ---

    def test_adx_slope_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """adx_slope is in reasonable range [-50, +50].

        ADX is in [0, 100], so 5-day change should typically be bounded.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        adx_slope = result["adx_slope"]

        assert adx_slope.min() >= -50, f"adx_slope too negative: {adx_slope.min()}"
        assert adx_slope.max() <= 50, f"adx_slope too positive: {adx_slope.max()}"

    def test_di_spread_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """di_spread is in [-100, +100] range (+DI minus -DI)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        di_spread = result["di_spread"]

        assert di_spread.min() >= -100, f"di_spread below -100: {di_spread.min()}"
        assert di_spread.max() <= 100, f"di_spread above 100: {di_spread.max()}"

    def test_aroon_oscillator_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """aroon_oscillator is in [-100, +100] range (Aroon Up - Aroon Down)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        aroon_osc = result["aroon_oscillator"]

        assert aroon_osc.min() >= -100, f"aroon_oscillator below -100: {aroon_osc.min()}"
        assert aroon_osc.max() <= 100, f"aroon_oscillator above 100: {aroon_osc.max()}"

    def test_price_pct_from_supertrend_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_supertrend is in reasonable range [-50, +50].

        Typically price stays within ~10-20% of SuperTrend, but allow wider range.
        """
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        pct_st = result["price_pct_from_supertrend"]

        assert pct_st.min() >= -50, f"price_pct_from_supertrend too negative: {pct_st.min()}"
        assert pct_st.max() <= 50, f"price_pct_from_supertrend too positive: {pct_st.max()}"

    def test_supertrend_direction_values(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """supertrend_direction contains only +1 or -1 values."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        direction = result["supertrend_direction"]

        unique_vals = direction.dropna().unique()
        for val in unique_vals:
            assert val in [1.0, -1.0], f"Unexpected direction value: {val}"

    # --- No-NaN test ---

    def test_chunk7_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in Chunk 7 indicator columns after warmup."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        chunk7_cols = [
            "adx_slope",
            "di_spread",
            "aroon_oscillator",
            "price_pct_from_supertrend",
            "supertrend_direction",
        ]
        for col in chunk7_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestVolumeIndicators:
    """Test volume-based indicators (rank 91-94, 97)."""

    # --- Existence tests ---

    def test_obv_slope_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """obv_slope column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "obv_slope" in result.columns

    def test_volume_price_trend_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_price_trend column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "volume_price_trend" in result.columns

    def test_kvo_histogram_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """kvo_histogram column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "kvo_histogram" in result.columns

    def test_accumulation_dist_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """accumulation_dist column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "accumulation_dist" in result.columns

    def test_buying_pressure_ratio_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """buying_pressure_ratio column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "buying_pressure_ratio" in result.columns

    # --- Value tests ---

    def test_obv_slope_finite(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """obv_slope values are all finite (no inf)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        obv_slope = result["obv_slope"]
        assert np.isfinite(obv_slope).all(), f"Non-finite values in obv_slope"

    def test_volume_price_trend_finite(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_price_trend values are all finite (no inf)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        vpt = result["volume_price_trend"]
        assert np.isfinite(vpt).all(), f"Non-finite values in volume_price_trend"

    def test_kvo_histogram_finite(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """kvo_histogram values are all finite (no inf)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        kvo = result["kvo_histogram"]
        assert np.isfinite(kvo).all(), f"Non-finite values in kvo_histogram"

    def test_accumulation_dist_finite(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """accumulation_dist values are all finite (no inf)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        ad = result["accumulation_dist"]
        assert np.isfinite(ad).all(), f"Non-finite values in accumulation_dist"

    def test_buying_pressure_ratio_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """buying_pressure_ratio is in [0, 1] range."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        bpr = result["buying_pressure_ratio"]

        assert bpr.min() >= 0, f"buying_pressure_ratio below 0: {bpr.min()}"
        assert bpr.max() <= 1, f"buying_pressure_ratio above 1: {bpr.max()}"

    def test_buying_pressure_ratio_semantic(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """buying_pressure_ratio: close near high → ratio > 0.5."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        bpr = result["buying_pressure_ratio"]

        # Mean should be around 0.5 for random data
        assert 0.3 < bpr.mean() < 0.7, f"buying_pressure_ratio mean unexpected: {bpr.mean()}"

    # --- No-NaN test ---

    def test_volume_indicators_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in volume indicator columns after warmup."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        volume_cols = [
            "obv_slope",
            "volume_price_trend",
            "kvo_histogram",
            "accumulation_dist",
            "buying_pressure_ratio",
        ]
        for col in volume_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestExpectancyMetrics:
    """Test expectancy and win rate indicators (rank 95-96)."""

    def test_expectancy_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """expectancy_20d column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "expectancy_20d" in result.columns

    def test_win_rate_20d_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """win_rate_20d column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "win_rate_20d" in result.columns

    def test_win_rate_20d_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """win_rate_20d is in [0, 1] range."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        win_rate = result["win_rate_20d"]

        assert win_rate.min() >= 0, f"win_rate_20d below 0: {win_rate.min()}"
        assert win_rate.max() <= 1, f"win_rate_20d above 1: {win_rate.max()}"

    def test_win_rate_20d_semantic(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """win_rate_20d: random walk should have ~50% win rate."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        win_rate = result["win_rate_20d"]

        # For random walk, win rate should be around 0.5
        assert 0.3 < win_rate.mean() < 0.7, f"win_rate_20d mean unexpected: {win_rate.mean()}"

    def test_expectancy_20d_finite(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """expectancy_20d values are all finite (no inf)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        exp = result["expectancy_20d"]
        assert np.isfinite(exp).all(), f"Non-finite values in expectancy_20d"

    def test_expectancy_metrics_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in expectancy indicator columns after warmup."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        exp_cols = ["expectancy_20d", "win_rate_20d"]
        for col in exp_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestSupportResistance:
    """Test support/resistance indicators (rank 98-100)."""

    def test_fib_range_position_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """fib_range_position column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "fib_range_position" in result.columns

    def test_prior_high_20d_dist_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """prior_high_20d_dist column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "prior_high_20d_dist" in result.columns

    def test_prior_low_20d_dist_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """prior_low_20d_dist column is present in output."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "prior_low_20d_dist" in result.columns

    def test_fib_range_position_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """fib_range_position is in [0, 1] range."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        fib = result["fib_range_position"]

        assert fib.min() >= 0, f"fib_range_position below 0: {fib.min()}"
        assert fib.max() <= 1, f"fib_range_position above 1: {fib.max()}"

    def test_prior_high_20d_dist_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """prior_high_20d_dist is always <= 0 (at or below prior high)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        high_dist = result["prior_high_20d_dist"]

        assert high_dist.max() <= 0, f"prior_high_20d_dist above 0: {high_dist.max()}"

    def test_prior_low_20d_dist_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """prior_low_20d_dist is always >= 0 (at or above prior low)."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        low_dist = result["prior_low_20d_dist"]

        assert low_dist.min() >= 0, f"prior_low_20d_dist below 0: {low_dist.min()}"

    def test_prior_distances_reasonable_range(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """prior distances are in reasonable range [-50, +50]."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        high_dist = result["prior_high_20d_dist"]
        low_dist = result["prior_low_20d_dist"]

        assert high_dist.min() >= -50, f"prior_high_20d_dist too negative: {high_dist.min()}"
        assert low_dist.max() <= 50, f"prior_low_20d_dist too positive: {low_dist.max()}"

    def test_sr_indicators_no_nan(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in S/R indicator columns after warmup."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)

        sr_cols = ["fib_range_position", "prior_high_20d_dist", "prior_low_20d_dist"]
        for col in sr_cols:
            assert not result[col].isna().any(), f"NaN in {col}"


class TestA100OutputShape:
    """Test output shape and structure for current chunk."""

    def test_output_has_date_column(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output DataFrame has Date column."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "Date" in result.columns

    def test_output_fewer_rows_than_input(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output has fewer rows due to warmup period."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert len(result) < len(sample_daily_df), "Warmup rows should be dropped"

    def test_no_nan_in_output(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in any column after warmup rows dropped."""
        result = tier_a100.build_feature_dataframe(sample_daily_df, sample_vix_df)
        nan_count = result.drop(columns=["Date"]).isnull().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"
