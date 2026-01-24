"""Tests for tier_a50 indicator module (indicators 21-50)."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import tier_a50


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

    VIX is required for VRP indicators (ranks 31-33).
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


class TestA50FeatureShape:
    """Test output shape and structure."""

    def test_a50_returns_50_features(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output has exactly 50 indicator columns plus Date."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # Expect: Date + 50 features = 51 columns
        assert result.shape[1] == 51, f"Expected 51 columns, got {result.shape[1]}"

    def test_a50_contains_all_a20_features(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """All 20 features from tier_a20 are present."""
        from src.features import tier_a20

        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        for feature in tier_a20.FEATURE_LIST:
            assert feature in result.columns, f"Missing a20 feature: {feature}"

    def test_a50_feature_list_length(self) -> None:
        """FEATURE_LIST has exactly 50 indicators."""
        assert len(tier_a50.FEATURE_LIST) == 50, f"Expected 50, got {len(tier_a50.FEATURE_LIST)}"

    def test_a50_addition_list_length(self) -> None:
        """A50_ADDITION_LIST has exactly 30 new indicators."""
        assert len(tier_a50.A50_ADDITION_LIST) == 30, f"Expected 30, got {len(tier_a50.A50_ADDITION_LIST)}"

    def test_a50_no_nan_after_warmup(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """No NaN values in output after warmup rows dropped."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        nan_count = result.drop(columns=["Date"]).isnull().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"

    def test_a50_fewer_rows_than_input(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """Output has fewer rows due to warmup period."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert len(result) < len(sample_daily_df), "Warmup rows should be dropped"
        # With 252-day indicators, expect ~40-50 rows from 300 input
        assert len(result) >= 30, f"Too few rows: {len(result)}"


class TestMomentumIndicators:
    """Test momentum indicators (ranks 21-25, 46-47)."""

    def test_return_1d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """return_1d = (close - close.shift(1)) / close.shift(1) * 100."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # Verify return values are in reasonable range (typically -10% to +10% daily)
        ret_1d = result["return_1d"]
        assert ret_1d.min() > -50, f"return_1d too negative: {ret_1d.min()}"
        assert ret_1d.max() < 50, f"return_1d too positive: {ret_1d.max()}"

    def test_return_5d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """return_5d = 5-day percentage return."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        ret_5d = result["return_5d"]
        # 5-day returns typically larger magnitude than 1-day
        assert ret_5d.std() > 0, "return_5d should have non-zero variance"

    def test_return_21d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """return_21d = 21-day (monthly) percentage return."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "return_21d" in result.columns

    def test_return_63d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """return_63d = 63-day (quarterly) percentage return."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "return_63d" in result.columns

    def test_return_252d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """return_252d = 252-day (annual) percentage return."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "return_252d" in result.columns

    def test_overnight_gap_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """overnight_gap = (open - prev_close) / prev_close * 100."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        gap = result["overnight_gap"]
        # Overnight gaps are typically small (-3% to +3%)
        assert gap.abs().mean() < 5, f"Mean absolute gap too large: {gap.abs().mean()}"

    def test_open_to_close_pct_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """open_to_close_pct = (close - open) / open * 100."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        intraday = result["open_to_close_pct"]
        assert intraday.std() > 0, "Intraday returns should have variance"


class TestRSIDerivatives:
    """Test RSI derivative indicators (ranks 38-39)."""

    def test_rsi_slope_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """rsi_slope = rate of change of RSI."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        rsi_slope = result["rsi_slope"]
        # Slope should be roughly centered around 0
        assert abs(rsi_slope.mean()) < 5, f"RSI slope mean too large: {rsi_slope.mean()}"

    def test_rsi_extreme_dist_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """rsi_extreme_dist = distance to nearest extreme (30 or 70)."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        dist = result["rsi_extreme_dist"]
        # Distance should be in range [0, 20] (RSI 50 is 20 from both extremes)
        assert dist.min() >= 0, f"Distance should be non-negative: {dist.min()}"
        assert dist.max() <= 20, f"Distance too large: {dist.max()}"


class TestMADerivatives:
    """Test moving average derivative indicators (ranks 40-42)."""

    def test_price_pct_from_sma_50(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_sma_50 = (close - sma_50) / sma_50 * 100."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        pct_from_sma = result["price_pct_from_sma_50"]
        # Typically within +/- 20% of SMA
        assert pct_from_sma.min() > -50, f"Too far below SMA50: {pct_from_sma.min()}"
        assert pct_from_sma.max() < 50, f"Too far above SMA50: {pct_from_sma.max()}"

    def test_price_pct_from_sma_200(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """price_pct_from_sma_200 = (close - sma_200) / sma_200 * 100."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        pct_from_sma = result["price_pct_from_sma_200"]
        assert pct_from_sma.std() > 0, "Should have non-zero variance"

    def test_sma_50_200_proximity(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sma_50_200_proximity = (sma_50 - sma_200) / sma_200 * 100."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        proximity = result["sma_50_200_proximity"]
        # Typically small values for golden/death cross proximity
        assert proximity.std() > 0, "Should have variance"


class TestVolatilityIndicators:
    """Test volatility indicators (ranks 43-45)."""

    def test_atr_pct_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_pct = ATR / close * 100 (normalized volatility)."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        atr_pct = result["atr_pct"]
        # ATR% typically 0.5-5% for equity indices
        assert atr_pct.min() > 0, "ATR% should be positive"
        assert atr_pct.max() < 20, f"ATR% too high: {atr_pct.max()}"

    def test_atr_pct_slope_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """atr_pct_slope = rate of change of ATR%."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        slope = result["atr_pct_slope"]
        # Slope should be roughly centered around 0
        assert abs(slope.mean()) < 1, f"ATR% slope mean too large: {slope.mean()}"

    def test_bb_width_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """bb_width = (upper_band - lower_band) / middle_band * 100."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        bb_width = result["bb_width"]
        assert bb_width.min() > 0, "BB width should be positive"


class TestVolumeIndicators:
    """Test volume indicators (ranks 48, 50)."""

    def test_volume_ratio_20d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """volume_ratio_20d = volume / 20-day SMA of volume."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        vol_ratio = result["volume_ratio_20d"]
        assert vol_ratio.min() > 0, "Volume ratio should be positive"
        # Average should be close to 1.0
        assert 0.5 < vol_ratio.mean() < 2.0, f"Mean volume ratio unusual: {vol_ratio.mean()}"

    def test_macd_histogram_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """macd_histogram = MACD line - signal line."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        macd_hist = result["macd_histogram"]
        # Histogram should be centered roughly around 0
        assert abs(macd_hist.mean()) < 5, f"MACD histogram mean too large: {macd_hist.mean()}"


class TestRiskMetrics:
    """Test risk metrics (ranks 34-37)."""

    def test_sharpe_20d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sharpe_20d = rolling 20-day Sharpe ratio."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        sharpe = result["sharpe_20d"]
        # Sharpe typically in range [-3, 3] for rolling windows
        assert sharpe.min() > -10, f"Sharpe too negative: {sharpe.min()}"
        assert sharpe.max() < 10, f"Sharpe too positive: {sharpe.max()}"

    def test_sharpe_60d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sharpe_60d = rolling 60-day Sharpe ratio."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sharpe_60d" in result.columns

    def test_sortino_20d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sortino_20d = rolling 20-day Sortino ratio."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        sortino = result["sortino_20d"]
        # Sortino can be higher than Sharpe when there's positive skew
        assert sortino.min() > -20, f"Sortino too negative: {sortino.min()}"

    def test_sortino_60d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """sortino_60d = rolling 60-day Sortino ratio."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "sortino_60d" in result.columns

    def test_risk_metrics_handle_zero_std(self) -> None:
        """Sharpe/Sortino handle zero std gracefully (no inf/nan)."""
        # Create data with constant returns in some periods
        dates = pd.date_range("2023-01-02", periods=300, freq="B")
        n = len(dates)
        # First 50 days constant, then normal
        close = np.concatenate([np.ones(50) * 100, 100 + np.cumsum(np.random.randn(250) * 0.5)])

        df = pd.DataFrame({
            "Date": dates,
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        })

        vix_df = pd.DataFrame({
            "Date": dates,
            "Open": [18.0] * n,
            "High": [19.0] * n,
            "Low": [17.0] * n,
            "Close": [18.0] * n,
            "Volume": [np.nan] * n,
        })

        result = tier_a50.build_feature_dataframe(df, vix_df)
        # Should not have inf values
        assert not np.isinf(result["sharpe_20d"]).any(), "sharpe_20d has inf values"
        assert not np.isinf(result["sortino_20d"]).any(), "sortino_20d has inf values"


class TestKVOIndicator:
    """Test KVO (Klinger Volume Oscillator) indicator (rank 49)."""

    def test_kvo_signal_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """kvo_signal is computed and has valid values."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "kvo_signal" in result.columns
        assert result["kvo_signal"].std() > 0, "KVO signal should have variance"


class TestQQEIndicators:
    """Test QQE (Quantitative Qualitative Estimation) indicators (ranks 26-28)."""

    def test_qqe_fast_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """qqe_fast indicator is computed."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "qqe_fast" in result.columns

    def test_qqe_slow_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """qqe_slow indicator is computed."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "qqe_slow" in result.columns

    def test_qqe_fast_slow_spread_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """qqe_fast_slow_spread = qqe_fast - qqe_slow."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "qqe_fast_slow_spread" in result.columns

    def test_qqe_range_bounds(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """QQE values should be in reasonable RSI-derived range [0, 100]."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # QQE is RSI-based, so values should be in 0-100 range
        assert result["qqe_fast"].min() >= 0, f"qqe_fast min too low: {result['qqe_fast'].min()}"
        assert result["qqe_fast"].max() <= 100, f"qqe_fast max too high: {result['qqe_fast'].max()}"
        assert result["qqe_slow"].min() >= 0, f"qqe_slow min too low: {result['qqe_slow'].min()}"
        assert result["qqe_slow"].max() <= 100, f"qqe_slow max too high: {result['qqe_slow'].max()}"


class TestSTCIndicators:
    """Test STC (Schaff Trend Cycle) indicators (ranks 29-30)."""

    def test_stc_value_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stc_value indicator is computed."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "stc_value" in result.columns

    def test_stc_from_50_exists(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stc_from_50 = stc_value - 50 (distance from neutral)."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "stc_from_50" in result.columns

    def test_stc_value_range_0_100(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """STC is a stochastic-based indicator, bounded [0, 100]."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        stc = result["stc_value"]
        assert stc.min() >= 0, f"STC min too low: {stc.min()}"
        assert stc.max() <= 100, f"STC max too high: {stc.max()}"

    def test_stc_from_50_signed(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """stc_from_50 should be in range [-50, 50]."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        stc_from_50 = result["stc_from_50"]
        assert stc_from_50.min() >= -50, f"stc_from_50 min too low: {stc_from_50.min()}"
        assert stc_from_50.max() <= 50, f"stc_from_50 max too high: {stc_from_50.max()}"


class TestVRPIndicators:
    """Test VRP (Volatility Risk Premium) indicators (ranks 31-33)."""

    def test_vrp_requires_vix_data(self, sample_daily_df: pd.DataFrame) -> None:
        """VRP calculation requires VIX data - should raise without it."""
        with pytest.raises(TypeError):
            # Should fail because vix_df is required
            tier_a50.build_feature_dataframe(sample_daily_df)  # type: ignore

    def test_vrp_10d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vrp_10d = implied vol (VIX) - realized vol (10-day)."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        vrp = result["vrp_10d"]
        # VRP is typically positive (implied > realized)
        assert vrp.mean() > -20, f"Mean VRP unusually negative: {vrp.mean()}"

    def test_vrp_21d_calculation(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """vrp_21d = implied vol (VIX) - realized vol (21-day)."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        assert "vrp_21d" in result.columns

    def test_implied_vs_realized_ratio(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """implied_vs_realized_ratio = VIX / realized_vol."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        ratio = result["implied_vs_realized_ratio"]
        # Ratio typically > 1 (implied > realized)
        assert ratio.min() > 0, "Ratio should be positive"
        assert ratio.max() < 20, f"Ratio unusually high: {ratio.max()}"

    def test_vix_date_alignment(
        self, sample_daily_df: pd.DataFrame, sample_vix_df: pd.DataFrame
    ) -> None:
        """VIX dates must align with price data dates."""
        result = tier_a50.build_feature_dataframe(sample_daily_df, sample_vix_df)
        # Result dates should be subset of input dates
        result_dates = set(result["Date"])
        input_dates = set(sample_daily_df["Date"])
        assert result_dates.issubset(input_dates), "Result dates not subset of input dates"
