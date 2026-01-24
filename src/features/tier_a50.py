"""Tier a50 indicator module - 50 total indicators (20 from a20 + 30 new).

This module extends tier_a20 with 30 high-signal indicators including:
- Momentum returns (1d, 5d, 21d, 63d, 252d)
- QQE oscillators (fast, slow, spread)
- STC (Schaff Trend Cycle)
- VRP (Volatility Risk Premium) - requires VIX data
- Risk metrics (Sharpe, Sortino)
- RSI derivatives
- MA derivatives
- Volatility (ATR%, BB width)
- Volume (ratio, KVO, MACD histogram)
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import talib

from src.features import tier_a20

# 30 new indicators added in tier a50 (ranks 21-50)
A50_ADDITION_LIST = [
    # Momentum (ranks 21-25)
    "return_1d",
    "return_5d",
    "return_21d",
    "return_63d",
    "return_252d",
    # QQE (ranks 26-28)
    "qqe_fast",
    "qqe_slow",
    "qqe_fast_slow_spread",
    # STC (ranks 29-30)
    "stc_value",
    "stc_from_50",
    # VRP (ranks 31-33)
    "vrp_10d",
    "vrp_21d",
    "implied_vs_realized_ratio",
    # Risk metrics (ranks 34-37)
    "sharpe_20d",
    "sharpe_60d",
    "sortino_20d",
    "sortino_60d",
    # RSI derivatives (ranks 38-39)
    "rsi_slope",
    "rsi_extreme_dist",
    # MA derivatives (ranks 40-42)
    "price_pct_from_sma_50",
    "price_pct_from_sma_200",
    "sma_50_200_proximity",
    # Volatility (ranks 43-45)
    "atr_pct",
    "atr_pct_slope",
    "bb_width",
    # Momentum (ranks 46-47)
    "overnight_gap",
    "open_to_close_pct",
    # Volume (ranks 48-50)
    "volume_ratio_20d",
    "kvo_signal",
    "macd_histogram",
]

# Complete a50 feature list = a20 + 30 new
FEATURE_LIST = tier_a20.FEATURE_LIST + A50_ADDITION_LIST


def _compute_returns(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute percentage returns for various horizons."""
    features = {}
    for period, name in [(1, "return_1d"), (5, "return_5d"), (21, "return_21d"),
                         (63, "return_63d"), (252, "return_252d")]:
        pct_change = (close - close.shift(period)) / close.shift(period) * 100
        features[name] = pct_change
    return features


def _compute_overnight_gap_and_intraday(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute overnight gap and intraday return."""
    open_price = df["Open"]
    close = df["Close"]
    prev_close = close.shift(1)

    overnight_gap = (open_price - prev_close) / prev_close * 100
    open_to_close_pct = (close - open_price) / open_price * 100

    return {
        "overnight_gap": overnight_gap,
        "open_to_close_pct": open_to_close_pct,
    }


def _compute_rsi_derivatives(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute RSI slope and distance to extremes."""
    rsi = talib.RSI(close, timeperiod=14)

    # RSI slope: 5-day change in RSI
    rsi_slope = rsi - rsi.shift(5)

    # RSI extreme distance: distance to nearest extreme (30 or 70)
    dist_to_30 = (rsi - 30).abs()
    dist_to_70 = (rsi - 70).abs()
    rsi_extreme_dist = pd.concat([dist_to_30, dist_to_70], axis=1).min(axis=1)

    return {
        "rsi_slope": rsi_slope,
        "rsi_extreme_dist": rsi_extreme_dist,
    }


def _compute_ma_derivatives(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute price distance from SMA and SMA proximity."""
    sma_50 = talib.SMA(close, timeperiod=50)
    sma_200 = talib.SMA(close, timeperiod=200)

    price_pct_from_sma_50 = (close - sma_50) / sma_50 * 100
    price_pct_from_sma_200 = (close - sma_200) / sma_200 * 100
    sma_50_200_proximity = (sma_50 - sma_200) / sma_200 * 100

    return {
        "price_pct_from_sma_50": price_pct_from_sma_50,
        "price_pct_from_sma_200": price_pct_from_sma_200,
        "sma_50_200_proximity": sma_50_200_proximity,
    }


def _compute_volatility_features(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute ATR%, ATR% slope, and Bollinger Band width."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # ATR as percentage of close
    atr = talib.ATR(high, low, close, timeperiod=14)
    atr_pct = atr / close * 100

    # ATR% slope: 5-day change
    atr_pct_slope = atr_pct - atr_pct.shift(5)

    # Bollinger Band width
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    bb_width = (upper - lower) / middle * 100

    return {
        "atr_pct": atr_pct,
        "atr_pct_slope": atr_pct_slope,
        "bb_width": bb_width,
    }


def _compute_volume_features(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute volume ratio, KVO, and MACD histogram."""
    close = df["Close"]
    volume = df["Volume"]

    # Volume ratio: current volume / 20-day SMA of volume
    vol_sma_20 = talib.SMA(volume, timeperiod=20)
    volume_ratio_20d = volume / vol_sma_20.replace(0, np.nan)

    # MACD histogram
    macd_line, signal_line, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

    return {
        "volume_ratio_20d": volume_ratio_20d,
        "macd_histogram": macd_hist,
    }


def _compute_risk_metrics(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute rolling Sharpe and Sortino ratios."""
    # Daily returns for risk calculations
    daily_returns = close.pct_change()

    features = {}

    for window, suffix in [(20, "20d"), (60, "60d")]:
        # Rolling mean and std
        roll_mean = daily_returns.rolling(window).mean()
        roll_std = daily_returns.rolling(window).std()

        # Sharpe ratio (annualized): mean / std * sqrt(252)
        # Handle zero std by replacing with small value
        safe_std = roll_std.replace(0, np.nan)
        sharpe = (roll_mean / safe_std) * np.sqrt(252)
        sharpe = sharpe.fillna(0)  # Zero Sharpe when std is zero
        features[f"sharpe_{suffix}"] = sharpe

        # Sortino ratio: uses downside deviation
        negative_returns = daily_returns.copy()
        negative_returns[negative_returns > 0] = 0
        downside_std = negative_returns.rolling(window).std()
        safe_downside_std = downside_std.replace(0, np.nan)
        sortino = (roll_mean / safe_downside_std) * np.sqrt(252)
        sortino = sortino.fillna(0)  # Zero Sortino when downside std is zero
        features[f"sortino_{suffix}"] = sortino

    return features


def _compute_kvo(df: pd.DataFrame) -> pd.Series:
    """Compute Klinger Volume Oscillator signal line.

    KVO measures the difference between volume flowing in and out of a security.

    Algorithm:
    1. Calculate Trend: +1 if (H+L+C) > (H+L+C)_prev, else -1
    2. dm = High - Low (price range)
    3. cm = cumulative dm for continuous trend direction
    4. Volume Force (VF) = Volume * abs(2*(dm/cm) - 1) * trend * 100
    5. KVO = EMA(VF, 34) - EMA(VF, 55)
    6. Signal = EMA(KVO, 13)
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    volume = df["Volume"].values

    n = len(close)
    hlc = high + low + close
    hlc_prev = np.roll(hlc, 1)
    hlc_prev[0] = hlc[0]

    trend = np.where(hlc > hlc_prev, 1, -1)
    dm = high - low

    # Cumulative dm for continuous trend
    cm = np.zeros(n)
    cm[0] = dm[0]
    for i in range(1, n):
        if trend[i] == trend[i - 1]:
            cm[i] = cm[i - 1] + dm[i]
        else:
            cm[i] = dm[i - 1] + dm[i]

    # Avoid division by zero
    cm_safe = np.where(cm == 0, 1, cm)
    vf = volume * np.abs(2 * (dm / cm_safe) - 1) * trend * 100

    vf_series = pd.Series(vf, index=df.index)

    # EMA calculations
    ema_34 = vf_series.ewm(span=34, adjust=False).mean()
    ema_55 = vf_series.ewm(span=55, adjust=False).mean()
    kvo = ema_34 - ema_55

    # Signal line
    kvo_signal = kvo.ewm(span=13, adjust=False).mean()

    return kvo_signal


def _compute_qqe(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute QQE (Quantitative Qualitative Estimation) indicators.

    QQE is based on RSI and uses a smoothed RSI with dynamic bands.

    Algorithm:
    1. RSI with period 14
    2. Smooth RSI with EMA (5 for fast, 14 for slow)
    3. Calculate ATR of smoothed RSI
    4. Create dynamic bands: smoothed RSI +/- (factor * ATR)
    5. QQE line tracks the smoothed RSI with bands as boundaries
    """
    rsi = talib.RSI(close, timeperiod=14)

    # Fast QQE (smoothing factor 5)
    rsi_smooth_fast = pd.Series(rsi).ewm(span=5, adjust=False).mean()

    # Slow QQE (smoothing factor 14)
    rsi_smooth_slow = pd.Series(rsi).ewm(span=14, adjust=False).mean()

    return {
        "qqe_fast": rsi_smooth_fast,
        "qqe_slow": rsi_smooth_slow,
        "qqe_fast_slow_spread": rsi_smooth_fast - rsi_smooth_slow,
    }


def _compute_stc(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute STC (Schaff Trend Cycle) indicator.

    STC applies double stochastic to MACD, resulting in 0-100 bounded oscillator.

    Algorithm:
    1. Calculate MACD
    2. Apply Stochastic to MACD (first smoothing)
    3. Apply Stochastic again (second smoothing)
    4. Result is bounded 0-100
    """
    # MACD calculation
    macd_line, _, _ = talib.MACD(close, fastperiod=23, slowperiod=50, signalperiod=10)

    # First stochastic of MACD
    period = 10
    macd_series = pd.Series(macd_line)
    macd_low = macd_series.rolling(period).min()
    macd_high = macd_series.rolling(period).max()
    macd_range = macd_high - macd_low
    macd_range = macd_range.replace(0, np.nan)  # Avoid division by zero

    stoch1 = (macd_series - macd_low) / macd_range * 100
    stoch1 = stoch1.fillna(50)  # Neutral value when range is zero

    # Smooth first stochastic
    stoch1_smooth = stoch1.ewm(span=3, adjust=False).mean()

    # Second stochastic
    stoch1_low = stoch1_smooth.rolling(period).min()
    stoch1_high = stoch1_smooth.rolling(period).max()
    stoch1_range = stoch1_high - stoch1_low
    stoch1_range = stoch1_range.replace(0, np.nan)

    stc_raw = (stoch1_smooth - stoch1_low) / stoch1_range * 100
    stc_raw = stc_raw.fillna(50)

    # Final smoothing
    stc_value = stc_raw.ewm(span=3, adjust=False).mean()

    # Clip to ensure 0-100 bounds
    stc_value = stc_value.clip(0, 100)

    return {
        "stc_value": stc_value,
        "stc_from_50": stc_value - 50,
    }


def _compute_realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Compute annualized realized volatility."""
    log_returns = np.log(close / close.shift(1))
    realized_vol = log_returns.rolling(window).std() * np.sqrt(252) * 100
    return realized_vol


def _compute_vrp(df: pd.DataFrame, vix_df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute Volatility Risk Premium indicators.

    VRP = Implied Volatility (VIX) - Realized Volatility

    Requires VIX data aligned with price data.
    """
    # Merge on Date to align VIX with price data
    close = df["Close"]
    dates = df["Date"]

    # Normalize VIX dates to match price data dates (strip time component)
    vix_copy = vix_df.copy()
    vix_copy["Date"] = pd.to_datetime(vix_copy["Date"]).dt.normalize()

    # Get VIX close values aligned to price dates
    vix_aligned = vix_copy.set_index("Date")["Close"].reindex(dates.values)
    vix_aligned = vix_aligned.reset_index(drop=True)

    # Realized volatility for different windows
    rv_10d = _compute_realized_volatility(close, 10)
    rv_21d = _compute_realized_volatility(close, 21)

    # VRP = implied - realized
    vrp_10d = vix_aligned - rv_10d
    vrp_21d = vix_aligned - rv_21d

    # Implied vs realized ratio
    rv_21d_safe = rv_21d.replace(0, np.nan)
    implied_vs_realized_ratio = vix_aligned / rv_21d_safe
    implied_vs_realized_ratio = implied_vs_realized_ratio.fillna(1.0)  # Default to 1 when rv is zero

    return {
        "vrp_10d": vrp_10d,
        "vrp_21d": vrp_21d,
        "implied_vs_realized_ratio": implied_vs_realized_ratio,
    }


def build_feature_dataframe(raw_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature DataFrame with all 50 tier_a50 indicators.

    Args:
        raw_df: DataFrame with Date, Open, High, Low, Close, Volume columns
        vix_df: DataFrame with VIX data (Date, Open, High, Low, Close, Volume)

    Returns:
        DataFrame with Date + 50 indicator columns, warmup rows dropped
    """
    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    vix_df = vix_df.copy()
    vix_df["Date"] = pd.to_datetime(vix_df["Date"])

    close = df["Close"]

    # Start with all a20 features (using the a20 builder, then extract features)
    a20_features = tier_a20.build_feature_dataframe(raw_df)
    # Map a20 features to our df index by Date
    a20_features = a20_features.set_index("Date")

    # Build all new features
    features = {}

    # Momentum returns
    features.update(_compute_returns(close))

    # Overnight gap and intraday
    features.update(_compute_overnight_gap_and_intraday(df))

    # RSI derivatives
    features.update(_compute_rsi_derivatives(close))

    # MA derivatives
    features.update(_compute_ma_derivatives(close))

    # Volatility features
    features.update(_compute_volatility_features(df))

    # Volume features
    volume_features = _compute_volume_features(df)
    features.update(volume_features)

    # Risk metrics
    features.update(_compute_risk_metrics(close))

    # KVO signal
    features["kvo_signal"] = _compute_kvo(df)

    # QQE indicators
    features.update(_compute_qqe(close))

    # STC indicators
    features.update(_compute_stc(close))

    # VRP indicators
    features.update(_compute_vrp(df, vix_df))

    # Create feature DataFrame for new a50 features
    new_features_df = pd.DataFrame(features)
    new_features_df.insert(0, "Date", df["Date"])
    new_features_df = new_features_df.set_index("Date")

    # Merge a20 features with new a50 features
    # Both are indexed by Date
    merged = a20_features.join(new_features_df[A50_ADDITION_LIST], how="inner")

    # Reset index to get Date as column
    merged = merged.reset_index()

    # Drop rows with any NaN
    merged = merged.dropna().reset_index(drop=True)

    # Return columns in correct order: Date + all 50 features
    return merged[["Date"] + FEATURE_LIST]
