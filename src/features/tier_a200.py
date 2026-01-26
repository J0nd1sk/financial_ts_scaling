"""Tier a200 indicator module - 200 total indicators (100 from a100 + 100 new).

This module extends tier_a100 with 100 additional indicators.
Currently implements Chunk 1 (ranks 101-120) and Chunk 2 (ranks 121-140).

Chunk 1 (rank 101-120): Extended MA Types
- tema_{9,20,50,100} - Triple EMA at various periods
- wma_{10,20,50,200} - Weighted MA at various periods
- kama_{10,20,50} - Kaufman Adaptive MA at various periods
- hma_{9,21,50} - Hull MA at various periods
- vwma_{10,20,50} - Volume-Weighted MA at various periods
- tema_20_slope - 5-day change in TEMA_20
- price_pct_from_tema_50 - % distance from TEMA_50
- price_pct_from_kama_20 - % distance from KAMA_20

Chunk 2 (rank 121-140): Duration Counters & Cross Proximity
- days_above/below_sma_{9,50,200} - Consecutive days price above/below SMA
- days_above/below_tema_20 - Consecutive days price above/below TEMA_20
- days_above/below_kama_20 - Consecutive days price above/below KAMA_20
- days_above/below_vwma_20 - Consecutive days price above/below VWMA_20
- days_since_sma_9_50_cross - Days since SMA_9 crossed SMA_50
- days_since_sma_50_200_cross - Days since golden/death cross
- days_since_tema_sma_50_cross - Days since TEMA_20 crossed SMA_50
- days_since_kama_sma_50_cross - Days since KAMA_20 crossed SMA_50
- days_since_sma_9_200_cross - Days since SMA_9 crossed SMA_200
- tema_20_sma_50_proximity - % difference between TEMA_20 and SMA_50
- kama_20_sma_50_proximity - % difference between KAMA_20 and SMA_50
- sma_9_200_proximity - % difference between SMA_9 and SMA_200
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import talib

from src.features import tier_a100

# 40 new indicators added in tier a200 Chunks 1-2 (ranks 101-140)
# Future chunks will add ranks 141-200
A200_ADDITION_LIST = [
    # Chunk 1: Extended MA Types (ranks 101-120)
    # TEMA - Triple Exponential Moving Average
    "tema_9",
    "tema_20",
    "tema_50",
    "tema_100",
    # WMA - Weighted Moving Average
    "wma_10",
    "wma_20",
    "wma_50",
    "wma_200",
    # KAMA - Kaufman Adaptive Moving Average
    "kama_10",
    "kama_20",
    "kama_50",
    # HMA - Hull Moving Average
    "hma_9",
    "hma_21",
    "hma_50",
    # VWMA - Volume-Weighted Moving Average
    "vwma_10",
    "vwma_20",
    "vwma_50",
    # Derived MA indicators
    "tema_20_slope",
    "price_pct_from_tema_50",
    "price_pct_from_kama_20",
    # Chunk 2: Duration Counters & Cross Proximity (ranks 121-140)
    # Duration counters - Price vs MA (ranks 121-132)
    "days_above_sma_9",
    "days_below_sma_9",
    "days_above_sma_50",
    "days_below_sma_50",
    "days_above_sma_200",
    "days_below_sma_200",
    "days_above_tema_20",
    "days_below_tema_20",
    "days_above_kama_20",
    "days_below_kama_20",
    "days_above_vwma_20",
    "days_below_vwma_20",
    # MA-to-MA cross recency (ranks 133-137)
    "days_since_sma_9_50_cross",
    "days_since_sma_50_200_cross",
    "days_since_tema_sma_50_cross",
    "days_since_kama_sma_50_cross",
    "days_since_sma_9_200_cross",
    # New MA proximity features (ranks 138-140)
    "tema_20_sma_50_proximity",
    "kama_20_sma_50_proximity",
    "sma_9_200_proximity",
]

# Complete a200 feature list = a100 (100) + 40 new (Chunks 1-2) = 140 total
FEATURE_LIST = tier_a100.FEATURE_LIST + A200_ADDITION_LIST


def _compute_tema(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute Triple Exponential Moving Average (TEMA) at various periods.

    TEMA reduces lag compared to simple EMA by using triple smoothing.
    Formula: TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))

    Args:
        close: Close price series

    Returns:
        Dict with tema_9, tema_20, tema_50, tema_100
    """
    close_arr = close.values

    features = {}
    for period in [9, 20, 50, 100]:
        tema = talib.TEMA(close_arr, timeperiod=period)
        features[f"tema_{period}"] = pd.Series(tema, index=close.index)

    return features


def _compute_wma(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute Weighted Moving Average (WMA) at various periods.

    WMA gives more weight to recent prices using linear weighting.

    Args:
        close: Close price series

    Returns:
        Dict with wma_10, wma_20, wma_50, wma_200
    """
    close_arr = close.values

    features = {}
    for period in [10, 20, 50, 200]:
        wma = talib.WMA(close_arr, timeperiod=period)
        features[f"wma_{period}"] = pd.Series(wma, index=close.index)

    return features


def _compute_kama(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute Kaufman Adaptive Moving Average (KAMA) at various periods.

    KAMA adapts its smoothing based on market noise/efficiency.
    Low noise = faster response, high noise = slower response.

    Args:
        close: Close price series

    Returns:
        Dict with kama_10, kama_20, kama_50
    """
    close_arr = close.values

    features = {}
    for period in [10, 20, 50]:
        kama = talib.KAMA(close_arr, timeperiod=period)
        features[f"kama_{period}"] = pd.Series(kama, index=close.index)

    return features


def _compute_hma(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute Hull Moving Average (HMA) at various periods.

    HMA reduces lag while maintaining smoothness using weighted moving averages.
    Formula: HMA = WMA(2*WMA(close, n/2) - WMA(close, n), sqrt(n))

    Note: HMA is not available in TA-Lib, so we implement it manually.

    Args:
        close: Close price series

    Returns:
        Dict with hma_9, hma_21, hma_50
    """
    close_arr = close.values

    features = {}
    for period in [9, 21, 50]:
        half_period = max(1, period // 2)
        sqrt_period = max(1, int(np.sqrt(period)))

        # Compute WMA components
        wma_half = talib.WMA(close_arr, timeperiod=half_period)
        wma_full = talib.WMA(close_arr, timeperiod=period)

        # Raw HMA = 2*WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full

        # Final HMA = WMA(raw_hma, sqrt(n))
        hma = talib.WMA(raw_hma, timeperiod=sqrt_period)

        features[f"hma_{period}"] = pd.Series(hma, index=close.index)

    return features


def _compute_vwma(close: pd.Series, volume: pd.Series) -> Mapping[str, pd.Series]:
    """Compute Volume-Weighted Moving Average (VWMA) at various periods.

    VWMA weights prices by volume, giving more importance to high-volume periods.
    Formula: VWMA = SUM(close * volume, n) / SUM(volume, n)

    Note: VWMA is not available in TA-Lib, so we implement it manually.

    Args:
        close: Close price series
        volume: Volume series

    Returns:
        Dict with vwma_10, vwma_20, vwma_50
    """
    features = {}

    # Price * Volume for numerator
    pv = close * volume

    for period in [10, 20, 50]:
        pv_sum = pv.rolling(window=period).sum()
        vol_sum = volume.rolling(window=period).sum()

        # Handle division by zero: if volume sum is 0, use simple close
        # This shouldn't happen with real data, but protects against edge cases
        vwma = pv_sum / vol_sum
        vwma = vwma.where(vol_sum != 0, close)

        features[f"vwma_{period}"] = vwma

    return features


def _compute_derived_ma(
    close: pd.Series,
    tema_features: Mapping[str, pd.Series],
    kama_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute derived MA indicators: slopes and percent distances.

    Args:
        close: Close price series
        tema_features: Dict containing tema_20, tema_50
        kama_features: Dict containing kama_20

    Returns:
        Dict with tema_20_slope, price_pct_from_tema_50, price_pct_from_kama_20
    """
    features = {}

    # TEMA_20 slope: 5-day change in TEMA_20
    tema_20 = tema_features["tema_20"]
    features["tema_20_slope"] = tema_20 - tema_20.shift(5)

    # Price percent from TEMA_50: (close - TEMA_50) / TEMA_50 * 100
    tema_50 = tema_features["tema_50"]
    features["price_pct_from_tema_50"] = (close - tema_50) / tema_50 * 100

    # Price percent from KAMA_20: (close - KAMA_20) / KAMA_20 * 100
    kama_20 = kama_features["kama_20"]
    features["price_pct_from_kama_20"] = (close - kama_20) / kama_20 * 100

    return features


# =============================================================================
# Chunk 2 computation functions (ranks 121-140)
# =============================================================================


def _consecutive_days_above_below(close: pd.Series, ma: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Compute consecutive days price is above/below a moving average.

    Uses the convention that price >= MA means "above".

    Args:
        close: Close price series
        ma: Moving average series

    Returns:
        Tuple of (days_above, days_below) Series
    """
    # Price is above MA if close >= ma (inclusive for "above")
    above_mask = close >= ma

    # Compute run lengths for both states
    days_above = pd.Series(0, index=close.index, dtype=int)
    days_below = pd.Series(0, index=close.index, dtype=int)

    above_count = 0
    below_count = 0

    for i in range(len(close)):
        if pd.isna(ma.iloc[i]):
            # During warmup, set both to 0
            days_above.iloc[i] = 0
            days_below.iloc[i] = 0
            above_count = 0
            below_count = 0
        elif above_mask.iloc[i]:
            above_count += 1
            below_count = 0
            days_above.iloc[i] = above_count
            days_below.iloc[i] = 0
        else:
            below_count += 1
            above_count = 0
            days_below.iloc[i] = below_count
            days_above.iloc[i] = 0

    return days_above, days_below


def _compute_duration_counters(
    close: pd.Series,
    tema_features: Mapping[str, pd.Series],
    kama_features: Mapping[str, pd.Series],
    vwma_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute duration counters: consecutive days above/below various MAs.

    These features capture overextension signals - prices tend to revert
    when above/below MA for extended periods.

    Args:
        close: Close price series
        tema_features: Dict containing tema_20
        kama_features: Dict containing kama_20
        vwma_features: Dict containing vwma_20

    Returns:
        Dict with days_above/below for sma_9, sma_50, sma_200, tema_20, kama_20, vwma_20
    """
    close_arr = close.values
    features = {}

    # Compute SMAs for duration tracking
    sma_9 = pd.Series(talib.SMA(close_arr, timeperiod=9), index=close.index)
    sma_50 = pd.Series(talib.SMA(close_arr, timeperiod=50), index=close.index)
    sma_200 = pd.Series(talib.SMA(close_arr, timeperiod=200), index=close.index)

    # SMA duration counters
    above, below = _consecutive_days_above_below(close, sma_9)
    features["days_above_sma_9"] = above
    features["days_below_sma_9"] = below

    above, below = _consecutive_days_above_below(close, sma_50)
    features["days_above_sma_50"] = above
    features["days_below_sma_50"] = below

    above, below = _consecutive_days_above_below(close, sma_200)
    features["days_above_sma_200"] = above
    features["days_below_sma_200"] = below

    # TEMA_20 duration counters
    above, below = _consecutive_days_above_below(close, tema_features["tema_20"])
    features["days_above_tema_20"] = above
    features["days_below_tema_20"] = below

    # KAMA_20 duration counters
    above, below = _consecutive_days_above_below(close, kama_features["kama_20"])
    features["days_above_kama_20"] = above
    features["days_below_kama_20"] = below

    # VWMA_20 duration counters
    above, below = _consecutive_days_above_below(close, vwma_features["vwma_20"])
    features["days_above_vwma_20"] = above
    features["days_below_vwma_20"] = below

    return features


def _ma_cross_recency(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    """Compute days since short MA crossed long MA (signed).

    Positive values = short MA above long MA (bullish signal)
    Negative values = short MA below long MA (bearish signal)
    Magnitude = days since last cross

    Args:
        short_ma: Shorter period moving average
        long_ma: Longer period moving average

    Returns:
        Series with signed days since cross
    """
    result = pd.Series(0, index=short_ma.index, dtype=int)

    # Determine if short > long (bullish)
    bullish_mask = short_ma >= long_ma

    days_count = 0
    prev_bullish = None

    for i in range(len(short_ma)):
        if pd.isna(short_ma.iloc[i]) or pd.isna(long_ma.iloc[i]):
            result.iloc[i] = 0
            days_count = 0
            prev_bullish = None
        else:
            current_bullish = bullish_mask.iloc[i]
            if prev_bullish is None:
                # First valid point
                days_count = 1
            elif current_bullish != prev_bullish:
                # Cross occurred - reset count
                days_count = 1
            else:
                days_count += 1

            # Sign: positive for bullish, negative for bearish
            result.iloc[i] = days_count if current_bullish else -days_count
            prev_bullish = current_bullish

    return result


def _compute_ma_cross_recency(
    close: pd.Series,
    tema_features: Mapping[str, pd.Series],
    kama_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute MA-to-MA cross recency features.

    Days since various MA crossover events, with sign indicating direction.
    Positive = short MA above long MA (bullish)
    Negative = short MA below long MA (bearish)

    Args:
        close: Close price series (for computing SMAs)
        tema_features: Dict containing tema_20
        kama_features: Dict containing kama_20

    Returns:
        Dict with days_since_*_cross features
    """
    close_arr = close.values
    features = {}

    # Compute SMAs
    sma_9 = pd.Series(talib.SMA(close_arr, timeperiod=9), index=close.index)
    sma_50 = pd.Series(talib.SMA(close_arr, timeperiod=50), index=close.index)
    sma_200 = pd.Series(talib.SMA(close_arr, timeperiod=200), index=close.index)

    # SMA 9 vs SMA 50
    features["days_since_sma_9_50_cross"] = _ma_cross_recency(sma_9, sma_50)

    # SMA 50 vs SMA 200 (golden/death cross)
    features["days_since_sma_50_200_cross"] = _ma_cross_recency(sma_50, sma_200)

    # TEMA 20 vs SMA 50
    features["days_since_tema_sma_50_cross"] = _ma_cross_recency(
        tema_features["tema_20"], sma_50
    )

    # KAMA 20 vs SMA 50
    features["days_since_kama_sma_50_cross"] = _ma_cross_recency(
        kama_features["kama_20"], sma_50
    )

    # SMA 9 vs SMA 200 (extreme momentum)
    features["days_since_sma_9_200_cross"] = _ma_cross_recency(sma_9, sma_200)

    return features


def _compute_new_proximity(
    close: pd.Series,
    tema_features: Mapping[str, pd.Series],
    kama_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute new MA proximity features.

    Proximity indicates how close two MAs are to crossing.
    Formula: (short_MA - long_MA) / long_MA * 100

    Args:
        close: Close price series (for computing SMAs)
        tema_features: Dict containing tema_20
        kama_features: Dict containing kama_20

    Returns:
        Dict with proximity features
    """
    close_arr = close.values
    features = {}

    # Compute SMAs
    sma_9 = pd.Series(talib.SMA(close_arr, timeperiod=9), index=close.index)
    sma_50 = pd.Series(talib.SMA(close_arr, timeperiod=50), index=close.index)
    sma_200 = pd.Series(talib.SMA(close_arr, timeperiod=200), index=close.index)

    # TEMA 20 vs SMA 50 proximity
    tema_20 = tema_features["tema_20"]
    features["tema_20_sma_50_proximity"] = (tema_20 - sma_50) / sma_50 * 100

    # KAMA 20 vs SMA 50 proximity
    kama_20 = kama_features["kama_20"]
    features["kama_20_sma_50_proximity"] = (kama_20 - sma_50) / sma_50 * 100

    # SMA 9 vs SMA 200 proximity (extreme divergence indicator)
    features["sma_9_200_proximity"] = (sma_9 - sma_200) / sma_200 * 100

    return features


def build_feature_dataframe(raw_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature DataFrame with all tier_a200 indicators.

    Args:
        raw_df: DataFrame with Date, Open, High, Low, Close, Volume columns
        vix_df: DataFrame with VIX data (Date, Open, High, Low, Close, Volume)

    Returns:
        DataFrame with Date + indicator columns, warmup rows dropped
    """
    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"]
    volume = df["Volume"].astype(float)

    # Start with all a100 features
    a100_features = tier_a100.build_feature_dataframe(raw_df, vix_df)
    a100_features = a100_features.set_index("Date")

    # Build all new a200 features
    features = {}

    # =========================================================================
    # Chunk 1: Extended MA Types (ranks 101-120)
    # =========================================================================

    # TEMA indicators
    tema_features = _compute_tema(close)
    features.update(tema_features)

    # WMA indicators
    features.update(_compute_wma(close))

    # KAMA indicators
    kama_features = _compute_kama(close)
    features.update(kama_features)

    # HMA indicators
    features.update(_compute_hma(close))

    # VWMA indicators
    vwma_features = _compute_vwma(close, volume)
    features.update(vwma_features)

    # Derived MA indicators (slope, pct_from)
    features.update(_compute_derived_ma(close, tema_features, kama_features))

    # =========================================================================
    # Chunk 2: Duration Counters & Cross Proximity (ranks 121-140)
    # =========================================================================

    # Duration counters: consecutive days above/below various MAs
    features.update(
        _compute_duration_counters(close, tema_features, kama_features, vwma_features)
    )

    # MA-to-MA cross recency: days since crossover events
    features.update(_compute_ma_cross_recency(close, tema_features, kama_features))

    # New MA proximity features
    features.update(_compute_new_proximity(close, tema_features, kama_features))

    # Create feature DataFrame for new a200 features
    new_features_df = pd.DataFrame(features)
    new_features_df.insert(0, "Date", df["Date"])
    new_features_df = new_features_df.set_index("Date")

    # Merge a100 features with new a200 features
    merged = a100_features.join(new_features_df[A200_ADDITION_LIST], how="inner")

    # Reset index to get Date as column
    merged = merged.reset_index()

    # Drop rows with any NaN
    merged = merged.dropna().reset_index(drop=True)

    # Return columns in correct order: Date + all features
    return merged[["Date"] + FEATURE_LIST]
