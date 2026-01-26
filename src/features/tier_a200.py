"""Tier a200 indicator module - 200 total indicators (100 from a100 + 100 new).

This module extends tier_a100 with 100 additional indicators.
Currently implements Chunk 1 (ranks 101-120).

Chunk 1 (rank 101-120): Extended MA Types
- tema_{9,20,50,100} - Triple EMA at various periods
- wma_{10,20,50,200} - Weighted MA at various periods
- kama_{10,20,50} - Kaufman Adaptive MA at various periods
- hma_{9,21,50} - Hull MA at various periods
- vwma_{10,20,50} - Volume-Weighted MA at various periods
- tema_20_slope - 5-day change in TEMA_20
- price_pct_from_tema_50 - % distance from TEMA_50
- price_pct_from_kama_20 - % distance from KAMA_20
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import talib

from src.features import tier_a100

# 20 new indicators added in tier a200 Chunk 1 (ranks 101-120)
# Future chunks will add ranks 121-200
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
]

# Complete a200 feature list = a100 (100) + 20 new (Chunk 1) = 120 total
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

    # Build all new a200 Chunk 1 features
    features = {}

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
    features.update(_compute_vwma(close, volume))

    # Derived MA indicators (slope, pct_from)
    features.update(_compute_derived_ma(close, tema_features, kama_features))

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
