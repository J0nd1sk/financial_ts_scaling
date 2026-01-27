"""Tier a500 indicator module - 500 total indicators (206 from a200 + 294 new).

This module extends tier_a200 with 294 additional indicators organized into
12 sub-chunks (6a through 11b), following the tier_a200 architecture pattern.

Sub-Chunk 6a (rank 207-230): MA Extended Part 1 (~24 features)
- sma_5, sma_14, sma_21, sma_63 - New SMA periods
- ema_5, ema_9, ema_50, ema_100, ema_200 - New EMA periods
- sma_5_slope, sma_21_slope, sma_63_slope - SMA slopes (5-day change)
- ema_9_slope, ema_50_slope, ema_100_slope - EMA slopes (5-day change)
- price_pct_from_sma_5, price_pct_from_sma_21 - Price distance from SMA
- price_pct_from_ema_9, price_pct_from_ema_50, price_pct_from_ema_100 - Price distance from EMA
- sma_5_21_proximity, sma_21_50_proximity, sma_63_200_proximity - SMA proximity
- ema_9_50_proximity - EMA proximity

Sub-Chunk 6b (rank 231-255): MA Durations/Crosses + OSC Extended (~25 features)
Sub-Chunk 7a (rank 256-278): VOL Complete (~23 features)
Sub-Chunk 7b (rank 279-300): VLM Complete (~22 features)
Sub-Chunk 8a (rank 301-323): TRD Complete (~23 features)
Sub-Chunk 8b (rank 324-345): SR Complete (~22 features)
Sub-Chunk 9a (rank 346-370): CDL Part 1 (~25 features)
Sub-Chunk 9b (rank 371-395): CDL Part 2 (~25 features)
Sub-Chunk 10a (rank 396-420): MTF Complete (~25 features)
Sub-Chunk 10b (rank 421-445): ENT Extended (~25 features)
Sub-Chunk 11a (rank 446-472): ADV Part 1 (~27 features)
Sub-Chunk 11b (rank 473-500): ADV Part 2 (~28 features)
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import talib

from src.features import tier_a200

# Sub-Chunk 6a: MA Extended Part 1 (ranks 207-230) - 24 features
CHUNK_6A_FEATURES = [
    # New SMA periods
    "sma_5",
    "sma_14",
    "sma_21",
    "sma_63",
    # New EMA periods
    "ema_5",
    "ema_9",
    "ema_50",
    "ema_100",
    "ema_200",
    # SMA slopes (5-day change)
    "sma_5_slope",
    "sma_21_slope",
    "sma_63_slope",
    # EMA slopes (5-day change)
    "ema_9_slope",
    "ema_50_slope",
    "ema_100_slope",
    # Price-to-SMA distance
    "price_pct_from_sma_5",
    "price_pct_from_sma_21",
    # Price-to-EMA distance
    "price_pct_from_ema_9",
    "price_pct_from_ema_50",
    "price_pct_from_ema_100",
    # SMA-to-SMA proximity
    "sma_5_21_proximity",
    "sma_21_50_proximity",
    "sma_63_200_proximity",
    # EMA-to-EMA proximity
    "ema_9_50_proximity",
]

# 294 new indicators added in tier a500 (ranks 207-500)
# Built incrementally across sub-chunks 6a through 11b
A500_ADDITION_LIST = (
    CHUNK_6A_FEATURES
    # + CHUNK_6B_FEATURES  # To be added
    # + CHUNK_7A_FEATURES  # To be added
    # ... remaining chunks
)

# Complete a500 feature list = a200 (206) + 294 new = 500 total
# Note: Currently only includes Sub-Chunk 6a (24 features)
FEATURE_LIST = tier_a200.FEATURE_LIST + A500_ADDITION_LIST


# =============================================================================
# Sub-Chunk 6a computation functions (ranks 207-230)
# =============================================================================


def _compute_new_sma(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute new SMA periods not in a200.

    Args:
        close: Close price series

    Returns:
        Dict with sma_5, sma_14, sma_21, sma_63
    """
    close_arr = close.values
    features = {}

    for period in [5, 14, 21, 63]:
        sma = talib.SMA(close_arr, timeperiod=period)
        features[f"sma_{period}"] = pd.Series(sma, index=close.index)

    return features


def _compute_new_ema(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute new EMA periods not in a200.

    Args:
        close: Close price series

    Returns:
        Dict with ema_5, ema_9, ema_50, ema_100, ema_200
    """
    close_arr = close.values
    features = {}

    for period in [5, 9, 50, 100, 200]:
        ema = talib.EMA(close_arr, timeperiod=period)
        features[f"ema_{period}"] = pd.Series(ema, index=close.index)

    return features


def _compute_ma_slopes(
    sma_features: Mapping[str, pd.Series],
    ema_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute 5-day slopes for new MAs.

    Args:
        sma_features: Dict containing sma_5, sma_21, sma_63
        ema_features: Dict containing ema_9, ema_50, ema_100

    Returns:
        Dict with slope features (5-day change in MA value)
    """
    features = {}

    # SMA slopes
    for period in [5, 21, 63]:
        sma = sma_features[f"sma_{period}"]
        features[f"sma_{period}_slope"] = sma - sma.shift(5)

    # EMA slopes
    for period in [9, 50, 100]:
        ema = ema_features[f"ema_{period}"]
        features[f"ema_{period}_slope"] = ema - ema.shift(5)

    return features


def _compute_price_ma_distance(
    close: pd.Series,
    sma_features: Mapping[str, pd.Series],
    ema_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute % distance from price to various MAs.

    Formula: (close - MA) / MA * 100

    Args:
        close: Close price series
        sma_features: Dict containing sma_5, sma_21
        ema_features: Dict containing ema_9, ema_50, ema_100

    Returns:
        Dict with price distance features
    """
    features = {}

    # Price to SMA distance
    for period in [5, 21]:
        sma = sma_features[f"sma_{period}"]
        features[f"price_pct_from_sma_{period}"] = (close - sma) / sma * 100

    # Price to EMA distance
    for period in [9, 50, 100]:
        ema = ema_features[f"ema_{period}"]
        features[f"price_pct_from_ema_{period}"] = (close - ema) / ema * 100

    return features


def _compute_ma_proximity(
    close: pd.Series,
    sma_features: Mapping[str, pd.Series],
    ema_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute MA-to-MA proximity (% difference between MAs).

    Proximity indicates how close two MAs are to crossing.
    Formula: (short_MA - long_MA) / long_MA * 100

    Args:
        close: Close price series (for computing SMA 50/200 from a200)
        sma_features: Dict containing sma_5, sma_21, sma_63
        ema_features: Dict containing ema_9, ema_50

    Returns:
        Dict with proximity features
    """
    close_arr = close.values
    features = {}

    # Need SMA 50 and 200 from a200 (not in sma_features)
    sma_50 = pd.Series(talib.SMA(close_arr, timeperiod=50), index=close.index)
    sma_200 = pd.Series(talib.SMA(close_arr, timeperiod=200), index=close.index)

    # SMA proximities
    sma_5 = sma_features["sma_5"]
    sma_21 = sma_features["sma_21"]
    sma_63 = sma_features["sma_63"]

    features["sma_5_21_proximity"] = (sma_5 - sma_21) / sma_21 * 100
    features["sma_21_50_proximity"] = (sma_21 - sma_50) / sma_50 * 100
    features["sma_63_200_proximity"] = (sma_63 - sma_200) / sma_200 * 100

    # EMA proximity
    ema_9 = ema_features["ema_9"]
    ema_50 = ema_features["ema_50"]
    features["ema_9_50_proximity"] = (ema_9 - ema_50) / ema_50 * 100

    return features


def _compute_chunk_6a(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute all Sub-Chunk 6a features.

    Args:
        close: Close price series

    Returns:
        Dict with all 24 Chunk 6a features
    """
    features = {}

    # Compute base MAs
    sma_features = _compute_new_sma(close)
    ema_features = _compute_new_ema(close)

    # Add base MAs to features
    features.update(sma_features)
    features.update(ema_features)

    # Compute derived features
    features.update(_compute_ma_slopes(sma_features, ema_features))
    features.update(_compute_price_ma_distance(close, sma_features, ema_features))
    features.update(_compute_ma_proximity(close, sma_features, ema_features))

    return features


def build_feature_dataframe(raw_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature DataFrame with all tier_a500 indicators.

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

    # Start with all a200 features
    a200_features = tier_a200.build_feature_dataframe(raw_df, vix_df)
    a200_features = a200_features.set_index("Date")

    # Build all new a500 features
    features = {}

    # =========================================================================
    # Sub-Chunk 6a: MA Extended Part 1 (ranks 207-230)
    # =========================================================================
    features.update(_compute_chunk_6a(close))

    # Create feature DataFrame for new a500 features
    new_features_df = pd.DataFrame(features)
    new_features_df.insert(0, "Date", df["Date"])
    new_features_df = new_features_df.set_index("Date")

    # Merge a200 features with new a500 features
    # Use A500_ADDITION_LIST to get only the new features
    merged = a200_features.join(new_features_df[list(A500_ADDITION_LIST)], how="inner")

    # Reset index to get Date as column
    merged = merged.reset_index()

    # Drop rows with any NaN
    merged = merged.dropna().reset_index(drop=True)

    # Return columns in correct order: Date + all features
    return merged[["Date"] + list(FEATURE_LIST)]
