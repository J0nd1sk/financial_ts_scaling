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

# Sub-Chunk 6b: MA Extended Part 2 + New Oscillators (ranks 231-255) - 25 features
CHUNK_6B_FEATURES = [
    # Duration counters for NEW 6a MAs (8 features)
    "days_above_ema_9",
    "days_below_ema_9",
    "days_above_ema_50",
    "days_below_ema_50",
    "days_above_sma_21",
    "days_below_sma_21",
    "days_above_sma_63",
    "days_below_sma_63",
    # Cross recency for NEW MA pairs (5 features)
    "days_since_ema_9_50_cross",
    "days_since_ema_50_200_cross",
    "days_since_sma_5_21_cross",
    "days_since_sma_21_63_cross",
    "days_since_ema_9_sma_50_cross",
    # MA acceleration (2nd derivative) (4 features)
    "ema_9_acceleration",
    "ema_50_acceleration",
    "sma_21_acceleration",
    "sma_63_acceleration",
    # New oscillator periods (4 features)
    "rsi_5",
    "rsi_21",
    "stoch_k_5",
    "stoch_d_5",
    # Oscillator derivatives (4 features)
    "rsi_5_slope",
    "rsi_21_slope",
    "stoch_k_5_slope",
    "rsi_5_21_spread",
]

# Sub-Chunk 7a: VOL Complete (ranks 256-278) - 23 features
CHUNK_7A_FEATURES = [
    # Extended ATR periods (4 features)
    "atr_5",
    "atr_21",
    "atr_5_pct",
    "atr_21_pct",
    # ATR dynamics (4 features)
    "atr_5_21_ratio",
    "atr_expansion_5d",
    "atr_acceleration",
    "atr_percentile_20d",
    # True Range features (3 features)
    "tr_pct",
    "tr_pct_zscore_20d",
    "consecutive_high_vol_days",
    # Alternative vol estimators (3 features)
    "rogers_satchell_volatility",
    "yang_zhang_volatility",
    "historical_volatility_10d",
    # Bollinger Band extended (4 features)
    "bb_width_slope",
    "bb_width_acceleration",
    "bb_width_percentile_20d",
    "price_bb_band_position",
    # Keltner Channel features (3 features)
    "kc_width",
    "kc_position",
    "bb_kc_ratio",
    # Volatility regime extended (2 features)
    "vol_regime_change_intensity",
    "vol_clustering_score",
]

# Sub-Chunk 7b: VLM Complete (ranks 279-300) - 22 features
CHUNK_7B_FEATURES = [
    # Volume Vectors (4 features)
    "volume_trend_3d",
    "volume_ma_ratio_5_20",
    "consecutive_decreasing_vol",
    "volume_acceleration",
    # VWAP Extended (5 features)
    "pct_from_vwap_20",
    "vwap_slope_5d",
    "vwap_pct_change_1d",
    "vwap_price_divergence",
    "price_vwap_cross_recency",
    # Volume Indicators (5 features)
    "cmf_20",
    "emv_14",
    "nvi_signal",
    "pvi_signal",
    "vpt_slope",
    # Volume-Price Confluence (4 features)
    "volume_spike_price_flat",
    "volume_price_spike_both",
    "sequential_vol_buildup_3d",
    "vol_breakout_confirmation",
    # Volume Regime (4 features)
    "volume_percentile_20d",
    "volume_zscore_20d",
    "avg_vol_up_vs_down_days",
    "volume_trend_strength",
]

# 294 new indicators added in tier a500 (ranks 207-500)
# Built incrementally across sub-chunks 6a through 11b
A500_ADDITION_LIST = (
    CHUNK_6A_FEATURES
    + CHUNK_6B_FEATURES
    + CHUNK_7A_FEATURES
    + CHUNK_7B_FEATURES
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


# =============================================================================
# Sub-Chunk 6b computation functions (ranks 231-255)
# =============================================================================


def _consecutive_days_above_below(close: pd.Series, ma: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Compute consecutive days price is above/below a moving average.

    Uses the convention that price >= MA means "above".
    Replicates the tier_a200 pattern for consistency.

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


def _ma_cross_recency(short_ma: pd.Series, long_ma: pd.Series) -> pd.Series:
    """Compute days since short MA crossed long MA (signed).

    Positive values = short MA above long MA (bullish signal)
    Negative values = short MA below long MA (bearish signal)
    Magnitude = days since last cross

    Replicates the tier_a200 pattern for consistency.

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


def _compute_6b_duration_counters(
    close: pd.Series,
    ema_features: Mapping[str, pd.Series],
    sma_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute duration counters for new 6a MAs.

    Features:
    - days_above/below_ema_9: Consecutive days price above/below EMA_9
    - days_above/below_ema_50: Consecutive days price above/below EMA_50
    - days_above/below_sma_21: Consecutive days price above/below SMA_21
    - days_above/below_sma_63: Consecutive days price above/below SMA_63

    Args:
        close: Close price series
        ema_features: Dict containing ema_9, ema_50
        sma_features: Dict containing sma_21, sma_63

    Returns:
        Dict with 8 duration counter features
    """
    features = {}

    # EMA_9 duration counters
    above, below = _consecutive_days_above_below(close, ema_features["ema_9"])
    features["days_above_ema_9"] = above
    features["days_below_ema_9"] = below

    # EMA_50 duration counters
    above, below = _consecutive_days_above_below(close, ema_features["ema_50"])
    features["days_above_ema_50"] = above
    features["days_below_ema_50"] = below

    # SMA_21 duration counters
    above, below = _consecutive_days_above_below(close, sma_features["sma_21"])
    features["days_above_sma_21"] = above
    features["days_below_sma_21"] = below

    # SMA_63 duration counters
    above, below = _consecutive_days_above_below(close, sma_features["sma_63"])
    features["days_above_sma_63"] = above
    features["days_below_sma_63"] = below

    return features


def _compute_6b_cross_recency(
    close: pd.Series,
    ema_features: Mapping[str, pd.Series],
    sma_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute MA-to-MA cross recency for new MA pairs.

    Features:
    - days_since_ema_9_50_cross: Signed days since EMA_9 crossed EMA_50
    - days_since_ema_50_200_cross: Signed days since EMA_50 crossed EMA_200
    - days_since_sma_5_21_cross: Signed days since SMA_5 crossed SMA_21
    - days_since_sma_21_63_cross: Signed days since SMA_21 crossed SMA_63
    - days_since_ema_9_sma_50_cross: Signed days since EMA_9 crossed SMA_50

    Sign convention: Positive = short MA >= long MA (bullish)

    Args:
        close: Close price series (for computing SMA_50)
        ema_features: Dict containing ema_9, ema_50, ema_200
        sma_features: Dict containing sma_5, sma_21, sma_63

    Returns:
        Dict with 5 cross recency features
    """
    close_arr = close.values
    features = {}

    # EMA 9 vs EMA 50
    features["days_since_ema_9_50_cross"] = _ma_cross_recency(
        ema_features["ema_9"], ema_features["ema_50"]
    )

    # EMA 50 vs EMA 200
    features["days_since_ema_50_200_cross"] = _ma_cross_recency(
        ema_features["ema_50"], ema_features["ema_200"]
    )

    # SMA 5 vs SMA 21
    features["days_since_sma_5_21_cross"] = _ma_cross_recency(
        sma_features["sma_5"], sma_features["sma_21"]
    )

    # SMA 21 vs SMA 63
    features["days_since_sma_21_63_cross"] = _ma_cross_recency(
        sma_features["sma_21"], sma_features["sma_63"]
    )

    # EMA 9 vs SMA 50 (need to compute SMA 50)
    sma_50 = pd.Series(talib.SMA(close_arr, timeperiod=50), index=close.index)
    features["days_since_ema_9_sma_50_cross"] = _ma_cross_recency(
        ema_features["ema_9"], sma_50
    )

    return features


def _compute_6b_acceleration(
    slope_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute MA acceleration features (2nd derivative of MAs).

    Acceleration is the 5-day change in slope (change in change).
    This captures momentum shifts before price changes.

    Features:
    - ema_9_acceleration: 5-day change in ema_9_slope
    - ema_50_acceleration: 5-day change in ema_50_slope
    - sma_21_acceleration: 5-day change in sma_21_slope
    - sma_63_acceleration: 5-day change in sma_63_slope

    Args:
        slope_features: Dict containing ema_9_slope, ema_50_slope, sma_21_slope, sma_63_slope

    Returns:
        Dict with 4 acceleration features
    """
    features = {}

    # EMA accelerations
    features["ema_9_acceleration"] = (
        slope_features["ema_9_slope"] - slope_features["ema_9_slope"].shift(5)
    )
    features["ema_50_acceleration"] = (
        slope_features["ema_50_slope"] - slope_features["ema_50_slope"].shift(5)
    )

    # SMA accelerations
    features["sma_21_acceleration"] = (
        slope_features["sma_21_slope"] - slope_features["sma_21_slope"].shift(5)
    )
    features["sma_63_acceleration"] = (
        slope_features["sma_63_slope"] - slope_features["sma_63_slope"].shift(5)
    )

    return features


def _compute_6b_oscillators(
    close: pd.Series, high: pd.Series, low: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute new oscillator period features.

    Features:
    - rsi_5: RSI with 5-day period (faster than standard RSI_14)
    - rsi_21: RSI with 21-day period (slower than standard RSI_14)
    - stoch_k_5: Stochastic %K with 5-day period
    - stoch_d_5: 3-day SMA of stoch_k_5 (smoothed)

    Args:
        close: Close price series
        high: High price series
        low: Low price series

    Returns:
        Dict with 4 oscillator features
    """
    close_arr = close.values
    high_arr = high.values
    low_arr = low.values
    features = {}

    # RSI with different periods
    features["rsi_5"] = pd.Series(talib.RSI(close_arr, timeperiod=5), index=close.index)
    features["rsi_21"] = pd.Series(talib.RSI(close_arr, timeperiod=21), index=close.index)

    # Stochastic with 5-day period
    # STOCH returns slowk and slowd; we want fastk and a custom smoothing
    # Use STOCHF for fast stochastic
    stoch_k, stoch_d = talib.STOCHF(
        high_arr, low_arr, close_arr,
        fastk_period=5, fastd_period=3, fastd_matype=0  # 0 = SMA
    )
    features["stoch_k_5"] = pd.Series(stoch_k, index=close.index)
    features["stoch_d_5"] = pd.Series(stoch_d, index=close.index)

    return features


def _compute_6b_oscillator_derivatives(
    osc_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute oscillator derivative features.

    Features:
    - rsi_5_slope: 5-day change in RSI_5
    - rsi_21_slope: 5-day change in RSI_21
    - stoch_k_5_slope: 5-day change in stoch_k_5
    - rsi_5_21_spread: RSI_5 - RSI_21 (range: -100 to 100)

    Args:
        osc_features: Dict containing rsi_5, rsi_21, stoch_k_5

    Returns:
        Dict with 4 oscillator derivative features
    """
    features = {}

    # RSI slopes
    features["rsi_5_slope"] = osc_features["rsi_5"] - osc_features["rsi_5"].shift(5)
    features["rsi_21_slope"] = osc_features["rsi_21"] - osc_features["rsi_21"].shift(5)

    # Stochastic slope
    features["stoch_k_5_slope"] = osc_features["stoch_k_5"] - osc_features["stoch_k_5"].shift(5)

    # RSI spread (short - long)
    features["rsi_5_21_spread"] = osc_features["rsi_5"] - osc_features["rsi_21"]

    return features


def _compute_chunk_6b(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    sma_features: Mapping[str, pd.Series],
    ema_features: Mapping[str, pd.Series],
    slope_features: Mapping[str, pd.Series],
) -> Mapping[str, pd.Series]:
    """Compute all Sub-Chunk 6b features.

    Args:
        close: Close price series
        high: High price series
        low: Low price series
        sma_features: Dict containing sma_5, sma_21, sma_63 from 6a
        ema_features: Dict containing ema_9, ema_50, ema_200 from 6a
        slope_features: Dict containing slope features from 6a

    Returns:
        Dict with all 25 Chunk 6b features
    """
    features = {}

    # Duration counters (8 features)
    features.update(_compute_6b_duration_counters(close, ema_features, sma_features))

    # Cross recency (5 features)
    features.update(_compute_6b_cross_recency(close, ema_features, sma_features))

    # Acceleration (4 features)
    features.update(_compute_6b_acceleration(slope_features))

    # New oscillators (4 features)
    osc_features = _compute_6b_oscillators(close, high, low)
    features.update(osc_features)

    # Oscillator derivatives (4 features)
    features.update(_compute_6b_oscillator_derivatives(osc_features))

    return features


# =============================================================================
# Sub-Chunk 7a computation functions (ranks 256-278)
# =============================================================================


def _compute_7a_atr_extended(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute extended ATR period features.

    Features:
    - atr_5: 5-day ATR (short-term volatility)
    - atr_21: 21-day ATR (medium-term volatility)
    - atr_5_pct: ATR_5 as % of close
    - atr_21_pct: ATR_21 as % of close

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 4 ATR extended features
    """
    high_arr = high.values
    low_arr = low.values
    close_arr = close.values
    features = {}

    # ATR with different periods
    atr_5 = pd.Series(talib.ATR(high_arr, low_arr, close_arr, timeperiod=5), index=close.index)
    atr_21 = pd.Series(talib.ATR(high_arr, low_arr, close_arr, timeperiod=21), index=close.index)

    features["atr_5"] = atr_5
    features["atr_21"] = atr_21
    features["atr_5_pct"] = atr_5 / close * 100
    features["atr_21_pct"] = atr_21 / close * 100

    return features


def _compute_7a_atr_dynamics(
    high: pd.Series, low: pd.Series, close: pd.Series,
    atr_extended: Mapping[str, pd.Series]
) -> Mapping[str, pd.Series]:
    """Compute ATR dynamics features.

    Features:
    - atr_5_21_ratio: ATR_5 / ATR_21 (short vs medium vol)
    - atr_expansion_5d: ATR_14 today / ATR_14 5 days ago
    - atr_acceleration: 5-day change in atr_pct_slope
    - atr_percentile_20d: 20-day rolling percentile of ATR_14

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        atr_extended: Dict containing atr_5, atr_21 from _compute_7a_atr_extended

    Returns:
        Dict with 4 ATR dynamics features
    """
    high_arr = high.values
    low_arr = low.values
    close_arr = close.values
    features = {}

    atr_5 = atr_extended["atr_5"]
    atr_21 = atr_extended["atr_21"]

    # ATR_5 / ATR_21 ratio
    features["atr_5_21_ratio"] = atr_5 / atr_21

    # ATR expansion (ATR_14 today / ATR_14 5 days ago)
    atr_14 = pd.Series(talib.ATR(high_arr, low_arr, close_arr, timeperiod=14), index=close.index)
    features["atr_expansion_5d"] = atr_14 / atr_14.shift(5)

    # ATR acceleration (change in atr_pct slope)
    atr_pct = atr_14 / close * 100
    atr_pct_slope = atr_pct - atr_pct.shift(5)
    features["atr_acceleration"] = atr_pct_slope - atr_pct_slope.shift(5)

    # 20-day rolling percentile of ATR_14
    def rolling_percentile(x):
        """Compute percentile rank within rolling window."""
        if len(x) < 2:
            return np.nan
        return (x < x.iloc[-1]).sum() / (len(x) - 1)

    features["atr_percentile_20d"] = atr_14.rolling(window=20, min_periods=20).apply(
        rolling_percentile, raw=False
    )

    return features


def _compute_7a_true_range(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute True Range features.

    Features:
    - tr_pct: True Range as % of close
    - tr_pct_zscore_20d: Z-score of TR% over 20 days
    - consecutive_high_vol_days: Days with TR% > 1.5× 20d avg

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 3 True Range features
    """
    high_arr = high.values
    low_arr = low.values
    close_arr = close.values
    features = {}

    # True Range (single day, using TRANGE from TA-Lib)
    tr = pd.Series(talib.TRANGE(high_arr, low_arr, close_arr), index=close.index)
    tr_pct = tr / close * 100
    features["tr_pct"] = tr_pct

    # Z-score of TR% over 20 days
    tr_mean = tr_pct.rolling(window=20, min_periods=20).mean()
    tr_std = tr_pct.rolling(window=20, min_periods=20).std()
    features["tr_pct_zscore_20d"] = (tr_pct - tr_mean) / tr_std

    # Consecutive days with TR% > 1.5× 20d average
    tr_avg_20d = tr_pct.rolling(window=20, min_periods=20).mean()
    high_vol_mask = tr_pct > (tr_avg_20d * 1.5)

    consecutive_high_vol = pd.Series(0, index=close.index, dtype=int)
    count = 0
    for i in range(len(close)):
        if pd.isna(high_vol_mask.iloc[i]) or pd.isna(tr_avg_20d.iloc[i]):
            consecutive_high_vol.iloc[i] = 0
            count = 0
        elif high_vol_mask.iloc[i]:
            count += 1
            consecutive_high_vol.iloc[i] = count
        else:
            count = 0
            consecutive_high_vol.iloc[i] = 0

    features["consecutive_high_vol_days"] = consecutive_high_vol

    return features


def _compute_7a_vol_estimators(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute alternative volatility estimator features.

    Features:
    - rogers_satchell_volatility: Rogers-Satchell estimator (handles drift)
    - yang_zhang_volatility: Yang-Zhang estimator (most efficient)
    - historical_volatility_10d: 10-day close-to-close annualized vol

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 3 volatility estimator features
    """
    features = {}
    window = 20  # Common window for estimators

    # Log returns for calculations
    log_high = np.log(high)
    log_low = np.log(low)
    log_close = np.log(close)
    log_open = np.log(open_price)

    # Rogers-Satchell volatility (handles drift)
    # RS = log(H/O) * log(H/C) + log(L/O) * log(L/C)
    rs_daily = (log_high - log_open) * (log_high - log_close) + \
               (log_low - log_open) * (log_low - log_close)
    # Take rolling mean and sqrt for volatility
    rs_var = rs_daily.rolling(window=window, min_periods=window).mean()
    # Annualize (sqrt(252) for daily data)
    features["rogers_satchell_volatility"] = np.sqrt(rs_var.clip(lower=0)) * np.sqrt(252)

    # Yang-Zhang volatility (most efficient estimator)
    # Combines overnight, open-to-close, and Rogers-Satchell components
    prev_close = close.shift(1)
    log_prev_close = np.log(prev_close)

    # Overnight volatility (close-to-open)
    overnight = log_open - log_prev_close
    overnight_var = overnight.rolling(window=window, min_periods=window).var()

    # Open-to-close volatility
    open_close = log_close - log_open
    open_close_var = open_close.rolling(window=window, min_periods=window).var()

    # Combine with k weighting factor (optimal k ≈ 0.34)
    k = 0.34
    yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
    features["yang_zhang_volatility"] = np.sqrt(yz_var.clip(lower=0)) * np.sqrt(252)

    # Historical volatility (10-day close-to-close)
    log_returns = log_close - log_prev_close
    hv_10d = log_returns.rolling(window=10, min_periods=10).std() * np.sqrt(252)
    features["historical_volatility_10d"] = hv_10d

    return features


def _compute_7a_bb_extended(
    close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute extended Bollinger Band features.

    Features:
    - bb_width_slope: 5-day change in BB width
    - bb_width_acceleration: 5-day change in BB width slope
    - bb_width_percentile_20d: 20-day rolling percentile of BB width
    - price_bb_band_position: (Close - Lower) / (Upper - Lower), 0-1

    Args:
        close: Close price series

    Returns:
        Dict with 4 extended BB features
    """
    close_arr = close.values
    features = {}

    # Get Bollinger Bands
    upper, middle, lower = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2)
    upper = pd.Series(upper, index=close.index)
    middle = pd.Series(middle, index=close.index)
    lower = pd.Series(lower, index=close.index)

    # BB width as % of middle
    bb_width = (upper - lower) / middle * 100

    # BB width slope (5-day change)
    bb_width_slope = bb_width - bb_width.shift(5)
    features["bb_width_slope"] = bb_width_slope

    # BB width acceleration (change in slope)
    features["bb_width_acceleration"] = bb_width_slope - bb_width_slope.shift(5)

    # 20-day rolling percentile of BB width
    def rolling_percentile(x):
        if len(x) < 2:
            return np.nan
        return (x < x.iloc[-1]).sum() / (len(x) - 1)

    features["bb_width_percentile_20d"] = bb_width.rolling(window=20, min_periods=20).apply(
        rolling_percentile, raw=False
    )

    # Price position within bands: (Close - Lower) / (Upper - Lower)
    band_range = upper - lower
    features["price_bb_band_position"] = (close - lower) / band_range

    return features


def _compute_7a_keltner_channel(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Keltner Channel features.

    Features:
    - kc_width: Keltner Channel width as % of price
    - kc_position: (Close - Lower) / (Upper - Lower), 0-1
    - bb_kc_ratio: BB width / KC width (squeeze intensity)

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 3 Keltner Channel features
    """
    high_arr = high.values
    low_arr = low.values
    close_arr = close.values
    features = {}

    # Keltner Channel: EMA(20) center, 1.5×ATR(10) bands
    kc_middle = pd.Series(talib.EMA(close_arr, timeperiod=20), index=close.index)
    atr_10 = pd.Series(talib.ATR(high_arr, low_arr, close_arr, timeperiod=10), index=close.index)
    kc_upper = kc_middle + 1.5 * atr_10
    kc_lower = kc_middle - 1.5 * atr_10

    # KC width as % of middle
    kc_width = (kc_upper - kc_lower) / kc_middle * 100
    features["kc_width"] = kc_width

    # Price position within KC bands
    kc_range = kc_upper - kc_lower
    features["kc_position"] = (close - kc_lower) / kc_range

    # BB/KC ratio for squeeze intensity
    # Get BB width
    upper, middle, lower = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_width = (pd.Series(upper, index=close.index) - pd.Series(lower, index=close.index)) / \
               pd.Series(middle, index=close.index) * 100

    features["bb_kc_ratio"] = bb_width / kc_width

    return features


def _compute_7a_vol_regime(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute extended volatility regime features.

    Features:
    - vol_regime_change_intensity: Magnitude of regime change (% ATR change)
    - vol_clustering_score: Autocorrelation of abs returns (GARCH-like)

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 2 volatility regime features
    """
    high_arr = high.values
    low_arr = low.values
    close_arr = close.values
    features = {}

    # Vol regime change intensity: % change in ATR (fast vs slow)
    atr_5 = pd.Series(talib.ATR(high_arr, low_arr, close_arr, timeperiod=5), index=close.index)
    atr_21 = pd.Series(talib.ATR(high_arr, low_arr, close_arr, timeperiod=21), index=close.index)

    # Intensity = (ATR_5 - ATR_21) / ATR_21 * 100
    # Positive = expanding vol, Negative = contracting vol
    features["vol_regime_change_intensity"] = (atr_5 - atr_21) / atr_21 * 100

    # Vol clustering score: autocorrelation of absolute returns
    # This measures GARCH-like behavior (vol today predicts vol tomorrow)
    returns = close.pct_change()
    abs_returns = returns.abs()

    # 20-day rolling correlation of abs_returns with lag-1 abs_returns
    abs_returns_lag1 = abs_returns.shift(1)

    def rolling_corr(x):
        """Compute correlation between current and lagged values."""
        if len(x) < 4:
            return np.nan
        current = x.iloc[1:]
        lagged = x.iloc[:-1]
        if current.std() == 0 or lagged.std() == 0:
            return 0.0
        return np.corrcoef(current, lagged)[0, 1]

    # Need to combine current and lagged in a window for correlation
    # Use a simpler approach: rolling correlation
    features["vol_clustering_score"] = abs_returns.rolling(window=20, min_periods=20).apply(
        lambda x: x.autocorr(lag=1) if len(x) >= 2 else np.nan,
        raw=False
    )

    return features


def _compute_chunk_7a(
    open_price: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> Mapping[str, pd.Series]:
    """Compute all Sub-Chunk 7a features.

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with all 23 Chunk 7a features
    """
    features = {}

    # Extended ATR periods (4 features)
    atr_extended = _compute_7a_atr_extended(high, low, close)
    features.update(atr_extended)

    # ATR dynamics (4 features)
    features.update(_compute_7a_atr_dynamics(high, low, close, atr_extended))

    # True Range features (3 features)
    features.update(_compute_7a_true_range(high, low, close))

    # Alternative vol estimators (3 features)
    features.update(_compute_7a_vol_estimators(open_price, high, low, close))

    # Bollinger Band extended (4 features)
    features.update(_compute_7a_bb_extended(close))

    # Keltner Channel features (3 features)
    features.update(_compute_7a_keltner_channel(high, low, close))

    # Volatility regime extended (2 features)
    features.update(_compute_7a_vol_regime(high, low, close))

    return features


# =============================================================================
# Sub-Chunk 7b computation functions (ranks 279-300)
# =============================================================================


def _compute_7b_volume_vectors(volume: pd.Series) -> Mapping[str, pd.Series]:
    """Compute volume vector features.

    Features:
    - volume_trend_3d: (vol - vol.shift(3)) / vol.rolling(20).std()
    - volume_ma_ratio_5_20: vol.rolling(5).mean() / vol.rolling(20).mean()
    - consecutive_decreasing_vol: Count of consecutive days vol < vol[t-1]
    - volume_acceleration: volume_trend_3d - volume_trend_3d.shift(3)

    Args:
        volume: Volume series

    Returns:
        Dict with 4 volume vector features
    """
    features = {}

    # Volume trend (3-day normalized)
    vol_std_20 = volume.rolling(window=20, min_periods=20).std()
    volume_trend_3d = (volume - volume.shift(3)) / vol_std_20
    features["volume_trend_3d"] = volume_trend_3d

    # Volume MA ratio (5/20)
    vol_ma_5 = volume.rolling(window=5, min_periods=5).mean()
    vol_ma_20 = volume.rolling(window=20, min_periods=20).mean()
    features["volume_ma_ratio_5_20"] = vol_ma_5 / vol_ma_20

    # Consecutive decreasing volume days
    decreasing_mask = volume < volume.shift(1)
    consecutive_decreasing = pd.Series(0, index=volume.index, dtype=int)
    count = 0
    for i in range(len(volume)):
        if pd.isna(decreasing_mask.iloc[i]):
            consecutive_decreasing.iloc[i] = 0
            count = 0
        elif decreasing_mask.iloc[i]:
            count += 1
            consecutive_decreasing.iloc[i] = count
        else:
            count = 0
            consecutive_decreasing.iloc[i] = 0
    features["consecutive_decreasing_vol"] = consecutive_decreasing

    # Volume acceleration (change in volume trend)
    features["volume_acceleration"] = volume_trend_3d - volume_trend_3d.shift(3)

    return features


def _compute_7b_vwap_extended(close: pd.Series, volume: pd.Series) -> Mapping[str, pd.Series]:
    """Compute VWAP extended features.

    Features:
    - pct_from_vwap_20: (close - vwap_20) / vwap_20 * 100
    - vwap_slope_5d: vwap_20 - vwap_20.shift(5)
    - vwap_pct_change_1d: (vwap_20 - vwap_20.shift(1)) / vwap_20.shift(1) * 100
    - vwap_price_divergence: 1 if sign(price_slope) != sign(vwap_slope), else 0
    - price_vwap_cross_recency: Signed days since close crossed vwap_20

    Args:
        close: Close price series
        volume: Volume series

    Returns:
        Dict with 5 VWAP extended features
    """
    features = {}

    # Compute 20-day rolling VWAP: sum(close * volume) / sum(volume)
    pv = close * volume
    vwap_20 = pv.rolling(window=20, min_periods=20).sum() / volume.rolling(window=20, min_periods=20).sum()

    # Pct from VWAP
    features["pct_from_vwap_20"] = (close - vwap_20) / vwap_20 * 100

    # VWAP slope (5-day change)
    vwap_slope = vwap_20 - vwap_20.shift(5)
    features["vwap_slope_5d"] = vwap_slope

    # VWAP pct change (1-day)
    features["vwap_pct_change_1d"] = (vwap_20 - vwap_20.shift(1)) / vwap_20.shift(1) * 100

    # Price-VWAP divergence: sign of price slope != sign of VWAP slope
    price_slope = close - close.shift(5)
    divergence = ((np.sign(price_slope) != np.sign(vwap_slope)) &
                  (~pd.isna(price_slope)) & (~pd.isna(vwap_slope))).astype(int)
    features["vwap_price_divergence"] = divergence

    # Price-VWAP cross recency (signed days since cross)
    # Positive = price above VWAP, Negative = price below VWAP
    result = pd.Series(0, index=close.index, dtype=int)
    bullish_mask = close >= vwap_20
    days_count = 0
    prev_bullish = None

    for i in range(len(close)):
        if pd.isna(vwap_20.iloc[i]):
            result.iloc[i] = 0
            days_count = 0
            prev_bullish = None
        else:
            current_bullish = bullish_mask.iloc[i]
            if prev_bullish is None:
                days_count = 1
            elif current_bullish != prev_bullish:
                days_count = 1
            else:
                days_count += 1
            result.iloc[i] = days_count if current_bullish else -days_count
            prev_bullish = current_bullish

    features["price_vwap_cross_recency"] = result

    return features


def _compute_7b_volume_indicators(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute volume indicator features.

    Features:
    - cmf_20: Chaikin Money Flow (20-period) - Range [-1, 1]
    - emv_14: Ease of Movement (14-period smoothed)
    - nvi_signal: Negative Volume Index vs 255-day EMA
    - pvi_signal: Positive Volume Index vs 255-day EMA
    - vpt_slope: 5-day slope of Volume Price Trend

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series

    Returns:
        Dict with 5 volume indicator features
    """
    features = {}

    # CMF (Chaikin Money Flow): CLV * Volume summed over period / Volume summed over period
    # CLV = ((Close - Low) - (High - Close)) / (High - Low)
    hl_range = high - low
    # Handle division by zero (when high == low)
    hl_range_safe = hl_range.replace(0, np.nan)
    clv = ((close - low) - (high - close)) / hl_range_safe
    clv = clv.fillna(0)  # If high == low, CLV is 0
    clv_vol = clv * volume
    cmf_20 = clv_vol.rolling(window=20, min_periods=20).sum() / volume.rolling(window=20, min_periods=20).sum()
    features["cmf_20"] = cmf_20

    # EMV (Ease of Movement)
    # distance = (H + L) / 2 - (H[t-1] + L[t-1]) / 2
    # box_ratio = (Volume / 1e6) / (H - L)
    # EMV = distance / box_ratio, then 14-period EMA
    mid_price = (high + low) / 2
    distance = mid_price - mid_price.shift(1)
    box_ratio = (volume / 1e6) / hl_range_safe
    box_ratio = box_ratio.replace([np.inf, -np.inf], np.nan)
    emv_raw = distance / box_ratio
    emv_raw = emv_raw.replace([np.inf, -np.inf], np.nan)
    emv_14 = emv_raw.ewm(span=14, adjust=False).mean()
    features["emv_14"] = emv_14

    # NVI (Negative Volume Index): updates only when volume decreases
    # Start at 1000, add return only on days when vol < vol[t-1]
    ret = close.pct_change()
    vol_decreased = volume < volume.shift(1)

    nvi = pd.Series(1000.0, index=close.index)
    for i in range(1, len(close)):
        if vol_decreased.iloc[i] and not pd.isna(ret.iloc[i]):
            nvi.iloc[i] = nvi.iloc[i-1] * (1 + ret.iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i-1]

    nvi_ema = nvi.ewm(span=255, min_periods=255, adjust=False).mean()
    nvi_signal = (nvi > nvi_ema).astype(int)
    features["nvi_signal"] = nvi_signal

    # PVI (Positive Volume Index): updates only when volume increases
    vol_increased = volume > volume.shift(1)

    pvi = pd.Series(1000.0, index=close.index)
    for i in range(1, len(close)):
        if vol_increased.iloc[i] and not pd.isna(ret.iloc[i]):
            pvi.iloc[i] = pvi.iloc[i-1] * (1 + ret.iloc[i])
        else:
            pvi.iloc[i] = pvi.iloc[i-1]

    pvi_ema = pvi.ewm(span=255, min_periods=255, adjust=False).mean()
    pvi_signal = (pvi > pvi_ema).astype(int)
    features["pvi_signal"] = pvi_signal

    # VPT (Volume Price Trend) slope
    # VPT = cumsum(volume * (close - close[t-1]) / close[t-1])
    vpt = (volume * ret).cumsum()
    vpt_slope = vpt - vpt.shift(5)
    features["vpt_slope"] = vpt_slope

    return features


def _compute_7b_vol_price_confluence(
    close: pd.Series, volume: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute volume-price confluence features.

    Features:
    - volume_spike_price_flat: 1 if vol_zscore > 2 AND abs(ret_zscore) < 0.5
    - volume_price_spike_both: 1 if vol_zscore > 2 AND abs(ret_zscore) > 2
    - sequential_vol_buildup_3d: Score: (days with vol > prior) / 3 over 3 days
    - vol_breakout_confirmation: 1 if (price > 20d high) AND (vol > 1.5x avg)

    Args:
        close: Close price series
        volume: Volume series

    Returns:
        Dict with 4 volume-price confluence features
    """
    features = {}

    # Compute z-scores
    vol_mean_20 = volume.rolling(window=20, min_periods=20).mean()
    vol_std_20 = volume.rolling(window=20, min_periods=20).std()
    vol_zscore = (volume - vol_mean_20) / vol_std_20

    ret = close.pct_change()
    ret_mean_20 = ret.rolling(window=20, min_periods=20).mean()
    ret_std_20 = ret.rolling(window=20, min_periods=20).std()
    ret_zscore = (ret - ret_mean_20) / ret_std_20

    # Volume spike + price flat
    volume_spike_price_flat = ((vol_zscore > 2) & (ret_zscore.abs() < 0.5)).astype(int)
    features["volume_spike_price_flat"] = volume_spike_price_flat

    # Both volume and price spike
    volume_price_spike_both = ((vol_zscore > 2) & (ret_zscore.abs() > 2)).astype(int)
    features["volume_price_spike_both"] = volume_price_spike_both

    # Sequential volume buildup (3-day score)
    # Count how many of the last 3 days had volume > prior day
    vol_increasing = (volume > volume.shift(1)).astype(int)
    sequential_buildup = (
        vol_increasing +
        vol_increasing.shift(1).fillna(0) +
        vol_increasing.shift(2).fillna(0)
    ) / 3
    features["sequential_vol_buildup_3d"] = sequential_buildup

    # Volume breakout confirmation
    # Price > 20-day high AND volume > 1.5x 20-day average
    price_high_20 = close.rolling(window=20, min_periods=20).max()
    # Compare price to previous day's 20-day high (not including today)
    price_breakout = close > price_high_20.shift(1)
    vol_high = volume > (vol_mean_20 * 1.5)
    vol_breakout_confirmation = (price_breakout & vol_high).astype(int)
    features["vol_breakout_confirmation"] = vol_breakout_confirmation

    return features


def _compute_7b_vol_regime(
    close: pd.Series, volume: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute volume regime features.

    Features:
    - volume_percentile_20d: Percentile rank of volume in 20-day window [0, 1]
    - volume_zscore_20d: (vol - vol.rolling(20).mean()) / vol.rolling(20).std()
    - avg_vol_up_vs_down_days: 20d avg: (vol on up days) / (vol on down days)
    - volume_trend_strength: 20d rolling corr(volume, abs(return)) [-1, 1]

    Args:
        close: Close price series
        volume: Volume series

    Returns:
        Dict with 4 volume regime features
    """
    features = {}

    # Volume percentile in 20-day window
    def rolling_percentile(x):
        if len(x) < 2:
            return np.nan
        return (x < x.iloc[-1]).sum() / (len(x) - 1)

    volume_percentile_20d = volume.rolling(window=20, min_periods=20).apply(
        rolling_percentile, raw=False
    )
    features["volume_percentile_20d"] = volume_percentile_20d

    # Volume z-score (20-day)
    vol_mean_20 = volume.rolling(window=20, min_periods=20).mean()
    vol_std_20 = volume.rolling(window=20, min_periods=20).std()
    features["volume_zscore_20d"] = (volume - vol_mean_20) / vol_std_20

    # Average volume on up days vs down days (20-day rolling)
    ret = close.pct_change()
    up_day = ret >= 0
    down_day = ret < 0

    # Rolling sum of volume on up days and down days
    vol_up = (volume * up_day).rolling(window=20, min_periods=20).sum()
    vol_down = (volume * down_day).rolling(window=20, min_periods=20).sum()
    up_count = up_day.rolling(window=20, min_periods=20).sum()
    down_count = down_day.rolling(window=20, min_periods=20).sum()

    avg_vol_up = vol_up / up_count
    avg_vol_down = vol_down / down_count
    # Handle division by zero (all up or all down days)
    avg_vol_up_vs_down = avg_vol_up / avg_vol_down.replace(0, np.nan)
    # Fill edge cases with 1.0 (neutral)
    avg_vol_up_vs_down = avg_vol_up_vs_down.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    features["avg_vol_up_vs_down_days"] = avg_vol_up_vs_down

    # Volume trend strength: correlation of volume with absolute returns
    abs_ret = ret.abs()

    def rolling_corr_func(x, y, window):
        """Compute rolling correlation."""
        return x.rolling(window=window, min_periods=window).corr(y)

    volume_trend_strength = rolling_corr_func(volume, abs_ret, 20)
    # Handle NaN from constant values
    volume_trend_strength = volume_trend_strength.fillna(0)
    features["volume_trend_strength"] = volume_trend_strength

    return features


def _compute_chunk_7b(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> Mapping[str, pd.Series]:
    """Compute all Sub-Chunk 7b features.

    Args:
        close: Close price series
        high: High price series
        low: Low price series
        volume: Volume series

    Returns:
        Dict with all 22 Chunk 7b features
    """
    features = {}

    # Volume Vectors (4 features)
    features.update(_compute_7b_volume_vectors(volume))

    # VWAP Extended (5 features)
    features.update(_compute_7b_vwap_extended(close, volume))

    # Volume Indicators (5 features)
    features.update(_compute_7b_volume_indicators(high, low, close, volume))

    # Volume-Price Confluence (4 features)
    features.update(_compute_7b_vol_price_confluence(close, volume))

    # Volume Regime (4 features)
    features.update(_compute_7b_vol_regime(close, volume))

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

    open_price = df["Open"]
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Start with all a200 features
    a200_features = tier_a200.build_feature_dataframe(raw_df, vix_df)
    a200_features = a200_features.set_index("Date")

    # Build all new a500 features
    features = {}

    # =========================================================================
    # Sub-Chunk 6a: MA Extended Part 1 (ranks 207-230)
    # =========================================================================

    # Compute base MAs for 6a (also needed for 6b)
    sma_features = _compute_new_sma(close)
    ema_features = _compute_new_ema(close)

    # Add base MAs to features
    features.update(sma_features)
    features.update(ema_features)

    # Compute derived 6a features
    slope_features = _compute_ma_slopes(sma_features, ema_features)
    features.update(slope_features)
    features.update(_compute_price_ma_distance(close, sma_features, ema_features))
    features.update(_compute_ma_proximity(close, sma_features, ema_features))

    # =========================================================================
    # Sub-Chunk 6b: MA Extended Part 2 + New Oscillators (ranks 231-255)
    # =========================================================================
    features.update(_compute_chunk_6b(
        close, high, low,
        sma_features, ema_features, slope_features
    ))

    # =========================================================================
    # Sub-Chunk 7a: VOL Complete (ranks 256-278)
    # =========================================================================
    features.update(_compute_chunk_7a(open_price, high, low, close))

    # =========================================================================
    # Sub-Chunk 7b: VLM Complete (ranks 279-300)
    # =========================================================================
    volume = df["Volume"]
    features.update(_compute_chunk_7b(close, high, low, volume))

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
