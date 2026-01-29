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

# Sub-Chunk 8b: SR Complete (ranks 324-345) - 22 features
CHUNK_8B_FEATURES = [
    # Rolling Range Position (4 features)
    "range_position_20d",
    "range_position_50d",
    "range_position_252d",
    "range_width_20d_pct",
    # Distance from Extremes (4 features)
    "pct_from_20d_high",
    "pct_from_20d_low",
    "pct_from_52w_high",
    "pct_from_52w_low",
    # Recency of Extremes (4 features)
    "days_since_20d_high",
    "days_since_20d_low",
    "days_since_50d_high",
    "days_since_50d_low",
    # Breakout/Breakdown Detection (4 features)
    "breakout_20d",
    "breakdown_20d",
    "breakout_strength_20d",
    "consecutive_new_highs_20d",
    # Range Dynamics (4 features)
    "range_expansion_10d",
    "range_contraction_score",
    "high_low_range_ratio",
    "support_test_count_20d",
    # Fibonacci Context (2 features)
    "fib_retracement_level",
    "distance_to_fib_50",
]

# Sub-Chunk 8a: TRD Complete (ranks 301-323) - 23 features
CHUNK_8A_FEATURES = [
    # ADX Extended (5 features)
    "plus_di_14",
    "minus_di_14",
    "adx_14_slope",
    "adx_acceleration",
    "di_cross_recency",
    # Trend Exhaustion (6 features)
    "avg_up_day_magnitude",
    "avg_down_day_magnitude",
    "up_down_magnitude_ratio",
    "trend_persistence_20d",
    "up_vs_down_momentum",
    "directional_bias_strength",
    # Trend Regime (5 features)
    "adx_regime",
    "price_trend_direction",
    "trend_alignment_score",
    "trend_regime_duration",
    "trend_strength_vs_vol",
    # Trend Channel (4 features)
    "linreg_slope_20d",
    "linreg_r_squared_20d",
    "price_linreg_deviation",
    "channel_width_linreg_20d",
    # Aroon Extended (3 features)
    "aroon_up_25",
    "aroon_down_25",
    "aroon_trend_strength",
]

# Sub-Chunk 9a: CDL Part 1 - Candlestick Patterns (ranks 346-370) - 25 features
CHUNK_9A_FEATURES = [
    # Group A: Engulfing Patterns (4 features)
    "bullish_engulfing",
    "bearish_engulfing",
    "engulfing_score",
    "consecutive_engulfing_count",
    # Group B: Wick Rejection (5 features)
    "hammer_indicator",
    "shooting_star_indicator",
    "hammer_score",
    "shooting_star_score",
    "wick_rejection_score",
    # Group C: Gap Analysis (5 features)
    "gap_size_pct",
    "gap_direction",
    "gap_filled_today",
    "gap_fill_pct",
    "significant_gap",
    # Group D: Inside/Outside Days (4 features)
    "inside_day",
    "outside_day",
    "consecutive_inside_days",
    "consecutive_outside_days",
    # Group E: Range Extremes (4 features)
    "narrow_range_day",
    "wide_range_day",
    "narrow_range_score",
    "consecutive_narrow_days",
    # Group F: Trend Days (3 features)
    "trend_day_indicator",
    "trend_day_direction",
    "consecutive_trend_days",
]

# Sub-Chunk 9b: CDL Part 2 - Candlestick Patterns (ranks 371-395) - 25 features
CHUNK_9B_FEATURES = [
    # Group A: Doji Patterns (5 features)
    "doji_strict_indicator",
    "doji_score",
    "doji_type",
    "consecutive_doji_count",
    "doji_after_trend",
    # Group B: Marubozu & Strong Candles (4 features)
    "marubozu_indicator",
    "marubozu_direction",
    "marubozu_strength",
    "consecutive_strong_candles",
    # Group C: Spinning Top & Indecision (4 features)
    "spinning_top_indicator",
    "spinning_top_score",
    "indecision_streak",
    "indecision_at_extreme",
    # Group D: Multi-Candle Patterns - Reversal (5 features)
    "morning_star_indicator",
    "evening_star_indicator",
    "three_white_soldiers",
    "three_black_crows",
    "harami_indicator",
    # Group E: Multi-Candle Patterns - Continuation (4 features)
    "piercing_line",
    "dark_cloud_cover",
    "tweezer_bottom",
    "tweezer_top",
    # Group F: Pattern Context (3 features)
    "reversal_pattern_count_5d",
    "pattern_alignment_score",
    "pattern_cluster_indicator",
]

# Sub-Chunk 10a: MTF Complete (ranks 396-420) - 25 features
CHUNK_10A_FEATURES = [
    # Weekly MA Features (3 features)
    "weekly_ma_slope",
    "weekly_ma_slope_acceleration",
    "price_pct_from_weekly_ma",
    # Weekly RSI Features (2 features)
    "weekly_rsi_slope",
    "weekly_rsi_slope_acceleration",
    # Weekly Bollinger Band Features (3 features)
    "weekly_bb_position",
    "weekly_bb_width",
    "weekly_bb_width_slope",
    # Daily-Weekly Alignment Features (3 features)
    "trend_alignment_daily_weekly",
    "rsi_alignment_daily_weekly",
    "vol_alignment_daily_weekly",
    # Extended Entropy Features (6 features)
    "permutation_entropy_slope",
    "permutation_entropy_acceleration",
    "sample_entropy_20d",
    "sample_entropy_slope",
    "sample_entropy_acceleration",
    "entropy_percentile_60d",
    "entropy_vol_ratio",
    "entropy_regime_score",
    # Complexity Features (6 features)
    "hurst_exponent_20d",
    "hurst_exponent_slope",
    "autocorr_lag1",
    "autocorr_lag5",
    "autocorr_partial_lag1",
    "fractal_dimension_20d",
]

# 294 new indicators added in tier a500 (ranks 207-500)
# Built incrementally across sub-chunks 6a through 11b
A500_ADDITION_LIST = (
    CHUNK_6A_FEATURES
    + CHUNK_6B_FEATURES
    + CHUNK_7A_FEATURES
    + CHUNK_7B_FEATURES
    + CHUNK_8A_FEATURES
    + CHUNK_8B_FEATURES
    + CHUNK_9A_FEATURES
    + CHUNK_9B_FEATURES
    + CHUNK_10A_FEATURES
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


# =============================================================================
# Sub-Chunk 8a computation functions (ranks 301-323)
# =============================================================================


def _compute_8a_adx_extended(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute ADX Extended features.

    Features:
    - plus_di_14: Raw +DI (0-100)
    - minus_di_14: Raw -DI (0-100)
    - adx_14_slope: ADX - ADX.shift(5)
    - adx_acceleration: adx_slope - adx_slope.shift(5)
    - di_cross_recency: Signed days since +DI crossed -DI

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 5 ADX Extended features
    """
    high_arr = high.values
    low_arr = low.values
    close_arr = close.values
    features = {}

    # Compute +DI and -DI using TA-Lib
    plus_di = pd.Series(talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=14), index=close.index)
    minus_di = pd.Series(talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=14), index=close.index)
    adx = pd.Series(talib.ADX(high_arr, low_arr, close_arr, timeperiod=14), index=close.index)

    features["plus_di_14"] = plus_di
    features["minus_di_14"] = minus_di

    # ADX slope (5-day change)
    adx_slope = adx - adx.shift(5)
    features["adx_14_slope"] = adx_slope

    # ADX acceleration (change in slope)
    features["adx_acceleration"] = adx_slope - adx_slope.shift(5)

    # DI cross recency: signed days since +DI crossed -DI
    # Positive = +DI > -DI (bullish), Negative = -DI > +DI (bearish)
    result = pd.Series(0, index=close.index, dtype=int)
    bullish_mask = plus_di >= minus_di
    days_count = 0
    prev_bullish = None

    for i in range(len(close)):
        if pd.isna(plus_di.iloc[i]) or pd.isna(minus_di.iloc[i]):
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

    features["di_cross_recency"] = result

    return features


def _compute_8a_trend_exhaustion(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute Trend Exhaustion features.

    Features:
    - avg_up_day_magnitude: 20d mean of returns where ret > 0
    - avg_down_day_magnitude: 20d mean of abs(returns where ret < 0)
    - up_down_magnitude_ratio: avg_up / avg_down
    - trend_persistence_20d: 20d max of consecutive up/down streaks
    - up_vs_down_momentum: sum(up returns) / sum(abs(down returns)) over 20d
    - directional_bias_strength: abs(up_days_ratio - 0.5) * 2

    Args:
        close: Close price series

    Returns:
        Dict with 6 Trend Exhaustion features
    """
    features = {}
    window = 20

    # Calculate returns
    ret = close.pct_change() * 100  # As percentage for easier interpretation

    # Separate up and down returns
    up_ret = ret.where(ret > 0, 0)
    down_ret = ret.where(ret < 0, 0).abs()  # Absolute value of down returns

    # Count up and down days
    up_day = (ret > 0).astype(int)
    down_day = (ret < 0).astype(int)

    # Rolling sums and counts
    up_sum = up_ret.rolling(window=window, min_periods=window).sum()
    down_sum = down_ret.rolling(window=window, min_periods=window).sum()
    up_count = up_day.rolling(window=window, min_periods=window).sum()
    down_count = down_day.rolling(window=window, min_periods=window).sum()

    # Average magnitude of up days
    avg_up = up_sum / up_count.replace(0, np.nan)
    avg_up = avg_up.fillna(0)  # If no up days, magnitude is 0
    features["avg_up_day_magnitude"] = avg_up

    # Average magnitude of down days
    avg_down = down_sum / down_count.replace(0, np.nan)
    avg_down = avg_down.fillna(0)  # If no down days, magnitude is 0
    features["avg_down_day_magnitude"] = avg_down

    # Up/down magnitude ratio (handle division by zero)
    ratio = avg_up / avg_down.replace(0, np.nan)
    ratio = ratio.fillna(1.0)  # If no down days, ratio is 1.0 (neutral)
    features["up_down_magnitude_ratio"] = ratio

    # Trend persistence: max consecutive streak in 20-day window
    # Calculate consecutive up/down streaks
    up_streak = pd.Series(0, index=close.index, dtype=int)
    down_streak = pd.Series(0, index=close.index, dtype=int)
    current_up = 0
    current_down = 0

    for i in range(len(close)):
        if pd.isna(ret.iloc[i]):
            up_streak.iloc[i] = 0
            down_streak.iloc[i] = 0
            current_up = 0
            current_down = 0
        elif ret.iloc[i] > 0:
            current_up += 1
            current_down = 0
            up_streak.iloc[i] = current_up
            down_streak.iloc[i] = 0
        elif ret.iloc[i] < 0:
            current_down += 1
            current_up = 0
            down_streak.iloc[i] = current_down
            up_streak.iloc[i] = 0
        else:  # ret == 0
            # Flat day - reset both streaks
            current_up = 0
            current_down = 0
            up_streak.iloc[i] = 0
            down_streak.iloc[i] = 0

    # Max streak in rolling window (max of up or down streak)
    max_streak = pd.concat([up_streak, down_streak], axis=1).max(axis=1)
    features["trend_persistence_20d"] = max_streak.rolling(window=window, min_periods=window).max()

    # Up vs down momentum
    momentum = up_sum / down_sum.replace(0, np.nan)
    momentum = momentum.fillna(1.0)  # If no down returns, momentum is neutral
    features["up_vs_down_momentum"] = momentum

    # Directional bias strength
    up_days_ratio = up_count / window
    directional_bias = (up_days_ratio - 0.5).abs() * 2
    features["directional_bias_strength"] = directional_bias

    return features


def _compute_8a_trend_regime(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Trend Regime features.

    Features:
    - adx_regime: 0 if ADX < 20, 1 if 20-25, 2 if > 25
    - price_trend_direction: sign(SMA_50_slope) with threshold
    - trend_alignment_score: 1 if DI-spread and price direction agree
    - trend_regime_duration: Days in current adx_regime
    - trend_strength_vs_vol: ADX / (ATR_pct * 10)

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 5 Trend Regime features
    """
    high_arr = high.values
    low_arr = low.values
    close_arr = close.values
    features = {}

    # Get ADX
    adx = pd.Series(talib.ADX(high_arr, low_arr, close_arr, timeperiod=14), index=close.index)

    # ADX regime: 0 = ranging (ADX < 20), 1 = weak trend (20-25), 2 = strong trend (> 25)
    adx_regime = pd.Series(0, index=close.index, dtype=int)
    adx_regime = adx_regime.where(adx < 20, 1)  # Set to 1 where ADX >= 20
    adx_regime = adx_regime.where(adx <= 25, 2)  # Set to 2 where ADX > 25
    # Handle NaN in ADX - set to 0 (ranging)
    adx_regime = adx_regime.fillna(0).astype(int)
    features["adx_regime"] = adx_regime

    # Price trend direction: based on SMA_50 slope
    sma_50 = pd.Series(talib.SMA(close_arr, timeperiod=50), index=close.index)
    sma_50_slope = sma_50 - sma_50.shift(5)

    # Threshold: slope > 0.5% of price is bullish, < -0.5% is bearish, else neutral
    slope_pct = sma_50_slope / sma_50 * 100
    price_trend_direction = pd.Series(0, index=close.index, dtype=int)
    price_trend_direction = price_trend_direction.where(slope_pct.abs() <= 0.5, 0)
    price_trend_direction = price_trend_direction.mask(slope_pct > 0.5, 1)
    price_trend_direction = price_trend_direction.mask(slope_pct < -0.5, -1)
    price_trend_direction = price_trend_direction.fillna(0).astype(int)
    features["price_trend_direction"] = price_trend_direction

    # Trend alignment: DI-spread direction agrees with price direction
    plus_di = pd.Series(talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=14), index=close.index)
    minus_di = pd.Series(talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=14), index=close.index)
    di_direction = np.sign(plus_di - minus_di)

    # Alignment: both bullish (DI > 0 and price = 1) or both bearish (DI < 0 and price = -1)
    # or both neutral
    alignment = ((di_direction > 0) & (price_trend_direction == 1)) | \
                ((di_direction < 0) & (price_trend_direction == -1)) | \
                ((di_direction == 0) & (price_trend_direction == 0))
    features["trend_alignment_score"] = alignment.astype(int)

    # Trend regime duration: consecutive days in current regime
    regime_duration = pd.Series(0, index=close.index, dtype=int)
    days_count = 0
    prev_regime = None

    for i in range(len(close)):
        current_regime = adx_regime.iloc[i]
        if pd.isna(adx.iloc[i]):
            regime_duration.iloc[i] = 1
            days_count = 1
            prev_regime = None
        else:
            if prev_regime is None or current_regime != prev_regime:
                days_count = 1
            else:
                days_count += 1
            regime_duration.iloc[i] = days_count
            prev_regime = current_regime

    features["trend_regime_duration"] = regime_duration

    # Trend strength vs volatility: ADX / (ATR_pct * 10)
    atr = pd.Series(talib.ATR(high_arr, low_arr, close_arr, timeperiod=14), index=close.index)
    atr_pct = atr / close * 100
    # Scale ATR_pct to make ratio more interpretable
    trend_vs_vol = adx / (atr_pct * 10).replace(0, np.nan)
    trend_vs_vol = trend_vs_vol.fillna(0)
    features["trend_strength_vs_vol"] = trend_vs_vol

    return features


def _compute_8a_trend_channel(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute Trend Channel features using linear regression.

    Features:
    - linreg_slope_20d: Linear regression slope annualized as %
    - linreg_r_squared_20d: R-squared (goodness of fit) in [0, 1]
    - price_linreg_deviation: (Close - linreg) / Close * 100
    - channel_width_linreg_20d: 2 * std(residuals) / Close * 100

    Args:
        close: Close price series

    Returns:
        Dict with 4 Trend Channel features
    """
    close_arr = close.values
    features = {}
    window = 20

    # Use TA-Lib's LINEARREG for regression value
    linreg = pd.Series(talib.LINEARREG(close_arr, timeperiod=window), index=close.index)
    linreg_slope = pd.Series(talib.LINEARREG_SLOPE(close_arr, timeperiod=window), index=close.index)

    # Annualized slope as percentage of price
    # slope is price change per day, annualize by * 252, then express as % of price
    linreg_slope_pct = (linreg_slope * 252) / close * 100
    features["linreg_slope_20d"] = linreg_slope_pct

    # R-squared: compute manually using rolling window
    def compute_r_squared(x):
        """Compute R-squared for a rolling window."""
        if len(x) < window:
            return np.nan
        y = np.array(x)
        t = np.arange(len(y))
        if np.std(y) == 0:
            return 1.0  # Perfect fit if no variance
        # Linear regression
        slope, intercept = np.polyfit(t, y, 1)
        y_pred = slope * t + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0
        return 1 - ss_res / ss_tot

    r_squared = close.rolling(window=window, min_periods=window).apply(
        compute_r_squared, raw=True
    )
    features["linreg_r_squared_20d"] = r_squared

    # Price deviation from linear regression
    deviation = (close - linreg) / close * 100
    features["price_linreg_deviation"] = deviation

    # Channel width: 2 * std of residuals as % of price
    def compute_channel_width(x, idx):
        """Compute channel width as 2*std of residuals."""
        if len(x) < window:
            return np.nan
        y = np.array(x)
        t = np.arange(len(y))
        # Linear regression
        slope, intercept = np.polyfit(t, y, 1)
        y_pred = slope * t + intercept
        residuals = y - y_pred
        return 2 * np.std(residuals)

    # Rolling std of residuals
    def channel_width_rolling(x):
        if len(x) < window:
            return np.nan
        y = np.array(x)
        t = np.arange(len(y))
        slope, intercept = np.polyfit(t, y, 1)
        y_pred = slope * t + intercept
        residuals = y - y_pred
        current_price = y[-1]
        return 2 * np.std(residuals) / current_price * 100

    channel_width = close.rolling(window=window, min_periods=window).apply(
        channel_width_rolling, raw=True
    )
    features["channel_width_linreg_20d"] = channel_width

    return features


def _compute_8a_aroon_extended(
    high: pd.Series, low: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Aroon Extended features.

    Features:
    - aroon_up_25: Aroon Up with 25-period
    - aroon_down_25: Aroon Down with 25-period
    - aroon_trend_strength: abs(aroon_up - aroon_down) / 100

    Args:
        high: High price series
        low: Low price series

    Returns:
        Dict with 3 Aroon Extended features
    """
    high_arr = high.values
    low_arr = low.values
    features = {}

    # TA-Lib AROON returns (aroon_down, aroon_up)
    aroon_down, aroon_up = talib.AROON(high_arr, low_arr, timeperiod=25)
    aroon_up = pd.Series(aroon_up, index=high.index)
    aroon_down = pd.Series(aroon_down, index=high.index)

    features["aroon_up_25"] = aroon_up
    features["aroon_down_25"] = aroon_down

    # Trend strength: magnitude of difference
    features["aroon_trend_strength"] = (aroon_up - aroon_down).abs() / 100

    return features


def _compute_chunk_8a(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> Mapping[str, pd.Series]:
    """Compute all Sub-Chunk 8a features.

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with all 23 Chunk 8a features
    """
    features = {}

    # ADX Extended (5 features)
    features.update(_compute_8a_adx_extended(high, low, close))

    # Trend Exhaustion (6 features)
    features.update(_compute_8a_trend_exhaustion(close))

    # Trend Regime (5 features)
    features.update(_compute_8a_trend_regime(high, low, close))

    # Trend Channel (4 features)
    features.update(_compute_8a_trend_channel(close))

    # Aroon Extended (3 features)
    features.update(_compute_8a_aroon_extended(high, low))

    return features


# =============================================================================
# Sub-Chunk 8b computation functions (ranks 324-345)
# =============================================================================


def _compute_8b_range_position(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Rolling Range Position features.

    Features:
    - range_position_20d: (Close - Low20) / (High20 - Low20), clipped to [0, 1]
    - range_position_50d: (Close - Low50) / (High50 - Low50), clipped to [0, 1]
    - range_position_252d: (Close - Low252) / (High252 - Low252), clipped to [0, 1]
    - range_width_20d_pct: (High20 - Low20) / Close * 100

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 4 Range Position features
    """
    features = {}

    for period, label in [(20, "20d"), (50, "50d"), (252, "252d")]:
        rolling_high = high.rolling(window=period, min_periods=period).max()
        rolling_low = low.rolling(window=period, min_periods=period).min()

        range_size = rolling_high - rolling_low
        # Handle zero range (flat price) by setting to NaN then filling
        range_size_safe = range_size.replace(0, np.nan)
        position = (close - rolling_low) / range_size_safe
        # Clip to [0, 1] to handle edge cases and fill NaN from zero range
        position = position.clip(lower=0, upper=1).fillna(0.5)  # 0.5 = middle if range is 0
        features[f"range_position_{label}"] = position

    # Range width as % of close (only for 20d)
    rolling_high_20 = high.rolling(window=20, min_periods=20).max()
    rolling_low_20 = low.rolling(window=20, min_periods=20).min()
    range_width = (rolling_high_20 - rolling_low_20) / close * 100
    features["range_width_20d_pct"] = range_width

    return features


def _compute_8b_distance_from_extremes(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Distance from Extremes features.

    Features:
    - pct_from_20d_high: (Close - High20) / High20 * 100 (always <= 0)
    - pct_from_20d_low: (Close - Low20) / Low20 * 100 (always >= 0)
    - pct_from_52w_high: (Close - High252) / High252 * 100 (always <= 0)
    - pct_from_52w_low: (Close - Low252) / Low252 * 100 (always >= 0)

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 4 Distance from Extremes features
    """
    features = {}

    # 20-day extremes
    high_20 = high.rolling(window=20, min_periods=20).max()
    low_20 = low.rolling(window=20, min_periods=20).min()
    features["pct_from_20d_high"] = (close - high_20) / high_20 * 100
    features["pct_from_20d_low"] = (close - low_20) / low_20 * 100

    # 52-week (252 trading days) extremes
    high_252 = high.rolling(window=252, min_periods=252).max()
    low_252 = low.rolling(window=252, min_periods=252).min()
    features["pct_from_52w_high"] = (close - high_252) / high_252 * 100
    features["pct_from_52w_low"] = (close - low_252) / low_252 * 100

    return features


def _compute_8b_recency_of_extremes(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Recency of Extremes features.

    Features:
    - days_since_20d_high: Days since close == 20d rolling high [0, 19]
    - days_since_20d_low: Days since close == 20d rolling low [0, 19]
    - days_since_50d_high: Days since close == 50d rolling high [0, 49]
    - days_since_50d_low: Days since close == 50d rolling low [0, 49]

    Note: Uses close price comparison with tolerance for floating point.

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 4 Recency of Extremes features
    """
    features = {}

    for period, max_days in [(20, 19), (50, 49)]:
        rolling_high = high.rolling(window=period, min_periods=period).max()
        rolling_low = low.rolling(window=period, min_periods=period).min()

        # Days since high: check if current high equals rolling high
        # Use idxmax to find the index of max within window
        days_since_high = pd.Series(0, index=close.index, dtype=int)
        days_since_low = pd.Series(0, index=close.index, dtype=int)

        for i in range(len(close)):
            if i < period - 1:
                days_since_high.iloc[i] = 0
                days_since_low.iloc[i] = 0
                continue

            # Look back within window to find most recent high/low
            window_high = high.iloc[max(0, i - period + 1):i + 1]
            window_low = low.iloc[max(0, i - period + 1):i + 1]

            # Find index of max/min (most recent if tie)
            high_idx = window_high[::-1].idxmax()  # Reverse to get most recent tie
            low_idx = window_low[::-1].idxmin()

            # Calculate days since
            days_since_high.iloc[i] = i - close.index.get_loc(high_idx)
            days_since_low.iloc[i] = i - close.index.get_loc(low_idx)

        # Clip to max range
        days_since_high = days_since_high.clip(upper=max_days)
        days_since_low = days_since_low.clip(upper=max_days)

        label = "20d" if period == 20 else "50d"
        features[f"days_since_{label}_high"] = days_since_high
        features[f"days_since_{label}_low"] = days_since_low

    return features


def _compute_8b_breakout_detection(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Breakout/Breakdown Detection features.

    Features:
    - breakout_20d: 1 if Close > previous 20d high, else 0
    - breakdown_20d: 1 if Close < previous 20d low, else 0
    - breakout_strength_20d: (Close - prev_high) / ATR when breaking out, else 0
    - consecutive_new_highs_20d: Consecutive days at new 20d high

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 4 Breakout Detection features
    """
    features = {}

    # Previous day's 20-day high/low (exclude today)
    prev_high_20 = high.rolling(window=20, min_periods=20).max().shift(1)
    prev_low_20 = low.rolling(window=20, min_periods=20).min().shift(1)

    # Breakout/breakdown binary
    breakout = (close > prev_high_20).astype(int)
    breakdown = (close < prev_low_20).astype(int)
    features["breakout_20d"] = breakout
    features["breakdown_20d"] = breakdown

    # Breakout strength: (Close - prev_high) / ATR (only on breakout days)
    # Compute ATR_14 for normalization
    high_arr = high.values
    low_arr = low.values
    close_arr = close.values
    atr_14 = pd.Series(talib.ATR(high_arr, low_arr, close_arr, timeperiod=14), index=close.index)

    breakout_amount = close - prev_high_20
    breakout_strength = (breakout_amount / atr_14).clip(lower=0)
    # Set to 0 when not breaking out
    breakout_strength = breakout_strength.where(breakout == 1, 0)
    features["breakout_strength_20d"] = breakout_strength

    # Consecutive new highs (close > prev 20d high for multiple days)
    consecutive_highs = pd.Series(0, index=close.index, dtype=int)
    count = 0
    for i in range(len(close)):
        if pd.isna(prev_high_20.iloc[i]):
            consecutive_highs.iloc[i] = 0
            count = 0
        elif close.iloc[i] > prev_high_20.iloc[i]:
            count += 1
            consecutive_highs.iloc[i] = count
        else:
            count = 0
            consecutive_highs.iloc[i] = 0
    features["consecutive_new_highs_20d"] = consecutive_highs

    return features


def _compute_8b_range_dynamics(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Range Dynamics features.

    Features:
    - range_expansion_10d: (10d range today) / (10d range 10 days ago)
    - range_contraction_score: Consecutive days of narrowing 10d range
    - high_low_range_ratio: (20d range) / (50d range)
    - support_test_count_20d: Days price touched Low20 within tolerance in 20d window

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 4 Range Dynamics features
    """
    features = {}

    # 10-day range
    range_10d = high.rolling(window=10, min_periods=10).max() - low.rolling(window=10, min_periods=10).min()
    range_10d_prev = range_10d.shift(10)

    # Range expansion: current range / range 10 days ago
    # Handle zero/NaN in denominator
    expansion = range_10d / range_10d_prev.replace(0, np.nan)
    expansion = expansion.fillna(1.0)  # If prev range is 0, assume expansion of 1
    features["range_expansion_10d"] = expansion

    # Range contraction score: consecutive days of narrowing range
    range_narrowing = range_10d < range_10d.shift(1)
    contraction_score = pd.Series(0, index=close.index, dtype=int)
    count = 0
    for i in range(len(close)):
        if pd.isna(range_narrowing.iloc[i]):
            contraction_score.iloc[i] = 0
            count = 0
        elif range_narrowing.iloc[i]:
            count += 1
            contraction_score.iloc[i] = count
        else:
            count = 0
            contraction_score.iloc[i] = 0
    features["range_contraction_score"] = contraction_score

    # High-low range ratio: 20d range / 50d range
    range_20d = high.rolling(window=20, min_periods=20).max() - low.rolling(window=20, min_periods=20).min()
    range_50d = high.rolling(window=50, min_periods=50).max() - low.rolling(window=50, min_periods=50).min()
    ratio = range_20d / range_50d.replace(0, np.nan)
    ratio = ratio.fillna(1.0)
    features["high_low_range_ratio"] = ratio

    # Support test count: days within tolerance of 20d low in 20d window
    # Tolerance: within 0.5% of Low20
    low_20 = low.rolling(window=20, min_periods=20).min()
    tolerance = low_20 * 0.005  # 0.5%
    near_support = (close - low_20).abs() <= tolerance

    # Rolling count of support tests in 20-day window
    support_test_count = near_support.astype(int).rolling(window=20, min_periods=20).sum()
    features["support_test_count_20d"] = support_test_count

    return features


def _compute_8b_fibonacci_context(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Fibonacci Context features.

    Features:
    - fib_retracement_level: Nearest fib level (0, 0.236, 0.382, 0.5, 0.618, 0.786, 1)
    - distance_to_fib_50: (Close - fib_50) / range, where fib_50 = Low20 + 0.5 * range

    Fibonacci levels computed from 20-day range.

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 2 Fibonacci Context features
    """
    features = {}

    # 20-day range for Fibonacci levels
    high_20 = high.rolling(window=20, min_periods=20).max()
    low_20 = low.rolling(window=20, min_periods=20).min()
    range_20 = high_20 - low_20

    # Fibonacci levels (standard retracement levels)
    fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

    # Calculate price position as retracement level: (Close - Low) / Range
    position = (close - low_20) / range_20.replace(0, np.nan)
    position = position.fillna(0.5)  # If range is 0, assume middle

    # Find nearest Fibonacci level
    def nearest_fib(pos):
        if pd.isna(pos):
            return 0.5
        return min(fib_levels, key=lambda x: abs(x - pos))

    fib_level = position.apply(nearest_fib)
    features["fib_retracement_level"] = fib_level

    # Distance to Fibonacci 50% level
    # fib_50 = Low20 + 0.5 * range
    fib_50 = low_20 + 0.5 * range_20
    # Normalize by range
    distance = (close - fib_50) / range_20.replace(0, np.nan)
    distance = distance.fillna(0.0)  # If range is 0, distance is 0
    features["distance_to_fib_50"] = distance

    return features


def _compute_chunk_8b(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> Mapping[str, pd.Series]:
    """Compute all Sub-Chunk 8b features.

    Args:
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with all 22 Chunk 8b features
    """
    features = {}

    # Range Position (4 features)
    features.update(_compute_8b_range_position(high, low, close))

    # Distance from Extremes (4 features)
    features.update(_compute_8b_distance_from_extremes(high, low, close))

    # Recency of Extremes (4 features)
    features.update(_compute_8b_recency_of_extremes(high, low, close))

    # Breakout/Breakdown Detection (4 features)
    features.update(_compute_8b_breakout_detection(high, low, close))

    # Range Dynamics (4 features)
    features.update(_compute_8b_range_dynamics(high, low, close))

    # Fibonacci Context (2 features)
    features.update(_compute_8b_fibonacci_context(high, low, close))

    return features


# =============================================================================
# Sub-Chunk 9a computation functions (ranks 346-370)
# CDL Part 1 - Candlestick Patterns
# =============================================================================


def _compute_9a_engulfing(
    open_price: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Engulfing Pattern features.

    Features:
    - bullish_engulfing: 1 if today's body engulfs yesterday's AND today up AND yesterday down
    - bearish_engulfing: 1 if today's body engulfs yesterday's AND today down AND yesterday up
    - engulfing_score: today_body / yesterday_body, clipped [0, 5]
    - consecutive_engulfing_count: Running count of consecutive engulfing days

    Args:
        open_price: Open price series
        close: Close price series

    Returns:
        Dict with 4 Engulfing Pattern features
    """
    features = {}

    # Body calculations (absolute value for size comparison)
    body = close - open_price
    body_size = np.abs(body)
    prev_body = body.shift(1)
    prev_body_size = body_size.shift(1)

    # Today's body bounds (min/max of open and close)
    body_low = np.minimum(open_price, close)
    body_high = np.maximum(open_price, close)
    prev_body_low = body_low.shift(1)
    prev_body_high = body_high.shift(1)

    # Engulfing: today's body completely contains yesterday's body
    today_engulfs = (body_low <= prev_body_low) & (body_high >= prev_body_high)

    # Direction checks
    today_up = body > 0
    today_down = body < 0
    yesterday_up = prev_body > 0
    yesterday_down = prev_body < 0

    # Bullish engulfing: today up, yesterday down, today engulfs
    bullish = (today_engulfs & today_up & yesterday_down).astype(int)
    features["bullish_engulfing"] = bullish

    # Bearish engulfing: today down, yesterday up, today engulfs
    bearish = (today_engulfs & today_down & yesterday_up).astype(int)
    features["bearish_engulfing"] = bearish

    # Engulfing score: today body / yesterday body (clipped 0-5)
    # Use small epsilon to avoid division by zero
    eps = 1e-8
    score = body_size / (prev_body_size + eps)
    score = score.clip(0, 5)
    features["engulfing_score"] = score.fillna(0)

    # Consecutive engulfing count
    any_engulfing = (bullish == 1) | (bearish == 1)
    consecutive = pd.Series(0, index=close.index, dtype=int)
    count = 0
    for i in range(len(close)):
        if any_engulfing.iloc[i]:
            count += 1
            consecutive.iloc[i] = count
        else:
            count = 0
            consecutive.iloc[i] = 0
    features["consecutive_engulfing_count"] = consecutive

    return features


def _compute_9a_wick_rejection(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Wick Rejection features.

    Features:
    - hammer_indicator: 1 if lower_wick >= 2×body AND upper_wick <= 0.3×body
    - shooting_star_indicator: 1 if upper_wick >= 2×body AND lower_wick <= 0.3×body
    - hammer_score: lower_wick / (body + ε), clipped [0, 10]
    - shooting_star_score: upper_wick / (body + ε), clipped [0, 10]
    - wick_rejection_score: max(hammer, shooting_star) × direction_sign

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 5 Wick Rejection features
    """
    features = {}
    eps = 1e-8

    # Body and wick calculations
    body_low = np.minimum(open_price, close)
    body_high = np.maximum(open_price, close)
    body_size = body_high - body_low

    upper_wick = high - body_high
    lower_wick = body_low - low

    # Hammer: long lower wick, short upper wick
    hammer_cond = (lower_wick >= 2 * body_size) & (upper_wick <= 0.3 * body_size)
    features["hammer_indicator"] = hammer_cond.astype(int)

    # Shooting star: long upper wick, short lower wick
    shooting_star_cond = (upper_wick >= 2 * body_size) & (lower_wick <= 0.3 * body_size)
    features["shooting_star_indicator"] = shooting_star_cond.astype(int)

    # Hammer score: lower_wick / body (clipped 0-10)
    hammer_score = lower_wick / (body_size + eps)
    hammer_score = hammer_score.clip(0, 10)
    features["hammer_score"] = hammer_score.fillna(0)

    # Shooting star score: upper_wick / body (clipped 0-10)
    shooting_star_score = upper_wick / (body_size + eps)
    shooting_star_score = shooting_star_score.clip(0, 10)
    features["shooting_star_score"] = shooting_star_score.fillna(0)

    # Wick rejection score: max(hammer, shooting_star) × direction
    # Positive for hammer (bullish rejection), negative for shooting star (bearish rejection)
    max_wick_score = np.maximum(hammer_score, shooting_star_score)
    # Direction: +1 if hammer dominates, -1 if shooting star dominates
    direction = np.where(
        hammer_score > shooting_star_score,
        1,
        np.where(shooting_star_score > hammer_score, -1, 0)
    )
    wick_rejection = max_wick_score * direction
    features["wick_rejection_score"] = pd.Series(wick_rejection, index=close.index).fillna(0)

    return features


def _compute_9a_gaps(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Gap Analysis features.

    Features:
    - gap_size_pct: (Open - prev_Close) / prev_Close × 100 (signed)
    - gap_direction: +1 if gap > 0.1%, -1 if < -0.1%, else 0
    - gap_filled_today: 1 if gap was partially/fully filled during day
    - gap_fill_pct: % of gap filled today (0-100)
    - significant_gap: 1 if abs(gap_size_pct) > 0.5%

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 5 Gap Analysis features
    """
    features = {}
    eps = 1e-8

    prev_close = close.shift(1)

    # Gap size as percentage
    gap_size = (open_price - prev_close) / (prev_close + eps) * 100
    features["gap_size_pct"] = gap_size.fillna(0)

    # Gap direction: +1 if > 0.1%, -1 if < -0.1%, else 0
    gap_direction = pd.Series(0, index=close.index, dtype=int)
    gap_direction = gap_direction.where(gap_size <= 0.1, 1)
    gap_direction = gap_direction.where(gap_size >= -0.1, -1)
    features["gap_direction"] = gap_direction.fillna(0).astype(int)

    # Gap filled today: did price reach prev_close during the day?
    # For gap up (open > prev_close): filled if low <= prev_close
    # For gap down (open < prev_close): filled if high >= prev_close
    gap_up = open_price > prev_close
    gap_down = open_price < prev_close
    filled = (gap_up & (low <= prev_close)) | (gap_down & (high >= prev_close))
    features["gap_filled_today"] = filled.fillna(False).astype(int)

    # Gap fill percentage (0-100)
    # For gap up: how much of the gap from prev_close to open was retraced
    # For gap down: similar but in opposite direction
    gap_amount = open_price - prev_close
    fill_amount = pd.Series(0.0, index=close.index)

    # Gap up: fill amount = open - low (how far price came down toward prev_close)
    # But capped at the gap size
    fill_amount = fill_amount.where(
        ~gap_up,
        np.minimum(open_price - low, gap_amount)
    )
    # Gap down: fill amount = high - open (how far price came up toward prev_close)
    # But capped at abs(gap)
    fill_amount = fill_amount.where(
        ~gap_down,
        np.minimum(high - open_price, np.abs(gap_amount))
    )

    # Convert to percentage (0-100)
    gap_fill_pct = np.abs(fill_amount) / (np.abs(gap_amount) + eps) * 100
    gap_fill_pct = gap_fill_pct.clip(0, 100)
    features["gap_fill_pct"] = gap_fill_pct.fillna(0)

    # Significant gap: abs(gap_size_pct) > 0.5%
    significant = (np.abs(gap_size) > 0.5).astype(int)
    features["significant_gap"] = significant.fillna(0).astype(int)

    return features


def _compute_9a_inside_outside(
    high: pd.Series, low: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Inside/Outside Day features.

    Features:
    - inside_day: 1 if H <= prev_H AND L >= prev_L
    - outside_day: 1 if H > prev_H AND L < prev_L
    - consecutive_inside_days: Running count of consecutive inside days
    - consecutive_outside_days: Running count of consecutive outside days

    Args:
        high: High price series
        low: Low price series

    Returns:
        Dict with 4 Inside/Outside features
    """
    features = {}

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    # Inside day: today's range is within yesterday's range
    inside = (high <= prev_high) & (low >= prev_low)
    features["inside_day"] = inside.fillna(False).astype(int)

    # Outside day: today's range engulfs yesterday's range
    outside = (high > prev_high) & (low < prev_low)
    features["outside_day"] = outside.fillna(False).astype(int)

    # Consecutive inside days
    consecutive_inside = pd.Series(0, index=high.index, dtype=int)
    count = 0
    for i in range(len(high)):
        if inside.iloc[i] if not pd.isna(inside.iloc[i]) else False:
            count += 1
            consecutive_inside.iloc[i] = count
        else:
            count = 0
            consecutive_inside.iloc[i] = 0
    features["consecutive_inside_days"] = consecutive_inside

    # Consecutive outside days
    consecutive_outside = pd.Series(0, index=high.index, dtype=int)
    count = 0
    for i in range(len(high)):
        if outside.iloc[i] if not pd.isna(outside.iloc[i]) else False:
            count += 1
            consecutive_outside.iloc[i] = count
        else:
            count = 0
            consecutive_outside.iloc[i] = 0
    features["consecutive_outside_days"] = consecutive_outside

    return features


def _compute_9a_range_extremes(
    high: pd.Series, low: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Range Extremes features.

    Features:
    - narrow_range_day: 1 if range < 0.5 × 10d avg range
    - wide_range_day: 1 if range > 2.0 × 10d avg range
    - narrow_range_score: 10d_avg_range / (today_range + ε), clipped [0, 10]
    - consecutive_narrow_days: Running count of consecutive narrow range days

    Args:
        high: High price series
        low: Low price series

    Returns:
        Dict with 4 Range Extremes features
    """
    features = {}
    eps = 1e-8

    # Today's range
    today_range = high - low

    # 10-day average range
    avg_range_10d = today_range.rolling(window=10, min_periods=10).mean()

    # Narrow range day: range < 0.5 × avg
    narrow = today_range < (0.5 * avg_range_10d)
    features["narrow_range_day"] = narrow.fillna(False).astype(int)

    # Wide range day: range > 2.0 × avg
    wide = today_range > (2.0 * avg_range_10d)
    features["wide_range_day"] = wide.fillna(False).astype(int)

    # Narrow range score: avg_range / today_range (higher = narrower)
    narrow_score = avg_range_10d / (today_range + eps)
    narrow_score = narrow_score.clip(0, 10)
    features["narrow_range_score"] = narrow_score.fillna(0)

    # Consecutive narrow range days
    consecutive_narrow = pd.Series(0, index=high.index, dtype=int)
    count = 0
    for i in range(len(high)):
        if narrow.iloc[i] if not pd.isna(narrow.iloc[i]) else False:
            count += 1
            consecutive_narrow.iloc[i] = count
        else:
            count = 0
            consecutive_narrow.iloc[i] = 0
    features["consecutive_narrow_days"] = consecutive_narrow

    return features


def _compute_9a_trend_days(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Trend Day features.

    Features:
    - trend_day_indicator: 1 if body_to_range > 0.7 AND body > 0.5% of open
    - trend_day_direction: +1 up, -1 down, 0 otherwise
    - consecutive_trend_days: Running count in same direction

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 3 Trend Day features
    """
    features = {}
    eps = 1e-8

    # Body and range calculations
    body = close - open_price
    body_size = np.abs(body)
    day_range = high - low

    # Body to range ratio
    body_to_range = body_size / (day_range + eps)

    # Body as percentage of open
    body_pct = body_size / (open_price + eps) * 100

    # Trend day: body_to_range > 0.7 AND body > 0.5% of open
    trend_day = (body_to_range > 0.7) & (body_pct > 0.5)
    features["trend_day_indicator"] = trend_day.fillna(False).astype(int)

    # Trend day direction: +1 if up trend day, -1 if down trend day, 0 otherwise
    direction = pd.Series(0, index=close.index, dtype=int)
    direction = direction.where(~(trend_day & (body > 0)), 1)
    direction = direction.where(~(trend_day & (body < 0)), -1)
    features["trend_day_direction"] = direction.fillna(0).astype(int)

    # Consecutive trend days in same direction
    consecutive = pd.Series(0, index=close.index, dtype=int)
    count = 0
    prev_direction = 0
    for i in range(len(close)):
        curr_dir = direction.iloc[i]
        if curr_dir != 0 and curr_dir == prev_direction:
            count += 1
            consecutive.iloc[i] = count
        elif curr_dir != 0:
            # New trend day in different direction
            count = 1
            consecutive.iloc[i] = count
            prev_direction = curr_dir
        else:
            count = 0
            consecutive.iloc[i] = 0
            prev_direction = 0
    features["consecutive_trend_days"] = consecutive

    return features


def _compute_chunk_9a(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute all Sub-Chunk 9a features.

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with all 25 Chunk 9a features
    """
    features = {}

    # Engulfing Patterns (4 features)
    features.update(_compute_9a_engulfing(open_price, close))

    # Wick Rejection (5 features)
    features.update(_compute_9a_wick_rejection(open_price, high, low, close))

    # Gap Analysis (5 features)
    features.update(_compute_9a_gaps(open_price, high, low, close))

    # Inside/Outside Days (4 features)
    features.update(_compute_9a_inside_outside(high, low))

    # Range Extremes (4 features)
    features.update(_compute_9a_range_extremes(high, low))

    # Trend Days (3 features)
    features.update(_compute_9a_trend_days(open_price, high, low, close))

    return features


# =============================================================================
# Sub-Chunk 9b computation functions (ranks 371-395)
# =============================================================================


def _compute_9b_doji(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Doji pattern features.

    Features:
    - doji_strict_indicator: 1 if body/range < 0.1 (small body relative to range)
    - doji_score: 1 - (body/range), higher = more doji-like [0,1]
    - doji_type: +1 dragonfly (long lower wick), -1 gravestone (long upper), 0 neutral
    - consecutive_doji_count: Running count of consecutive doji days
    - doji_after_trend: 1 if doji follows 3+ trend days in same direction

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 5 Doji features
    """
    features = {}
    eps = 1e-8

    # Body and range calculations
    body = close - open_price
    body_size = np.abs(body)
    day_range = high - low

    # Body to range ratio
    body_ratio = body_size / (day_range + eps)

    # Doji indicator: body/range < 0.1
    is_doji = (body_ratio < 0.1).fillna(False)
    features["doji_strict_indicator"] = is_doji.astype(int)

    # Doji score: 1 - body_ratio, clipped to [0, 1]
    doji_score = (1 - body_ratio).clip(0, 1)
    features["doji_score"] = doji_score.fillna(0.0)

    # Doji type: dragonfly (+1), gravestone (-1), neutral (0)
    upper_wick = high - np.maximum(open_price, close)
    lower_wick = np.minimum(open_price, close) - low

    # For doji type, compare wick lengths
    # Dragonfly: long lower wick (lower > 2x upper)
    # Gravestone: long upper wick (upper > 2x lower)
    doji_type = pd.Series(0, index=close.index, dtype=int)
    dragonfly = is_doji & (lower_wick > 2 * upper_wick + eps)
    gravestone = is_doji & (upper_wick > 2 * lower_wick + eps)
    doji_type = doji_type.where(~dragonfly, 1)
    doji_type = doji_type.where(~gravestone, -1)
    features["doji_type"] = doji_type.fillna(0).astype(int)

    # Consecutive doji count
    consecutive = pd.Series(0, index=close.index, dtype=int)
    count = 0
    for i in range(len(close)):
        if is_doji.iloc[i]:
            count += 1
            consecutive.iloc[i] = count
        else:
            count = 0
            consecutive.iloc[i] = 0
    features["consecutive_doji_count"] = consecutive

    # Doji after trend: check if 3+ consecutive days in same direction before doji
    # Direction based on close-to-close
    daily_direction = (close > close.shift(1)).astype(int) - (close < close.shift(1)).astype(int)

    # Count consecutive same direction days
    trend_count = pd.Series(0, index=close.index, dtype=int)
    prev_trend_count = pd.Series(0, index=close.index, dtype=int)
    curr_count = 0
    curr_dir = 0
    for i in range(len(close)):
        if i == 0:
            prev_trend_count.iloc[i] = 0
            curr_count = 0
            curr_dir = 0
        else:
            prev_trend_count.iloc[i] = curr_count
            dir_today = daily_direction.iloc[i]
            if dir_today != 0 and dir_today == curr_dir:
                curr_count += 1
            elif dir_today != 0:
                curr_count = 1
                curr_dir = dir_today
            else:
                curr_count = 0
                curr_dir = 0
        trend_count.iloc[i] = curr_count

    # Doji after trend: doji AND previous trend_count >= 3
    doji_after_trend = is_doji & (prev_trend_count >= 3)
    features["doji_after_trend"] = doji_after_trend.fillna(False).astype(int)

    return features


def _compute_9b_marubozu(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Marubozu & Strong Candle features.

    Features:
    - marubozu_indicator: 1 if (upper_wick + lower_wick) / range < 0.1
    - marubozu_direction: +1 bullish (close>open), -1 bearish, 0 if not marubozu
    - marubozu_strength: body / range (0.9+ for marubozu)
    - consecutive_strong_candles: Running count of body/range > 0.7 in same direction

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 4 Marubozu features
    """
    features = {}
    eps = 1e-8

    # Body and range calculations
    body = close - open_price
    body_size = np.abs(body)
    day_range = high - low

    # Wick calculations
    upper_wick = high - np.maximum(open_price, close)
    lower_wick = np.minimum(open_price, close) - low
    total_wick = upper_wick + lower_wick

    # Wick to range ratio
    wick_ratio = total_wick / (day_range + eps)

    # Marubozu indicator: total wick < 10% of range
    is_marubozu = (wick_ratio < 0.1).fillna(False)
    features["marubozu_indicator"] = is_marubozu.astype(int)

    # Marubozu direction
    marubozu_direction = pd.Series(0, index=close.index, dtype=int)
    bullish_marubozu = is_marubozu & (body > 0)
    bearish_marubozu = is_marubozu & (body < 0)
    marubozu_direction = marubozu_direction.where(~bullish_marubozu, 1)
    marubozu_direction = marubozu_direction.where(~bearish_marubozu, -1)
    features["marubozu_direction"] = marubozu_direction.fillna(0).astype(int)

    # Marubozu strength: body / range, clipped to [0, 1]
    marubozu_strength = (body_size / (day_range + eps)).clip(0, 1)
    features["marubozu_strength"] = marubozu_strength.fillna(0.0)

    # Consecutive strong candles (body/range > 0.7 in same direction)
    body_ratio = body_size / (day_range + eps)
    is_strong = body_ratio > 0.7
    direction = np.sign(body).fillna(0).astype(int)

    consecutive = pd.Series(0, index=close.index, dtype=int)
    count = 0
    prev_dir = 0
    for i in range(len(close)):
        curr_strong = is_strong.iloc[i]
        curr_dir = direction.iloc[i]
        if curr_strong and curr_dir != 0 and curr_dir == prev_dir:
            count += 1
            consecutive.iloc[i] = count
        elif curr_strong and curr_dir != 0:
            count = 1
            consecutive.iloc[i] = count
            prev_dir = curr_dir
        else:
            count = 0
            consecutive.iloc[i] = 0
            prev_dir = 0
    features["consecutive_strong_candles"] = consecutive

    return features


def _compute_9b_spinning_top(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Spinning Top & Indecision features.

    Features:
    - spinning_top_indicator: 1 if body/range < 0.3 AND both wicks > body
    - spinning_top_score: (upper_wick + lower_wick) / (body + eps), capped [0, 10]
    - indecision_streak: Running count of spinning_top OR doji days
    - indecision_at_extreme: 1 if spinning_top AND near 20d high/low (within 2%)

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 4 Spinning Top features
    """
    features = {}
    eps = 1e-8

    # Body and range calculations
    body = close - open_price
    body_size = np.abs(body)
    day_range = high - low

    # Wick calculations
    upper_wick = high - np.maximum(open_price, close)
    lower_wick = np.minimum(open_price, close) - low

    # Body to range ratio
    body_ratio = body_size / (day_range + eps)

    # Spinning top: small body (< 0.3) AND both wicks > body
    is_spinning_top = (body_ratio < 0.3) & (upper_wick > body_size) & (lower_wick > body_size)
    is_spinning_top = is_spinning_top.fillna(False)
    features["spinning_top_indicator"] = is_spinning_top.astype(int)

    # Spinning top score: total wicks / body, capped at 10
    spinning_score = (upper_wick + lower_wick) / (body_size + eps)
    spinning_score = spinning_score.clip(0, 10)
    features["spinning_top_score"] = spinning_score.fillna(0.0)

    # Doji indicator (needed for indecision streak)
    is_doji = (body_ratio < 0.1).fillna(False)

    # Indecision streak: consecutive spinning top or doji
    is_indecision = is_spinning_top | is_doji
    consecutive = pd.Series(0, index=close.index, dtype=int)
    count = 0
    for i in range(len(close)):
        if is_indecision.iloc[i]:
            count += 1
            consecutive.iloc[i] = count
        else:
            count = 0
            consecutive.iloc[i] = 0
    features["indecision_streak"] = consecutive

    # Indecision at extreme: spinning top near 20d high or low (within 2%)
    high_20d = high.rolling(window=20, min_periods=1).max()
    low_20d = low.rolling(window=20, min_periods=1).min()

    near_high = (high >= high_20d * 0.98)
    near_low = (low <= low_20d * 1.02)

    at_extreme = is_spinning_top & (near_high | near_low)
    features["indecision_at_extreme"] = at_extreme.fillna(False).astype(int)

    return features


def _compute_9b_reversal_patterns(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Multi-Candle Reversal Pattern features.

    Features:
    - morning_star_indicator: 1 if: down candle, small body, up candle close > mid of first
    - evening_star_indicator: 1 if: up candle, small body, down candle close < mid of first
    - three_white_soldiers: 1 if 3 consecutive up days, each closing near high
    - three_black_crows: 1 if 3 consecutive down days, each closing near low
    - harami_indicator: +1 bullish (down→up contained), -1 bearish (up→down contained), 0 none

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 5 Reversal Pattern features
    """
    features = {}
    eps = 1e-8

    # Body calculations
    body = close - open_price
    body_size = np.abs(body)
    day_range = high - low

    # Body ratio
    body_ratio = body_size / (day_range + eps)

    # Morning Star: [down candle] [small body] [up candle closes above mid of day 1]
    # Day -2: bearish (close < open)
    # Day -1: small body (body_ratio < 0.3)
    # Day 0: bullish AND close > midpoint of day -2
    day_m2_bearish = close.shift(2) < open_price.shift(2)
    day_m1_small = body_ratio.shift(1) < 0.3
    day_0_bullish = close > open_price
    day_m2_mid = (open_price.shift(2) + close.shift(2)) / 2
    day_0_above_mid = close > day_m2_mid

    morning_star = day_m2_bearish & day_m1_small & day_0_bullish & day_0_above_mid
    features["morning_star_indicator"] = morning_star.fillna(False).astype(int)

    # Evening Star: [up candle] [small body] [down candle closes below mid of day 1]
    day_m2_bullish = close.shift(2) > open_price.shift(2)
    day_0_bearish = close < open_price
    day_0_below_mid = close < day_m2_mid

    evening_star = day_m2_bullish & day_m1_small & day_0_bearish & day_0_below_mid
    features["evening_star_indicator"] = evening_star.fillna(False).astype(int)

    # Three White Soldiers: 3 up days, each closing near high (upper wick < 30% of range)
    upper_wick = high - np.maximum(open_price, close)
    upper_wick_ratio = upper_wick / (day_range + eps)

    is_up_day = close > open_price
    closes_near_high = upper_wick_ratio < 0.3

    three_white = (
        is_up_day & is_up_day.shift(1) & is_up_day.shift(2) &
        closes_near_high & closes_near_high.shift(1) & closes_near_high.shift(2)
    )
    features["three_white_soldiers"] = three_white.fillna(False).astype(int)

    # Three Black Crows: 3 down days, each closing near low (lower wick < 30% of range)
    lower_wick = np.minimum(open_price, close) - low
    lower_wick_ratio = lower_wick / (day_range + eps)

    is_down_day = close < open_price
    closes_near_low = lower_wick_ratio < 0.3

    three_black = (
        is_down_day & is_down_day.shift(1) & is_down_day.shift(2) &
        closes_near_low & closes_near_low.shift(1) & closes_near_low.shift(2)
    )
    features["three_black_crows"] = three_black.fillna(False).astype(int)

    # Harami: today's body contained within yesterday's body
    # Bullish harami: yesterday down, today up, today contained
    # Bearish harami: yesterday up, today down, today contained
    today_open = open_price
    today_close = close
    yest_open = open_price.shift(1)
    yest_close = close.shift(1)

    yest_high_body = np.maximum(yest_open, yest_close)
    yest_low_body = np.minimum(yest_open, yest_close)
    today_high_body = np.maximum(today_open, today_close)
    today_low_body = np.minimum(today_open, today_close)

    # Today's body contained within yesterday's body
    body_contained = (today_high_body <= yest_high_body) & (today_low_body >= yest_low_body)

    yest_bearish = yest_close < yest_open
    yest_bullish = yest_close > yest_open
    today_bullish = today_close > today_open
    today_bearish = today_close < today_open

    bullish_harami = body_contained & yest_bearish & today_bullish
    bearish_harami = body_contained & yest_bullish & today_bearish

    harami = pd.Series(0, index=close.index, dtype=int)
    harami = harami.where(~bullish_harami, 1)
    harami = harami.where(~bearish_harami, -1)
    features["harami_indicator"] = harami.fillna(0).astype(int)

    return features


def _compute_9b_continuation_patterns(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Multi-Candle Continuation Pattern features.

    Features:
    - piercing_line: 1 if down day then up day closing above mid of prior
    - dark_cloud_cover: 1 if up day then down day closing below mid of prior
    - tweezer_bottom: 1 if two lows match (within 0.1%) after downtrend
    - tweezer_top: 1 if two highs match (within 0.1%) after uptrend

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with 4 Continuation Pattern features
    """
    features = {}
    eps = 1e-8

    # Basic direction
    is_up_day = close > open_price
    is_down_day = close < open_price

    # Yesterday's values
    yest_open = open_price.shift(1)
    yest_close = close.shift(1)
    yest_high = high.shift(1)
    yest_low = low.shift(1)

    yest_mid = (yest_open + yest_close) / 2

    # Piercing Line: yesterday down, today up, today closes above yesterday's midpoint
    # Also: today opens below yesterday's low (gap down) - classic definition
    yest_down = yest_close < yest_open
    today_up = close > open_price
    today_opens_low = open_price < yest_low
    today_closes_above_mid = close > yest_mid
    today_closes_below_yest_open = close < yest_open  # Doesn't fully engulf

    piercing = yest_down & today_up & today_closes_above_mid
    features["piercing_line"] = piercing.fillna(False).astype(int)

    # Dark Cloud Cover: yesterday up, today down, today closes below yesterday's midpoint
    yest_up = yest_close > yest_open
    today_down = close < open_price
    today_closes_below_mid = close < yest_mid

    dark_cloud = yest_up & today_down & today_closes_below_mid
    features["dark_cloud_cover"] = dark_cloud.fillna(False).astype(int)

    # Tweezer Bottom: two lows match (within 0.1%) - typically after downtrend
    # Check 3-day downtrend before pattern
    low_match = np.abs(low - yest_low) / (yest_low + eps) < 0.001

    # Simple downtrend: 3 lower closes before
    downtrend = (close.shift(2) > close.shift(1)) & (close.shift(3) > close.shift(2))

    tweezer_bottom = low_match & downtrend
    features["tweezer_bottom"] = tweezer_bottom.fillna(False).astype(int)

    # Tweezer Top: two highs match (within 0.1%) - typically after uptrend
    high_match = np.abs(high - yest_high) / (yest_high + eps) < 0.001

    # Simple uptrend: 3 higher closes before
    uptrend = (close.shift(2) < close.shift(1)) & (close.shift(3) < close.shift(2))

    tweezer_top = high_match & uptrend
    features["tweezer_top"] = tweezer_top.fillna(False).astype(int)

    return features


def _compute_9b_pattern_context(
    reversal_features: Mapping[str, pd.Series],
    close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute Pattern Context features.

    Features:
    - reversal_pattern_count_5d: Count of reversal patterns in last 5 days
    - pattern_alignment_score: +1 if bullish patterns dominate, -1 if bearish, 0 mixed
    - pattern_cluster_indicator: 1 if 2+ patterns in last 3 days

    Args:
        reversal_features: Dict with reversal pattern features from 9b
        close: Close price series (for index)

    Returns:
        Dict with 3 Pattern Context features
    """
    features = {}

    # Get individual pattern indicators
    morning_star = reversal_features.get("morning_star_indicator", pd.Series(0, index=close.index))
    evening_star = reversal_features.get("evening_star_indicator", pd.Series(0, index=close.index))
    three_white = reversal_features.get("three_white_soldiers", pd.Series(0, index=close.index))
    three_black = reversal_features.get("three_black_crows", pd.Series(0, index=close.index))
    harami = reversal_features.get("harami_indicator", pd.Series(0, index=close.index))

    # Bullish patterns: morning_star, three_white_soldiers, bullish harami
    bullish_patterns = (
        morning_star.astype(int) +
        three_white.astype(int) +
        (harami == 1).astype(int)
    )

    # Bearish patterns: evening_star, three_black_crows, bearish harami
    bearish_patterns = (
        evening_star.astype(int) +
        three_black.astype(int) +
        (harami == -1).astype(int)
    )

    # All reversal patterns count
    all_patterns = bullish_patterns + bearish_patterns

    # Reversal pattern count in 5-day window
    pattern_count_5d = all_patterns.rolling(window=5, min_periods=1).sum()
    features["reversal_pattern_count_5d"] = pattern_count_5d.fillna(0).astype(int)

    # Pattern alignment score: compare bullish vs bearish in 5-day window
    bullish_5d = bullish_patterns.rolling(window=5, min_periods=1).sum()
    bearish_5d = bearish_patterns.rolling(window=5, min_periods=1).sum()

    alignment = pd.Series(0, index=close.index, dtype=int)
    alignment = alignment.where(~(bullish_5d > bearish_5d), 1)
    alignment = alignment.where(~(bearish_5d > bullish_5d), -1)
    features["pattern_alignment_score"] = alignment.fillna(0).astype(int)

    # Pattern cluster: 2+ patterns in last 3 days
    pattern_count_3d = all_patterns.rolling(window=3, min_periods=1).sum()
    cluster = (pattern_count_3d >= 2).astype(int)
    features["pattern_cluster_indicator"] = cluster.fillna(0).astype(int)

    return features


def _compute_chunk_9b(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute all Sub-Chunk 9b features.

    Args:
        open_price: Open price series
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with all 25 Chunk 9b features
    """
    features = {}

    # Doji Patterns (5 features)
    features.update(_compute_9b_doji(open_price, high, low, close))

    # Marubozu & Strong Candles (4 features)
    features.update(_compute_9b_marubozu(open_price, high, low, close))

    # Spinning Top & Indecision (4 features)
    features.update(_compute_9b_spinning_top(open_price, high, low, close))

    # Multi-Candle Reversal Patterns (5 features)
    reversal_features = _compute_9b_reversal_patterns(open_price, high, low, close)
    features.update(reversal_features)

    # Multi-Candle Continuation Patterns (4 features)
    features.update(_compute_9b_continuation_patterns(open_price, high, low, close))

    # Pattern Context (3 features) - depends on reversal features
    features.update(_compute_9b_pattern_context(reversal_features, close))

    return features


# =============================================================================
# Sub-Chunk 10a computation functions (ranks 396-420)
# =============================================================================

WEEKLY_FREQ = "W-MON"


def _resample_to_weekly_a500(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV data to weekly for MTF features.

    Args:
        df: DataFrame with Date, Open, High, Low, Close, Volume

    Returns:
        Weekly aggregated DataFrame
    """
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])
    df_copy = df_copy.set_index("Date")

    weekly = (
        df_copy
        .resample(WEEKLY_FREQ, label="left", closed="left")
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
    )
    weekly = weekly[weekly["Close"].notnull()]
    return weekly


def _compute_10a_weekly_ma_features(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute weekly MA features for MTF analysis.

    Features:
    - weekly_ma_slope: Rate of change in weekly SMA (5-week change / 5)
    - weekly_ma_slope_acceleration: Change in weekly MA slope
    - price_pct_from_weekly_ma: Daily close vs weekly MA (signed %)

    Args:
        df: Daily DataFrame with OHLCV

    Returns:
        Dict with weekly MA features (daily-aligned)
    """
    features = {}

    # Resample to weekly
    weekly = _resample_to_weekly_a500(df)
    if len(weekly) < 20:
        # Not enough weekly data
        n = len(df)
        features["weekly_ma_slope"] = pd.Series(np.nan, index=range(n))
        features["weekly_ma_slope_acceleration"] = pd.Series(np.nan, index=range(n))
        features["price_pct_from_weekly_ma"] = pd.Series(np.nan, index=range(n))
        return features

    # Compute 20-week SMA on weekly data
    weekly_sma = talib.SMA(weekly["Close"].values, timeperiod=20)
    weekly_sma_series = pd.Series(weekly_sma, index=weekly.index)

    # Weekly MA slope: 5-week change as fraction
    weekly_ma_slope = (weekly_sma_series - weekly_sma_series.shift(5)) / weekly_sma_series.shift(5)

    # Weekly MA slope acceleration: change in slope
    weekly_ma_slope_accel = weekly_ma_slope - weekly_ma_slope.shift(1)

    # Shift by 1 week to avoid lookahead bias
    weekly_sma_series = weekly_sma_series.shift(1)
    weekly_ma_slope = weekly_ma_slope.shift(1)
    weekly_ma_slope_accel = weekly_ma_slope_accel.shift(1)

    # Align weekly features to daily index
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])
    daily_index = df_copy["Date"]

    # Forward-fill weekly data to daily
    weekly_sma_daily = weekly_sma_series.reindex(daily_index).ffill()
    weekly_slope_daily = weekly_ma_slope.reindex(daily_index).ffill()
    weekly_accel_daily = weekly_ma_slope_accel.reindex(daily_index).ffill()

    # Reset index for output
    features["weekly_ma_slope"] = weekly_slope_daily.reset_index(drop=True)
    features["weekly_ma_slope_acceleration"] = weekly_accel_daily.reset_index(drop=True)

    # Price % from weekly MA (daily close vs weekly MA)
    close = df_copy["Close"].values
    weekly_ma_values = weekly_sma_daily.values
    price_pct_from_weekly_ma = (close - weekly_ma_values) / weekly_ma_values
    features["price_pct_from_weekly_ma"] = pd.Series(price_pct_from_weekly_ma)

    return features


def _compute_10a_weekly_rsi_features(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute weekly RSI features for MTF analysis.

    Features:
    - weekly_rsi_slope: 5-week change in weekly RSI
    - weekly_rsi_slope_acceleration: Change in weekly RSI slope

    Args:
        df: Daily DataFrame with OHLCV

    Returns:
        Dict with weekly RSI features (daily-aligned)
    """
    features = {}

    # Resample to weekly
    weekly = _resample_to_weekly_a500(df)
    if len(weekly) < 20:
        n = len(df)
        features["weekly_rsi_slope"] = pd.Series(np.nan, index=range(n))
        features["weekly_rsi_slope_acceleration"] = pd.Series(np.nan, index=range(n))
        return features

    # Compute weekly RSI
    weekly_rsi = talib.RSI(weekly["Close"].values, timeperiod=14)
    weekly_rsi_series = pd.Series(weekly_rsi, index=weekly.index)

    # Weekly RSI slope: 5-week change
    weekly_rsi_slope = weekly_rsi_series - weekly_rsi_series.shift(5)

    # Weekly RSI slope acceleration: change in slope
    weekly_rsi_slope_accel = weekly_rsi_slope - weekly_rsi_slope.shift(1)

    # Shift by 1 week to avoid lookahead
    weekly_rsi_slope = weekly_rsi_slope.shift(1)
    weekly_rsi_slope_accel = weekly_rsi_slope_accel.shift(1)

    # Align to daily index
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])
    daily_index = df_copy["Date"]

    weekly_rsi_slope_daily = weekly_rsi_slope.reindex(daily_index).ffill()
    weekly_rsi_accel_daily = weekly_rsi_slope_accel.reindex(daily_index).ffill()

    features["weekly_rsi_slope"] = weekly_rsi_slope_daily.reset_index(drop=True)
    features["weekly_rsi_slope_acceleration"] = weekly_rsi_accel_daily.reset_index(drop=True)

    return features


def _compute_10a_weekly_bb_features(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute weekly Bollinger Band features for MTF analysis.

    Features:
    - weekly_bb_position: Position in weekly BB (0 = lower, 1 = upper)
    - weekly_bb_width: Weekly BB width as fraction of price
    - weekly_bb_width_slope: Change in weekly BB width

    Args:
        df: Daily DataFrame with OHLCV

    Returns:
        Dict with weekly BB features (daily-aligned)
    """
    features = {}

    # Resample to weekly
    weekly = _resample_to_weekly_a500(df)
    if len(weekly) < 20:
        n = len(df)
        features["weekly_bb_position"] = pd.Series(np.nan, index=range(n))
        features["weekly_bb_width"] = pd.Series(np.nan, index=range(n))
        features["weekly_bb_width_slope"] = pd.Series(np.nan, index=range(n))
        return features

    # Compute weekly Bollinger Bands
    upper, middle, lower = talib.BBANDS(
        weekly["Close"].values, timeperiod=20, nbdevup=2, nbdevdn=2
    )
    upper_series = pd.Series(upper, index=weekly.index)
    middle_series = pd.Series(middle, index=weekly.index)
    lower_series = pd.Series(lower, index=weekly.index)

    # Weekly BB position: (close - lower) / (upper - lower)
    weekly_close = weekly["Close"]
    bb_range = upper_series - lower_series
    bb_position = (weekly_close - lower_series) / bb_range.replace(0, np.nan)

    # Weekly BB width as fraction of price
    bb_width = bb_range / middle_series

    # BB width slope: 5-week change
    bb_width_slope = bb_width - bb_width.shift(5)

    # Shift by 1 week to avoid lookahead
    bb_position = bb_position.shift(1)
    bb_width = bb_width.shift(1)
    bb_width_slope = bb_width_slope.shift(1)

    # Align to daily index
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])
    daily_index = df_copy["Date"]

    bb_position_daily = bb_position.reindex(daily_index).ffill()
    bb_width_daily = bb_width.reindex(daily_index).ffill()
    bb_width_slope_daily = bb_width_slope.reindex(daily_index).ffill()

    features["weekly_bb_position"] = bb_position_daily.reset_index(drop=True)
    features["weekly_bb_width"] = bb_width_daily.reset_index(drop=True)
    features["weekly_bb_width_slope"] = bb_width_slope_daily.reset_index(drop=True)

    return features


def _compute_10a_alignment_features(
    df: pd.DataFrame, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute daily-weekly alignment features.

    Features:
    - trend_alignment_daily_weekly: +1 both up, -1 both down, 0 mixed
    - rsi_alignment_daily_weekly: Daily RSI - Weekly RSI (divergence)
    - vol_alignment_daily_weekly: Daily vol percentile - Weekly vol percentile

    Args:
        df: Daily DataFrame with OHLCV
        close: Daily close prices

    Returns:
        Dict with alignment features
    """
    features = {}
    n = len(close)

    # Resample to weekly
    weekly = _resample_to_weekly_a500(df)

    if len(weekly) < 20:
        features["trend_alignment_daily_weekly"] = pd.Series(np.nan, index=range(n))
        features["rsi_alignment_daily_weekly"] = pd.Series(np.nan, index=range(n))
        features["vol_alignment_daily_weekly"] = pd.Series(np.nan, index=range(n))
        return features

    # === Trend Alignment ===
    # Daily trend: 20-day SMA slope
    daily_sma_20 = talib.SMA(close.values, timeperiod=20)
    daily_trend = np.sign(daily_sma_20 - np.roll(daily_sma_20, 5))
    daily_trend[:5] = np.nan

    # Weekly trend: 4-week SMA slope
    weekly_sma_4 = talib.SMA(weekly["Close"].values, timeperiod=4)
    weekly_trend = np.sign(weekly_sma_4 - np.roll(weekly_sma_4, 1))
    weekly_trend[:1] = np.nan
    weekly_trend_series = pd.Series(weekly_trend, index=weekly.index).shift(1)

    # Align weekly trend to daily
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])
    daily_index = df_copy["Date"]
    weekly_trend_daily = weekly_trend_series.reindex(daily_index).ffill().values

    # Trend alignment: +1 if both up, -1 if both down, 0 if mixed
    trend_alignment = np.where(
        (daily_trend > 0) & (weekly_trend_daily > 0), 1,
        np.where((daily_trend < 0) & (weekly_trend_daily < 0), -1, 0)
    )
    features["trend_alignment_daily_weekly"] = pd.Series(trend_alignment)

    # === RSI Alignment ===
    daily_rsi = talib.RSI(close.values, timeperiod=14)
    weekly_rsi = talib.RSI(weekly["Close"].values, timeperiod=14)
    weekly_rsi_series = pd.Series(weekly_rsi, index=weekly.index).shift(1)
    weekly_rsi_daily = weekly_rsi_series.reindex(daily_index).ffill().values

    rsi_alignment = daily_rsi - weekly_rsi_daily
    features["rsi_alignment_daily_weekly"] = pd.Series(rsi_alignment)

    # === Volatility Alignment ===
    # Daily volatility: ATR% percentile over 60 days
    daily_atr = talib.ATR(df_copy["High"].values, df_copy["Low"].values, close.values, timeperiod=14)
    daily_atr_pct = daily_atr / close.values
    daily_vol_pctile = pd.Series(daily_atr_pct).rolling(60).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5,
        raw=False
    ).values

    # Weekly volatility: ATR% percentile over 12 weeks
    weekly_atr = talib.ATR(weekly["High"].values, weekly["Low"].values, weekly["Close"].values, timeperiod=4)
    weekly_atr_pct = weekly_atr / weekly["Close"].values
    weekly_vol_pctile = pd.Series(weekly_atr_pct).rolling(12).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5,
        raw=False
    )
    weekly_vol_pctile_series = pd.Series(weekly_vol_pctile.values, index=weekly.index).shift(1)
    weekly_vol_daily = weekly_vol_pctile_series.reindex(daily_index).ffill().values

    vol_alignment = daily_vol_pctile - weekly_vol_daily
    features["vol_alignment_daily_weekly"] = pd.Series(vol_alignment)

    return features


def _sample_entropy(series: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Compute sample entropy of a time series.

    Sample entropy measures complexity/irregularity without counting self-matches.
    Lower values = more regular, higher values = more complex/random.

    Args:
        series: 1D array of values
        m: Embedding dimension (default 2)
        r: Tolerance as fraction of std (default 0.2)

    Returns:
        Sample entropy value (non-negative)
    """
    n = len(series)
    if n < m + 1:
        return np.nan

    # Normalize tolerance by std
    std = np.std(series)
    if std == 0:
        return 0.0
    tolerance = r * std

    # Count template matches for dimension m and m+1
    def count_matches(dim: int) -> int:
        count = 0
        for i in range(n - dim):
            template_i = series[i:i + dim]
            for j in range(i + 1, n - dim):
                template_j = series[j:j + dim]
                if np.max(np.abs(template_i - template_j)) < tolerance:
                    count += 1
        return count

    b = count_matches(m)
    a = count_matches(m + 1)

    if b == 0 or a == 0:
        return 0.0

    return -np.log(a / b)


def _hurst_exponent_rs(series: np.ndarray) -> float:
    """Compute Hurst exponent using R/S (Rescaled Range) method.

    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending

    Args:
        series: 1D array of values

    Returns:
        Hurst exponent in [0, 1]
    """
    n = len(series)
    if n < 10:
        return np.nan

    # Use log-spaced subset sizes
    max_k = n // 2
    min_k = 10
    if max_k <= min_k:
        return 0.5

    sizes = np.unique(np.logspace(np.log10(min_k), np.log10(max_k), 10).astype(int))
    rs_values = []
    valid_sizes = []

    for size in sizes:
        if size > n:
            continue
        # Divide series into subseries of this size
        num_subseries = n // size
        if num_subseries < 1:
            continue

        rs_list = []
        for i in range(num_subseries):
            subseries = series[i * size:(i + 1) * size]
            mean_val = np.mean(subseries)
            deviations = subseries - mean_val
            cumsum = np.cumsum(deviations)
            r = np.max(cumsum) - np.min(cumsum)
            s = np.std(subseries, ddof=1)
            if s > 0:
                rs_list.append(r / s)

        if len(rs_list) > 0:
            rs_values.append(np.mean(rs_list))
            valid_sizes.append(size)

    if len(valid_sizes) < 2:
        return 0.5

    # Linear regression of log(R/S) vs log(n)
    log_sizes = np.log(valid_sizes)
    log_rs = np.log(rs_values)

    # Simple linear regression
    slope, _ = np.polyfit(log_sizes, log_rs, 1)

    # Clip to valid range
    return float(np.clip(slope, 0, 1))


def _fractal_dimension_higuchi(series: np.ndarray, k_max: int = 10) -> float:
    """Compute fractal dimension using Higuchi's method.

    For 1D time series: FD in [1, 2]
    FD = 1: Smooth line
    FD = 2: Space-filling curve

    Args:
        series: 1D array of values
        k_max: Maximum interval (default 10)

    Returns:
        Fractal dimension in [1, 2]
    """
    n = len(series)
    if n < k_max * 2:
        return np.nan

    lk_values = []
    k_values = []

    for k in range(1, k_max + 1):
        lm_sum = 0
        for m in range(1, k + 1):
            # Compute length for this (k, m) pair
            indices = np.arange(m - 1, n, k)
            if len(indices) < 2:
                continue
            subseries = series[indices]
            length = np.sum(np.abs(np.diff(subseries))) * (n - 1) / (k * (len(indices) - 1) * k)
            lm_sum += length
        if lm_sum > 0:
            lk = lm_sum / k
            lk_values.append(lk)
            k_values.append(k)

    if len(k_values) < 2:
        return 1.5

    # Linear regression of log(L(k)) vs log(1/k)
    log_k_inv = -np.log(k_values)
    log_lk = np.log(lk_values)

    slope, _ = np.polyfit(log_k_inv, log_lk, 1)

    # Clip to valid range [1, 2]
    return float(np.clip(slope, 1, 2))


def _compute_10a_entropy_features(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute extended entropy features.

    Features:
    - permutation_entropy_slope: 5-day change in permutation entropy (order 4)
    - permutation_entropy_acceleration: Change in entropy slope
    - sample_entropy_20d: Sample entropy over 20-day window
    - sample_entropy_slope: 5-day change in sample entropy
    - sample_entropy_acceleration: Change in sample entropy slope
    - entropy_percentile_60d: Historical percentile of permutation entropy
    - entropy_vol_ratio: Entropy / normalized ATR%
    - entropy_regime_score: Continuous -1 (low) to +1 (high) entropy regime

    Args:
        close: Close price series

    Returns:
        Dict with entropy features
    """
    features = {}
    n = len(close)
    close_arr = close.values

    # Import permutation entropy from tier_a200
    from src.features.tier_a200 import _permutation_entropy

    # Compute rolling permutation entropy (order 4, 20-day window)
    perm_entropy = pd.Series(np.nan, index=range(n))
    for i in range(20, n):
        window = close_arr[i - 20:i]
        perm_entropy.iloc[i] = _permutation_entropy(window, order=4)

    # Permutation entropy slope: 5-day change
    perm_entropy_slope = perm_entropy - perm_entropy.shift(5)
    features["permutation_entropy_slope"] = perm_entropy_slope

    # Permutation entropy acceleration: change in slope
    perm_entropy_accel = perm_entropy_slope - perm_entropy_slope.shift(5)
    features["permutation_entropy_acceleration"] = perm_entropy_accel

    # Sample entropy (20-day rolling)
    sample_ent = pd.Series(np.nan, index=range(n))
    for i in range(20, n):
        window = close_arr[i - 20:i]
        sample_ent.iloc[i] = _sample_entropy(window, m=2, r=0.2)

    features["sample_entropy_20d"] = sample_ent

    # Sample entropy slope: 5-day change
    sample_ent_slope = sample_ent - sample_ent.shift(5)
    features["sample_entropy_slope"] = sample_ent_slope

    # Sample entropy acceleration
    sample_ent_accel = sample_ent_slope - sample_ent_slope.shift(5)
    features["sample_entropy_acceleration"] = sample_ent_accel

    # Entropy percentile: rolling 60-day percentile of permutation entropy
    entropy_pctile = perm_entropy.rolling(60).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5,
        raw=False
    )
    features["entropy_percentile_60d"] = entropy_pctile

    # Entropy / volatility ratio
    # Use returns volatility as denominator
    returns = pd.Series(close_arr).pct_change()
    vol_20d = returns.rolling(20).std()
    # Normalize vol to [0, 1] range for ratio
    vol_normalized = vol_20d / vol_20d.rolling(60).max().replace(0, np.nan)
    entropy_vol_ratio = perm_entropy / vol_normalized.replace(0, np.nan)
    # Clip extreme ratios
    entropy_vol_ratio = entropy_vol_ratio.clip(0, 10)
    features["entropy_vol_ratio"] = entropy_vol_ratio

    # Entropy regime score: -1 (low entropy) to +1 (high entropy)
    # Based on percentile: (percentile - 0.5) * 2
    entropy_regime = (entropy_pctile - 0.5) * 2
    features["entropy_regime_score"] = entropy_regime

    return features


def _compute_10a_complexity_features(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute complexity features (Hurst, autocorrelation, fractal dimension).

    Features:
    - hurst_exponent_20d: Trending vs mean-reverting indicator
    - hurst_exponent_slope: Change in Hurst exponent
    - autocorr_lag1: Return autocorrelation at lag 1
    - autocorr_lag5: Return autocorrelation at lag 5
    - autocorr_partial_lag1: Partial autocorrelation at lag 1
    - fractal_dimension_20d: Price path complexity

    Args:
        close: Close price series

    Returns:
        Dict with complexity features
    """
    features = {}
    n = len(close)
    close_arr = close.values

    # Compute returns for autocorrelation
    returns = pd.Series(close_arr).pct_change()

    # Hurst exponent (20-day rolling)
    hurst = pd.Series(np.nan, index=range(n))
    for i in range(20, n):
        window = close_arr[i - 20:i]
        hurst.iloc[i] = _hurst_exponent_rs(window)

    features["hurst_exponent_20d"] = hurst

    # Hurst exponent slope: 5-day change
    hurst_slope = hurst - hurst.shift(5)
    features["hurst_exponent_slope"] = hurst_slope

    # Autocorrelation lag 1 (20-day rolling)
    autocorr_lag1 = returns.rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x.dropna()) > 2 else np.nan,
        raw=False
    )
    features["autocorr_lag1"] = autocorr_lag1

    # Autocorrelation lag 5 (20-day rolling)
    autocorr_lag5 = returns.rolling(20).apply(
        lambda x: x.autocorr(lag=5) if len(x.dropna()) > 6 else np.nan,
        raw=False
    )
    features["autocorr_lag5"] = autocorr_lag5

    # Partial autocorrelation lag 1
    # For simplicity, use autocorr_lag1 as approximation (exact PACF requires more complex computation)
    # PACF(1) = ACF(1) for lag 1
    features["autocorr_partial_lag1"] = autocorr_lag1.copy()

    # Fractal dimension (20-day rolling)
    fractal_dim = pd.Series(np.nan, index=range(n))
    for i in range(20, n):
        window = close_arr[i - 20:i]
        fractal_dim.iloc[i] = _fractal_dimension_higuchi(window, k_max=5)

    features["fractal_dimension_20d"] = fractal_dim

    return features


def _compute_chunk_10a(
    df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series
) -> Mapping[str, pd.Series]:
    """Compute all Sub-Chunk 10a features (MTF + Entropy + Complexity).

    Args:
        df: Full DataFrame with OHLCV
        high: High price series
        low: Low price series
        close: Close price series

    Returns:
        Dict with all 25 Chunk 10a features
    """
    features = {}

    # Weekly MA Features (3 features)
    features.update(_compute_10a_weekly_ma_features(df))

    # Weekly RSI Features (2 features)
    features.update(_compute_10a_weekly_rsi_features(df))

    # Weekly BB Features (3 features)
    features.update(_compute_10a_weekly_bb_features(df))

    # Alignment Features (3 features)
    features.update(_compute_10a_alignment_features(df, close))

    # Entropy Features (8 features)
    features.update(_compute_10a_entropy_features(close))

    # Complexity Features (6 features)
    features.update(_compute_10a_complexity_features(close))

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

    # =========================================================================
    # Sub-Chunk 8a: TRD Complete (ranks 301-323)
    # =========================================================================
    features.update(_compute_chunk_8a(high, low, close))

    # =========================================================================
    # Sub-Chunk 8b: SR Complete (ranks 324-345)
    # =========================================================================
    features.update(_compute_chunk_8b(high, low, close))

    # =========================================================================
    # Sub-Chunk 9a: CDL Part 1 - Candlestick Patterns (ranks 346-370)
    # =========================================================================
    features.update(_compute_chunk_9a(open_price, high, low, close))

    # =========================================================================
    # Sub-Chunk 9b: CDL Part 2 - Candlestick Patterns (ranks 371-395)
    # =========================================================================
    features.update(_compute_chunk_9b(open_price, high, low, close))

    # =========================================================================
    # Sub-Chunk 10a: MTF Complete (ranks 396-420)
    # =========================================================================
    features.update(_compute_chunk_10a(df, high, low, close))

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
