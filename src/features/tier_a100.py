"""Tier a100 indicator module - 100 total indicators (50 from a50 + 50 new).

This module extends tier_a50 with 50 additional indicators including:

Chunk 1 (rank 51-52): Momentum derivatives
- return_1d_acceleration - change in 1-day return (momentum shift)
- return_5d_acceleration - change in 5-day return

Chunk 2 (rank 53-56): QQE/STC derivatives
- qqe_slope - 5-day change in QQE fast line
- qqe_extreme_dist - distance to nearest extreme (20 or 80)
- stc_slope - 5-day change in STC value
- stc_extreme_dist - distance to nearest extreme (25 or 75)

Chunk 3 (rank 57-64): Standard oscillators
- demarker_value - DeMarker oscillator [0, 1]
- demarker_from_half - DeMarker deviation from 0.5 [-0.5, +0.5]
- stoch_k_14 - Stochastic %K (14-period) [0, 100]
- stoch_d_14 - Stochastic %D (3-period SMA of %K) [0, 100]
- stoch_extreme_dist - Distance to nearest extreme (20 or 80) [0, 30]
- cci_14 - Commodity Channel Index (14-period) [unbounded]
- mfi_14 - Money Flow Index (14-period) [0, 100]
- williams_r_14 - Williams %R (14-period) [-100, 0]

Chunk 4 (rank 65-73): VRP + Risk metrics
- vrp_5d - VIX minus 5-day realized volatility
- vrp_slope - daily change in vrp_5d
- sharpe_252d - 252-day annualized Sharpe ratio
- sortino_252d - 252-day annualized Sortino ratio
- sharpe_slope_20d - daily change in 20-day Sharpe
- sortino_slope_20d - daily change in 20-day Sortino
- var_95 - 5th percentile of 20-day returns (Value at Risk)
- var_99 - 1st percentile of 20-day returns
- cvar_95 - mean of returns below var_95 (Conditional VaR)

Chunk 5 (rank 74-80): MA extensions
- sma_9_50_proximity - % difference between SMA_9 and SMA_50
- sma_50_slope - 5-day change in SMA_50
- sma_200_slope - 5-day change in SMA_200
- days_since_sma_50_cross - days since price crossed SMA_50
- days_since_sma_200_cross - days since price crossed SMA_200
- ema_12 - 12-period EMA value
- ema_26 - 26-period EMA value

Chunk 6 (rank 81-85): Advanced volatility
- atr_pct_percentile_60d - 60-day rolling percentile of ATR%
- bb_width_percentile_60d - 60-day rolling percentile of Bollinger Band width
- parkinson_volatility - Parkinson volatility estimator (20-day, annualized %)
- garman_klass_volatility - Garman-Klass volatility estimator (20-day, annualized %)
- vol_of_vol - std(realized_vol) over 20 days

Chunk 7 (rank 86-90): Trend indicators
- adx_slope - 5-day change in ADX (trend strength momentum)
- di_spread - +DI minus -DI (directional bias) [-100, +100]
- aroon_oscillator - Aroon Up - Aroon Down (25-period) [-100, +100]
- price_pct_from_supertrend - % distance from SuperTrend (signed)
- supertrend_direction - +1 bullish, -1 bearish

Chunk 8 (rank 91-100): Volume + Momentum + S/R
- obv_slope - 5-day change in OBV (normalized by std)
- volume_price_trend - cumsum(Volume × pct_change(Close))
- kvo_histogram - KVO minus Signal (Klinger Volume Oscillator)
- accumulation_dist - Accumulation/Distribution Line
- expectancy_20d - win_rate × avg_gain - (1-win_rate) × abs(avg_loss)
- win_rate_20d - Count(positive returns) / 20 [0, 1]
- buying_pressure_ratio - (Close - Low) / (High - Low) [0, 1]
- fib_range_position - (Close - Low_44d) / (High_44d - Low_44d) [0, 1]
- prior_high_20d_dist - (Close - High_20d) / High_20d × 100 [≤ 0]
- prior_low_20d_dist - (Close - Low_20d) / Low_20d × 100 [≥ 0]
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import talib

from src.features import tier_a50

# 50 new indicators added in tier a100 (ranks 51-100)
# Built incrementally across chunks
A100_ADDITION_LIST = [
    # Chunk 1: Momentum derivatives (ranks 51-52)
    "return_1d_acceleration",
    "return_5d_acceleration",
    # Chunk 2: QQE/STC derivatives (ranks 53-56)
    "qqe_slope",
    "qqe_extreme_dist",
    "stc_slope",
    "stc_extreme_dist",
    # Chunk 3: Standard oscillators (ranks 57-64)
    "demarker_value",
    "demarker_from_half",
    "stoch_k_14",
    "stoch_d_14",
    "stoch_extreme_dist",
    "cci_14",
    "mfi_14",
    "williams_r_14",
    # Chunk 4: VRP + Risk metrics (ranks 65-73)
    "vrp_5d",
    "vrp_slope",
    "sharpe_252d",
    "sortino_252d",
    "sharpe_slope_20d",
    "sortino_slope_20d",
    "var_95",
    "var_99",
    "cvar_95",
    # Chunk 5: MA extensions (ranks 74-80)
    "sma_9_50_proximity",
    "sma_50_slope",
    "sma_200_slope",
    "days_since_sma_50_cross",
    "days_since_sma_200_cross",
    "ema_12",
    "ema_26",
    # Chunk 6: Advanced volatility (ranks 81-85)
    "atr_pct_percentile_60d",
    "bb_width_percentile_60d",
    "parkinson_volatility",
    "garman_klass_volatility",
    "vol_of_vol",
    # Chunk 7: Trend indicators (ranks 86-90)
    "adx_slope",
    "di_spread",
    "aroon_oscillator",
    "price_pct_from_supertrend",
    "supertrend_direction",
    # Chunk 8: Volume + Momentum + S/R (ranks 91-100)
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

# Complete a100 feature list = a50 (50) + 50 new = 100 total
FEATURE_LIST = tier_a50.FEATURE_LIST + A100_ADDITION_LIST


def _compute_return_accelerations(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute acceleration (change in return) for momentum shift detection.

    Acceleration is the first derivative of returns, measuring how quickly
    momentum is changing. Positive acceleration = momentum increasing.

    Args:
        close: Close price series

    Returns:
        Dict with return_1d_acceleration and return_5d_acceleration
    """
    features = {}

    # 1-day return
    return_1d = (close - close.shift(1)) / close.shift(1) * 100
    # 1-day acceleration = change in 1-day return
    features["return_1d_acceleration"] = return_1d - return_1d.shift(1)

    # 5-day return
    return_5d = (close - close.shift(5)) / close.shift(5) * 100
    # 5-day acceleration = change in 5-day return
    features["return_5d_acceleration"] = return_5d - return_5d.shift(1)

    return features


def _compute_qqe_stc_derivatives(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute QQE and STC derivative indicators.

    Derives from tier_a50's qqe_fast (0-100) and stc_value (0-100).

    Args:
        close: Close price series

    Returns:
        Dict with qqe_slope, qqe_extreme_dist, stc_slope, stc_extreme_dist
    """
    # Get QQE values from a50 helper
    qqe_features = tier_a50._compute_qqe(close)
    qqe_fast = qqe_features["qqe_fast"]

    # Get STC values from a50 helper
    stc_features = tier_a50._compute_stc(close)
    stc_value = stc_features["stc_value"]

    features = {}

    # QQE slope: 5-day change in qqe_fast
    features["qqe_slope"] = qqe_fast - qqe_fast.shift(5)

    # QQE extreme distance: min distance to 20 or 80
    dist_to_20 = (qqe_fast - 20).abs()
    dist_to_80 = (qqe_fast - 80).abs()
    features["qqe_extreme_dist"] = pd.concat([dist_to_20, dist_to_80], axis=1).min(axis=1)

    # STC slope: 5-day change in stc_value
    features["stc_slope"] = stc_value - stc_value.shift(5)

    # STC extreme distance: min distance to 25 or 75
    dist_to_25 = (stc_value - 25).abs()
    dist_to_75 = (stc_value - 75).abs()
    features["stc_extreme_dist"] = pd.concat([dist_to_25, dist_to_75], axis=1).min(axis=1)

    return features


def _compute_standard_oscillators(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute standard oscillator indicators.

    Args:
        df: DataFrame with High, Low, Close, Volume columns

    Returns:
        Dict with demarker_value, demarker_from_half, stoch_k_14, stoch_d_14,
        stoch_extreme_dist, cci_14, mfi_14, williams_r_14
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    volume = df["Volume"].values.astype(float)

    features = {}

    # --- DeMarker (manual calculation - not in talib) ---
    # DeMax = max(High - High[t-1], 0)
    # DeMin = max(Low[t-1] - Low, 0)
    # DeMarker = SMA(DeMax, 14) / (SMA(DeMax, 14) + SMA(DeMin, 14))
    high_series = pd.Series(high)
    low_series = pd.Series(low)

    de_max = (high_series - high_series.shift(1)).clip(lower=0)
    de_min = (low_series.shift(1) - low_series).clip(lower=0)

    sma_de_max = de_max.rolling(window=14).mean()
    sma_de_min = de_min.rolling(window=14).mean()

    # Handle division by zero: fill with 0.5 (neutral value)
    denominator = sma_de_max + sma_de_min
    demarker = sma_de_max / denominator
    demarker = demarker.fillna(0.5)  # NaN from division by zero -> neutral
    demarker = demarker.where(denominator != 0, 0.5)  # Explicit zero denominator -> neutral

    features["demarker_value"] = demarker
    features["demarker_from_half"] = demarker - 0.5

    # --- Stochastic (talib) ---
    # fastk_period=14, slowk_period=3, slowd_period=3
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=14,
        slowk_period=3,
        slowk_matype=0,  # SMA
        slowd_period=3,
        slowd_matype=0,  # SMA
    )
    features["stoch_k_14"] = pd.Series(slowk)
    features["stoch_d_14"] = pd.Series(slowd)

    # Stochastic extreme distance: min distance to 20 or 80
    stoch_k_series = pd.Series(slowk)
    dist_to_20 = (stoch_k_series - 20).abs()
    dist_to_80 = (stoch_k_series - 80).abs()
    features["stoch_extreme_dist"] = pd.concat([dist_to_20, dist_to_80], axis=1).min(axis=1)

    # --- CCI (talib) ---
    cci = talib.CCI(high, low, close, timeperiod=14)
    features["cci_14"] = pd.Series(cci)

    # --- MFI (talib) ---
    mfi = talib.MFI(high, low, close, volume, timeperiod=14)
    features["mfi_14"] = pd.Series(mfi)

    # --- Williams %R (talib) ---
    willr = talib.WILLR(high, low, close, timeperiod=14)
    features["williams_r_14"] = pd.Series(willr)

    return features


def _compute_vrp_extensions(df: pd.DataFrame, vix_df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute VRP extension indicators.

    VRP = Implied Volatility (VIX) - Realized Volatility

    Args:
        df: DataFrame with Date, Close columns
        vix_df: DataFrame with VIX data

    Returns:
        Dict with vrp_5d, vrp_slope
    """
    close = df["Close"]

    # Align VIX to price data by date (same pattern as tier_a50._compute_vrp)
    vix_df_copy = vix_df.copy()
    vix_df_copy["Date"] = pd.to_datetime(vix_df_copy["Date"])

    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])

    # Reindex VIX to match df dates
    vix_aligned = (
        vix_df_copy.set_index("Date")["Close"]
        .reindex(df_copy["Date"])
        .values
    )
    vix_aligned = pd.Series(vix_aligned, index=close.index)

    # 5-day realized volatility (reuse tier_a50 helper)
    rv_5d = tier_a50._compute_realized_volatility(close, 5)

    # VRP = implied - realized
    vrp_5d = vix_aligned - rv_5d

    features = {}
    features["vrp_5d"] = vrp_5d
    features["vrp_slope"] = vrp_5d - vrp_5d.shift(1)

    return features


def _compute_extended_risk_metrics(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute extended risk metrics: Sharpe, Sortino, and their slopes.

    Args:
        close: Close price series

    Returns:
        Dict with sharpe_252d, sortino_252d, sharpe_slope_20d, sortino_slope_20d
    """
    features = {}

    # Daily returns (percentage)
    returns = close.pct_change() * 100

    # --- 252-day Sharpe ratio (annualized) ---
    # Sharpe = mean(returns) / std(returns) * sqrt(252)
    rolling_mean_252 = returns.rolling(252).mean()
    rolling_std_252 = returns.rolling(252).std()

    # Handle division by zero
    rolling_std_252_safe = rolling_std_252.replace(0, np.nan)
    sharpe_252d = rolling_mean_252 / rolling_std_252_safe * np.sqrt(252)
    sharpe_252d = sharpe_252d.fillna(0)
    features["sharpe_252d"] = sharpe_252d

    # --- 252-day Sortino ratio (annualized) ---
    # Sortino = mean(returns) / downside_std * sqrt(252)
    # Downside std = std of negative returns only
    def downside_std(x: pd.Series) -> float:
        """Compute standard deviation of negative values only."""
        negatives = x[x < 0]
        if len(negatives) == 0:
            return np.nan  # No downside risk
        return negatives.std()

    rolling_downside_std_252 = returns.rolling(252).apply(downside_std, raw=False)
    rolling_downside_std_252_safe = rolling_downside_std_252.replace(0, np.nan)
    sortino_252d = rolling_mean_252 / rolling_downside_std_252_safe * np.sqrt(252)
    sortino_252d = sortino_252d.fillna(0)
    features["sortino_252d"] = sortino_252d

    # --- 20-day Sharpe slope ---
    # First compute 20-day Sharpe, then take daily difference
    rolling_mean_20 = returns.rolling(20).mean()
    rolling_std_20 = returns.rolling(20).std()
    rolling_std_20_safe = rolling_std_20.replace(0, np.nan)
    sharpe_20d = rolling_mean_20 / rolling_std_20_safe * np.sqrt(252)
    sharpe_20d = sharpe_20d.fillna(0)
    features["sharpe_slope_20d"] = sharpe_20d - sharpe_20d.shift(1)

    # --- 20-day Sortino slope ---
    rolling_downside_std_20 = returns.rolling(20).apply(downside_std, raw=False)
    rolling_downside_std_20_safe = rolling_downside_std_20.replace(0, np.nan)
    sortino_20d = rolling_mean_20 / rolling_downside_std_20_safe * np.sqrt(252)
    sortino_20d = sortino_20d.fillna(0)
    features["sortino_slope_20d"] = sortino_20d - sortino_20d.shift(1)

    return features


def _compute_var_cvar(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute Value at Risk and Conditional VaR indicators.

    Args:
        close: Close price series

    Returns:
        Dict with var_95, var_99, cvar_95
    """
    features = {}

    # Daily returns (percentage)
    returns = close.pct_change() * 100

    # --- VaR 95 (5th percentile of returns over 20 days) ---
    var_95 = returns.rolling(20).quantile(0.05)
    features["var_95"] = var_95

    # --- VaR 99 (1st percentile of returns over 20 days) ---
    var_99 = returns.rolling(20).quantile(0.01)
    features["var_99"] = var_99

    # --- CVaR 95 (mean of returns below VaR 95) ---
    # Conditional VaR = Expected Shortfall = mean of tail losses
    def cvar_func(x: pd.Series) -> float:
        """Compute CVaR (mean of returns below 5th percentile)."""
        threshold = np.percentile(x, 5)
        tail = x[x <= threshold]
        if len(tail) == 0:
            return threshold  # Fallback to VaR if no returns below threshold
        return tail.mean()

    cvar_95 = returns.rolling(20).apply(cvar_func, raw=False)
    features["cvar_95"] = cvar_95

    return features


def _compute_ma_extensions(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute MA extension indicators: proximity, slopes, and EMAs.

    Args:
        close: Close price series

    Returns:
        Dict with sma_9_50_proximity, sma_50_slope, sma_200_slope, ema_12, ema_26
    """
    close_arr = close.values

    features = {}

    # Compute SMAs using talib
    sma_9 = pd.Series(talib.SMA(close_arr, timeperiod=9), index=close.index)
    sma_50 = pd.Series(talib.SMA(close_arr, timeperiod=50), index=close.index)
    sma_200 = pd.Series(talib.SMA(close_arr, timeperiod=200), index=close.index)

    # SMA 9/50 proximity: % difference between SMA_9 and SMA_50
    # (SMA_9 - SMA_50) / SMA_50 * 100
    features["sma_9_50_proximity"] = (sma_9 - sma_50) / sma_50 * 100

    # SMA slopes: 5-day change in SMA value
    features["sma_50_slope"] = sma_50 - sma_50.shift(5)
    features["sma_200_slope"] = sma_200 - sma_200.shift(5)

    # EMAs using talib
    features["ema_12"] = pd.Series(talib.EMA(close_arr, timeperiod=12), index=close.index)
    features["ema_26"] = pd.Series(talib.EMA(close_arr, timeperiod=26), index=close.index)

    return features


def _compute_days_since_cross(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute days since price crossed SMA levels.

    A cross occurs when price moves from one side of the MA to the other.
    The count resets to 0 on each cross and increments daily.

    Args:
        close: Close price series

    Returns:
        Dict with days_since_sma_50_cross, days_since_sma_200_cross
    """
    close_arr = close.values

    features = {}

    # Compute SMAs
    sma_50 = pd.Series(talib.SMA(close_arr, timeperiod=50), index=close.index)
    sma_200 = pd.Series(talib.SMA(close_arr, timeperiod=200), index=close.index)

    def days_since_cross(price: pd.Series, ma: pd.Series) -> pd.Series:
        """Count days since price crossed the moving average.

        Uses sign changes to detect crosses, then cumsum trick to count days.
        """
        # Position: 1 if above MA, -1 if below, 0 if equal
        position = np.sign(price - ma)

        # Detect crosses: position changed from previous day
        # Cross occurs when position != previous position (and neither is 0)
        cross = (position != position.shift(1)) & (position != 0) & (position.shift(1) != 0)

        # Create groups: each cross starts a new group
        # cumsum of cross gives group number
        groups = cross.cumsum()

        # Count days within each group using cumcount pattern
        # For each group, count = position within group (0, 1, 2, ...)
        result = groups.groupby(groups).cumcount()

        # Handle initial period before first cross
        # If no cross has occurred, days_since = index position (days from start)
        first_valid_idx = ma.first_valid_index()
        if first_valid_idx is not None:
            first_valid_pos = close.index.get_loc(first_valid_idx)
            # Before first valid MA, set to 0
            result.iloc[:first_valid_pos] = 0

        return result

    features["days_since_sma_50_cross"] = days_since_cross(close, sma_50)
    features["days_since_sma_200_cross"] = days_since_cross(close, sma_200)

    return features


def _compute_advanced_volatility(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute advanced volatility indicators.

    Includes percentile-based volatility rankings and alternative volatility
    estimators (Parkinson, Garman-Klass) which are more efficient than
    close-to-close volatility.

    Args:
        df: DataFrame with Open, High, Low, Close columns

    Returns:
        Dict with atr_pct_percentile_60d, bb_width_percentile_60d,
        parkinson_volatility, garman_klass_volatility, vol_of_vol
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    open_price = df["Open"].values

    features = {}

    # --- ATR percentile (60-day rolling percentile of ATR%) ---
    atr = pd.Series(talib.ATR(high, low, close, timeperiod=14), index=df.index)
    atr_pct = atr / pd.Series(close, index=df.index) * 100

    def percentile_rank(x: pd.Series) -> float:
        """Compute percentile rank of last value within the window."""
        if len(x) == 0 or pd.isna(x.iloc[-1]):
            return np.nan
        return (x.iloc[-1] <= x).sum() / len(x) * 100

    features["atr_pct_percentile_60d"] = atr_pct.rolling(60).apply(
        percentile_rank, raw=False
    )

    # --- BB width percentile (60-day rolling percentile of BB width%) ---
    upper, middle, lower = talib.BBANDS(close, timeperiod=20)
    bb_width = pd.Series((upper - lower) / middle * 100, index=df.index)
    features["bb_width_percentile_60d"] = bb_width.rolling(60).apply(
        percentile_rank, raw=False
    )

    # --- Parkinson volatility (20-day, annualized %) ---
    # Parkinson estimator: sqrt(1/(4*ln(2)) * mean(log(H/L)^2))
    # More efficient than close-to-close as it uses intraday range
    high_series = pd.Series(high, index=df.index)
    low_series = pd.Series(low, index=df.index)
    log_hl = np.log(high_series / low_series)
    log_hl_sq = log_hl ** 2

    # Rolling 20-day Parkinson variance
    parkinson_var = log_hl_sq.rolling(20).mean() / (4 * np.log(2))
    # Annualize: sqrt(var * 252) * 100
    features["parkinson_volatility"] = np.sqrt(parkinson_var * 252) * 100

    # --- Garman-Klass volatility (20-day, annualized %) ---
    # GK estimator: sqrt(0.5*log(H/L)^2 - (2*ln(2)-1)*log(C/O)^2)
    # Even more efficient as it incorporates open/close prices
    open_series = pd.Series(open_price, index=df.index)
    close_series = pd.Series(close, index=df.index)
    log_co = np.log(close_series / open_series)

    # Daily GK variance = 0.5*log(H/L)^2 - (2*ln(2)-1)*log(C/O)^2
    gk_daily = 0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co ** 2

    # Rolling 20-day mean, clip to 0 to handle rare negative values
    gk_var = gk_daily.rolling(20).mean().clip(lower=0)
    # Annualize: sqrt(var * 252) * 100
    features["garman_klass_volatility"] = np.sqrt(gk_var * 252) * 100

    # --- Vol of vol (20-day std of 5-day realized volatility) ---
    # Measures how stable/unstable volatility is
    returns = close_series.pct_change() * 100
    realized_vol_5d = returns.rolling(5).std() * np.sqrt(252)
    features["vol_of_vol"] = realized_vol_5d.rolling(20).std()

    return features


def _compute_trend_indicators(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute trend strength and direction indicators.

    Includes ADX (trend strength), DI spread (directional bias),
    Aroon oscillator (time-based trend), and SuperTrend (ATR-based trend).

    Args:
        df: DataFrame with High, Low, Close columns

    Returns:
        Dict with adx_14, di_spread, aroon_oscillator,
        price_pct_from_supertrend, supertrend_direction
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values

    features = {}

    # --- ADX slope and DI (talib) ---
    # Note: adx_14 already exists in tier_a50, so we add adx_slope instead
    adx = pd.Series(talib.ADX(high, low, close, timeperiod=14), index=df.index)
    features["adx_slope"] = adx - adx.shift(5)

    plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
    features["di_spread"] = pd.Series(plus_di - minus_di, index=df.index)

    # --- Aroon Oscillator (talib) ---
    aroon_down, aroon_up = talib.AROON(high, low, timeperiod=25)
    features["aroon_oscillator"] = pd.Series(aroon_up - aroon_down, index=df.index)

    # --- SuperTrend (manual - not in talib) ---
    # SuperTrend = ATR-based bands, flips on close crossing band
    atr_period = 10
    multiplier = 3.0

    atr = pd.Series(talib.ATR(high, low, close, timeperiod=atr_period), index=df.index)
    hl2 = (pd.Series(high, index=df.index) + pd.Series(low, index=df.index)) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    close_series = pd.Series(close, index=df.index)

    # Initialize SuperTrend and direction arrays
    supertrend = np.empty(len(df))
    supertrend[:] = np.nan
    direction = np.empty(len(df))
    direction[:] = np.nan

    # Find first valid index (where ATR is valid)
    first_valid_pos = atr_period  # ATR needs atr_period bars
    if first_valid_pos < len(df):
        # Initialize with bearish assumption (price below upper band)
        supertrend[first_valid_pos] = upper_band.iloc[first_valid_pos]
        direction[first_valid_pos] = -1

    # Calculate SuperTrend iteratively
    for i in range(first_valid_pos + 1, len(df)):
        if np.isnan(atr.iloc[i]):
            continue

        prev_supertrend = supertrend[i - 1]
        prev_direction = direction[i - 1]

        if prev_direction == 1:  # Previous bullish (price above lower band)
            # Use lower band, adjust upward only (trailing stop)
            curr_lower = max(lower_band.iloc[i], prev_supertrend)
            if close_series.iloc[i] >= curr_lower:
                # Stay bullish
                supertrend[i] = curr_lower
                direction[i] = 1
            else:
                # Switch to bearish
                supertrend[i] = upper_band.iloc[i]
                direction[i] = -1
        else:  # Previous bearish (price below upper band)
            # Use upper band, adjust downward only (trailing stop)
            curr_upper = min(upper_band.iloc[i], prev_supertrend)
            if close_series.iloc[i] <= curr_upper:
                # Stay bearish
                supertrend[i] = curr_upper
                direction[i] = -1
            else:
                # Switch to bullish
                supertrend[i] = lower_band.iloc[i]
                direction[i] = 1

    supertrend_series = pd.Series(supertrend, index=df.index)
    direction_series = pd.Series(direction, index=df.index)

    # Percent distance from SuperTrend (positive = above, negative = below)
    features["price_pct_from_supertrend"] = (
        (close_series - supertrend_series) / supertrend_series * 100
    )
    features["supertrend_direction"] = direction_series

    return features


def _compute_volume_indicators(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute volume-based indicators.

    Includes OBV slope, Volume Price Trend, Klinger Volume Oscillator,
    Accumulation/Distribution, and Buying Pressure Ratio.

    Args:
        df: DataFrame with High, Low, Close, Volume columns

    Returns:
        Dict with obv_slope, volume_price_trend, kvo_histogram,
        accumulation_dist, buying_pressure_ratio
    """
    high = df["High"].values
    low = df["Low"].values
    close = df["Close"].values
    volume = df["Volume"].values.astype(float)

    close_series = pd.Series(close, index=df.index)
    high_series = pd.Series(high, index=df.index)
    low_series = pd.Series(low, index=df.index)
    volume_series = pd.Series(volume, index=df.index)

    features = {}

    # --- OBV slope (5-day change in OBV, normalized by std) ---
    obv = pd.Series(talib.OBV(close, volume), index=df.index)
    obv_change_5d = obv - obv.shift(5)
    # Normalize by rolling std to make comparable across different volume regimes
    obv_std = obv_change_5d.rolling(20).std()
    obv_std_safe = obv_std.replace(0, np.nan)
    features["obv_slope"] = (obv_change_5d / obv_std_safe).fillna(0)

    # --- Volume Price Trend (VPT) ---
    # VPT = cumsum(Volume × pct_change(Close))
    pct_change = close_series.pct_change()
    vpt = (volume_series * pct_change).cumsum()
    # Normalize by rolling mean to make comparable
    vpt_mean = vpt.rolling(20).mean()
    vpt_mean_safe = vpt_mean.replace(0, np.nan).abs()
    features["volume_price_trend"] = (vpt / vpt_mean_safe).fillna(0)

    # --- Klinger Volume Oscillator (KVO) histogram ---
    # KVO = EMA34(VolumeForce) - EMA55(VolumeForce)
    # VolumeForce = Volume × |2×(dm/cm) - 1| × trend × 100
    # where dm = High - Low, cm = cumulative dm for trend
    # Simplified: VolumeForce = Volume × sign(typical price change)

    # Typical price
    typical = (high_series + low_series + close_series) / 3
    typical_change = typical - typical.shift(1)

    # Trend direction: +1 if typical price up, -1 if down
    trend = np.sign(typical_change)

    # Volume Force (simplified)
    dm = high_series - low_series
    volume_force = volume_series * dm * trend

    # KVO = EMA34 - EMA55 of Volume Force
    ema34_vf = volume_force.ewm(span=34, adjust=False).mean()
    ema55_vf = volume_force.ewm(span=55, adjust=False).mean()
    kvo = ema34_vf - ema55_vf

    # Signal line = EMA13 of KVO
    kvo_signal = kvo.ewm(span=13, adjust=False).mean()

    # Histogram = KVO - Signal
    kvo_histogram = kvo - kvo_signal
    # Normalize by rolling std
    kvo_std = kvo_histogram.rolling(20).std()
    kvo_std_safe = kvo_std.replace(0, np.nan)
    features["kvo_histogram"] = (kvo_histogram / kvo_std_safe).fillna(0)

    # --- Accumulation/Distribution Line (AD) ---
    ad = pd.Series(talib.AD(high, low, close, volume), index=df.index)
    # Normalize by rolling mean (absolute value)
    ad_mean = ad.rolling(20).mean().abs()
    ad_mean_safe = ad_mean.replace(0, np.nan)
    features["accumulation_dist"] = (ad / ad_mean_safe).fillna(0)

    # --- Buying Pressure Ratio ---
    # (Close - Low) / (High - Low)
    range_hl = high_series - low_series
    # Handle division by zero: fill with 0.5 (neutral) when High == Low
    buying_pressure = (close_series - low_series) / range_hl
    buying_pressure = buying_pressure.where(range_hl != 0, 0.5)
    features["buying_pressure_ratio"] = buying_pressure

    return features


def _compute_expectancy_metrics(close: pd.Series) -> Mapping[str, pd.Series]:
    """Compute trading expectancy and win rate metrics.

    Expectancy = win_rate × avg_gain - (1 - win_rate) × abs(avg_loss)

    Args:
        close: Close price series

    Returns:
        Dict with expectancy_20d, win_rate_20d
    """
    features = {}

    # Daily returns (percentage)
    returns = close.pct_change() * 100

    # Win rate: proportion of positive returns over 20 days
    def win_rate(x: pd.Series) -> float:
        """Compute win rate (proportion of positive returns)."""
        if len(x) == 0:
            return 0.5
        return (x > 0).sum() / len(x)

    features["win_rate_20d"] = returns.rolling(20).apply(win_rate, raw=False)

    # Expectancy: win_rate × avg_gain - (1 - win_rate) × abs(avg_loss)
    def expectancy(x: pd.Series) -> float:
        """Compute expectancy from returns."""
        if len(x) == 0:
            return 0.0

        wins = x[x > 0]
        losses = x[x < 0]

        win_rate_val = len(wins) / len(x) if len(x) > 0 else 0.5
        avg_gain = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0

        return win_rate_val * avg_gain - (1 - win_rate_val) * abs(avg_loss)

    features["expectancy_20d"] = returns.rolling(20).apply(expectancy, raw=False)

    return features


def _compute_sr_indicators(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    """Compute support/resistance indicators.

    Includes Fibonacci range position and distances from prior highs/lows.

    Args:
        df: DataFrame with High, Low, Close columns

    Returns:
        Dict with fib_range_position, prior_high_20d_dist, prior_low_20d_dist
    """
    close = pd.Series(df["Close"].values, index=df.index)
    high = pd.Series(df["High"].values, index=df.index)
    low = pd.Series(df["Low"].values, index=df.index)

    features = {}

    # --- Fibonacci range position (44-day lookback per IronBot spec) ---
    # Position within the 44-day high-low range
    high_44d = high.rolling(44).max()
    low_44d = low.rolling(44).min()
    range_44d = high_44d - low_44d

    # (Close - Low_44d) / (High_44d - Low_44d)
    # Handle division by zero: fill with 0.5 (middle) when range is 0
    fib_position = (close - low_44d) / range_44d
    fib_position = fib_position.where(range_44d != 0, 0.5)
    features["fib_range_position"] = fib_position

    # --- Prior high 20d distance ---
    # (Close - High_20d) / High_20d × 100
    # This is always <= 0 because close cannot exceed the rolling max high
    high_20d = high.rolling(20).max()
    features["prior_high_20d_dist"] = (close - high_20d) / high_20d * 100

    # --- Prior low 20d distance ---
    # (Close - Low_20d) / Low_20d × 100
    # This is always >= 0 because close cannot be below the rolling min low
    low_20d = low.rolling(20).min()
    features["prior_low_20d_dist"] = (close - low_20d) / low_20d * 100

    return features


def build_feature_dataframe(raw_df: pd.DataFrame, vix_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature DataFrame with all tier_a100 indicators.

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

    # Start with all a50 features
    a50_features = tier_a50.build_feature_dataframe(raw_df, vix_df)
    a50_features = a50_features.set_index("Date")

    # Build all new a100 features
    features = {}

    # Chunk 1: Momentum derivatives
    features.update(_compute_return_accelerations(close))

    # Chunk 2: QQE/STC derivatives
    features.update(_compute_qqe_stc_derivatives(close))

    # Chunk 3: Standard oscillators
    features.update(_compute_standard_oscillators(df))

    # Chunk 4: VRP + Risk metrics
    features.update(_compute_vrp_extensions(df, vix_df))
    features.update(_compute_extended_risk_metrics(close))
    features.update(_compute_var_cvar(close))

    # Chunk 5: MA extensions
    features.update(_compute_ma_extensions(close))
    features.update(_compute_days_since_cross(close))

    # Chunk 6: Advanced volatility
    features.update(_compute_advanced_volatility(df))

    # Chunk 7: Trend indicators
    features.update(_compute_trend_indicators(df))

    # Chunk 8: Volume + Momentum + S/R
    features.update(_compute_volume_indicators(df))
    features.update(_compute_expectancy_metrics(close))
    features.update(_compute_sr_indicators(df))

    # Create feature DataFrame for new a100 features
    new_features_df = pd.DataFrame(features)
    new_features_df.insert(0, "Date", df["Date"])
    new_features_df = new_features_df.set_index("Date")

    # Merge a50 features with new a100 features
    merged = a50_features.join(new_features_df[A100_ADDITION_LIST], how="inner")

    # Reset index to get Date as column
    merged = merged.reset_index()

    # Drop rows with any NaN
    merged = merged.dropna().reset_index(drop=True)

    # Return columns in correct order: Date + all features
    return merged[["Date"] + FEATURE_LIST]
