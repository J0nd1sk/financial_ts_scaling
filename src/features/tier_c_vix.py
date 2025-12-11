"""VIX feature engineering for tier c.

Computes 8 VIX-derived features for volatility context:
- vix_close: Raw VIX close value
- vix_sma_10: 10-day simple moving average
- vix_sma_20: 20-day simple moving average
- vix_percentile_60d: 60-day rolling percentile rank [0-100]
- vix_zscore_20d: 20-day rolling z-score
- vix_regime: Categorical - 'low' (<15), 'normal' (15-25), 'high' (>=25)
- vix_change_1d: 1-day percent change
- vix_change_5d: 5-day percent change
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VIX_FEATURE_LIST = [
    "vix_close",
    "vix_sma_10",
    "vix_sma_20",
    "vix_percentile_60d",
    "vix_zscore_20d",
    "vix_regime",
    "vix_change_1d",
    "vix_change_5d",
]

# Regime thresholds (standard VIX interpretation)
VIX_LOW_THRESHOLD = 15
VIX_HIGH_THRESHOLD = 25

# Lookback periods
SMA_SHORT = 10
SMA_LONG = 20
PERCENTILE_WINDOW = 60
ZSCORE_WINDOW = 20
CHANGE_SHORT = 1
CHANGE_LONG = 5


def _compute_rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling percentile rank (0-100 scale)."""

    def percentile_rank(x: np.ndarray) -> float:
        """Rank of last value as percentile of window."""
        if len(x) < window:
            return np.nan
        current = x[-1]
        rank = (x[:-1] < current).sum()
        return (rank / (len(x) - 1)) * 100

    return series.rolling(window).apply(percentile_rank, raw=True)


def _compute_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score with zero-std handling."""
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()

    # Handle zero std case: when std=0, z-score is 0 (no deviation from mean)
    zscore = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore = zscore.fillna(0)

    return zscore


def _classify_regime(close: pd.Series) -> pd.Series:
    """Classify VIX into regime categories."""
    conditions = [
        close < VIX_LOW_THRESHOLD,
        (close >= VIX_LOW_THRESHOLD) & (close < VIX_HIGH_THRESHOLD),
        close >= VIX_HIGH_THRESHOLD,
    ]
    choices = ["low", "normal", "high"]
    return pd.Series(np.select(conditions, choices, default="normal"), index=close.index)


def _compute_percent_change(series: pd.Series, periods: int) -> pd.Series:
    """Compute percent change over N periods."""
    lagged = series.shift(periods)
    return ((series - lagged) / lagged) * 100


def build_vix_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build VIX features from raw VIX OHLCV data.

    Args:
        df: DataFrame with Date, Open, High, Low, Close columns

    Returns:
        DataFrame with Date and 8 VIX features, warmup rows dropped
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"]

    features = {
        "vix_close": close,
        "vix_sma_10": close.rolling(SMA_SHORT).mean(),
        "vix_sma_20": close.rolling(SMA_LONG).mean(),
        "vix_percentile_60d": _compute_rolling_percentile(close, PERCENTILE_WINDOW),
        "vix_zscore_20d": _compute_rolling_zscore(close, ZSCORE_WINDOW),
        "vix_regime": _classify_regime(close),
        "vix_change_1d": _compute_percent_change(close, CHANGE_SHORT),
        "vix_change_5d": _compute_percent_change(close, CHANGE_LONG),
    }

    result = pd.DataFrame(features)
    result.insert(0, "Date", df["Date"])

    # Drop warmup rows (longest lookback is 60 days for percentile)
    result = result.dropna(subset=["vix_percentile_60d"]).reset_index(drop=True)

    return result[["Date"] + VIX_FEATURE_LIST]
