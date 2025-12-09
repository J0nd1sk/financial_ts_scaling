"""Indicator calculation utilities for SPY features."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import talib

DEFAULT_VWAP_WINDOW = 20
WEEKLY_FREQ = "W-MON"

FEATURE_LIST = [
    "dema_9",
    "dema_10",
    "sma_12",
    "dema_20",
    "dema_25",
    "sma_50",
    "dema_90",
    "sma_100",
    "sma_200",
    "rsi_daily",
    "rsi_weekly",
    "stochrsi_daily",
    "stochrsi_weekly",
    "macd_line",
    "obv",
    "adosc",
    "atr_14",
    "adx_14",
    "bb_percent_b",
    "vwap_20",
]


def load_raw_data(path: Path) -> pd.DataFrame:
    """Read raw parquet file and return sorted DataFrame."""
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    return df.reset_index(drop=True)


def _compute_moving_averages(close: pd.Series) -> Mapping[str, pd.Series]:
    periods_dema = [9, 10, 20, 25, 90]
    periods_sma = [12, 50, 100, 200]
    features = {}
    for period in periods_dema:
        features[f"dema_{period}"] = talib.DEMA(close, timeperiod=period)
    for period in periods_sma:
        features[f"sma_{period}"] = talib.SMA(close, timeperiod=period)
    return features


def _compute_daily_oscillators(close: pd.Series) -> Mapping[str, pd.Series]:
    rsi_daily = talib.RSI(close, timeperiod=14)
    stoch_k, _ = talib.STOCHRSI(
        close,
        timeperiod=14,
        fastk_period=5,
        fastd_period=3,
        fastd_matype=0,
    )
    return {
        "rsi_daily": rsi_daily,
        "stochrsi_daily": stoch_k,
    }


def _resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.set_index("Date")
        .resample(WEEKLY_FREQ, label="left", closed="left")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna(subset=["Close"])
    )
    weekly = weekly[weekly["Close"].notnull()]
    return weekly


def _compute_weekly_indicators(df: pd.DataFrame) -> pd.DataFrame:
    weekly = _resample_to_weekly(df)
    weekly_rsi = talib.RSI(weekly["Close"], timeperiod=14)
    stoch_k, _ = talib.STOCHRSI(
        weekly["Close"],
        timeperiod=14,
        fastk_period=5,
        fastd_period=3,
        fastd_matype=0,
    )
    weekly_features = pd.DataFrame(
        {
            "rsi_weekly": weekly_rsi,
            "stochrsi_weekly": stoch_k,
        },
        index=weekly.index,
    )
    daily_index = df["Date"]
    weekly_features = weekly_features.reindex(daily_index).ffill()
    # Align index with daily RangeIndex to avoid index union when concatenating features.
    weekly_features = weekly_features.reset_index(drop=True)
    return weekly_features


def _compute_other_indicators(df: pd.DataFrame) -> Mapping[str, pd.Series]:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    macd_line, _, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    obv = talib.OBV(close, volume)
    adosc = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
    atr = talib.ATR(high, low, close, timeperiod=14)
    adx = talib.ADX(high, low, close, timeperiod=14)
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    percent_b = (close - lower) / (upper - lower)
    vwap = _rolling_vwap(close, volume, window=DEFAULT_VWAP_WINDOW)
    return {
        "macd_line": macd_line,
        "obv": obv,
        "adosc": adosc,
        "atr_14": atr,
        "adx_14": adx,
        "bb_percent_b": percent_b,
        "vwap_20": vwap,
    }


def _rolling_vwap(close: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    price_volume = close * volume
    pv_sum = price_volume.rolling(window).sum()
    volume_sum = volume.rolling(window).sum()
    vwap = pv_sum / volume_sum.replace(0, np.nan)
    return vwap


def build_feature_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    close = df["Close"]
    features = {}
    features.update(_compute_moving_averages(close))
    features.update(_compute_daily_oscillators(close))

    weekly_features = _compute_weekly_indicators(df)
    features.update(weekly_features.to_dict(orient="series"))
    features.update(_compute_other_indicators(df))

    feature_df = pd.DataFrame(features)
    feature_df.insert(0, "Date", df["Date"])
    merged = pd.concat([df[["Date"]], feature_df.drop(columns=["Date"])], axis=1)
    merged = merged.dropna().reset_index(drop=True)
    return merged[["Date"] + FEATURE_LIST]

