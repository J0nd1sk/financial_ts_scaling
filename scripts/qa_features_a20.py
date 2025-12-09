#!/usr/bin/env python3
"""Quick QA checks for SPY a20 feature dataset.

Validations:
- Required columns present: Date + 20 indicators
- No nulls / no infs
- Dates monotonic increasing, no duplicates
- Basic value sanity:
  - Prices/averages/VWAP > 0
  - ATR/ADX > 0
  - %B in [0 - eps, 1 + eps]
  - RSI/StochRSI in [0 - eps, 100 + eps]
  - OBV/ADOSC finite
Optional: spot-check a few indicators against raw for a small sample.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

FEATURE_COLUMNS: Sequence[str] = [
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


def fail(msg: str) -> None:
    raise SystemExit(f"❌ QA failed: {msg}")


def check_columns(df: pd.DataFrame) -> None:
    expected = {"Date", *FEATURE_COLUMNS}
    missing = expected - set(df.columns)
    extra = set(df.columns) - expected
    if missing:
        fail(f"Missing columns: {sorted(missing)}")
    if extra:
        fail(f"Unexpected columns: {sorted(extra)}")


def check_nulls_infs(df: pd.DataFrame) -> None:
    if df.isnull().any().any():
        cols = df.columns[df.isnull().any()].tolist()
        fail(f"Nulls present in columns: {cols}")
    if np.isinf(df.select_dtypes(include=[float, int])).any().any():
        cols = df.columns[np.isinf(df.select_dtypes(include=[float, int])).any()].tolist()
        fail(f"Infinite values present in columns: {cols}")


def check_dates(df: pd.DataFrame) -> None:
    if not df["Date"].is_monotonic_increasing:
        fail("Date column is not monotonic increasing")
    if df["Date"].duplicated().any():
        fail("Duplicate dates found")


def check_value_ranges(df: pd.DataFrame, eps: float = 1e-6) -> None:
    # Positive-only series
    positive_cols = [
        "dema_9",
        "dema_10",
        "sma_12",
        "dema_20",
        "dema_25",
        "sma_50",
        "dema_90",
        "sma_100",
        "sma_200",
        "atr_14",
        "adx_14",
        "vwap_20",
    ]
    for col in positive_cols:
        if (df[col] <= 0).any():
            fail(f"Non-positive values in {col}")

    # %B can exceed [0,1] when price moves outside bands; just bound to a reasonable range.
    if ((df["bb_percent_b"] < -2) | (df["bb_percent_b"] > 2)).any():
        fail("bb_percent_b outside plausible range [-2, 2]")

    # RSI / StochRSI between 0 and 100
    bounded_cols = ["rsi_daily", "rsi_weekly", "stochrsi_daily", "stochrsi_weekly"]
    for col in bounded_cols:
        if ((df[col] < -eps) | (df[col] > 100 + eps)).any():
            fail(f"{col} outside [0,100] (with small epsilon)")

    # OBV/ADOSC finite (already checked for inf, but reassert)
    for col in ["obv", "adosc", "macd_line"]:
        if not np.isfinite(df[col]).all():
            fail(f"Non-finite values in {col}")


def main() -> int:
    parser = argparse.ArgumentParser(description="QA checks for SPY a20 features.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/processed/v1/SPY_features_a20.parquet"),
        help="Path to processed a20 feature parquet.",
    )
    args = parser.parse_args()

    if not args.path.exists():
        fail(f"File not found: {args.path}")

    df = pd.read_parquet(args.path)
    check_columns(df)
    check_nulls_infs(df)
    check_dates(df)
    check_value_ranges(df)

    print(f"✅ QA passed: {len(df)} rows, {len(df.columns)} columns at {args.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

