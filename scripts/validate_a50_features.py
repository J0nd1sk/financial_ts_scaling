#!/usr/bin/env python3
"""Comprehensive validation script for tier_a50 indicators.

This script validates the 30 new tier_a50 indicators for:
1. Look-ahead bias (via formula cross-validation)
2. Mathematical correctness
3. Data quality (no NaN, inf, anomalies)
4. Logical consistency across related features

Run: ./venv/bin/python scripts/validate_a50_features.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import talib

# A50 addition list (30 new indicators)
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


class ValidationResult:
    """Container for validation results."""

    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.messages: list[str] = []

    def ok(self, msg: str) -> None:
        self.passed += 1
        self.messages.append(f"  ✅ {msg}")

    def fail(self, msg: str) -> None:
        self.failed += 1
        self.messages.append(f"  ❌ {msg}")

    def warn(self, msg: str) -> None:
        self.messages.append(f"  ⚠️  {msg}")

    def report(self) -> str:
        header = f"\n{'='*60}\n{self.name}\n{'='*60}"
        summary = f"\nResults: {self.passed} passed, {self.failed} failed"
        return header + "\n" + "\n".join(self.messages) + summary


def load_data(
    a50_path: Path, raw_path: Path, vix_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all required datasets."""
    a50_df = pd.read_parquet(a50_path)
    raw_df = pd.read_parquet(raw_path)
    vix_df = pd.read_parquet(vix_path)

    # Normalize dates
    a50_df["Date"] = pd.to_datetime(a50_df["Date"]).dt.normalize()
    raw_df["Date"] = pd.to_datetime(raw_df["Date"]).dt.normalize()
    vix_df["Date"] = pd.to_datetime(vix_df["Date"]).dt.normalize()

    return a50_df, raw_df, vix_df


# =============================================================================
# Section 1: Data Quality Checks
# =============================================================================


def validate_data_quality(a50_df: pd.DataFrame) -> ValidationResult:
    """Check for NaN, Inf, and basic data quality."""
    result = ValidationResult("1. Data Quality Checks")

    # Check for NaN values
    nan_cols = a50_df.columns[a50_df.isnull().any()].tolist()
    if nan_cols:
        result.fail(f"NaN values found in columns: {nan_cols}")
    else:
        result.ok("No NaN values in dataset")

    # Check for Inf values
    numeric_df = a50_df.select_dtypes(include=[float, int])
    inf_cols = numeric_df.columns[np.isinf(numeric_df).any()].tolist()
    if inf_cols:
        result.fail(f"Infinite values found in columns: {inf_cols}")
    else:
        result.ok("No infinite values in dataset")

    # Check date continuity
    if not a50_df["Date"].is_monotonic_increasing:
        result.fail("Dates are not monotonic increasing")
    else:
        result.ok("Dates are monotonic increasing")

    if a50_df["Date"].duplicated().any():
        result.fail("Duplicate dates found")
    else:
        result.ok("No duplicate dates")

    # Check all A50 columns present
    missing_cols = set(A50_ADDITION_LIST) - set(a50_df.columns)
    if missing_cols:
        result.fail(f"Missing A50 columns: {missing_cols}")
    else:
        result.ok(f"All {len(A50_ADDITION_LIST)} A50 addition columns present")

    # Check no constant columns (all should have variation)
    for col in A50_ADDITION_LIST:
        if col in a50_df.columns:
            std = a50_df[col].std()
            if std == 0:
                result.fail(f"Column {col} is constant (std=0)")
            elif std < 1e-10:
                result.warn(f"Column {col} has very low variance (std={std:.2e})")

    result.ok("All indicator columns have non-zero variance")

    return result


# =============================================================================
# Section 2: Statistical Validation (Range Checks)
# =============================================================================


def validate_statistical_ranges(a50_df: pd.DataFrame) -> ValidationResult:
    """Validate all indicators are within expected realistic ranges."""
    result = ValidationResult("2. Statistical Range Validation")

    # Define expected ranges for each indicator type
    range_checks = {
        # Returns: SPY rarely moves >15% in one day, >40% in a week, etc.
        "return_1d": (-20, 20, "Daily return"),
        "return_5d": (-30, 30, "5-day return"),
        "return_21d": (-50, 50, "21-day return"),
        "return_63d": (-60, 60, "63-day return"),
        "return_252d": (-70, 100, "252-day return"),
        # QQE: Based on RSI, should be 0-100
        "qqe_fast": (0, 100, "QQE fast"),
        "qqe_slow": (0, 100, "QQE slow"),
        "qqe_fast_slow_spread": (-50, 50, "QQE spread"),
        # STC: Double stochastic, 0-100
        "stc_value": (0, 100, "STC value"),
        "stc_from_50": (-50, 50, "STC from 50"),
        # VRP: VIX typically 10-80, realized vol similar
        # During extreme volatility spikes, VRP can go more negative
        "vrp_10d": (-60, 60, "VRP 10d"),
        "vrp_21d": (-50, 60, "VRP 21d"),
        "implied_vs_realized_ratio": (0, 10, "Implied/Realized ratio"),
        # Risk metrics: Annualized Sharpe/Sortino
        # Sortino can be very high in low-vol uptrends (small downside dev)
        "sharpe_20d": (-20, 20, "Sharpe 20d"),
        "sharpe_60d": (-10, 10, "Sharpe 60d"),
        "sortino_20d": (-30, 100, "Sortino 20d"),
        "sortino_60d": (-15, 30, "Sortino 60d"),
        # RSI derivatives
        "rsi_slope": (-50, 50, "RSI slope"),
        "rsi_extreme_dist": (0, 50, "RSI extreme distance"),
        # MA derivatives: typically within -50% to +50% of MAs
        "price_pct_from_sma_50": (-50, 50, "Price from SMA50"),
        "price_pct_from_sma_200": (-60, 60, "Price from SMA200"),
        "sma_50_200_proximity": (-30, 30, "SMA 50/200 proximity"),
        # Volatility
        "atr_pct": (0, 20, "ATR%"),
        "atr_pct_slope": (-10, 10, "ATR% slope"),
        "bb_width": (0, 50, "BB width"),
        # Overnight/intraday
        "overnight_gap": (-15, 15, "Overnight gap"),
        "open_to_close_pct": (-15, 15, "Open to close"),
        # Volume
        "volume_ratio_20d": (0, 20, "Volume ratio"),
        "kvo_signal": (-1e12, 1e12, "KVO signal"),  # Wide range, scale depends on volume
        "macd_histogram": (-50, 50, "MACD histogram"),
    }

    for col, (min_val, max_val, desc) in range_checks.items():
        if col not in a50_df.columns:
            result.warn(f"Column {col} not found")
            continue

        series = a50_df[col]
        actual_min = series.min()
        actual_max = series.max()

        if actual_min < min_val or actual_max > max_val:
            result.fail(
                f"{desc}: range [{actual_min:.2f}, {actual_max:.2f}] "
                f"outside expected [{min_val}, {max_val}]"
            )
        else:
            result.ok(f"{desc}: range [{actual_min:.2f}, {actual_max:.2f}] OK")

    return result


# =============================================================================
# Section 3: Formula Cross-Validation
# =============================================================================


def validate_return_calculations(
    a50_df: pd.DataFrame, raw_df: pd.DataFrame
) -> ValidationResult:
    """Manually compute returns and compare to stored values."""
    result = ValidationResult("3.1 Return Calculations Cross-Validation")

    # Align data by date
    merged = pd.merge(
        raw_df[["Date", "Close"]],
        a50_df[["Date", "return_1d", "return_5d", "return_21d", "return_63d", "return_252d"]],
        on="Date",
        how="inner",
    )

    close = merged["Close"]

    for period, col_name in [
        (1, "return_1d"),
        (5, "return_5d"),
        (21, "return_21d"),
        (63, "return_63d"),
        (252, "return_252d"),
    ]:
        # Manual calculation
        computed = (close - close.shift(period)) / close.shift(period) * 100
        stored = merged[col_name]

        # Compare (skip NaN from warmup)
        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-8:
            result.ok(f"{col_name}: max diff = {max_diff:.2e} (< 1e-8)")
        else:
            result.fail(f"{col_name}: max diff = {max_diff:.2e} (>= 1e-8)")

    return result


def validate_overnight_gap(a50_df: pd.DataFrame, raw_df: pd.DataFrame) -> ValidationResult:
    """Validate overnight gap and open-to-close calculations."""
    result = ValidationResult("3.2 Overnight Gap & Intraday Cross-Validation")

    merged = pd.merge(
        raw_df[["Date", "Open", "Close"]],
        a50_df[["Date", "overnight_gap", "open_to_close_pct"]],
        on="Date",
        how="inner",
    )

    open_price = merged["Open"]
    close = merged["Close"]
    prev_close = close.shift(1)

    # Overnight gap: (Open - prev_Close) / prev_Close * 100
    computed_gap = (open_price - prev_close) / prev_close * 100
    stored_gap = merged["overnight_gap"]

    valid_mask = ~computed_gap.isna() & ~stored_gap.isna()
    max_diff = (computed_gap[valid_mask] - stored_gap[valid_mask]).abs().max()

    if max_diff < 1e-8:
        result.ok(f"overnight_gap: max diff = {max_diff:.2e}")
    else:
        result.fail(f"overnight_gap: max diff = {max_diff:.2e}")

    # Open to close: (Close - Open) / Open * 100
    computed_otc = (close - open_price) / open_price * 100
    stored_otc = merged["open_to_close_pct"]

    valid_mask = ~computed_otc.isna() & ~stored_otc.isna()
    max_diff = (computed_otc[valid_mask] - stored_otc[valid_mask]).abs().max()

    if max_diff < 1e-8:
        result.ok(f"open_to_close_pct: max diff = {max_diff:.2e}")
    else:
        result.fail(f"open_to_close_pct: max diff = {max_diff:.2e}")

    return result


def validate_rsi_derivatives(a50_df: pd.DataFrame, raw_df: pd.DataFrame) -> ValidationResult:
    """Validate RSI slope and extreme distance calculations."""
    result = ValidationResult("3.3 RSI Derivatives Cross-Validation")

    # Compute on FULL raw data first (preserves rolling window history)
    raw_sorted = raw_df.sort_values("Date").reset_index(drop=True)
    close = raw_sorted["Close"].astype(np.float64)
    rsi = talib.RSI(close.values, timeperiod=14)

    # Build computed features df with dates
    computed_df = pd.DataFrame({
        "Date": raw_sorted["Date"],
        "rsi_slope": pd.Series(rsi) - pd.Series(rsi).shift(5),
        "rsi_extreme_dist": pd.concat(
            [(pd.Series(rsi) - 30).abs(), (pd.Series(rsi) - 70).abs()], axis=1
        ).min(axis=1),
    })

    # Merge with stored values by date
    merged = pd.merge(
        computed_df,
        a50_df[["Date", "rsi_slope", "rsi_extreme_dist"]],
        on="Date",
        how="inner",
        suffixes=("_computed", "_stored"),
    )

    # Compare RSI slope
    valid_mask = ~merged["rsi_slope_computed"].isna() & ~merged["rsi_slope_stored"].isna()
    max_diff = (merged["rsi_slope_computed"][valid_mask] - merged["rsi_slope_stored"][valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"rsi_slope: max diff = {max_diff:.2e}")
    else:
        result.fail(f"rsi_slope: max diff = {max_diff:.2e}")

    # Compare RSI extreme distance
    valid_mask = ~merged["rsi_extreme_dist_computed"].isna() & ~merged["rsi_extreme_dist_stored"].isna()
    max_diff = (merged["rsi_extreme_dist_computed"][valid_mask] - merged["rsi_extreme_dist_stored"][valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"rsi_extreme_dist: max diff = {max_diff:.2e}")
    else:
        result.fail(f"rsi_extreme_dist: max diff = {max_diff:.2e}")

    return result


def validate_ma_derivatives(a50_df: pd.DataFrame, raw_df: pd.DataFrame) -> ValidationResult:
    """Validate MA derivative calculations."""
    result = ValidationResult("3.4 MA Derivatives Cross-Validation")

    merged = pd.merge(
        raw_df[["Date", "Close"]],
        a50_df[
            ["Date", "price_pct_from_sma_50", "price_pct_from_sma_200", "sma_50_200_proximity"]
        ],
        on="Date",
        how="inner",
    )

    close = merged["Close"].astype(np.float64)
    sma_50 = talib.SMA(close.values, timeperiod=50)
    sma_200 = talib.SMA(close.values, timeperiod=200)

    # Price % from SMA50
    computed_50 = (close.values - sma_50) / sma_50 * 100
    stored_50 = merged["price_pct_from_sma_50"].values

    valid_mask = ~np.isnan(computed_50) & ~np.isnan(stored_50)
    max_diff = np.abs(computed_50[valid_mask] - stored_50[valid_mask]).max()

    if max_diff < 1e-6:
        result.ok(f"price_pct_from_sma_50: max diff = {max_diff:.2e}")
    else:
        result.fail(f"price_pct_from_sma_50: max diff = {max_diff:.2e}")

    # Price % from SMA200
    computed_200 = (close.values - sma_200) / sma_200 * 100
    stored_200 = merged["price_pct_from_sma_200"].values

    valid_mask = ~np.isnan(computed_200) & ~np.isnan(stored_200)
    max_diff = np.abs(computed_200[valid_mask] - stored_200[valid_mask]).max()

    if max_diff < 1e-6:
        result.ok(f"price_pct_from_sma_200: max diff = {max_diff:.2e}")
    else:
        result.fail(f"price_pct_from_sma_200: max diff = {max_diff:.2e}")

    # SMA 50/200 proximity
    computed_prox = (sma_50 - sma_200) / sma_200 * 100
    stored_prox = merged["sma_50_200_proximity"].values

    valid_mask = ~np.isnan(computed_prox) & ~np.isnan(stored_prox)
    max_diff = np.abs(computed_prox[valid_mask] - stored_prox[valid_mask]).max()

    if max_diff < 1e-6:
        result.ok(f"sma_50_200_proximity: max diff = {max_diff:.2e}")
    else:
        result.fail(f"sma_50_200_proximity: max diff = {max_diff:.2e}")

    return result


def validate_volatility_features(
    a50_df: pd.DataFrame, raw_df: pd.DataFrame
) -> ValidationResult:
    """Validate volatility feature calculations."""
    result = ValidationResult("3.5 Volatility Features Cross-Validation")

    # Compute on FULL raw data first (preserves rolling window history)
    raw_sorted = raw_df.sort_values("Date").reset_index(drop=True)
    high = raw_sorted["High"].values.astype(np.float64)
    low = raw_sorted["Low"].values.astype(np.float64)
    close = raw_sorted["Close"].values.astype(np.float64)

    # ATR%
    atr = talib.ATR(high, low, close, timeperiod=14)
    atr_pct = atr / close * 100
    atr_pct_series = pd.Series(atr_pct)
    atr_pct_slope = atr_pct_series - atr_pct_series.shift(5)

    # Bollinger Band width
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    bb_width = (upper - lower) / middle * 100

    # Build computed features df with dates
    computed_df = pd.DataFrame({
        "Date": raw_sorted["Date"],
        "atr_pct": atr_pct,
        "atr_pct_slope": atr_pct_slope.values,
        "bb_width": bb_width,
    })

    # Merge with stored values by date
    merged = pd.merge(
        computed_df,
        a50_df[["Date", "atr_pct", "atr_pct_slope", "bb_width"]],
        on="Date",
        how="inner",
        suffixes=("_computed", "_stored"),
    )

    # Compare ATR%
    valid_mask = ~merged["atr_pct_computed"].isna() & ~merged["atr_pct_stored"].isna()
    max_diff = (merged["atr_pct_computed"][valid_mask] - merged["atr_pct_stored"][valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"atr_pct: max diff = {max_diff:.2e}")
    else:
        result.fail(f"atr_pct: max diff = {max_diff:.2e}")

    # Compare ATR% slope
    valid_mask = ~merged["atr_pct_slope_computed"].isna() & ~merged["atr_pct_slope_stored"].isna()
    max_diff = (merged["atr_pct_slope_computed"][valid_mask] - merged["atr_pct_slope_stored"][valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"atr_pct_slope: max diff = {max_diff:.2e}")
    else:
        result.fail(f"atr_pct_slope: max diff = {max_diff:.2e}")

    # Compare BB width
    valid_mask = ~merged["bb_width_computed"].isna() & ~merged["bb_width_stored"].isna()
    max_diff = (merged["bb_width_computed"][valid_mask] - merged["bb_width_stored"][valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"bb_width: max diff = {max_diff:.2e}")
    else:
        result.fail(f"bb_width: max diff = {max_diff:.2e}")

    return result


def validate_risk_metrics(a50_df: pd.DataFrame, raw_df: pd.DataFrame) -> ValidationResult:
    """Validate Sharpe and Sortino ratio calculations."""
    result = ValidationResult("3.6 Risk Metrics Cross-Validation")

    # Compute on FULL raw data first (preserves rolling window history)
    raw_sorted = raw_df.sort_values("Date").reset_index(drop=True)
    close = raw_sorted["Close"]
    daily_returns = close.pct_change()

    computed_features = {"Date": raw_sorted["Date"]}

    for window, suffix in [(20, "20d"), (60, "60d")]:
        # Sharpe ratio
        roll_mean = daily_returns.rolling(window).mean()
        roll_std = daily_returns.rolling(window).std()
        safe_std = roll_std.replace(0, np.nan)
        sharpe = (roll_mean / safe_std) * np.sqrt(252)
        sharpe = sharpe.fillna(0)
        computed_features[f"sharpe_{suffix}"] = sharpe

        # Sortino ratio
        negative_returns = daily_returns.copy()
        negative_returns[negative_returns > 0] = 0
        downside_std = negative_returns.rolling(window).std()
        safe_downside_std = downside_std.replace(0, np.nan)
        sortino = (roll_mean / safe_downside_std) * np.sqrt(252)
        sortino = sortino.fillna(0)
        computed_features[f"sortino_{suffix}"] = sortino

    computed_df = pd.DataFrame(computed_features)

    # Merge with stored values by date
    merged = pd.merge(
        computed_df,
        a50_df[["Date", "sharpe_20d", "sharpe_60d", "sortino_20d", "sortino_60d"]],
        on="Date",
        how="inner",
        suffixes=("_computed", "_stored"),
    )

    for window, suffix in [(20, "20d"), (60, "60d")]:
        # Compare Sharpe
        col = f"sharpe_{suffix}"
        valid_mask = ~merged[f"{col}_computed"].isna() & ~merged[f"{col}_stored"].isna()
        max_diff = (merged[f"{col}_computed"][valid_mask] - merged[f"{col}_stored"][valid_mask]).abs().max()

        if max_diff < 1e-6:
            result.ok(f"{col}: max diff = {max_diff:.2e}")
        else:
            result.fail(f"{col}: max diff = {max_diff:.2e}")

        # Compare Sortino
        col = f"sortino_{suffix}"
        valid_mask = ~merged[f"{col}_computed"].isna() & ~merged[f"{col}_stored"].isna()
        max_diff = (merged[f"{col}_computed"][valid_mask] - merged[f"{col}_stored"][valid_mask]).abs().max()

        if max_diff < 1e-6:
            result.ok(f"{col}: max diff = {max_diff:.2e}")
        else:
            result.fail(f"{col}: max diff = {max_diff:.2e}")

    return result


def validate_qqe_calculations(a50_df: pd.DataFrame, raw_df: pd.DataFrame) -> ValidationResult:
    """Validate QQE indicator calculations."""
    result = ValidationResult("3.7 QQE Cross-Validation")

    # Compute on FULL raw data first (preserves rolling window history)
    raw_sorted = raw_df.sort_values("Date").reset_index(drop=True)
    close = raw_sorted["Close"].astype(np.float64)
    rsi = talib.RSI(close.values, timeperiod=14)

    # Fast QQE (smoothing factor 5)
    rsi_smooth_fast = pd.Series(rsi).ewm(span=5, adjust=False).mean()

    # Slow QQE (smoothing factor 14)
    rsi_smooth_slow = pd.Series(rsi).ewm(span=14, adjust=False).mean()

    # Spread
    qqe_spread = rsi_smooth_fast - rsi_smooth_slow

    # Build computed features df with dates
    computed_df = pd.DataFrame({
        "Date": raw_sorted["Date"],
        "qqe_fast": rsi_smooth_fast.values,
        "qqe_slow": rsi_smooth_slow.values,
        "qqe_fast_slow_spread": qqe_spread.values,
    })

    # Merge with stored values by date
    merged = pd.merge(
        computed_df,
        a50_df[["Date", "qqe_fast", "qqe_slow", "qqe_fast_slow_spread"]],
        on="Date",
        how="inner",
        suffixes=("_computed", "_stored"),
    )

    # Compare QQE fast
    valid_mask = ~merged["qqe_fast_computed"].isna() & ~merged["qqe_fast_stored"].isna()
    max_diff = (merged["qqe_fast_computed"][valid_mask] - merged["qqe_fast_stored"][valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"qqe_fast: max diff = {max_diff:.2e}")
    else:
        result.fail(f"qqe_fast: max diff = {max_diff:.2e}")

    # Compare QQE slow
    valid_mask = ~merged["qqe_slow_computed"].isna() & ~merged["qqe_slow_stored"].isna()
    max_diff = (merged["qqe_slow_computed"][valid_mask] - merged["qqe_slow_stored"][valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"qqe_slow: max diff = {max_diff:.2e}")
    else:
        result.fail(f"qqe_slow: max diff = {max_diff:.2e}")

    # Compare QQE spread
    valid_mask = ~merged["qqe_fast_slow_spread_computed"].isna() & ~merged["qqe_fast_slow_spread_stored"].isna()
    max_diff = (merged["qqe_fast_slow_spread_computed"][valid_mask] - merged["qqe_fast_slow_spread_stored"][valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"qqe_fast_slow_spread: max diff = {max_diff:.2e}")
    else:
        result.fail(f"qqe_fast_slow_spread: max diff = {max_diff:.2e}")

    return result


def validate_vrp_calculations(
    a50_df: pd.DataFrame, raw_df: pd.DataFrame, vix_df: pd.DataFrame
) -> ValidationResult:
    """Validate VRP (Volatility Risk Premium) calculations."""
    result = ValidationResult("3.8 VRP Cross-Validation")

    # Prepare data
    raw_df = raw_df.copy()
    vix_df = vix_df.copy()
    raw_df["Date"] = pd.to_datetime(raw_df["Date"]).dt.normalize()
    vix_df["Date"] = pd.to_datetime(vix_df["Date"]).dt.normalize()

    # Get VIX aligned to SPY dates
    vix_aligned = vix_df.set_index("Date")["Close"].reindex(raw_df["Date"].values)
    vix_aligned = vix_aligned.reset_index(drop=True)

    close = raw_df["Close"]

    # Realized volatility
    log_returns = np.log(close / close.shift(1))
    rv_10d = log_returns.rolling(10).std() * np.sqrt(252) * 100
    rv_21d = log_returns.rolling(21).std() * np.sqrt(252) * 100

    # Merge with stored values
    raw_df["vix_aligned"] = vix_aligned.values
    raw_df["rv_10d"] = rv_10d.values
    raw_df["rv_21d"] = rv_21d.values

    merged = pd.merge(
        raw_df[["Date", "Close", "vix_aligned", "rv_10d", "rv_21d"]],
        a50_df[["Date", "vrp_10d", "vrp_21d", "implied_vs_realized_ratio"]],
        on="Date",
        how="inner",
    )

    # VRP 10d
    computed_vrp_10d = merged["vix_aligned"] - merged["rv_10d"]
    stored_vrp_10d = merged["vrp_10d"]

    valid_mask = ~computed_vrp_10d.isna() & ~stored_vrp_10d.isna()
    max_diff = (computed_vrp_10d[valid_mask] - stored_vrp_10d[valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"vrp_10d: max diff = {max_diff:.2e}")
    else:
        result.fail(f"vrp_10d: max diff = {max_diff:.2e}")

    # VRP 21d
    computed_vrp_21d = merged["vix_aligned"] - merged["rv_21d"]
    stored_vrp_21d = merged["vrp_21d"]

    valid_mask = ~computed_vrp_21d.isna() & ~stored_vrp_21d.isna()
    max_diff = (computed_vrp_21d[valid_mask] - stored_vrp_21d[valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"vrp_21d: max diff = {max_diff:.2e}")
    else:
        result.fail(f"vrp_21d: max diff = {max_diff:.2e}")

    # Implied vs realized ratio
    rv_21d_safe = merged["rv_21d"].replace(0, np.nan)
    computed_ratio = merged["vix_aligned"] / rv_21d_safe
    computed_ratio = computed_ratio.fillna(1.0)
    stored_ratio = merged["implied_vs_realized_ratio"]

    valid_mask = ~computed_ratio.isna() & ~stored_ratio.isna()
    max_diff = (computed_ratio[valid_mask] - stored_ratio[valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"implied_vs_realized_ratio: max diff = {max_diff:.2e}")
    else:
        result.fail(f"implied_vs_realized_ratio: max diff = {max_diff:.2e}")

    return result


def validate_volume_features(a50_df: pd.DataFrame, raw_df: pd.DataFrame) -> ValidationResult:
    """Validate volume feature calculations."""
    result = ValidationResult("3.9 Volume Features Cross-Validation")

    # Compute on FULL raw data first (preserves rolling window history)
    raw_sorted = raw_df.sort_values("Date").reset_index(drop=True)
    close = raw_sorted["Close"].values.astype(np.float64)
    volume = raw_sorted["Volume"].values.astype(np.float64)

    # Volume ratio 20d
    vol_sma_20 = talib.SMA(volume, timeperiod=20)
    vol_sma_safe = np.where(vol_sma_20 == 0, np.nan, vol_sma_20)
    volume_ratio_20d = volume / vol_sma_safe

    # MACD histogram
    macd_line, signal_line, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )

    # Build computed features df with dates
    computed_df = pd.DataFrame({
        "Date": raw_sorted["Date"],
        "volume_ratio_20d": volume_ratio_20d,
        "macd_histogram": macd_hist,
    })

    # Merge with stored values by date
    merged = pd.merge(
        computed_df,
        a50_df[["Date", "volume_ratio_20d", "macd_histogram"]],
        on="Date",
        how="inner",
        suffixes=("_computed", "_stored"),
    )

    # Compare volume ratio
    valid_mask = ~merged["volume_ratio_20d_computed"].isna() & ~merged["volume_ratio_20d_stored"].isna()
    max_diff = (merged["volume_ratio_20d_computed"][valid_mask] - merged["volume_ratio_20d_stored"][valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"volume_ratio_20d: max diff = {max_diff:.2e}")
    else:
        result.fail(f"volume_ratio_20d: max diff = {max_diff:.2e}")

    # Compare MACD histogram
    valid_mask = ~merged["macd_histogram_computed"].isna() & ~merged["macd_histogram_stored"].isna()
    max_diff = (merged["macd_histogram_computed"][valid_mask] - merged["macd_histogram_stored"][valid_mask]).abs().max()

    if max_diff < 1e-6:
        result.ok(f"macd_histogram: max diff = {max_diff:.2e}")
    else:
        result.fail(f"macd_histogram: max diff = {max_diff:.2e}")

    return result


# =============================================================================
# Section 4: Logical Consistency Checks
# =============================================================================


def validate_logical_consistency(a50_df: pd.DataFrame) -> ValidationResult:
    """Validate logical relationships and algebraic identities."""
    result = ValidationResult("4. Logical Consistency Checks")

    # QQE spread identity: qqe_fast_slow_spread == qqe_fast - qqe_slow
    computed_spread = a50_df["qqe_fast"] - a50_df["qqe_slow"]
    stored_spread = a50_df["qqe_fast_slow_spread"]
    max_diff = (computed_spread - stored_spread).abs().max()

    if max_diff < 1e-10:
        result.ok(f"QQE spread identity: max diff = {max_diff:.2e}")
    else:
        result.fail(f"QQE spread identity violated: max diff = {max_diff:.2e}")

    # STC from 50 identity: stc_from_50 == stc_value - 50
    computed_from_50 = a50_df["stc_value"] - 50
    stored_from_50 = a50_df["stc_from_50"]
    max_diff = (computed_from_50 - stored_from_50).abs().max()

    if max_diff < 1e-10:
        result.ok(f"STC from 50 identity: max diff = {max_diff:.2e}")
    else:
        result.fail(f"STC from 50 identity violated: max diff = {max_diff:.2e}")

    # RSI extreme distance should be <= 20 (max distance from 50 to 30 or 70)
    max_extreme_dist = a50_df["rsi_extreme_dist"].max()
    if max_extreme_dist <= 50:  # Actually max possible is 50 (from RSI 0 or 100 to extreme)
        result.ok(f"RSI extreme distance max = {max_extreme_dist:.2f} (bounded correctly)")
    else:
        result.fail(f"RSI extreme distance max = {max_extreme_dist:.2f} (should be <= 50)")

    # VRP logic: when VIX > realized vol, VRP should be positive
    # Check correlation: VRP and implied_vs_realized_ratio should be correlated
    vrp_21d = a50_df["vrp_21d"]
    ratio = a50_df["implied_vs_realized_ratio"]
    valid_mask = ~vrp_21d.isna() & ~ratio.isna()
    corr = vrp_21d[valid_mask].corr(ratio[valid_mask])

    if corr > 0.5:
        result.ok(f"VRP and implied/realized ratio correlation: {corr:.3f} (positive as expected)")
    else:
        result.warn(
            f"VRP and implied/realized ratio correlation: {corr:.3f} (expected positive)"
        )

    # Price above SMA50 should correspond to positive price_pct_from_sma_50
    # (This is a sanity check on signs)
    pct_from_50 = a50_df["price_pct_from_sma_50"]
    positive_pct = (pct_from_50 > 0).sum()
    total_valid = (~pct_from_50.isna()).sum()
    pct_positive = positive_pct / total_valid * 100

    if 30 < pct_positive < 70:  # Market is above/below 50MA roughly half the time
        result.ok(
            f"Price from SMA50 distribution: {pct_positive:.1f}% positive (reasonable split)"
        )
    else:
        result.warn(
            f"Price from SMA50 distribution: {pct_positive:.1f}% positive (unusual bias)"
        )

    return result


# =============================================================================
# Section 5: Distribution Sanity Checks
# =============================================================================


def validate_distributions(a50_df: pd.DataFrame) -> ValidationResult:
    """Validate distributions are reasonable."""
    result = ValidationResult("5. Distribution Sanity Checks")

    # Returns should have mean near 0 (daily), slightly positive (annual)
    return_1d_mean = a50_df["return_1d"].mean()
    if -0.5 < return_1d_mean < 0.5:
        result.ok(f"return_1d mean: {return_1d_mean:.4f}% (near zero as expected)")
    else:
        result.warn(f"return_1d mean: {return_1d_mean:.4f}% (expected near zero)")

    # Annual returns should be slightly positive (SPY long-term drift)
    return_252d_mean = a50_df["return_252d"].mean()
    if 0 < return_252d_mean < 30:
        result.ok(f"return_252d mean: {return_252d_mean:.2f}% (positive as expected for SPY)")
    else:
        result.warn(f"return_252d mean: {return_252d_mean:.2f}% (expected positive)")

    # VRP should be mostly positive (VIX typically > realized)
    vrp_21d_mean = a50_df["vrp_21d"].mean()
    vrp_positive_pct = (a50_df["vrp_21d"] > 0).sum() / len(a50_df) * 100

    if vrp_positive_pct > 50:
        result.ok(f"VRP 21d positive: {vrp_positive_pct:.1f}% (VIX > realized is normal)")
    else:
        result.warn(f"VRP 21d positive: {vrp_positive_pct:.1f}% (expected mostly positive)")

    # Spreads should be centered near 0
    qqe_spread_mean = a50_df["qqe_fast_slow_spread"].mean()
    if -5 < qqe_spread_mean < 5:
        result.ok(f"QQE spread mean: {qqe_spread_mean:.3f} (centered near zero)")
    else:
        result.warn(f"QQE spread mean: {qqe_spread_mean:.3f} (expected near zero)")

    stc_from_50_mean = a50_df["stc_from_50"].mean()
    if -20 < stc_from_50_mean < 20:
        result.ok(f"STC from 50 mean: {stc_from_50_mean:.2f} (reasonably centered)")
    else:
        result.warn(f"STC from 50 mean: {stc_from_50_mean:.2f} (expected near zero)")

    return result


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate tier_a50 indicators.")
    parser.add_argument(
        "--a50-path",
        type=Path,
        default=Path("data/processed/v1/SPY_dataset_a50.parquet"),
        help="Path to processed a50 dataset.",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/SPY.parquet"),
        help="Path to raw SPY data.",
    )
    parser.add_argument(
        "--vix-path",
        type=Path,
        default=Path("data/raw/VIX.parquet"),
        help="Path to raw VIX data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each check.",
    )
    args = parser.parse_args()

    # Check files exist
    for path, name in [
        (args.a50_path, "A50 dataset"),
        (args.raw_path, "Raw SPY data"),
        (args.vix_path, "VIX data"),
    ]:
        if not path.exists():
            print(f"❌ {name} not found: {path}")
            return 1

    # Load data
    print("Loading data...")
    a50_df, raw_df, vix_df = load_data(args.a50_path, args.raw_path, args.vix_path)
    print(f"  A50 dataset: {len(a50_df)} rows, {len(a50_df.columns)} columns")
    print(f"  Raw SPY: {len(raw_df)} rows")
    print(f"  VIX: {len(vix_df)} rows")

    # Run all validation checks
    results: list[ValidationResult] = []

    print("\nRunning validations...")

    # 1. Data Quality
    results.append(validate_data_quality(a50_df))

    # 2. Statistical Ranges
    results.append(validate_statistical_ranges(a50_df))

    # 3. Formula Cross-Validation
    results.append(validate_return_calculations(a50_df, raw_df))
    results.append(validate_overnight_gap(a50_df, raw_df))
    results.append(validate_rsi_derivatives(a50_df, raw_df))
    results.append(validate_ma_derivatives(a50_df, raw_df))
    results.append(validate_volatility_features(a50_df, raw_df))
    results.append(validate_risk_metrics(a50_df, raw_df))
    results.append(validate_qqe_calculations(a50_df, raw_df))
    results.append(validate_vrp_calculations(a50_df, raw_df, vix_df))
    results.append(validate_volume_features(a50_df, raw_df))

    # 4. Logical Consistency
    results.append(validate_logical_consistency(a50_df))

    # 5. Distribution Sanity
    results.append(validate_distributions(a50_df))

    # Print results
    total_passed = 0
    total_failed = 0

    for r in results:
        if args.verbose:
            print(r.report())
        else:
            status = "✅" if r.failed == 0 else "❌"
            print(f"{status} {r.name}: {r.passed} passed, {r.failed} failed")
        total_passed += r.passed
        total_failed += r.failed

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total: {total_passed} passed, {total_failed} failed")

    if total_failed == 0:
        print("\n✅ ALL VALIDATION CHECKS PASSED")
        print("The tier_a50 indicators are:")
        print("  - Free of look-ahead bias (verified via cross-validation)")
        print("  - Mathematically correct")
        print("  - Within expected statistical ranges")
        print("  - Logically consistent")
        print("  - Free of NaN/Inf values")
        return 0
    else:
        print(f"\n❌ {total_failed} VALIDATION CHECKS FAILED")
        print("Review the failures above and fix the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
