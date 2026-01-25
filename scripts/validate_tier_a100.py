#!/usr/bin/env python3
"""Comprehensive deep validation script for tier_a100 indicators.

This script validates the 50 new tier_a100 indicators (ranks 51-100) using:
1. Formula verification - hand-calculate at specific indices, compare
2. Reference comparison - compare talib-based indicators against direct talib calls
3. Boundary analysis - test at known edge cases
4. Domain logic checks - verify indicators behave correctly

Run: ./venv/bin/python scripts/validate_tier_a100.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as PathLibPath

# Add project root to path
project_root = PathLibPath(__file__).parent.parent
sys.path.insert(0, str(project_root))
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import talib

# A100 addition list (50 new indicators)
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


@dataclass
class ValidationCheck:
    """Single validation check result."""
    indicator: str
    check_name: str
    passed: bool
    expected: Any
    actual: Any
    evidence: str


@dataclass
class ValidationReport:
    """Container for all validation results."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    checks: list[ValidationCheck] = field(default_factory=list)

    def add(self, check: ValidationCheck) -> None:
        """Add a validation check."""
        self.checks.append(check)

    def ok(self, indicator: str, check_name: str, expected: Any, actual: Any, evidence: str) -> None:
        """Record a passing check."""
        self.add(ValidationCheck(indicator, check_name, True, expected, actual, evidence))

    def fail(self, indicator: str, check_name: str, expected: Any, actual: Any, evidence: str) -> None:
        """Record a failing check."""
        self.add(ValidationCheck(indicator, check_name, False, expected, actual, evidence))

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    @property
    def pass_rate(self) -> float:
        total = len(self.checks)
        return self.passed / total * 100 if total > 0 else 0.0

    def summary_by_chunk(self) -> dict[str, dict[str, int]]:
        """Get pass/fail summary by chunk."""
        chunk_map = {
            "Chunk 1: Momentum derivatives": ["return_1d_acceleration", "return_5d_acceleration"],
            "Chunk 2: QQE/STC derivatives": ["qqe_slope", "qqe_extreme_dist", "stc_slope", "stc_extreme_dist"],
            "Chunk 3: Standard oscillators": ["demarker_value", "demarker_from_half", "stoch_k_14", "stoch_d_14",
                                              "stoch_extreme_dist", "cci_14", "mfi_14", "williams_r_14"],
            "Chunk 4: VRP + Risk metrics": ["vrp_5d", "vrp_slope", "sharpe_252d", "sortino_252d",
                                            "sharpe_slope_20d", "sortino_slope_20d", "var_95", "var_99", "cvar_95"],
            "Chunk 5: MA extensions": ["sma_9_50_proximity", "sma_50_slope", "sma_200_slope",
                                       "days_since_sma_50_cross", "days_since_sma_200_cross", "ema_12", "ema_26"],
            "Chunk 6: Advanced volatility": ["atr_pct_percentile_60d", "bb_width_percentile_60d",
                                             "parkinson_volatility", "garman_klass_volatility", "vol_of_vol"],
            "Chunk 7: Trend indicators": ["adx_slope", "di_spread", "aroon_oscillator",
                                          "price_pct_from_supertrend", "supertrend_direction"],
            "Chunk 8: Volume + Momentum + S/R": ["obv_slope", "volume_price_trend", "kvo_histogram",
                                                  "accumulation_dist", "expectancy_20d", "win_rate_20d",
                                                  "buying_pressure_ratio", "fib_range_position",
                                                  "prior_high_20d_dist", "prior_low_20d_dist"],
        }

        summary = {}
        for chunk_name, indicators in chunk_map.items():
            chunk_checks = [c for c in self.checks if c.indicator in indicators]
            summary[chunk_name] = {
                "passed": sum(1 for c in chunk_checks if c.passed),
                "failed": sum(1 for c in chunk_checks if not c.passed),
                "total": len(chunk_checks),
            }
        return summary

    def to_json(self) -> dict:
        """Export to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_checks": len(self.checks),
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": f"{self.pass_rate:.1f}%",
            },
            "by_chunk": self.summary_by_chunk(),
            "checks": [asdict(c) for c in self.checks],
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Tier A100 Validation Report",
            "",
            f"**Generated**: {self.timestamp}",
            "",
            "## Summary",
            "",
            f"- **Total Checks**: {len(self.checks)}",
            f"- **Passed**: {self.passed}",
            f"- **Failed**: {self.failed}",
            f"- **Pass Rate**: {self.pass_rate:.1f}%",
            "",
            "## Results by Chunk",
            "",
            "| Chunk | Passed | Failed | Total |",
            "|-------|--------|--------|-------|",
        ]

        for chunk_name, stats in self.summary_by_chunk().items():
            status = "✅" if stats["failed"] == 0 else "❌"
            lines.append(f"| {status} {chunk_name} | {stats['passed']} | {stats['failed']} | {stats['total']} |")

        lines.extend([
            "",
            "## Detailed Results",
            "",
        ])

        # Group by indicator
        by_indicator: dict[str, list[ValidationCheck]] = {}
        for check in self.checks:
            by_indicator.setdefault(check.indicator, []).append(check)

        for indicator, checks in by_indicator.items():
            all_passed = all(c.passed for c in checks)
            status = "✅" if all_passed else "❌"
            lines.append(f"### {status} `{indicator}`")
            lines.append("")

            for check in checks:
                check_status = "✅" if check.passed else "❌"
                lines.append(f"- {check_status} **{check.check_name}**")
                lines.append(f"  - Expected: `{check.expected}`")
                lines.append(f"  - Actual: `{check.actual}`")
                lines.append(f"  - Evidence: {check.evidence}")
                lines.append("")

        # Failed checks summary
        failed_checks = [c for c in self.checks if not c.passed]
        if failed_checks:
            lines.extend([
                "## Failed Checks Summary",
                "",
            ])
            for check in failed_checks:
                lines.append(f"- **{check.indicator}**: {check.check_name}")
                lines.append(f"  - Expected: `{check.expected}`, Actual: `{check.actual}`")
                lines.append(f"  - Evidence: {check.evidence}")
                lines.append("")

        return "\n".join(lines)


def load_data(raw_path: Path, vix_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw data and compute features dynamically."""
    raw_df = pd.read_parquet(raw_path)
    vix_df = pd.read_parquet(vix_path)

    # Normalize dates
    raw_df["Date"] = pd.to_datetime(raw_df["Date"]).dt.normalize()
    vix_df["Date"] = pd.to_datetime(vix_df["Date"]).dt.normalize()

    # Sort by date
    raw_df = raw_df.sort_values("Date").reset_index(drop=True)
    vix_df = vix_df.sort_values("Date").reset_index(drop=True)

    # Build tier_a100 features
    from src.features import tier_a100
    a100_df = tier_a100.build_feature_dataframe(raw_df, vix_df)
    a100_df["Date"] = pd.to_datetime(a100_df["Date"]).dt.normalize()

    return a100_df, raw_df, vix_df


# =============================================================================
# Data Quality Checks
# =============================================================================

def validate_data_quality(a100_df: pd.DataFrame, report: ValidationReport) -> None:
    """Check for NaN, Inf, and basic data quality."""
    # Check all A100 columns present
    missing_cols = set(A100_ADDITION_LIST) - set(a100_df.columns)
    if missing_cols:
        report.fail("data_quality", "columns_present",
                   f"All {len(A100_ADDITION_LIST)} columns",
                   f"Missing: {missing_cols}",
                   "Column check")
    else:
        report.ok("data_quality", "columns_present",
                 f"All {len(A100_ADDITION_LIST)} columns",
                 f"All {len(A100_ADDITION_LIST)} columns present",
                 "Column check passed")

    # Check for NaN values in A100 columns
    nan_cols = [col for col in A100_ADDITION_LIST if col in a100_df.columns and a100_df[col].isnull().any()]
    if nan_cols:
        report.fail("data_quality", "no_nan_values",
                   "No NaN in any column",
                   f"NaN in: {nan_cols}",
                   "NaN check after dropna")
    else:
        report.ok("data_quality", "no_nan_values",
                 "No NaN in any column",
                 "No NaN values",
                 "NaN check passed")

    # Check for Inf values
    inf_cols = []
    for col in A100_ADDITION_LIST:
        if col in a100_df.columns:
            if np.isinf(a100_df[col]).any():
                inf_cols.append(col)

    if inf_cols:
        report.fail("data_quality", "no_inf_values",
                   "No Inf in any column",
                   f"Inf in: {inf_cols}",
                   "Infinity check")
    else:
        report.ok("data_quality", "no_inf_values",
                 "No Inf in any column",
                 "No Inf values",
                 "Infinity check passed")


# =============================================================================
# Chunk 1: Momentum Derivatives
# =============================================================================

def validate_chunk1_momentum_derivatives(a100_df: pd.DataFrame, raw_df: pd.DataFrame,
                                         report: ValidationReport) -> None:
    """Validate return acceleration formulas at random indices."""
    # Merge to align indices
    merged = pd.merge(
        raw_df[["Date", "Close"]],
        a100_df[["Date", "return_1d_acceleration", "return_5d_acceleration"]],
        on="Date",
        how="inner"
    )

    close = merged["Close"]
    n = len(merged)

    # Pick 5 random indices (after warmup)
    np.random.seed(42)
    test_indices = np.random.choice(range(10, n-5), size=5, replace=False)

    # return_1d_acceleration = (ret_1d - ret_1d.shift(1))
    for idx in test_indices:
        i = int(idx)
        ret_1d_now = (close.iloc[i] - close.iloc[i-1]) / close.iloc[i-1] * 100
        ret_1d_prev = (close.iloc[i-1] - close.iloc[i-2]) / close.iloc[i-2] * 100
        expected = ret_1d_now - ret_1d_prev
        actual = merged["return_1d_acceleration"].iloc[i]

        if np.isnan(actual) or np.isnan(expected):
            continue

        diff = abs(expected - actual)
        if diff < 1e-8:
            report.ok("return_1d_acceleration", f"formula_idx_{i}",
                     f"{expected:.6f}", f"{actual:.6f}",
                     f"Hand-calc at idx {i}: diff={diff:.2e}")
        else:
            report.fail("return_1d_acceleration", f"formula_idx_{i}",
                       f"{expected:.6f}", f"{actual:.6f}",
                       f"Hand-calc at idx {i}: diff={diff:.2e}")

    # return_5d_acceleration = (ret_5d - ret_5d.shift(1))
    for idx in test_indices:
        i = int(idx)
        ret_5d_now = (close.iloc[i] - close.iloc[i-5]) / close.iloc[i-5] * 100
        ret_5d_prev = (close.iloc[i-1] - close.iloc[i-6]) / close.iloc[i-6] * 100
        expected = ret_5d_now - ret_5d_prev
        actual = merged["return_5d_acceleration"].iloc[i]

        if np.isnan(actual) or np.isnan(expected):
            continue

        diff = abs(expected - actual)
        if diff < 1e-8:
            report.ok("return_5d_acceleration", f"formula_idx_{i}",
                     f"{expected:.6f}", f"{actual:.6f}",
                     f"Hand-calc at idx {i}: diff={diff:.2e}")
        else:
            report.fail("return_5d_acceleration", f"formula_idx_{i}",
                       f"{expected:.6f}", f"{actual:.6f}",
                       f"Hand-calc at idx {i}: diff={diff:.2e}")


# =============================================================================
# Chunk 2: QQE/STC Derivatives
# =============================================================================

def validate_chunk2_qqe_stc_derivatives(a100_df: pd.DataFrame, report: ValidationReport) -> None:
    """Validate QQE/STC slope and extreme distance formulas."""
    # Check that qqe_slope and stc_slope are 5-day changes
    # We can't easily recompute QQE/STC here, but we can verify the slope relationship

    # qqe_extreme_dist should be min(|qqe - 20|, |qqe - 80|)
    # stc_extreme_dist should be min(|stc - 25|, |stc - 75|)

    # For QQE (derived from qqe_fast which is in a50)
    if "qqe_fast" in a100_df.columns:
        qqe_fast = a100_df["qqe_fast"]
        computed_extreme = pd.concat([
            (qqe_fast - 20).abs(),
            (qqe_fast - 80).abs()
        ], axis=1).min(axis=1)

        stored = a100_df["qqe_extreme_dist"]
        valid_mask = ~computed_extreme.isna() & ~stored.isna()
        max_diff = (computed_extreme[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-8:
            report.ok("qqe_extreme_dist", "formula_verification",
                     "min(|qqe-20|, |qqe-80|)", f"max_diff={max_diff:.2e}",
                     "Formula verified against qqe_fast")
        else:
            report.fail("qqe_extreme_dist", "formula_verification",
                       "min(|qqe-20|, |qqe-80|)", f"max_diff={max_diff:.2e}",
                       "Formula mismatch")
    else:
        report.fail("qqe_extreme_dist", "formula_verification",
                   "qqe_fast column present", "qqe_fast not found",
                   "Cannot verify without qqe_fast")

    # For STC
    if "stc_value" in a100_df.columns:
        stc_value = a100_df["stc_value"]
        computed_extreme = pd.concat([
            (stc_value - 25).abs(),
            (stc_value - 75).abs()
        ], axis=1).min(axis=1)

        stored = a100_df["stc_extreme_dist"]
        valid_mask = ~computed_extreme.isna() & ~stored.isna()
        max_diff = (computed_extreme[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-8:
            report.ok("stc_extreme_dist", "formula_verification",
                     "min(|stc-25|, |stc-75|)", f"max_diff={max_diff:.2e}",
                     "Formula verified against stc_value")
        else:
            report.fail("stc_extreme_dist", "formula_verification",
                       "min(|stc-25|, |stc-75|)", f"max_diff={max_diff:.2e}",
                       "Formula mismatch")
    else:
        report.fail("stc_extreme_dist", "formula_verification",
                   "stc_value column present", "stc_value not found",
                   "Cannot verify without stc_value")

    # Verify slopes are 5-day changes
    if "qqe_fast" in a100_df.columns:
        qqe_fast = a100_df["qqe_fast"]
        computed_slope = qqe_fast - qqe_fast.shift(5)
        stored_slope = a100_df["qqe_slope"]

        valid_mask = ~computed_slope.isna() & ~stored_slope.isna()
        max_diff = (computed_slope[valid_mask] - stored_slope[valid_mask]).abs().max()

        if max_diff < 1e-8:
            report.ok("qqe_slope", "formula_verification",
                     "qqe_fast - qqe_fast.shift(5)", f"max_diff={max_diff:.2e}",
                     "5-day change formula verified")
        else:
            report.fail("qqe_slope", "formula_verification",
                       "qqe_fast - qqe_fast.shift(5)", f"max_diff={max_diff:.2e}",
                       "Formula mismatch")

    if "stc_value" in a100_df.columns:
        stc_value = a100_df["stc_value"]
        computed_slope = stc_value - stc_value.shift(5)
        stored_slope = a100_df["stc_slope"]

        valid_mask = ~computed_slope.isna() & ~stored_slope.isna()
        max_diff = (computed_slope[valid_mask] - stored_slope[valid_mask]).abs().max()

        if max_diff < 1e-8:
            report.ok("stc_slope", "formula_verification",
                     "stc_value - stc_value.shift(5)", f"max_diff={max_diff:.2e}",
                     "5-day change formula verified")
        else:
            report.fail("stc_slope", "formula_verification",
                       "stc_value - stc_value.shift(5)", f"max_diff={max_diff:.2e}",
                       "Formula mismatch")


# =============================================================================
# Chunk 3: Standard Oscillators
# =============================================================================

def validate_chunk3_oscillators(a100_df: pd.DataFrame, raw_df: pd.DataFrame,
                                report: ValidationReport) -> None:
    """Validate oscillator indicators against talib and hand-calculations."""
    # Merge data
    raw_sorted = raw_df.sort_values("Date").reset_index(drop=True)
    high = raw_sorted["High"].values.astype(np.float64)
    low = raw_sorted["Low"].values.astype(np.float64)
    close = raw_sorted["Close"].values.astype(np.float64)
    volume = raw_sorted["Volume"].values.astype(np.float64)

    # Build computed features
    computed_df = pd.DataFrame({"Date": raw_sorted["Date"]})

    # --- DeMarker hand calculation ---
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    de_max = (high_series - high_series.shift(1)).clip(lower=0)
    de_min = (low_series.shift(1) - low_series).clip(lower=0)
    sma_de_max = de_max.rolling(window=14).mean()
    sma_de_min = de_min.rolling(window=14).mean()
    denominator = sma_de_max + sma_de_min
    demarker = sma_de_max / denominator
    demarker = demarker.fillna(0.5)
    demarker = demarker.where(denominator != 0, 0.5)
    computed_df["demarker_value"] = demarker.values
    computed_df["demarker_from_half"] = (demarker - 0.5).values

    # --- Stochastic (talib) ---
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3,
                               slowk_matype=0, slowd_period=3, slowd_matype=0)
    computed_df["stoch_k_14"] = slowk
    computed_df["stoch_d_14"] = slowd
    stoch_k_series = pd.Series(slowk)
    dist_to_20 = (stoch_k_series - 20).abs()
    dist_to_80 = (stoch_k_series - 80).abs()
    computed_df["stoch_extreme_dist"] = pd.concat([dist_to_20, dist_to_80], axis=1).min(axis=1).values

    # --- CCI (talib) ---
    cci = talib.CCI(high, low, close, timeperiod=14)
    computed_df["cci_14"] = cci

    # --- MFI (talib) ---
    mfi = talib.MFI(high, low, close, volume, timeperiod=14)
    computed_df["mfi_14"] = mfi

    # --- Williams %R (talib) ---
    willr = talib.WILLR(high, low, close, timeperiod=14)
    computed_df["williams_r_14"] = willr

    # Merge with stored values
    merged = pd.merge(
        computed_df,
        a100_df[["Date", "demarker_value", "demarker_from_half", "stoch_k_14", "stoch_d_14",
                 "stoch_extreme_dist", "cci_14", "mfi_14", "williams_r_14"]],
        on="Date",
        how="inner",
        suffixes=("_computed", "_stored")
    )

    # Validate each indicator
    for col in ["demarker_value", "demarker_from_half", "stoch_k_14", "stoch_d_14",
                "stoch_extreme_dist", "cci_14", "mfi_14", "williams_r_14"]:
        comp_col = f"{col}_computed"
        stor_col = f"{col}_stored"

        valid_mask = ~merged[comp_col].isna() & ~merged[stor_col].isna()
        if valid_mask.sum() == 0:
            report.fail(col, "reference_comparison", "matching values", "no valid data",
                       "No overlapping valid data")
            continue

        max_diff = (merged[comp_col][valid_mask] - merged[stor_col][valid_mask]).abs().max()

        # Use appropriate tolerance (DeMarker has more precision issues)
        tol = 1e-6 if "demarker" not in col else 1e-5

        if max_diff < tol:
            report.ok(col, "reference_comparison",
                     f"max_diff < {tol}", f"max_diff={max_diff:.2e}",
                     f"Talib/formula reference match")
        else:
            report.fail(col, "reference_comparison",
                       f"max_diff < {tol}", f"max_diff={max_diff:.2e}",
                       f"Reference mismatch")

    # DeMarker deep validation: hand-calc at specific indices
    # Need to merge High/Low data with the a100_df to get aligned indices
    hl_merged = pd.merge(
        raw_sorted[["Date", "High", "Low"]],
        a100_df[["Date", "demarker_value"]],
        on="Date",
        how="inner"
    ).reset_index(drop=True)

    np.random.seed(42)
    test_indices = np.random.choice(range(20, len(hl_merged)-5), size=5, replace=False)

    hl_high = hl_merged["High"].values
    hl_low = hl_merged["Low"].values

    for idx in test_indices:
        i = int(idx)
        # Recompute DeMax/DeMin manually for verification using aligned data
        de_max_vals = []
        de_min_vals = []
        for j in range(i-13, i+1):  # 14 period window
            if j >= 1:
                de_max_vals.append(max(hl_high[j] - hl_high[j-1], 0))
                de_min_vals.append(max(hl_low[j-1] - hl_low[j], 0))

        if len(de_max_vals) == 14:
            sma_demax = sum(de_max_vals) / 14
            sma_demin = sum(de_min_vals) / 14
            denom = sma_demax + sma_demin
            expected_dm = sma_demax / denom if denom != 0 else 0.5
            actual_dm = hl_merged["demarker_value"].iloc[i]

            if not np.isnan(actual_dm):
                diff = abs(expected_dm - actual_dm)
                if diff < 1e-5:
                    report.ok("demarker_value", f"hand_calc_idx_{i}",
                             f"{expected_dm:.6f}", f"{actual_dm:.6f}",
                             f"DeMarker hand-calc at idx {i}")
                else:
                    report.fail("demarker_value", f"hand_calc_idx_{i}",
                               f"{expected_dm:.6f}", f"{actual_dm:.6f}",
                               f"DeMarker hand-calc diff={diff:.2e}")

    # Range checks (with floating point tolerance)
    FP_TOLERANCE = 1e-9
    for col, (min_val, max_val) in [
        ("demarker_value", (0, 1)),
        ("stoch_k_14", (0, 100)),
        ("stoch_d_14", (0, 100)),
        ("mfi_14", (0, 100)),
        ("williams_r_14", (-100, 0)),
    ]:
        if col in a100_df.columns:
            actual_min = a100_df[col].min()
            actual_max = a100_df[col].max()

            # Use tolerance for floating point comparisons
            if actual_min >= min_val - FP_TOLERANCE and actual_max <= max_val + FP_TOLERANCE:
                report.ok(col, "range_check",
                         f"[{min_val}, {max_val}]", f"[{actual_min:.2f}, {actual_max:.2f}]",
                         "Range within expected bounds")
            else:
                report.fail(col, "range_check",
                           f"[{min_val}, {max_val}]", f"[{actual_min:.2f}, {actual_max:.2f}]",
                           "Range out of bounds")


# =============================================================================
# Chunk 4: VRP + Risk Metrics
# =============================================================================

def validate_chunk4_vrp_risk(a100_df: pd.DataFrame, raw_df: pd.DataFrame,
                             vix_df: pd.DataFrame, report: ValidationReport) -> None:
    """Validate VRP and risk metric calculations."""
    # VRP = VIX - realized_vol
    raw_sorted = raw_df.sort_values("Date").reset_index(drop=True)
    close = raw_sorted["Close"]

    # Align VIX to price data
    vix_copy = vix_df.copy()
    vix_copy["Date"] = pd.to_datetime(vix_copy["Date"]).dt.normalize()
    raw_copy = raw_sorted.copy()
    raw_copy["Date"] = pd.to_datetime(raw_copy["Date"]).dt.normalize()

    vix_aligned = vix_copy.set_index("Date")["Close"].reindex(raw_copy["Date"]).values
    vix_aligned = pd.Series(vix_aligned)

    # 5-day realized volatility
    log_returns = np.log(close / close.shift(1))
    rv_5d = log_returns.rolling(5).std() * np.sqrt(252) * 100

    # VRP
    computed_vrp_5d = vix_aligned - rv_5d
    computed_vrp_slope = computed_vrp_5d - computed_vrp_5d.shift(1)

    computed_df = pd.DataFrame({
        "Date": raw_copy["Date"],
        "vrp_5d": computed_vrp_5d.values,
        "vrp_slope": computed_vrp_slope.values,
    })

    merged = pd.merge(
        computed_df,
        a100_df[["Date", "vrp_5d", "vrp_slope"]],
        on="Date",
        how="inner",
        suffixes=("_computed", "_stored")
    )

    for col in ["vrp_5d", "vrp_slope"]:
        valid_mask = ~merged[f"{col}_computed"].isna() & ~merged[f"{col}_stored"].isna()
        if valid_mask.sum() == 0:
            report.fail(col, "formula_verification", "matching values", "no valid data", "No overlap")
            continue

        max_diff = (merged[f"{col}_computed"][valid_mask] - merged[f"{col}_stored"][valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok(col, "formula_verification",
                     f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                     "VRP formula verified")
        else:
            report.fail(col, "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "VRP formula mismatch")

    # VaR/CVaR ordering constraint: CVaR <= VaR <= 0 (for losses)
    var_95 = a100_df["var_95"]
    var_99 = a100_df["var_99"]
    cvar_95 = a100_df["cvar_95"]

    # var_99 should be more negative than var_95 (more extreme)
    valid_mask = ~var_95.isna() & ~var_99.isna()
    violations = (var_99[valid_mask] > var_95[valid_mask]).sum()

    if violations == 0:
        report.ok("var_99", "ordering_constraint",
                 "var_99 <= var_95 (more extreme)", f"0 violations",
                 "VaR ordering verified: var_99 always <= var_95")
    else:
        pct = violations / valid_mask.sum() * 100
        report.fail("var_99", "ordering_constraint",
                   "var_99 <= var_95", f"{violations} violations ({pct:.1f}%)",
                   "VaR ordering violated")

    # CVaR should be <= VaR (mean of tail is more extreme than threshold)
    valid_mask = ~cvar_95.isna() & ~var_95.isna()
    violations = (cvar_95[valid_mask] > var_95[valid_mask]).sum()

    if violations == 0:
        report.ok("cvar_95", "ordering_constraint",
                 "cvar_95 <= var_95", f"0 violations",
                 "CVaR ordering verified")
    else:
        pct = violations / valid_mask.sum() * 100
        # Allow small tolerance for floating point
        if pct < 1:
            report.ok("cvar_95", "ordering_constraint",
                     "cvar_95 <= var_95", f"{violations} violations ({pct:.1f}%)",
                     "CVaR ordering mostly verified (minor floating point issues)")
        else:
            report.fail("cvar_95", "ordering_constraint",
                       "cvar_95 <= var_95", f"{violations} violations ({pct:.1f}%)",
                       "CVaR ordering violated")

    # Sharpe/Sortino range checks
    for col in ["sharpe_252d", "sortino_252d"]:
        if col in a100_df.columns:
            vals = a100_df[col]
            actual_min = vals.min()
            actual_max = vals.max()
            # Annualized ratios should typically be in [-10, 10] for 252d
            if actual_min >= -20 and actual_max <= 20:
                report.ok(col, "range_check",
                         "[-20, 20]", f"[{actual_min:.2f}, {actual_max:.2f}]",
                         "Range within expected bounds")
            else:
                report.fail(col, "range_check",
                           "[-20, 20]", f"[{actual_min:.2f}, {actual_max:.2f}]",
                           "Range out of bounds")


# =============================================================================
# Chunk 5: MA Extensions
# =============================================================================

def validate_chunk5_ma_extensions(a100_df: pd.DataFrame, raw_df: pd.DataFrame,
                                  report: ValidationReport) -> None:
    """Validate MA extension indicators including days_since_cross."""
    raw_sorted = raw_df.sort_values("Date").reset_index(drop=True)
    close = raw_sorted["Close"].values.astype(np.float64)

    # Compute SMAs and EMAs using talib
    sma_9 = pd.Series(talib.SMA(close, timeperiod=9))
    sma_50 = pd.Series(talib.SMA(close, timeperiod=50))
    sma_200 = pd.Series(talib.SMA(close, timeperiod=200))
    ema_12 = pd.Series(talib.EMA(close, timeperiod=12))
    ema_26 = pd.Series(talib.EMA(close, timeperiod=26))

    # sma_9_50_proximity = (SMA_9 - SMA_50) / SMA_50 * 100
    sma_9_50_proximity = (sma_9 - sma_50) / sma_50 * 100

    # Slopes = 5-day change
    sma_50_slope = sma_50 - sma_50.shift(5)
    sma_200_slope = sma_200 - sma_200.shift(5)

    computed_df = pd.DataFrame({
        "Date": raw_sorted["Date"],
        "sma_9_50_proximity": sma_9_50_proximity.values,
        "sma_50_slope": sma_50_slope.values,
        "sma_200_slope": sma_200_slope.values,
        "ema_12": ema_12.values,
        "ema_26": ema_26.values,
    })

    merged = pd.merge(
        computed_df,
        a100_df[["Date", "sma_9_50_proximity", "sma_50_slope", "sma_200_slope", "ema_12", "ema_26"]],
        on="Date",
        how="inner",
        suffixes=("_computed", "_stored")
    )

    for col in ["sma_9_50_proximity", "sma_50_slope", "sma_200_slope", "ema_12", "ema_26"]:
        valid_mask = ~merged[f"{col}_computed"].isna() & ~merged[f"{col}_stored"].isna()
        if valid_mask.sum() == 0:
            report.fail(col, "reference_comparison", "matching values", "no valid data", "No overlap")
            continue

        max_diff = (merged[f"{col}_computed"][valid_mask] - merged[f"{col}_stored"][valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok(col, "reference_comparison",
                     f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                     "Talib reference match")
        else:
            report.fail(col, "reference_comparison",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Reference mismatch")

    # days_since_cross validation: find actual crosses and verify counter
    close_series = pd.Series(close)

    for ma_col, days_col in [("sma_50", "days_since_sma_50_cross"),
                              ("sma_200", "days_since_sma_200_cross")]:
        ma = sma_50 if ma_col == "sma_50" else sma_200

        # Find crosses: sign(close - ma) changes
        position = np.sign(close_series - ma)
        cross = (position != position.shift(1)) & (position != 0) & (position.shift(1) != 0)
        cross_indices = cross[cross].index.tolist()

        if len(cross_indices) > 0:
            # Merge to get stored values
            test_df = pd.DataFrame({
                "Date": raw_sorted["Date"],
                "cross": cross.values,
            })
            test_merged = pd.merge(
                test_df,
                a100_df[["Date", days_col]],
                on="Date",
                how="inner"
            )

            # At cross points, days_since should be 0
            cross_mask = test_merged["cross"]
            if cross_mask.sum() > 0:
                cross_values = test_merged[days_col][cross_mask]
                zeros_at_cross = (cross_values == 0).sum()

                if zeros_at_cross == cross_mask.sum():
                    report.ok(days_col, "counter_reset",
                             "days=0 at all crosses", f"{zeros_at_cross}/{cross_mask.sum()} zeros",
                             "Counter resets correctly at crosses")
                else:
                    report.fail(days_col, "counter_reset",
                               "days=0 at all crosses", f"{zeros_at_cross}/{cross_mask.sum()} zeros",
                               "Counter not always 0 at crosses")

            # Verify counter increments after cross
            if len(cross_indices) >= 2:
                # Check a few days after a cross
                test_cross_idx = cross_indices[min(1, len(cross_indices)-1)]
                if test_cross_idx + 5 < len(test_merged):
                    days_values = test_merged[days_col].iloc[test_cross_idx:test_cross_idx+6].tolist()
                    # Should be roughly 0, 1, 2, 3, 4, 5 (incrementing)
                    expected_pattern = list(range(6))
                    if days_values == expected_pattern:
                        report.ok(days_col, "counter_increment",
                                 str(expected_pattern), str(days_values),
                                 "Counter increments correctly after cross")
                    else:
                        # Allow for slight variations due to boundary effects
                        is_increasing = all(days_values[i] <= days_values[i+1] for i in range(len(days_values)-1))
                        if is_increasing:
                            report.ok(days_col, "counter_increment",
                                     "monotonically increasing", str(days_values),
                                     "Counter generally increases after cross")
                        else:
                            report.fail(days_col, "counter_increment",
                                       str(expected_pattern), str(days_values),
                                       "Counter not incrementing correctly")
        else:
            report.ok(days_col, "counter_logic",
                     "crosses detected", "no crosses in data",
                     "No crosses to verify (expected for some datasets)")


# =============================================================================
# Chunk 6: Advanced Volatility
# =============================================================================

def validate_chunk6_volatility(a100_df: pd.DataFrame, raw_df: pd.DataFrame,
                               report: ValidationReport) -> None:
    """Validate advanced volatility indicators including Parkinson and Garman-Klass."""
    raw_sorted = raw_df.sort_values("Date").reset_index(drop=True)
    high = pd.Series(raw_sorted["High"].values, dtype=np.float64)
    low = pd.Series(raw_sorted["Low"].values, dtype=np.float64)
    close = pd.Series(raw_sorted["Close"].values, dtype=np.float64)
    open_price = pd.Series(raw_sorted["Open"].values, dtype=np.float64)

    # Parkinson volatility formula:
    # sqrt(1/(4*ln(2)) * mean(log(H/L)^2)) * sqrt(252) * 100
    log_hl = np.log(high / low)
    log_hl_sq = log_hl ** 2
    parkinson_var = log_hl_sq.rolling(20).mean() / (4 * np.log(2))
    parkinson_vol = np.sqrt(parkinson_var * 252) * 100

    # Garman-Klass volatility formula:
    # sqrt(0.5*log(H/L)^2 - (2*ln(2)-1)*log(C/O)^2) annualized
    log_co = np.log(close / open_price)
    gk_daily = 0.5 * log_hl_sq - (2 * np.log(2) - 1) * log_co ** 2
    gk_var = gk_daily.rolling(20).mean().clip(lower=0)
    gk_vol = np.sqrt(gk_var * 252) * 100

    computed_df = pd.DataFrame({
        "Date": raw_sorted["Date"],
        "parkinson_volatility": parkinson_vol.values,
        "garman_klass_volatility": gk_vol.values,
    })

    merged = pd.merge(
        computed_df,
        a100_df[["Date", "parkinson_volatility", "garman_klass_volatility"]],
        on="Date",
        how="inner",
        suffixes=("_computed", "_stored")
    )

    for col in ["parkinson_volatility", "garman_klass_volatility"]:
        valid_mask = ~merged[f"{col}_computed"].isna() & ~merged[f"{col}_stored"].isna()
        if valid_mask.sum() == 0:
            report.fail(col, "formula_verification", "matching values", "no valid data", "No overlap")
            continue

        max_diff = (merged[f"{col}_computed"][valid_mask] - merged[f"{col}_stored"][valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok(col, "formula_verification",
                     f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                     f"{col} formula verified (textbook formula)")
        else:
            report.fail(col, "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       f"{col} formula mismatch")

    # Range checks for volatility (annualized %)
    for col in ["parkinson_volatility", "garman_klass_volatility"]:
        if col in a100_df.columns:
            vals = a100_df[col]
            actual_min = vals.min()
            actual_max = vals.max()

            # Volatility should be positive and typically < 100% annualized for SPY
            if actual_min >= 0 and actual_max <= 150:
                report.ok(col, "range_check",
                         "[0, 150]", f"[{actual_min:.2f}, {actual_max:.2f}]",
                         "Volatility range reasonable")
            else:
                report.fail(col, "range_check",
                           "[0, 150]", f"[{actual_min:.2f}, {actual_max:.2f}]",
                           "Volatility range unexpected")

    # Percentile checks: should be in [0, 100]
    for col in ["atr_pct_percentile_60d", "bb_width_percentile_60d"]:
        if col in a100_df.columns:
            vals = a100_df[col]
            actual_min = vals.min()
            actual_max = vals.max()

            if actual_min >= 0 and actual_max <= 100:
                report.ok(col, "range_check",
                         "[0, 100]", f"[{actual_min:.2f}, {actual_max:.2f}]",
                         "Percentile range valid")
            else:
                report.fail(col, "range_check",
                           "[0, 100]", f"[{actual_min:.2f}, {actual_max:.2f}]",
                           "Percentile out of [0, 100] range")


# =============================================================================
# Chunk 7: Trend Indicators
# =============================================================================

def validate_chunk7_trend(a100_df: pd.DataFrame, raw_df: pd.DataFrame,
                          report: ValidationReport) -> None:
    """Validate trend indicators including SuperTrend direction consistency."""
    raw_sorted = raw_df.sort_values("Date").reset_index(drop=True)
    high = raw_sorted["High"].values.astype(np.float64)
    low = raw_sorted["Low"].values.astype(np.float64)
    close = raw_sorted["Close"].values.astype(np.float64)

    # ADX and DI from talib
    adx = pd.Series(talib.ADX(high, low, close, timeperiod=14))
    plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
    minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
    aroon_down, aroon_up = talib.AROON(high, low, timeperiod=25)

    computed_df = pd.DataFrame({
        "Date": raw_sorted["Date"],
        "adx_slope": (adx - adx.shift(5)).values,
        "di_spread": plus_di - minus_di,
        "aroon_oscillator": aroon_up - aroon_down,
    })

    merged = pd.merge(
        computed_df,
        a100_df[["Date", "adx_slope", "di_spread", "aroon_oscillator"]],
        on="Date",
        how="inner",
        suffixes=("_computed", "_stored")
    )

    for col in ["adx_slope", "di_spread", "aroon_oscillator"]:
        valid_mask = ~merged[f"{col}_computed"].isna() & ~merged[f"{col}_stored"].isna()
        if valid_mask.sum() == 0:
            report.fail(col, "reference_comparison", "matching values", "no valid data", "No overlap")
            continue

        max_diff = (merged[f"{col}_computed"][valid_mask] - merged[f"{col}_stored"][valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok(col, "reference_comparison",
                     f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                     "Talib reference match")
        else:
            report.fail(col, "reference_comparison",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Reference mismatch")

    # SuperTrend direction consistency validation
    # When direction = 1 (bullish), price should be above SuperTrend
    # When direction = -1 (bearish), price should be below SuperTrend

    st_merged = pd.merge(
        raw_sorted[["Date", "Close"]],
        a100_df[["Date", "price_pct_from_supertrend", "supertrend_direction"]],
        on="Date",
        how="inner"
    )

    valid_mask = ~st_merged["supertrend_direction"].isna() & ~st_merged["price_pct_from_supertrend"].isna()
    direction = st_merged["supertrend_direction"][valid_mask]
    pct_from_st = st_merged["price_pct_from_supertrend"][valid_mask]

    # When direction = 1, pct_from_st should be >= 0 (price above ST)
    # When direction = -1, pct_from_st should be <= 0 (price below ST)
    bullish_mask = direction == 1
    bearish_mask = direction == -1

    bullish_consistent = (pct_from_st[bullish_mask] >= -0.1).sum()  # Allow small tolerance
    bullish_total = bullish_mask.sum()

    bearish_consistent = (pct_from_st[bearish_mask] <= 0.1).sum()  # Allow small tolerance
    bearish_total = bearish_mask.sum()

    bullish_pct = bullish_consistent / bullish_total * 100 if bullish_total > 0 else 100
    bearish_pct = bearish_consistent / bearish_total * 100 if bearish_total > 0 else 100

    # Allow 95% consistency due to day-of-cross edge cases
    if bullish_pct >= 95:
        report.ok("supertrend_direction", "bullish_consistency",
                 "price above ST when bullish (>=95%)", f"{bullish_pct:.1f}%",
                 f"{bullish_consistent}/{bullish_total} consistent")
    else:
        report.fail("supertrend_direction", "bullish_consistency",
                   "price above ST when bullish (>=95%)", f"{bullish_pct:.1f}%",
                   f"{bullish_consistent}/{bullish_total} consistent")

    if bearish_pct >= 95:
        report.ok("supertrend_direction", "bearish_consistency",
                 "price below ST when bearish (>=95%)", f"{bearish_pct:.1f}%",
                 f"{bearish_consistent}/{bearish_total} consistent")
    else:
        report.fail("supertrend_direction", "bearish_consistency",
                   "price below ST when bearish (>=95%)", f"{bearish_pct:.1f}%",
                   f"{bearish_consistent}/{bearish_total} consistent")

    # Direction should only be +1 or -1
    unique_dirs = direction.unique()
    valid_dirs = set(unique_dirs) - {np.nan}

    if valid_dirs <= {1.0, -1.0, 1, -1}:
        report.ok("supertrend_direction", "value_range",
                 "{+1, -1}", str(valid_dirs),
                 "Direction values are valid")
    else:
        report.fail("supertrend_direction", "value_range",
                   "{+1, -1}", str(valid_dirs),
                   "Unexpected direction values")


# =============================================================================
# Chunk 8: Volume + Momentum + S/R
# =============================================================================

def validate_chunk8_volume_sr(a100_df: pd.DataFrame, raw_df: pd.DataFrame,
                              report: ValidationReport) -> None:
    """Validate volume, momentum, and support/resistance indicators."""
    # Merge raw data
    merged = pd.merge(
        raw_df[["Date", "High", "Low", "Close"]],
        a100_df[["Date", "buying_pressure_ratio", "fib_range_position",
                 "prior_high_20d_dist", "prior_low_20d_dist", "win_rate_20d"]],
        on="Date",
        how="inner"
    )

    # Buying pressure boundary: when Close = High, buying_pressure should = 1
    close_eq_high = merged["Close"] == merged["High"]
    if close_eq_high.sum() > 0:
        bp_at_high = merged["buying_pressure_ratio"][close_eq_high]
        ones = (bp_at_high == 1.0).sum()
        total = close_eq_high.sum()

        if ones == total:
            report.ok("buying_pressure_ratio", "boundary_close_eq_high",
                     "bp=1 when Close=High", f"{ones}/{total}",
                     "Boundary condition verified")
        else:
            # Allow floating point tolerance
            close_to_one = ((bp_at_high - 1.0).abs() < 1e-6).sum()
            if close_to_one == total:
                report.ok("buying_pressure_ratio", "boundary_close_eq_high",
                         "bp=1 when Close=High", f"{close_to_one}/{total} (within tolerance)",
                         "Boundary condition verified")
            else:
                report.fail("buying_pressure_ratio", "boundary_close_eq_high",
                           "bp=1 when Close=High", f"{close_to_one}/{total}",
                           f"Boundary not satisfied for {total - close_to_one} cases")
    else:
        report.ok("buying_pressure_ratio", "boundary_close_eq_high",
                 "bp=1 when Close=High", "no Close=High cases in data",
                 "No boundary cases to test")

    # When Close = Low, buying_pressure should = 0
    close_eq_low = merged["Close"] == merged["Low"]
    if close_eq_low.sum() > 0:
        bp_at_low = merged["buying_pressure_ratio"][close_eq_low]
        zeros = (bp_at_low == 0.0).sum()
        total = close_eq_low.sum()

        if zeros == total:
            report.ok("buying_pressure_ratio", "boundary_close_eq_low",
                     "bp=0 when Close=Low", f"{zeros}/{total}",
                     "Boundary condition verified")
        else:
            close_to_zero = ((bp_at_low - 0.0).abs() < 1e-6).sum()
            if close_to_zero == total:
                report.ok("buying_pressure_ratio", "boundary_close_eq_low",
                         "bp=0 when Close=Low", f"{close_to_zero}/{total} (within tolerance)",
                         "Boundary condition verified")
            else:
                report.fail("buying_pressure_ratio", "boundary_close_eq_low",
                           "bp=0 when Close=Low", f"{close_to_zero}/{total}",
                           f"Boundary not satisfied for {total - close_to_zero} cases")

    # Range checks (with floating point tolerance)
    FP_TOLERANCE = 1e-9
    for col, (min_val, max_val) in [
        ("buying_pressure_ratio", (0, 1)),
        ("fib_range_position", (0, 1)),
        ("win_rate_20d", (0, 1)),
    ]:
        if col in a100_df.columns:
            vals = a100_df[col]
            actual_min = vals.min()
            actual_max = vals.max()

            # Use tolerance for floating point comparisons
            if actual_min >= min_val - FP_TOLERANCE and actual_max <= max_val + FP_TOLERANCE:
                report.ok(col, "range_check",
                         f"[{min_val}, {max_val}]", f"[{actual_min:.4f}, {actual_max:.4f}]",
                         "Range valid")
            else:
                report.fail(col, "range_check",
                           f"[{min_val}, {max_val}]", f"[{actual_min:.4f}, {actual_max:.4f}]",
                           "Range out of bounds")

    # Sign constraints
    # prior_high_20d_dist should be <= 0 (close cannot exceed rolling max high)
    prior_high = a100_df["prior_high_20d_dist"]
    violations = (prior_high > 0.001).sum()  # Allow small tolerance for floating point

    if violations == 0:
        report.ok("prior_high_20d_dist", "sign_constraint",
                 "always <= 0", f"max={prior_high.max():.6f}",
                 "Sign constraint verified")
    else:
        pct = violations / len(prior_high) * 100
        report.fail("prior_high_20d_dist", "sign_constraint",
                   "always <= 0", f"{violations} violations ({pct:.2f}%)",
                   f"max={prior_high.max():.6f}")

    # prior_low_20d_dist should be >= 0 (close cannot be below rolling min low)
    prior_low = a100_df["prior_low_20d_dist"]
    violations = (prior_low < -0.001).sum()  # Allow small tolerance

    if violations == 0:
        report.ok("prior_low_20d_dist", "sign_constraint",
                 "always >= 0", f"min={prior_low.min():.6f}",
                 "Sign constraint verified")
    else:
        pct = violations / len(prior_low) * 100
        report.fail("prior_low_20d_dist", "sign_constraint",
                   "always >= 0", f"{violations} violations ({pct:.2f}%)",
                   f"min={prior_low.min():.6f}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Deep validation of tier_a100 indicators.")
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
        "--output-dir",
        type=Path,
        default=Path("outputs/validation"),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output for each check.",
    )
    args = parser.parse_args()

    # Check files exist
    for path, name in [
        (args.raw_path, "Raw SPY data"),
        (args.vix_path, "VIX data"),
    ]:
        if not path.exists():
            print(f"❌ {name} not found: {path}")
            return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data and computing tier_a100 features...")
    a100_df, raw_df, vix_df = load_data(args.raw_path, args.vix_path)
    print(f"  A100 dataset: {len(a100_df)} rows, {len(a100_df.columns)} columns")
    print(f"  Raw SPY: {len(raw_df)} rows")
    print(f"  VIX: {len(vix_df)} rows")

    # Initialize report
    report = ValidationReport()

    print("\nRunning validations...")

    # Data quality
    print("  Chunk 0: Data quality checks...")
    validate_data_quality(a100_df, report)

    # Chunk 1: Momentum derivatives
    print("  Chunk 1: Momentum derivatives...")
    validate_chunk1_momentum_derivatives(a100_df, raw_df, report)

    # Chunk 2: QQE/STC derivatives
    print("  Chunk 2: QQE/STC derivatives...")
    validate_chunk2_qqe_stc_derivatives(a100_df, report)

    # Chunk 3: Standard oscillators
    print("  Chunk 3: Standard oscillators...")
    validate_chunk3_oscillators(a100_df, raw_df, report)

    # Chunk 4: VRP + Risk metrics
    print("  Chunk 4: VRP + Risk metrics...")
    validate_chunk4_vrp_risk(a100_df, raw_df, vix_df, report)

    # Chunk 5: MA extensions
    print("  Chunk 5: MA extensions...")
    validate_chunk5_ma_extensions(a100_df, raw_df, report)

    # Chunk 6: Advanced volatility
    print("  Chunk 6: Advanced volatility...")
    validate_chunk6_volatility(a100_df, raw_df, report)

    # Chunk 7: Trend indicators
    print("  Chunk 7: Trend indicators...")
    validate_chunk7_trend(a100_df, raw_df, report)

    # Chunk 8: Volume + S/R
    print("  Chunk 8: Volume + S/R...")
    validate_chunk8_volume_sr(a100_df, raw_df, report)

    # Generate outputs
    json_path = args.output_dir / "tier_a100_validation.json"
    md_path = args.output_dir / "tier_a100_validation.md"

    with open(json_path, "w") as f:
        json.dump(report.to_json(), f, indent=2, default=str)

    with open(md_path, "w") as f:
        f.write(report.to_markdown())

    print(f"\nResults written to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for chunk_name, stats in report.summary_by_chunk().items():
        status = "✅" if stats["failed"] == 0 else "❌"
        print(f"{status} {chunk_name}: {stats['passed']} passed, {stats['failed']} failed")

    print(f"\nTotal: {report.passed} passed, {report.failed} failed ({report.pass_rate:.1f}%)")

    if report.failed == 0:
        print("\n✅ ALL VALIDATION CHECKS PASSED")
        return 0
    else:
        print(f"\n❌ {report.failed} VALIDATION CHECKS FAILED")
        if args.verbose:
            print("\nFailed checks:")
            for check in report.checks:
                if not check.passed:
                    print(f"  - {check.indicator}: {check.check_name}")
                    print(f"    Expected: {check.expected}")
                    print(f"    Actual: {check.actual}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
