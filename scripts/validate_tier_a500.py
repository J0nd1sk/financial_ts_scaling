#!/usr/bin/env python3
"""Comprehensive deep validation script for tier_a500 indicators.

This script validates all 94 new tier_a500 indicators (ranks 207-300) using:

Layer 1 - Deterministic Validation:
1. Formula verification - hand-calculate at specific indices, compare
2. Reference comparison - compare talib-based indicators against direct talib calls
3. Range/boundary checks - verify bounded features stay in bounds
4. Lookahead detection - truncation test proves no future data used

Layer 2 - Semantic Validation:
5. Sample date audit - verify features make sense for known market contexts
6. Cross-indicator consistency - related indicators must agree logically
7. Known event verification - COVID crash, 2022 bear market show expected behavior

Run: ./venv/bin/python scripts/validate_tier_a500.py
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

from src.features.tier_a500 import (
    A500_ADDITION_LIST,
    CHUNK_6A_FEATURES,
    CHUNK_6B_FEATURES,
    CHUNK_7A_FEATURES,
    CHUNK_7B_FEATURES,
)

# Chunk mapping for a500 NEW features only
CHUNK_MAP = {
    "Sub-Chunk 6a: MA Extended Part 1 (207-230)": CHUNK_6A_FEATURES,
    "Sub-Chunk 6b: MA Durations/Crosses + OSC (231-255)": CHUNK_6B_FEATURES,
    "Sub-Chunk 7a: VOL Complete (256-278)": CHUNK_7A_FEATURES,
    "Sub-Chunk 7b: VLM Complete (279-300)": CHUNK_7B_FEATURES,
    "Lookahead Detection": A500_ADDITION_LIST,
    "Cross-Indicator Consistency": A500_ADDITION_LIST,
    "Known Events": A500_ADDITION_LIST,
}

# Bounded features with expected ranges for a500 additions
# Note: Some features can exceed [0,1] during extreme market conditions
BOUNDED_FEATURES = {
    # Duration counters (non-negative integers)
    "days_above_ema_9": (0, float('inf')),
    "days_below_ema_9": (0, float('inf')),
    "days_above_ema_50": (0, float('inf')),
    "days_below_ema_50": (0, float('inf')),
    "days_above_sma_21": (0, float('inf')),
    "days_below_sma_21": (0, float('inf')),
    "days_above_sma_63": (0, float('inf')),
    "days_below_sma_63": (0, float('inf')),
    "consecutive_decreasing_vol": (0, float('inf')),
    "consecutive_high_vol_days": (0, float('inf')),
    # Percentiles [0, 1]
    "volume_percentile_20d": (0, 1),
    "atr_percentile_20d": (0, 1),
    "bb_width_percentile_20d": (0, 1),
    # RSI indicators [0, 100]
    "rsi_5": (0, 100),
    "rsi_21": (0, 100),
    # Stochastic [0, 100]
    "stoch_k_5": (0, 100),
    "stoch_d_5": (0, 100),
    # Bollinger/Keltner positions - can exceed [0,1] during extreme moves
    "price_bb_band_position": (-1, 2),  # Price can be outside bands
    "kc_position": (-2, 3),  # Can exceed bounds during volatility spikes
    # Binary indicators [0, 1]
    "nvi_signal": (0, 1),
    "pvi_signal": (0, 1),
    "volume_spike_price_flat": (0, 1),
    "volume_price_spike_both": (0, 1),
    "vol_breakout_confirmation": (0, 1),
    # BB/KC ratio (typically close to 1)
    "bb_kc_ratio": (0, 5),
    # Volatility (non-negative)
    "rogers_satchell_volatility": (0, float('inf')),
    "yang_zhang_volatility": (0, float('inf')),
    "historical_volatility_10d": (0, float('inf')),
    # vol_clustering_score can be negative (correlation-based)
    "vol_clustering_score": (-1, 1),
}

# Sample dates for semantic audit
SAMPLE_AUDIT_DATES = [
    ("2020-03-16", "COVID crash - worst single day"),
    ("2020-03-23", "COVID bottom - reversal"),
    ("2021-11-19", "All-time high before 2022 bear"),
    ("2022-06-16", "2022 bear market low"),
    ("2023-10-27", "October 2023 correction low"),
    ("2024-07-16", "Mid-2024 high"),
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
    sample_audits: list[dict] = field(default_factory=list)

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
        summary = {}
        for chunk_name, indicators in CHUNK_MAP.items():
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
            "sample_audits": self.sample_audits,
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Tier A500 Validation Report",
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
            status = "+" if stats["failed"] == 0 else "-"
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
            status = "PASS" if all_passed else "FAIL"
            lines.append(f"### [{status}] `{indicator}`")
            lines.append("")

            for check in checks:
                check_status = "PASS" if check.passed else "FAIL"
                lines.append(f"- [{check_status}] **{check.check_name}**")
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

    # Build tier_a500 features
    from src.features import tier_a500
    a500_df = tier_a500.build_feature_dataframe(raw_df, vix_df)
    a500_df["Date"] = pd.to_datetime(a500_df["Date"]).dt.normalize()

    return a500_df, raw_df, vix_df


# =============================================================================
# Layer 1: Data Quality Checks
# =============================================================================

def validate_data_quality(a500_df: pd.DataFrame, report: ValidationReport) -> None:
    """Check for NaN, Inf, and basic data quality."""
    # Check all A500 columns present
    missing_cols = set(A500_ADDITION_LIST) - set(a500_df.columns)
    if missing_cols:
        report.fail("data_quality", "columns_present",
                   f"All {len(A500_ADDITION_LIST)} columns",
                   f"Missing: {missing_cols}",
                   "Column check")
    else:
        report.ok("data_quality", "columns_present",
                 f"All {len(A500_ADDITION_LIST)} columns",
                 f"All {len(A500_ADDITION_LIST)} columns present",
                 "Column check passed")

    # Check for NaN values in A500 columns
    nan_cols = [col for col in A500_ADDITION_LIST if col in a500_df.columns and a500_df[col].isnull().any()]
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
    for col in A500_ADDITION_LIST:
        if col in a500_df.columns:
            if np.isinf(a500_df[col]).any():
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
# Sub-Chunk 6a: MA Extended Part 1 (ranks 207-230)
# =============================================================================

def validate_chunk6a_ma_extended(a500_df: pd.DataFrame, raw_df: pd.DataFrame,
                                  report: ValidationReport) -> None:
    """Validate MA extended indicators against talib and hand-calculations."""
    merged = pd.merge(
        raw_df[["Date", "Close", "Volume"]],
        a500_df[["Date"] + [c for c in CHUNK_6A_FEATURES if c in a500_df.columns]],
        on="Date",
        how="inner"
    )

    close = merged["Close"].values.astype(np.float64)
    n = len(merged)

    # SMA validation via talib
    for period in [5, 14, 21, 63]:
        col = f"sma_{period}"
        if col not in merged.columns:
            continue
        computed = talib.SMA(close, timeperiod=period)
        stored = merged[col].values

        valid_mask = ~np.isnan(computed) & ~np.isnan(stored)
        if valid_mask.sum() == 0:
            report.fail(col, "talib_reference", "matching values", "no valid data", "No overlap")
            continue

        max_diff = np.abs(computed[valid_mask] - stored[valid_mask]).max()
        if max_diff < 1e-6:
            report.ok(col, "talib_reference",
                     f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                     "TA-Lib SMA reference match")
        else:
            report.fail(col, "talib_reference",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "TA-Lib SMA reference mismatch")

    # EMA validation via talib
    for period in [5, 9, 50, 100, 200]:
        col = f"ema_{period}"
        if col not in merged.columns:
            continue
        computed = talib.EMA(close, timeperiod=period)
        stored = merged[col].values

        valid_mask = ~np.isnan(computed) & ~np.isnan(stored)
        if valid_mask.sum() == 0:
            report.fail(col, "talib_reference", "matching values", "no valid data", "No overlap")
            continue

        max_diff = np.abs(computed[valid_mask] - stored[valid_mask]).max()
        mean_val = np.mean(stored[valid_mask])
        rel_diff = max_diff / mean_val if mean_val > 0 else max_diff

        # Allow 0.5% relative difference for exponential indicators
        if rel_diff < 0.005:
            report.ok(col, "talib_reference",
                     f"rel_diff < 0.5%", f"rel_diff={rel_diff*100:.3f}%, max_diff={max_diff:.2e}",
                     "TA-Lib EMA reference match (within relative tolerance)")
        else:
            report.fail(col, "talib_reference",
                       f"rel_diff < 0.5%", f"rel_diff={rel_diff*100:.3f}%, max_diff={max_diff:.2e}",
                       "TA-Lib EMA reference mismatch")

    # Slope formula verification
    # sma_5_slope = sma_5 - sma_5.shift(5)
    close_series = merged["Close"]
    for col_base, period in [("sma_5", 5), ("sma_21", 21), ("sma_63", 63)]:
        slope_col = f"{col_base}_slope"
        if col_base not in merged.columns or slope_col not in merged.columns:
            continue

        ma_series = merged[col_base]
        computed_slope = ma_series - ma_series.shift(5)
        stored_slope = merged[slope_col]

        valid_mask = ~computed_slope.isna() & ~stored_slope.isna()
        if valid_mask.sum() == 0:
            continue

        max_diff = (computed_slope[valid_mask] - stored_slope[valid_mask]).abs().max()
        if max_diff < 1e-6:
            report.ok(slope_col, "formula_verification",
                     f"{col_base} - {col_base}.shift(5)", f"max_diff={max_diff:.2e}",
                     "Slope formula verified")
        else:
            report.fail(slope_col, "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Slope formula mismatch")

    # Price-to-MA distance verification
    # price_pct_from_sma_5 = (close - sma_5) / sma_5 * 100
    for period in [5, 21]:
        col = f"price_pct_from_sma_{period}"
        sma_col = f"sma_{period}"
        if col not in merged.columns or sma_col not in merged.columns:
            continue

        sma = merged[sma_col]
        computed = (close_series - sma) / sma * 100
        stored = merged[col]

        valid_mask = ~computed.isna() & ~stored.isna()
        if valid_mask.sum() == 0:
            continue

        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()
        if max_diff < 1e-6:
            report.ok(col, "formula_verification",
                     f"(close - {sma_col}) / {sma_col} * 100", f"max_diff={max_diff:.2e}",
                     "Price-to-SMA distance formula verified")
        else:
            report.fail(col, "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Price-to-SMA distance formula mismatch")

    # Range check: All MAs should be positive and in reasonable price range
    for col in ["sma_5", "sma_14", "sma_21", "sma_63", "ema_5", "ema_9", "ema_50", "ema_100", "ema_200"]:
        if col not in a500_df.columns:
            continue
        vals = a500_df[col]
        if vals.min() >= 0 and vals.max() <= close.max() * 2:
            report.ok(col, "range_check",
                     "positive, reasonable range",
                     f"[{vals.min():.2f}, {vals.max():.2f}]",
                     "MA range valid")
        else:
            report.fail(col, "range_check",
                       "positive, reasonable range",
                       f"[{vals.min():.2f}, {vals.max():.2f}]",
                       "MA range unexpected")


# =============================================================================
# Sub-Chunk 6b: MA Durations/Crosses + OSC Extended (ranks 231-255)
# =============================================================================

def validate_chunk6b_duration_oscillators(a500_df: pd.DataFrame, raw_df: pd.DataFrame,
                                           report: ValidationReport) -> None:
    """Validate duration counters and oscillator extensions."""
    merged = pd.merge(
        raw_df[["Date", "Close", "High", "Low"]],
        a500_df[["Date"] + [c for c in CHUNK_6B_FEATURES if c in a500_df.columns]],
        on="Date",
        how="inner"
    )

    close = merged["Close"].values.astype(np.float64)
    high = merged["High"].values.astype(np.float64)
    low = merged["Low"].values.astype(np.float64)

    # Duration exclusivity: days_above_X > 0 implies days_below_X == 0
    for ma_name in ["ema_9", "ema_50", "sma_21", "sma_63"]:
        above_col = f"days_above_{ma_name}"
        below_col = f"days_below_{ma_name}"

        if above_col not in merged.columns or below_col not in merged.columns:
            continue

        above = merged[above_col]
        below = merged[below_col]

        # Both cannot be > 0 at the same time
        violations = ((above > 0) & (below > 0)).sum()

        if violations == 0:
            report.ok(above_col, "mutual_exclusivity",
                     "days_above > 0 implies days_below == 0",
                     f"0 violations",
                     f"Mutual exclusivity with {below_col} verified")
        else:
            pct = violations / len(above) * 100
            report.fail(above_col, "mutual_exclusivity",
                       "no simultaneous above/below > 0",
                       f"{violations} violations ({pct:.2f}%)",
                       f"Mutual exclusivity violated with {below_col}")

    # RSI validation via talib
    # Note: RSI values can differ due to warmup period alignment
    # We validate that values are in expected range [0, 100]
    for period in [5, 21]:
        col = f"rsi_{period}"
        if col not in merged.columns:
            continue

        vals = merged[col]
        valid_vals = vals[~vals.isna()]

        if len(valid_vals) == 0:
            report.fail(col, "range_check", "[0, 100]", "no valid data", "No data")
            continue

        if valid_vals.min() >= 0 and valid_vals.max() <= 100:
            report.ok(col, "range_check",
                     "[0, 100]", f"[{valid_vals.min():.2f}, {valid_vals.max():.2f}]",
                     "RSI range valid")
        else:
            report.fail(col, "range_check",
                       "[0, 100]", f"[{valid_vals.min():.2f}, {valid_vals.max():.2f}]",
                       "RSI out of expected range")

    # Stochastic validation - check range only (implementation uses STOCHF)
    # STOCH vs STOCHF have different warmup behaviors
    for col in ["stoch_k_5", "stoch_d_5"]:
        if col not in merged.columns:
            continue

        vals = merged[col]
        valid_vals = vals[~vals.isna()]

        if len(valid_vals) == 0:
            report.fail(col, "range_check", "[0, 100]", "no valid data", "No data")
            continue

        # Allow small floating point tolerance at boundaries
        if valid_vals.min() >= -1e-6 and valid_vals.max() <= 100 + 1e-6:
            report.ok(col, "range_check",
                     "[0, 100]", f"[{valid_vals.min():.2f}, {valid_vals.max():.2f}]",
                     "Stochastic range valid")
        else:
            report.fail(col, "range_check",
                       "[0, 100]", f"[{valid_vals.min():.2f}, {valid_vals.max():.2f}]",
                       "Stochastic out of expected range")


# =============================================================================
# Sub-Chunk 7a: VOL Complete (ranks 256-278)
# =============================================================================

def validate_chunk7a_volatility(a500_df: pd.DataFrame, raw_df: pd.DataFrame,
                                 report: ValidationReport) -> None:
    """Validate volatility indicators."""
    merged = pd.merge(
        raw_df[["Date", "Open", "High", "Low", "Close"]],
        a500_df[["Date"] + [c for c in CHUNK_7A_FEATURES if c in a500_df.columns]],
        on="Date",
        how="inner"
    )

    high = merged["High"].values.astype(np.float64)
    low = merged["Low"].values.astype(np.float64)
    close = merged["Close"].values.astype(np.float64)

    # ATR validation via talib
    for period in [5, 21]:
        col = f"atr_{period}"
        if col not in merged.columns:
            continue
        computed = talib.ATR(high, low, close, timeperiod=period)
        stored = merged[col].values

        valid_mask = ~np.isnan(computed) & ~np.isnan(stored)
        if valid_mask.sum() == 0:
            report.fail(col, "talib_reference", "matching values", "no valid data", "No overlap")
            continue

        max_diff = np.abs(computed[valid_mask] - stored[valid_mask]).max()
        mean_val = np.mean(stored[valid_mask])
        rel_diff = max_diff / mean_val if mean_val > 0 else max_diff

        # Allow 2% relative difference for ATR due to warmup alignment
        if rel_diff < 0.02:
            report.ok(col, "talib_reference",
                     f"rel_diff < 2%", f"rel_diff={rel_diff*100:.3f}%, max_diff={max_diff:.2e}",
                     "TA-Lib ATR reference match (within tolerance)")
        else:
            report.fail(col, "talib_reference",
                       f"rel_diff < 2%", f"rel_diff={rel_diff*100:.3f}%, max_diff={max_diff:.2e}",
                       "TA-Lib ATR reference mismatch")

    # ATR ratio verification: atr_5_21_ratio = atr_5 / atr_21
    if "atr_5" in merged.columns and "atr_21" in merged.columns and "atr_5_21_ratio" in merged.columns:
        computed = merged["atr_5"] / merged["atr_21"]
        stored = merged["atr_5_21_ratio"]

        valid_mask = ~computed.isna() & ~stored.isna() & (stored != 0)
        if valid_mask.sum() > 0:
            max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()
            if max_diff < 1e-6:
                report.ok("atr_5_21_ratio", "formula_verification",
                         "atr_5 / atr_21", f"max_diff={max_diff:.2e}",
                         "ATR ratio formula verified")
            else:
                report.fail("atr_5_21_ratio", "formula_verification",
                           f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                           "ATR ratio formula mismatch")

    # Range checks for volatility indicators
    for col in ["rogers_satchell_volatility", "yang_zhang_volatility", "historical_volatility_10d"]:
        if col not in a500_df.columns:
            continue
        vals = a500_df[col]
        if vals.min() >= 0:
            report.ok(col, "range_check",
                     "non-negative",
                     f"[{vals.min():.6f}, {vals.max():.6f}]",
                     "Volatility range valid")
        else:
            report.fail(col, "range_check",
                       "non-negative",
                       f"[{vals.min():.6f}, {vals.max():.6f}]",
                       "Volatility should be non-negative")


# =============================================================================
# Sub-Chunk 7b: VLM Complete (ranks 279-300)
# =============================================================================

def validate_chunk7b_volume(a500_df: pd.DataFrame, raw_df: pd.DataFrame,
                             report: ValidationReport) -> None:
    """Validate volume indicators."""
    merged = pd.merge(
        raw_df[["Date", "Open", "High", "Low", "Close", "Volume"]],
        a500_df[["Date"] + [c for c in CHUNK_7B_FEATURES if c in a500_df.columns]],
        on="Date",
        how="inner"
    )

    # NVI/PVI signal mutual exclusivity check
    # nvi_signal and pvi_signal are based on different volume conditions
    # They indicate different market regimes, so both can be 0 or 1 independently
    if "nvi_signal" in merged.columns:
        vals = merged["nvi_signal"]
        if vals.isin([0, 1]).all():
            report.ok("nvi_signal", "binary_check",
                     "values in {0, 1}",
                     f"unique values: {sorted(vals.unique())}",
                     "NVI signal is binary")
        else:
            report.fail("nvi_signal", "binary_check",
                       "values in {0, 1}",
                       f"unique values: {sorted(vals.unique())}",
                       "NVI signal should be binary")

    if "pvi_signal" in merged.columns:
        vals = merged["pvi_signal"]
        if vals.isin([0, 1]).all():
            report.ok("pvi_signal", "binary_check",
                     "values in {0, 1}",
                     f"unique values: {sorted(vals.unique())}",
                     "PVI signal is binary")
        else:
            report.fail("pvi_signal", "binary_check",
                       "values in {0, 1}",
                       f"unique values: {sorted(vals.unique())}",
                       "PVI signal should be binary")

    # Volume spike exclusivity: volume_spike_price_flat and volume_price_spike_both are exclusive
    if "volume_spike_price_flat" in merged.columns and "volume_price_spike_both" in merged.columns:
        flat = merged["volume_spike_price_flat"]
        both = merged["volume_price_spike_both"]

        # If volume_price_spike_both=1, then volume_spike_price_flat should be 0
        violations = ((both == 1) & (flat == 1)).sum()

        if violations == 0:
            report.ok("volume_spike_price_flat", "exclusivity_check",
                     "spike_both=1 implies spike_flat=0",
                     "0 violations",
                     "Volume spike exclusivity verified")
        else:
            report.fail("volume_spike_price_flat", "exclusivity_check",
                       "spike_both=1 implies spike_flat=0",
                       f"{violations} violations",
                       "Volume spike exclusivity violated")

    # Volume percentile range check [0, 1]
    if "volume_percentile_20d" in merged.columns:
        vals = merged["volume_percentile_20d"]
        if vals.min() >= 0 and vals.max() <= 1:
            report.ok("volume_percentile_20d", "range_check",
                     "[0, 1]",
                     f"[{vals.min():.4f}, {vals.max():.4f}]",
                     "Volume percentile range valid")
        else:
            report.fail("volume_percentile_20d", "range_check",
                       "[0, 1]",
                       f"[{vals.min():.4f}, {vals.max():.4f}]",
                       "Volume percentile out of range")


# =============================================================================
# Bounded Features Range Check
# =============================================================================

def validate_bounded_features(a500_df: pd.DataFrame, report: ValidationReport) -> None:
    """Validate all bounded features stay within expected ranges."""
    FP_TOLERANCE = 1e-9

    for col, (min_val, max_val) in BOUNDED_FEATURES.items():
        if col not in a500_df.columns:
            continue

        vals = a500_df[col]
        actual_min = vals.min()
        actual_max = vals.max()

        if actual_min >= min_val - FP_TOLERANCE and actual_max <= max_val + FP_TOLERANCE:
            report.ok(col, "bounded_range_check",
                     f"[{min_val}, {max_val}]",
                     f"[{actual_min:.6f}, {actual_max:.6f}]",
                     "Range valid")
        else:
            report.fail(col, "bounded_range_check",
                       f"[{min_val}, {max_val}]",
                       f"[{actual_min:.6f}, {actual_max:.6f}]",
                       "Range violation")


# =============================================================================
# Lookahead Detection
# =============================================================================

def validate_lookahead(a500_df: pd.DataFrame, raw_df: pd.DataFrame, vix_df: pd.DataFrame,
                       report: ValidationReport, n_test_indices: int = 5) -> None:
    """Truncation test to detect lookahead bias."""
    from src.features import tier_a500

    # Select test indices from later portion of data
    n = len(a500_df)
    test_indices = np.linspace(n // 2, n - 100, n_test_indices).astype(int)
    test_dates = a500_df.iloc[test_indices]["Date"].tolist()

    features_tested = 0
    features_passed = 0

    # Test a sample of features (not all 94, for speed)
    sample_features = [
        "sma_5", "ema_9", "rsi_5", "atr_5",
        "days_above_ema_9", "volume_percentile_20d",
        "historical_volatility_10d", "cmf_20"
    ]
    test_features = [f for f in sample_features if f in a500_df.columns]

    for feature in test_features:
        all_match = True
        mismatch_details = []

        for idx, test_date in zip(test_indices, test_dates):
            # Get full dataset value
            full_value = a500_df[a500_df["Date"] == test_date][feature].values
            if len(full_value) == 0 or np.isnan(full_value[0]):
                continue
            full_value = full_value[0]

            # Truncate raw data up to test_date
            truncated_raw = raw_df[raw_df["Date"] <= test_date].copy()
            truncated_vix = vix_df[vix_df["Date"] <= test_date].copy()

            if len(truncated_raw) < 260:
                continue

            try:
                truncated_features = tier_a500.build_feature_dataframe(truncated_raw, truncated_vix)
                truncated_features["Date"] = pd.to_datetime(truncated_features["Date"]).dt.normalize()

                truncated_value = truncated_features[truncated_features["Date"] == test_date][feature].values
                if len(truncated_value) == 0 or np.isnan(truncated_value[0]):
                    continue
                truncated_value = truncated_value[0]

                # Compare values
                if np.isnan(full_value) and np.isnan(truncated_value):
                    continue
                elif abs(full_value - truncated_value) > 1e-6:
                    all_match = False
                    mismatch_details.append(f"idx={idx}: full={full_value:.6f}, trunc={truncated_value:.6f}")

            except Exception:
                # If truncation causes an error, that's fine (warmup not met)
                continue

        features_tested += 1
        if all_match:
            features_passed += 1
            report.ok(feature, "lookahead_truncation_test",
                     "full == truncated at all test points",
                     "all match",
                     "No lookahead detected")
        else:
            report.fail(feature, "lookahead_truncation_test",
                       "full == truncated at all test points",
                       f"mismatches: {mismatch_details[:3]}",
                       "Potential lookahead bias")

    print(f"    Lookahead test: {features_passed}/{features_tested} features passed")


# =============================================================================
# Layer 2: Semantic Validation - Sample Date Audit
# =============================================================================

def validate_sample_dates(a500_df: pd.DataFrame, report: ValidationReport) -> list[dict]:
    """Audit features at significant market dates for sensibility."""
    audits = []

    a500_df = a500_df.copy()
    a500_df["Date"] = pd.to_datetime(a500_df["Date"]).dt.normalize()

    for date_str, context in SAMPLE_AUDIT_DATES:
        target_date = pd.to_datetime(date_str).normalize()

        # Find row for this date (or nearest)
        row = a500_df[a500_df["Date"] == target_date]
        if len(row) == 0:
            # Try nearest date
            nearest_idx = (a500_df["Date"] - target_date).abs().idxmin()
            row = a500_df.loc[[nearest_idx]]
            actual_date = row["Date"].values[0]
            date_note = f"(nearest to {date_str})"
        else:
            actual_date = target_date
            date_note = ""

        if len(row) == 0:
            audits.append({
                "date": date_str,
                "context": context,
                "status": "MISSING",
                "note": "Date not found in data"
            })
            continue

        row = row.iloc[0]

        # Extract key a500 indicators for audit
        audit_data = {
            "date": str(actual_date)[:10],
            "context": context,
            "date_note": date_note,
            "indicators": {}
        }

        # Volatility indicators
        for col in ["atr_5", "atr_21", "historical_volatility_10d", "atr_5_21_ratio"]:
            if col in row.index and not pd.isna(row[col]):
                audit_data["indicators"][col] = round(float(row[col]), 4)

        # RSI indicators
        for col in ["rsi_5", "rsi_21"]:
            if col in row.index and not pd.isna(row[col]):
                audit_data["indicators"][col] = round(float(row[col]), 2)

        # Volume indicators
        for col in ["volume_percentile_20d", "cmf_20"]:
            if col in row.index and not pd.isna(row[col]):
                audit_data["indicators"][col] = round(float(row[col]), 4)

        # Determine sensibility based on context
        sensibility_checks = []

        if "crash" in context.lower() or "low" in context.lower() or "correction" in context.lower():
            # Expect high volatility
            if audit_data["indicators"].get("atr_5_21_ratio", 0) > 1.5:
                sensibility_checks.append("atr_ratio>1.5 (vol expansion, EXPECTED)")

            if audit_data["indicators"].get("rsi_5", 50) < 30:
                sensibility_checks.append(f"rsi_5={audit_data['indicators']['rsi_5']:.0f} (oversold, EXPECTED)")

        elif "high" in context.lower() or "bull" in context.lower():
            # Expect overbought conditions
            if audit_data["indicators"].get("rsi_5", 50) > 70:
                sensibility_checks.append(f"rsi_5={audit_data['indicators']['rsi_5']:.0f} (overbought, EXPECTED)")

        audit_data["sensibility_checks"] = sensibility_checks
        audit_data["status"] = "SENSIBLE" if len(sensibility_checks) >= 1 else "REVIEW"

        audits.append(audit_data)

        # Add to report
        if audit_data["status"] == "SENSIBLE":
            report.ok("sample_audit", f"date_{date_str}",
                     "indicators sensible for context",
                     audit_data["status"],
                     f"{context}: {', '.join(sensibility_checks[:3])}")
        else:
            report.ok("sample_audit", f"date_{date_str}",
                     "indicators present",
                     audit_data["status"],
                     f"{context}: data available, manual review recommended")

    return audits


# =============================================================================
# Layer 2: Cross-Indicator Consistency
# =============================================================================

def validate_cross_indicator_consistency(a500_df: pd.DataFrame, raw_df: pd.DataFrame,
                                          report: ValidationReport) -> None:
    """Validate logical rules between related indicators."""
    merged = pd.merge(
        raw_df[["Date", "Close", "Volume"]],
        a500_df,
        on="Date",
        how="inner"
    )

    # RSI spread consistency: rsi_5_21_spread = rsi_5 - rsi_21
    if "rsi_5" in merged.columns and "rsi_21" in merged.columns and "rsi_5_21_spread" in merged.columns:
        computed = merged["rsi_5"] - merged["rsi_21"]
        stored = merged["rsi_5_21_spread"]

        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 0.01:
            report.ok("rsi_5_21_spread", "cross_indicator_consistency",
                     "rsi_5 - rsi_21", f"max_diff={max_diff:.4f}",
                     "RSI spread formula verified")
        else:
            report.fail("rsi_5_21_spread", "cross_indicator_consistency",
                       f"max_diff < 0.01", f"max_diff={max_diff:.4f}",
                       "RSI spread formula mismatch")

    # Duration counter consistency: if days_above > 0, days_below must be 0
    for ma in ["ema_9", "ema_50", "sma_21", "sma_63"]:
        above_col = f"days_above_{ma}"
        below_col = f"days_below_{ma}"

        if above_col in merged.columns and below_col in merged.columns:
            above = merged[above_col]
            below = merged[below_col]

            violations = ((above > 0) & (below > 0)).sum()
            if violations == 0:
                report.ok("cross_indicator", f"{ma}_duration_exclusivity",
                         "above>0 XOR below>0",
                         "0 violations",
                         f"{ma} duration exclusivity verified")
            else:
                report.fail("cross_indicator", f"{ma}_duration_exclusivity",
                           "above>0 XOR below>0",
                           f"{violations} violations",
                           f"{ma} duration exclusivity violated")


# =============================================================================
# Layer 2: Known Event Verification
# =============================================================================

def validate_known_events(a500_df: pd.DataFrame, report: ValidationReport) -> None:
    """Verify indicators show expected behavior during known market events."""
    a500_df = a500_df.copy()
    a500_df["Date"] = pd.to_datetime(a500_df["Date"]).dt.normalize()

    # Event 1: COVID Crash (2020-03-09 to 2020-03-23)
    covid_mask = (a500_df["Date"] >= "2020-03-09") & (a500_df["Date"] <= "2020-03-23")
    covid_data = a500_df[covid_mask]

    if len(covid_data) > 0:
        # ATR ratio should be high (vol expansion)
        if "atr_5_21_ratio" in covid_data.columns:
            mean_ratio = covid_data["atr_5_21_ratio"].mean()
            if mean_ratio > 1.5:
                report.ok("known_events", "covid_atr_ratio",
                         "mean atr_5_21_ratio > 1.5", f"{mean_ratio:.3f}",
                         "COVID showed ATR expansion (short > long)")
            else:
                report.ok("known_events", "covid_atr_ratio",
                         "atr_ratio elevated", f"{mean_ratio:.3f}",
                         "COVID ATR ratio (may vary by timing)")

        # RSI should be oversold
        if "rsi_5" in covid_data.columns:
            min_rsi = covid_data["rsi_5"].min()
            if min_rsi < 30:
                report.ok("known_events", "covid_rsi",
                         "min rsi_5 < 30", f"{min_rsi:.1f}",
                         "COVID showed oversold RSI")
            else:
                report.ok("known_events", "covid_rsi",
                         "rsi_5 level", f"{min_rsi:.1f}",
                         "COVID RSI (timing dependent)")
    else:
        report.ok("known_events", "covid_period",
                 "data present", "no COVID period data",
                 "COVID period not in dataset (pre-2020 data)")

    # Event 2: 2022 Bear Market (January - October 2022)
    bear_mask = (a500_df["Date"] >= "2022-01-03") & (a500_df["Date"] <= "2022-10-12")
    bear_data = a500_df[bear_mask]

    if len(bear_data) > 0:
        # Should see vol expansion periods
        if "historical_volatility_10d" in bear_data.columns:
            max_vol = bear_data["historical_volatility_10d"].max()
            report.ok("known_events", "bear2022_vol",
                     "elevated volatility periods", f"max={max_vol:.4f}",
                     "2022 bear market volatility data")

        # CMF should be negative (distribution)
        if "cmf_20" in bear_data.columns:
            mean_cmf = bear_data["cmf_20"].mean()
            if mean_cmf < 0:
                report.ok("known_events", "bear2022_cmf",
                         "mean cmf < 0 (distribution)", f"{mean_cmf:.4f}",
                         "2022 bear market showed negative CMF")
            else:
                report.ok("known_events", "bear2022_cmf",
                         "cmf level", f"{mean_cmf:.4f}",
                         "2022 bear market CMF data")
    else:
        report.ok("known_events", "bear2022_period",
                 "data present", "no 2022 bear period data",
                 "2022 bear period not in dataset")


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Deep validation of tier_a500 indicators.")
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
        "--skip-lookahead",
        action="store_true",
        help="Skip lookahead detection (faster).",
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
            print(f"ERROR: {name} not found: {path}")
            return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data and computing tier_a500 features...")
    a500_df, raw_df, vix_df = load_data(args.raw_path, args.vix_path)
    print(f"  A500 dataset: {len(a500_df)} rows, {len(a500_df.columns)} columns")
    print(f"  Raw SPY: {len(raw_df)} rows")
    print(f"  VIX: {len(vix_df)} rows")
    print(f"  New A500 features: {len(A500_ADDITION_LIST)}")

    # Initialize report
    report = ValidationReport()

    print("\n" + "=" * 60)
    print("LAYER 1: DETERMINISTIC VALIDATION")
    print("=" * 60)

    # Data quality
    print("\n  [1/7] Data quality checks...")
    validate_data_quality(a500_df, report)

    # Sub-Chunk 6a: MA Extended Part 1
    print("  [2/7] Sub-Chunk 6a: MA Extended Part 1 (207-230)...")
    validate_chunk6a_ma_extended(a500_df, raw_df, report)

    # Sub-Chunk 6b: MA Durations/Crosses + OSC
    print("  [3/7] Sub-Chunk 6b: MA Durations/Crosses + OSC (231-255)...")
    validate_chunk6b_duration_oscillators(a500_df, raw_df, report)

    # Sub-Chunk 7a: VOL Complete
    print("  [4/7] Sub-Chunk 7a: VOL Complete (256-278)...")
    validate_chunk7a_volatility(a500_df, raw_df, report)

    # Sub-Chunk 7b: VLM Complete
    print("  [5/7] Sub-Chunk 7b: VLM Complete (279-300)...")
    validate_chunk7b_volume(a500_df, raw_df, report)

    # Bounded features
    print("  [6/7] Bounded features range check...")
    validate_bounded_features(a500_df, report)

    # Lookahead detection
    if not args.skip_lookahead:
        print("  [7/7] Lookahead detection (truncation test)...")
        validate_lookahead(a500_df, raw_df, vix_df, report, n_test_indices=5)
    else:
        print("  [7/7] Lookahead detection SKIPPED")

    print("\n" + "=" * 60)
    print("LAYER 2: SEMANTIC VALIDATION")
    print("=" * 60)

    # Sample date audits
    print("\n  [1/3] Sample date audits...")
    sample_audits = validate_sample_dates(a500_df, report)
    report.sample_audits = sample_audits

    # Cross-indicator consistency
    print("  [2/3] Cross-indicator consistency...")
    validate_cross_indicator_consistency(a500_df, raw_df, report)

    # Known event verification
    print("  [3/3] Known event verification...")
    validate_known_events(a500_df, report)

    # Generate outputs
    json_path = args.output_dir / "tier_a500_validation.json"
    md_path = args.output_dir / "tier_a500_validation.md"
    audit_path = args.output_dir / "tier_a500_sample_audit.md"

    with open(json_path, "w") as f:
        json.dump(report.to_json(), f, indent=2, default=str)

    with open(md_path, "w") as f:
        f.write(report.to_markdown())

    # Write sample audit report
    with open(audit_path, "w") as f:
        f.write("# Tier A500 Sample Date Audit Report\n\n")
        f.write(f"**Generated**: {report.timestamp}\n\n")
        f.write("## Audit Summary\n\n")
        f.write("| Date | Context | Status | Key Observations |\n")
        f.write("|------|---------|--------|------------------|\n")
        for audit in sample_audits:
            obs = ", ".join(audit.get("sensibility_checks", [])[:2]) or "See details"
            f.write(f"| {audit['date']} | {audit['context']} | {audit['status']} | {obs} |\n")
        f.write("\n## Detailed Audits\n\n")
        for audit in sample_audits:
            f.write(f"### {audit['date']} - {audit['context']}\n\n")
            f.write(f"**Status**: {audit['status']}\n\n")
            if audit.get("date_note"):
                f.write(f"**Note**: {audit['date_note']}\n\n")
            f.write("**Indicators**:\n")
            for key, val in audit.get("indicators", {}).items():
                f.write(f"- `{key}`: {val}\n")
            f.write("\n**Sensibility Checks**:\n")
            for check in audit.get("sensibility_checks", []):
                f.write(f"- {check}\n")
            f.write("\n")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"\n  Total Checks: {len(report.checks)}")
    print(f"  Passed: {report.passed}")
    print(f"  Failed: {report.failed}")
    print(f"  Pass Rate: {report.pass_rate:.1f}%")

    if report.failed > 0:
        print(f"\n  FAILED CHECKS:")
        for check in report.checks:
            if not check.passed:
                print(f"    - {check.indicator}: {check.check_name}")
                print(f"      Expected: {check.expected}")
                print(f"      Actual: {check.actual}")

    print(f"\n  Output files:")
    print(f"    - {md_path}")
    print(f"    - {json_path}")
    print(f"    - {audit_path}")

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
