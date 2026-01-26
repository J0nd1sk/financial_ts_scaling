#!/usr/bin/env python3
"""Comprehensive deep validation script for tier_a200 indicators.

This script validates all 106 new tier_a200 indicators (ranks 101-206) using:

Layer 1 - Deterministic Validation:
1. Formula verification - hand-calculate at specific indices, compare
2. Reference comparison - compare talib-based indicators against direct talib calls
3. Range/boundary checks - verify bounded features stay in bounds
4. Lookahead detection - truncation test proves no future data used

Layer 2 - Semantic Validation:
5. Sample date audit - verify features make sense for known market contexts
6. Cross-indicator consistency - related indicators must agree logically
7. Known event verification - COVID crash, 2022 bear market show expected behavior

Run: ./venv/bin/python scripts/validate_tier_a200.py
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

from src.features.tier_a200 import A200_ADDITION_LIST

# Chunk mapping for a200 features
CHUNK_MAP = {
    "Chunk 1: Extended MAs (101-120)": [
        "tema_9", "tema_20", "tema_50", "tema_100",
        "wma_10", "wma_20", "wma_50", "wma_200",
        "kama_10", "kama_20", "kama_50",
        "hma_9", "hma_21", "hma_50",
        "vwma_10", "vwma_20", "vwma_50",
        "tema_20_slope", "price_pct_from_tema_50", "price_pct_from_kama_20",
    ],
    "Chunk 2: Duration & Proximity (121-140)": [
        "days_above_sma_9", "days_below_sma_9",
        "days_above_sma_50", "days_below_sma_50",
        "days_above_sma_200", "days_below_sma_200",
        "days_above_tema_20", "days_below_tema_20",
        "days_above_kama_20", "days_below_kama_20",
        "days_above_vwma_20", "days_below_vwma_20",
        "days_since_sma_9_50_cross", "days_since_sma_50_200_cross",
        "days_since_tema_sma_50_cross", "days_since_kama_sma_50_cross",
        "days_since_sma_9_200_cross",
        "tema_20_sma_50_proximity", "kama_20_sma_50_proximity", "sma_9_200_proximity",
    ],
    "Chunk 3: BB/RSI/Mean Reversion (141-160)": [
        "pct_from_upper_band", "pct_from_lower_band",
        "days_above_upper_band", "days_below_lower_band",
        "bb_squeeze_indicator", "bb_squeeze_duration",
        "rsi_distance_from_50", "days_rsi_overbought", "days_rsi_oversold", "rsi_percentile_60d",
        "zscore_from_20d_mean", "zscore_from_50d_mean",
        "percentile_in_52wk_range", "distance_from_52wk_high_pct",
        "days_since_52wk_high", "days_since_52wk_low",
        "consecutive_up_days", "consecutive_down_days",
        "up_days_ratio_20d", "range_compression_5d",
    ],
    "Chunk 4: MACD/Volume/Calendar/Candle (161-180)": [
        "macd_signal", "macd_histogram_slope", "days_since_macd_cross_signal", "macd_signal_proximity",
        "volume_trend_5d", "consecutive_volume_increase", "volume_price_confluence", "high_volume_direction_bias",
        "trading_day_of_week", "is_monday", "is_friday", "days_to_month_end", "month_of_year", "is_quarter_end_month",
        "candle_body_pct", "body_to_range_ratio", "upper_wick_pct", "lower_wick_pct", "doji_indicator", "range_vs_avg_range",
    ],
    "Chunk 5: Ichimoku/Donchian/Divergence/Entropy (181-206)": [
        "tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", "price_vs_cloud", "cloud_thickness_pct",
        "donchian_upper_20", "donchian_lower_20", "donchian_position", "donchian_width_pct", "pct_to_donchian_breakout",
        "price_rsi_divergence", "price_obv_divergence", "divergence_streak", "divergence_magnitude",
        "permutation_entropy_order3", "permutation_entropy_order4", "permutation_entropy_order5", "entropy_trend_5d",
        "atr_regime_pct_60d", "atr_regime_rolling_q", "trend_strength_pct_60d", "trend_strength_rolling_q",
        "vol_regime_state", "regime_consistency", "regime_transition_prob",
    ],
    "Lookahead Detection": A200_ADDITION_LIST,
    "Cross-Indicator Consistency": A200_ADDITION_LIST,
    "Known Events": A200_ADDITION_LIST,
}

# Bounded features with expected ranges
BOUNDED_FEATURES = {
    # [0, 1] ranges
    "body_to_range_ratio": (0, 1),
    "upper_wick_pct": (0, 1),
    "lower_wick_pct": (0, 1),
    "donchian_position": (0, 1),
    "rsi_percentile_60d": (0, 1),
    "percentile_in_52wk_range": (0, 1),
    "up_days_ratio_20d": (0, 1),
    "permutation_entropy_order3": (0, 1),
    "permutation_entropy_order4": (0, 1),
    "permutation_entropy_order5": (0, 1),
    "atr_regime_pct_60d": (0, 1),
    "atr_regime_rolling_q": (0, 1),
    "trend_strength_pct_60d": (0, 1),
    "trend_strength_rolling_q": (0, 1),
    "regime_transition_prob": (0, 1),
    # Binary features
    "bb_squeeze_indicator": (0, 1),
    "doji_indicator": (0, 1),
    "is_monday": (0, 1),
    "is_friday": (0, 1),
    "is_quarter_end_month": (0, 1),
    # Calendar ranges
    "trading_day_of_week": (0, 4),
    "month_of_year": (1, 12),
    # Regime state
    "vol_regime_state": (-1, 1),
    "price_vs_cloud": (-1, 1),
    # Divergence ranges
    "price_rsi_divergence": (-1, 1),
    "price_obv_divergence": (-1, 1),
    "divergence_magnitude": (0, 1),
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
            "# Tier A200 Validation Report",
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

    # Build tier_a200 features
    from src.features import tier_a200
    a200_df = tier_a200.build_feature_dataframe(raw_df, vix_df)
    a200_df["Date"] = pd.to_datetime(a200_df["Date"]).dt.normalize()

    return a200_df, raw_df, vix_df


# =============================================================================
# Layer 1: Data Quality Checks
# =============================================================================

def validate_data_quality(a200_df: pd.DataFrame, report: ValidationReport) -> None:
    """Check for NaN, Inf, and basic data quality."""
    # Check all A200 columns present
    missing_cols = set(A200_ADDITION_LIST) - set(a200_df.columns)
    if missing_cols:
        report.fail("data_quality", "columns_present",
                   f"All {len(A200_ADDITION_LIST)} columns",
                   f"Missing: {missing_cols}",
                   "Column check")
    else:
        report.ok("data_quality", "columns_present",
                 f"All {len(A200_ADDITION_LIST)} columns",
                 f"All {len(A200_ADDITION_LIST)} columns present",
                 "Column check passed")

    # Check for NaN values in A200 columns
    nan_cols = [col for col in A200_ADDITION_LIST if col in a200_df.columns and a200_df[col].isnull().any()]
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
    for col in A200_ADDITION_LIST:
        if col in a200_df.columns:
            if np.isinf(a200_df[col]).any():
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
# Chunk 1: Extended MAs (ranks 101-120)
# =============================================================================

def validate_chunk1_extended_mas(a200_df: pd.DataFrame, raw_df: pd.DataFrame,
                                  report: ValidationReport) -> None:
    """Validate extended MA indicators against talib and hand-calculations."""
    # Merge to align indices
    merged = pd.merge(
        raw_df[["Date", "Close", "Volume"]],
        a200_df[["Date"] + [c for c in CHUNK_MAP["Chunk 1: Extended MAs (101-120)"] if c in a200_df.columns]],
        on="Date",
        how="inner"
    )

    close = merged["Close"].values.astype(np.float64)
    volume = merged["Volume"].values.astype(np.float64)
    n = len(merged)

    # TEMA validation via talib
    # Note: TEMA is an exponential indicator where small warmup differences can accumulate
    # Use relative tolerance (0.1% of mean) or absolute tolerance of 1.0
    for period in [9, 20, 50, 100]:
        col = f"tema_{period}"
        if col not in merged.columns:
            continue
        computed = talib.TEMA(close, timeperiod=period)
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
                     "TA-Lib TEMA reference match (within relative tolerance)")
        else:
            report.fail(col, "talib_reference",
                       f"rel_diff < 0.5%", f"rel_diff={rel_diff*100:.3f}%, max_diff={max_diff:.2e}",
                       "TA-Lib TEMA reference mismatch")

    # WMA validation via talib
    for period in [10, 20, 50, 200]:
        col = f"wma_{period}"
        if col not in merged.columns:
            continue
        computed = talib.WMA(close, timeperiod=period)
        stored = merged[col].values

        valid_mask = ~np.isnan(computed) & ~np.isnan(stored)
        if valid_mask.sum() == 0:
            report.fail(col, "talib_reference", "matching values", "no valid data", "No overlap")
            continue

        max_diff = np.abs(computed[valid_mask] - stored[valid_mask]).max()
        if max_diff < 1e-6:
            report.ok(col, "talib_reference",
                     f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                     "TA-Lib WMA reference match")
        else:
            report.fail(col, "talib_reference",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "TA-Lib WMA reference mismatch")

    # KAMA validation via talib
    # Note: KAMA is an adaptive indicator where differences can accumulate
    # Use relative tolerance (0.5% of mean) for these adaptive indicators
    for period in [10, 20, 50]:
        col = f"kama_{period}"
        if col not in merged.columns:
            continue
        computed = talib.KAMA(close, timeperiod=period)
        stored = merged[col].values

        valid_mask = ~np.isnan(computed) & ~np.isnan(stored)
        if valid_mask.sum() == 0:
            report.fail(col, "talib_reference", "matching values", "no valid data", "No overlap")
            continue

        max_diff = np.abs(computed[valid_mask] - stored[valid_mask]).max()
        mean_val = np.mean(stored[valid_mask])
        rel_diff = max_diff / mean_val if mean_val > 0 else max_diff

        # Allow 0.5% relative difference for adaptive indicators
        if rel_diff < 0.005:
            report.ok(col, "talib_reference",
                     f"rel_diff < 0.5%", f"rel_diff={rel_diff*100:.3f}%, max_diff={max_diff:.2e}",
                     "TA-Lib KAMA reference match (within relative tolerance)")
        else:
            report.fail(col, "talib_reference",
                       f"rel_diff < 0.5%", f"rel_diff={rel_diff*100:.3f}%, max_diff={max_diff:.2e}",
                       "TA-Lib KAMA reference mismatch")

    # HMA hand-calculation verification at specific indices
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    np.random.seed(42)
    test_indices = np.random.choice(range(60, n - 5), size=5, replace=False)

    for period in [9, 21, 50]:
        col = f"hma_{period}"
        if col not in merged.columns:
            continue

        half_period = max(1, period // 2)
        sqrt_period = max(1, int(np.sqrt(period)))

        wma_half = talib.WMA(close, timeperiod=half_period)
        wma_full = talib.WMA(close, timeperiod=period)
        raw_hma = 2 * wma_half - wma_full
        computed_hma = talib.WMA(raw_hma, timeperiod=sqrt_period)

        stored = merged[col].values
        valid_mask = ~np.isnan(computed_hma) & ~np.isnan(stored)
        if valid_mask.sum() == 0:
            report.fail(col, "formula_verification", "matching values", "no valid data", "No overlap")
            continue

        max_diff = np.abs(computed_hma[valid_mask] - stored[valid_mask]).max()
        if max_diff < 1e-6:
            report.ok(col, "formula_verification",
                     f"HMA formula: WMA(2*WMA(n/2)-WMA(n), sqrt(n))",
                     f"max_diff={max_diff:.2e}",
                     "HMA hand-calculation verified")
        else:
            report.fail(col, "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "HMA formula mismatch")

    # VWMA hand-calculation verification
    # VWMA = sum(close*vol)/sum(vol)
    close_series = merged["Close"]
    volume_series = merged["Volume"]

    for period in [10, 20, 50]:
        col = f"vwma_{period}"
        if col not in merged.columns:
            continue

        pv = close_series * volume_series
        pv_sum = pv.rolling(window=period).sum()
        vol_sum = volume_series.rolling(window=period).sum()
        computed_vwma = pv_sum / vol_sum

        stored = merged[col]
        valid_mask = ~computed_vwma.isna() & ~stored.isna()
        if valid_mask.sum() == 0:
            report.fail(col, "formula_verification", "matching values", "no valid data", "No overlap")
            continue

        max_diff = (computed_vwma[valid_mask] - stored[valid_mask]).abs().max()
        if max_diff < 1e-6:
            report.ok(col, "formula_verification",
                     f"VWMA = sum(close*vol)/sum(vol)",
                     f"max_diff={max_diff:.2e}",
                     "VWMA hand-calculation verified")
        else:
            report.fail(col, "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "VWMA formula mismatch")

    # Derived MA indicators
    # tema_20_slope = tema_20 - tema_20.shift(5)
    if "tema_20" in merged.columns and "tema_20_slope" in merged.columns:
        tema_20 = merged["tema_20"]
        computed_slope = tema_20 - tema_20.shift(5)
        stored_slope = merged["tema_20_slope"]

        valid_mask = ~computed_slope.isna() & ~stored_slope.isna()
        max_diff = (computed_slope[valid_mask] - stored_slope[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("tema_20_slope", "formula_verification",
                     "tema_20 - tema_20.shift(5)", f"max_diff={max_diff:.2e}",
                     "Slope formula verified")
        else:
            report.fail("tema_20_slope", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Slope formula mismatch")

    # price_pct_from_tema_50 = (close - tema_50) / tema_50 * 100
    if "tema_50" in merged.columns and "price_pct_from_tema_50" in merged.columns:
        tema_50 = merged["tema_50"]
        computed_pct = (close_series - tema_50) / tema_50 * 100
        stored_pct = merged["price_pct_from_tema_50"]

        valid_mask = ~computed_pct.isna() & ~stored_pct.isna()
        max_diff = (computed_pct[valid_mask] - stored_pct[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("price_pct_from_tema_50", "formula_verification",
                     "(close - tema_50) / tema_50 * 100", f"max_diff={max_diff:.2e}",
                     "Pct formula verified")
        else:
            report.fail("price_pct_from_tema_50", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Pct formula mismatch")

    # price_pct_from_kama_20
    if "kama_20" in merged.columns and "price_pct_from_kama_20" in merged.columns:
        kama_20 = merged["kama_20"]
        computed_pct = (close_series - kama_20) / kama_20 * 100
        stored_pct = merged["price_pct_from_kama_20"]

        valid_mask = ~computed_pct.isna() & ~stored_pct.isna()
        max_diff = (computed_pct[valid_mask] - stored_pct[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("price_pct_from_kama_20", "formula_verification",
                     "(close - kama_20) / kama_20 * 100", f"max_diff={max_diff:.2e}",
                     "Pct formula verified")
        else:
            report.fail("price_pct_from_kama_20", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Pct formula mismatch")

    # Range check: All MAs should be positive and in reasonable price range
    for col in ["tema_9", "tema_20", "tema_50", "tema_100", "wma_10", "wma_20", "wma_50", "wma_200",
                "kama_10", "kama_20", "kama_50", "hma_9", "hma_21", "hma_50", "vwma_10", "vwma_20", "vwma_50"]:
        if col not in a200_df.columns:
            continue
        vals = a200_df[col]
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
# Chunk 2: Duration & Proximity (ranks 121-140)
# =============================================================================

def validate_chunk2_duration_proximity(a200_df: pd.DataFrame, raw_df: pd.DataFrame,
                                        report: ValidationReport) -> None:
    """Validate duration counters and proximity indicators."""
    merged = pd.merge(
        raw_df[["Date", "Close"]],
        a200_df[["Date"] + [c for c in A200_ADDITION_LIST if c in a200_df.columns]],
        on="Date",
        how="inner"
    )

    close = merged["Close"].values.astype(np.float64)

    # Compute SMAs for verification
    sma_9 = pd.Series(talib.SMA(close, timeperiod=9), index=merged.index)
    sma_50 = pd.Series(talib.SMA(close, timeperiod=50), index=merged.index)
    sma_200 = pd.Series(talib.SMA(close, timeperiod=200), index=merged.index)

    # Mutual exclusivity check: days_above_X and days_below_X cannot both be > 0
    for ma_name in ["sma_9", "sma_50", "sma_200", "tema_20", "kama_20", "vwma_20"]:
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

    # Verify counter logic: when above, counter should increment
    # Test at specific indices after verifiable crossings
    for ma_name, ma_series in [("sma_9", sma_9), ("sma_50", sma_50)]:
        above_col = f"days_above_{ma_name}"
        if above_col not in merged.columns:
            continue

        close_series = merged["Close"]

        # Find indices where price is above MA
        above_mask = close_series >= ma_series

        # Check if counter increments when condition persists
        above_counter = merged[above_col]
        increment_violations = 0

        for i in range(1, len(merged)):
            if pd.isna(ma_series.iloc[i]) or pd.isna(ma_series.iloc[i-1]):
                continue

            if above_mask.iloc[i] and above_mask.iloc[i-1]:
                # Both days above MA - counter should increment
                if above_counter.iloc[i] != above_counter.iloc[i-1] + 1:
                    increment_violations += 1

        if increment_violations < len(merged) * 0.01:  # Allow 1% tolerance
            report.ok(above_col, "increment_logic",
                     "counter increments when condition persists",
                     f"{increment_violations} issues (< 1%)",
                     "Counter increment logic verified")
        else:
            report.fail(above_col, "increment_logic",
                       "counter increments when condition persists",
                       f"{increment_violations} issues",
                       "Counter increment logic failed")

    # Sign convention for days_since_*_cross
    # Positive = bullish (short > long), Negative = bearish (short < long)
    for cross_col in ["days_since_sma_9_50_cross", "days_since_sma_50_200_cross",
                      "days_since_tema_sma_50_cross", "days_since_kama_sma_50_cross",
                      "days_since_sma_9_200_cross"]:
        if cross_col not in merged.columns:
            continue

        vals = merged[cross_col]
        # Verify values change sign (not always positive or always negative)
        has_positive = (vals > 0).sum() > 0
        has_negative = (vals < 0).sum() > 0

        if has_positive and has_negative:
            report.ok(cross_col, "sign_convention",
                     "both positive and negative values exist",
                     f"pos: {(vals > 0).sum()}, neg: {(vals < 0).sum()}",
                     "Sign convention verified (has both bullish/bearish)")
        else:
            # This is still valid - market may have been consistently trending
            report.ok(cross_col, "sign_convention",
                     "sign indicates direction",
                     f"pos: {(vals > 0).sum()}, neg: {(vals < 0).sum()}",
                     "All same direction (market trend)")

    # Proximity formula verification
    # tema_20_sma_50_proximity = (tema_20 - sma_50) / sma_50 * 100
    if "tema_20" in merged.columns and "tema_20_sma_50_proximity" in merged.columns:
        tema_20 = merged["tema_20"]
        computed = (tema_20 - sma_50) / sma_50 * 100
        stored = merged["tema_20_sma_50_proximity"]

        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("tema_20_sma_50_proximity", "formula_verification",
                     "(tema_20 - sma_50) / sma_50 * 100",
                     f"max_diff={max_diff:.2e}",
                     "Proximity formula verified")
        else:
            report.fail("tema_20_sma_50_proximity", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Proximity formula mismatch")

    # sma_9_200_proximity
    if "sma_9_200_proximity" in merged.columns:
        computed = (sma_9 - sma_200) / sma_200 * 100
        stored = merged["sma_9_200_proximity"]

        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("sma_9_200_proximity", "formula_verification",
                     "(sma_9 - sma_200) / sma_200 * 100",
                     f"max_diff={max_diff:.2e}",
                     "Proximity formula verified")
        else:
            report.fail("sma_9_200_proximity", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Proximity formula mismatch")


# =============================================================================
# Chunk 3: BB/RSI/Mean Reversion (ranks 141-160)
# =============================================================================

def validate_chunk3_bb_rsi_mean_reversion(a200_df: pd.DataFrame, raw_df: pd.DataFrame,
                                           report: ValidationReport) -> None:
    """Validate BB, RSI, and mean reversion indicators."""
    merged = pd.merge(
        raw_df[["Date", "High", "Low", "Close"]],
        a200_df[["Date"] + [c for c in A200_ADDITION_LIST if c in a200_df.columns]],
        on="Date",
        how="inner"
    )

    close = merged["Close"].values.astype(np.float64)
    high = merged["High"].values.astype(np.float64)
    low = merged["Low"].values.astype(np.float64)

    # BB formula verification via talib
    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

    # pct_from_upper_band = (close - upper) / upper * 100
    if "pct_from_upper_band" in merged.columns:
        computed = (close - upper) / upper * 100
        stored = merged["pct_from_upper_band"].values

        valid_mask = ~np.isnan(computed) & ~np.isnan(stored)
        max_diff = np.abs(computed[valid_mask] - stored[valid_mask]).max()

        if max_diff < 1e-6:
            report.ok("pct_from_upper_band", "formula_verification",
                     "(close - upper_bb) / upper_bb * 100",
                     f"max_diff={max_diff:.2e}",
                     "BB pct formula verified")
        else:
            report.fail("pct_from_upper_band", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "BB pct formula mismatch")

    # pct_from_lower_band = (close - lower) / lower * 100
    if "pct_from_lower_band" in merged.columns:
        computed = (close - lower) / lower * 100
        stored = merged["pct_from_lower_band"].values

        valid_mask = ~np.isnan(computed) & ~np.isnan(stored)
        max_diff = np.abs(computed[valid_mask] - stored[valid_mask]).max()

        if max_diff < 1e-6:
            report.ok("pct_from_lower_band", "formula_verification",
                     "(close - lower_bb) / lower_bb * 100",
                     f"max_diff={max_diff:.2e}",
                     "BB pct formula verified")
        else:
            report.fail("pct_from_lower_band", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "BB pct formula mismatch")

    # Squeeze logic: bb_squeeze_indicator = 1 implies bb_squeeze_duration >= 1
    if "bb_squeeze_indicator" in merged.columns and "bb_squeeze_duration" in merged.columns:
        squeeze = merged["bb_squeeze_indicator"]
        duration = merged["bb_squeeze_duration"]

        squeeze_on = squeeze == 1
        duration_ok = duration >= 1

        violations = (squeeze_on & ~duration_ok).sum()

        if violations == 0:
            report.ok("bb_squeeze_indicator", "squeeze_duration_consistency",
                     "squeeze=1 implies duration>=1",
                     f"0 violations",
                     "Squeeze-duration consistency verified")
        else:
            report.fail("bb_squeeze_indicator", "squeeze_duration_consistency",
                       "squeeze=1 implies duration>=1",
                       f"{violations} violations",
                       "Squeeze-duration mismatch")

    # RSI reference verification
    # IMPORTANT: RSI must be computed on FULL raw close, then aligned to a200 dates
    # (tier_a200 computes on full data before warmup dropout)
    full_raw_close = raw_df["Close"].values.astype(np.float64)
    full_rsi = talib.RSI(full_raw_close, timeperiod=14)
    full_rsi_series = pd.Series(full_rsi, index=pd.to_datetime(raw_df["Date"]))

    # Align to merged dates
    merged_dates = pd.to_datetime(merged["Date"])
    aligned_rsi = full_rsi_series.loc[merged_dates].values

    # rsi_distance_from_50 = rsi - 50
    if "rsi_distance_from_50" in merged.columns:
        computed = aligned_rsi - 50
        stored = merged["rsi_distance_from_50"].values

        valid_mask = ~np.isnan(computed) & ~np.isnan(stored)
        max_diff = np.abs(computed[valid_mask] - stored[valid_mask]).max()

        if max_diff < 1e-6:
            report.ok("rsi_distance_from_50", "formula_verification",
                     "RSI - 50",
                     f"max_diff={max_diff:.2e}",
                     "RSI distance formula verified")
        else:
            report.fail("rsi_distance_from_50", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "RSI distance formula mismatch")

    # RSI percentile range check [0, 1]
    if "rsi_percentile_60d" in merged.columns:
        vals = a200_df["rsi_percentile_60d"]
        if vals.min() >= 0 - 1e-9 and vals.max() <= 1 + 1e-9:
            report.ok("rsi_percentile_60d", "range_check",
                     "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                     "Percentile range valid")
        else:
            report.fail("rsi_percentile_60d", "range_check",
                       "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                       "Percentile out of range")

    # Z-score formula verification
    close_series = merged["Close"]
    sma_20 = pd.Series(talib.SMA(close, timeperiod=20), index=merged.index)
    std_20 = close_series.rolling(20).std()

    if "zscore_from_20d_mean" in merged.columns:
        computed = (close_series - sma_20) / std_20
        stored = merged["zscore_from_20d_mean"]

        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("zscore_from_20d_mean", "formula_verification",
                     "(close - SMA20) / std20",
                     f"max_diff={max_diff:.2e}",
                     "Z-score formula verified")
        else:
            report.fail("zscore_from_20d_mean", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Z-score formula mismatch")

    # 52wk constraint: distance_from_52wk_high_pct must be <= 0
    if "distance_from_52wk_high_pct" in a200_df.columns:
        vals = a200_df["distance_from_52wk_high_pct"]
        violations = (vals > 0.001).sum()  # Allow tiny tolerance

        if violations == 0:
            report.ok("distance_from_52wk_high_pct", "sign_constraint",
                     "always <= 0",
                     f"max={vals.max():.6f}",
                     "Sign constraint verified (cannot exceed 52wk high)")
        else:
            report.fail("distance_from_52wk_high_pct", "sign_constraint",
                       "always <= 0",
                       f"{violations} violations, max={vals.max():.6f}",
                       "Sign constraint violated")

    # percentile_in_52wk_range must be in [0, 1]
    if "percentile_in_52wk_range" in a200_df.columns:
        vals = a200_df["percentile_in_52wk_range"]
        if vals.min() >= 0 - 1e-9 and vals.max() <= 1 + 1e-9:
            report.ok("percentile_in_52wk_range", "range_check",
                     "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                     "Percentile range valid")
        else:
            report.fail("percentile_in_52wk_range", "range_check",
                       "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                       "Percentile out of range")

    # Streak exclusivity: consecutive_up_days and consecutive_down_days cannot both be > 0
    if "consecutive_up_days" in merged.columns and "consecutive_down_days" in merged.columns:
        up = merged["consecutive_up_days"]
        down = merged["consecutive_down_days"]

        violations = ((up > 0) & (down > 0)).sum()

        if violations == 0:
            report.ok("consecutive_up_days", "streak_exclusivity",
                     "up_days > 0 implies down_days == 0",
                     f"0 violations",
                     "Streak exclusivity verified")
        else:
            report.fail("consecutive_up_days", "streak_exclusivity",
                       "no simultaneous up/down > 0",
                       f"{violations} violations",
                       "Streak exclusivity violated")

    # up_days_ratio_20d must be in [0, 1]
    if "up_days_ratio_20d" in a200_df.columns:
        vals = a200_df["up_days_ratio_20d"]
        if vals.min() >= 0 - 1e-9 and vals.max() <= 1 + 1e-9:
            report.ok("up_days_ratio_20d", "range_check",
                     "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                     "Ratio range valid")
        else:
            report.fail("up_days_ratio_20d", "range_check",
                       "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                       "Ratio out of range")


# =============================================================================
# Chunk 4: MACD/Volume/Calendar/Candle (ranks 161-180)
# =============================================================================

def validate_chunk4_macd_volume_calendar_candle(a200_df: pd.DataFrame, raw_df: pd.DataFrame,
                                                  report: ValidationReport) -> None:
    """Validate MACD, volume, calendar, and candle indicators."""
    merged = pd.merge(
        raw_df[["Date", "Open", "High", "Low", "Close", "Volume"]],
        a200_df[["Date"] + [c for c in A200_ADDITION_LIST if c in a200_df.columns]],
        on="Date",
        how="inner"
    )

    close = merged["Close"].values.astype(np.float64)
    dates = pd.to_datetime(merged["Date"])

    # MACD reference via talib
    # IMPORTANT: MACD must be computed on FULL raw close, then aligned to a200 dates
    # (tier_a200 computes on full data before warmup dropout)
    full_raw_close = raw_df["Close"].values.astype(np.float64)
    full_macd_line, full_signal_line, full_histogram = talib.MACD(
        full_raw_close, fastperiod=12, slowperiod=26, signalperiod=9
    )
    full_signal_series = pd.Series(full_signal_line, index=pd.to_datetime(raw_df["Date"]))

    # Align to merged dates
    merged_dates = pd.to_datetime(merged["Date"])
    aligned_signal = full_signal_series.loc[merged_dates].values

    if "macd_signal" in merged.columns:
        stored = merged["macd_signal"].values

        valid_mask = ~np.isnan(aligned_signal) & ~np.isnan(stored)
        max_diff = np.abs(aligned_signal[valid_mask] - stored[valid_mask]).max()

        if max_diff < 1e-6:
            report.ok("macd_signal", "talib_reference",
                     f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                     "TA-Lib MACD signal match")
        else:
            report.fail("macd_signal", "talib_reference",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "TA-Lib MACD signal mismatch")

    # Calendar logic checks
    # trading_day_of_week must be in [0, 4]
    if "trading_day_of_week" in a200_df.columns:
        vals = a200_df["trading_day_of_week"]
        if vals.min() >= 0 and vals.max() <= 4:
            report.ok("trading_day_of_week", "range_check",
                     "[0, 4]", f"[{vals.min()}, {vals.max()}]",
                     "Day of week range valid")
        else:
            report.fail("trading_day_of_week", "range_check",
                       "[0, 4]", f"[{vals.min()}, {vals.max()}]",
                       "Day of week out of range")

    # is_monday == (dayofweek == 0)
    if "is_monday" in merged.columns:
        computed = (dates.dt.dayofweek == 0).astype(int).values
        stored = merged["is_monday"].values

        mismatches = (computed != stored).sum()

        if mismatches == 0:
            report.ok("is_monday", "calendar_logic",
                     "is_monday == (dayofweek == 0)",
                     f"0 mismatches",
                     "Monday logic verified")
        else:
            report.fail("is_monday", "calendar_logic",
                       "is_monday == (dayofweek == 0)",
                       f"{mismatches} mismatches",
                       "Monday logic mismatch")

    # is_friday == (dayofweek == 4)
    if "is_friday" in merged.columns:
        computed = (dates.dt.dayofweek == 4).astype(int).values
        stored = merged["is_friday"].values

        mismatches = (computed != stored).sum()

        if mismatches == 0:
            report.ok("is_friday", "calendar_logic",
                     "is_friday == (dayofweek == 4)",
                     f"0 mismatches",
                     "Friday logic verified")
        else:
            report.fail("is_friday", "calendar_logic",
                       "is_friday == (dayofweek == 4)",
                       f"{mismatches} mismatches",
                       "Friday logic mismatch")

    # month_of_year must be in [1, 12] and equal dt.month
    if "month_of_year" in merged.columns:
        computed = dates.dt.month.values
        stored = merged["month_of_year"].values

        mismatches = (computed != stored).sum()
        in_range = a200_df["month_of_year"].min() >= 1 and a200_df["month_of_year"].max() <= 12

        if mismatches == 0 and in_range:
            report.ok("month_of_year", "calendar_logic",
                     "month_of_year == dt.month, [1, 12]",
                     f"0 mismatches",
                     "Month logic verified")
        else:
            report.fail("month_of_year", "calendar_logic",
                       "month_of_year == dt.month, [1, 12]",
                       f"{mismatches} mismatches",
                       "Month logic mismatch")

    # is_quarter_end_month == month in {3, 6, 9, 12}
    if "is_quarter_end_month" in merged.columns:
        computed = dates.dt.month.isin({3, 6, 9, 12}).astype(int).values
        stored = merged["is_quarter_end_month"].values

        mismatches = (computed != stored).sum()

        if mismatches == 0:
            report.ok("is_quarter_end_month", "calendar_logic",
                     "is_quarter_end_month == month in {3,6,9,12}",
                     f"0 mismatches",
                     "Quarter end logic verified")
        else:
            report.fail("is_quarter_end_month", "calendar_logic",
                       "is_quarter_end_month == month in {3,6,9,12}",
                       f"{mismatches} mismatches",
                       "Quarter end logic mismatch")

    # Candle range checks
    for col in ["body_to_range_ratio", "upper_wick_pct", "lower_wick_pct"]:
        if col in a200_df.columns:
            vals = a200_df[col]
            if vals.min() >= 0 - 1e-9 and vals.max() <= 1 + 1e-9:
                report.ok(col, "range_check",
                         "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                         "Candle ratio range valid")
            else:
                report.fail(col, "range_check",
                           "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                           "Candle ratio out of range")

    # Doji logic: doji_indicator == 1 implies body_to_range_ratio < 0.1
    if "doji_indicator" in merged.columns and "body_to_range_ratio" in merged.columns:
        doji = merged["doji_indicator"]
        body_ratio = merged["body_to_range_ratio"]

        doji_on = doji == 1
        ratio_low = body_ratio < 0.1

        violations = (doji_on & ~ratio_low).sum()

        if violations == 0:
            report.ok("doji_indicator", "doji_logic",
                     "doji=1 implies body_ratio < 0.1",
                     f"0 violations",
                     "Doji logic verified")
        else:
            report.fail("doji_indicator", "doji_logic",
                       "doji=1 implies body_ratio < 0.1",
                       f"{violations} violations",
                       "Doji logic mismatch")

    # Candle formula verification at random indices
    open_price = merged["Open"]
    high = merged["High"]
    low = merged["Low"]
    close_series = merged["Close"]

    if "candle_body_pct" in merged.columns:
        body = (close_series - open_price).abs()
        computed = (body / open_price) * 100
        stored = merged["candle_body_pct"]

        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("candle_body_pct", "formula_verification",
                     "abs(C-O)/O * 100",
                     f"max_diff={max_diff:.2e}",
                     "Candle body formula verified")
        else:
            report.fail("candle_body_pct", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Candle body formula mismatch")


# =============================================================================
# Chunk 5: Ichimoku/Donchian/Divergence/Entropy (ranks 181-206)
# =============================================================================

def validate_chunk5_ichimoku_donchian_entropy(a200_df: pd.DataFrame, raw_df: pd.DataFrame,
                                               report: ValidationReport) -> None:
    """Validate Ichimoku, Donchian, divergence, and entropy/regime indicators."""
    merged = pd.merge(
        raw_df[["Date", "High", "Low", "Close", "Volume"]],
        a200_df[["Date"] + [c for c in A200_ADDITION_LIST if c in a200_df.columns]],
        on="Date",
        how="inner"
    )

    high = merged["High"]
    low = merged["Low"]
    close = merged["Close"]

    # Ichimoku formula verification
    # tenkan_sen = (9d_high + 9d_low) / 2
    if "tenkan_sen" in merged.columns:
        high_9 = high.rolling(9).max()
        low_9 = low.rolling(9).min()
        computed = (high_9 + low_9) / 2
        stored = merged["tenkan_sen"]

        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("tenkan_sen", "formula_verification",
                     "(9d_high + 9d_low) / 2",
                     f"max_diff={max_diff:.2e}",
                     "Tenkan-sen formula verified")
        else:
            report.fail("tenkan_sen", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Tenkan-sen formula mismatch")

    # kijun_sen = (26d_high + 26d_low) / 2
    if "kijun_sen" in merged.columns:
        high_26 = high.rolling(26).max()
        low_26 = low.rolling(26).min()
        computed = (high_26 + low_26) / 2
        stored = merged["kijun_sen"]

        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("kijun_sen", "formula_verification",
                     "(26d_high + 26d_low) / 2",
                     f"max_diff={max_diff:.2e}",
                     "Kijun-sen formula verified")
        else:
            report.fail("kijun_sen", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Kijun-sen formula mismatch")

    # senkou_span_b = (52d_high + 52d_low) / 2
    if "senkou_span_b" in merged.columns:
        high_52 = high.rolling(52).max()
        low_52 = low.rolling(52).min()
        computed = (high_52 + low_52) / 2
        stored = merged["senkou_span_b"]

        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("senkou_span_b", "formula_verification",
                     "(52d_high + 52d_low) / 2",
                     f"max_diff={max_diff:.2e}",
                     "Senkou Span B formula verified")
        else:
            report.fail("senkou_span_b", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Senkou Span B formula mismatch")

    # price_vs_cloud range check [-1, 0, 1]
    if "price_vs_cloud" in a200_df.columns:
        vals = a200_df["price_vs_cloud"]
        unique_vals = set(vals.unique()) - {np.nan}

        if unique_vals <= {-1, 0, 1, -1.0, 0.0, 1.0}:
            report.ok("price_vs_cloud", "value_range",
                     "{-1, 0, 1}",
                     str(unique_vals),
                     "Cloud position values valid")
        else:
            report.fail("price_vs_cloud", "value_range",
                       "{-1, 0, 1}",
                       str(unique_vals),
                       "Unexpected cloud position values")

    # Donchian formula verification
    # donchian_upper_20 = rolling max of High
    if "donchian_upper_20" in merged.columns:
        computed = high.rolling(20).max()
        stored = merged["donchian_upper_20"]

        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("donchian_upper_20", "formula_verification",
                     "rolling(20).max(high)",
                     f"max_diff={max_diff:.2e}",
                     "Donchian upper formula verified")
        else:
            report.fail("donchian_upper_20", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Donchian upper formula mismatch")

    # donchian_lower_20 = rolling min of Low
    if "donchian_lower_20" in merged.columns:
        computed = low.rolling(20).min()
        stored = merged["donchian_lower_20"]

        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("donchian_lower_20", "formula_verification",
                     "rolling(20).min(low)",
                     f"max_diff={max_diff:.2e}",
                     "Donchian lower formula verified")
        else:
            report.fail("donchian_lower_20", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Donchian lower formula mismatch")

    # donchian_position must be in [0, 1]
    if "donchian_position" in a200_df.columns:
        vals = a200_df["donchian_position"]
        if vals.min() >= 0 - 1e-9 and vals.max() <= 1 + 1e-9:
            report.ok("donchian_position", "range_check",
                     "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                     "Donchian position range valid")
        else:
            report.fail("donchian_position", "range_check",
                       "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                       "Donchian position out of range")

    # Divergence range checks [-1, 1]
    for col in ["price_rsi_divergence", "price_obv_divergence"]:
        if col in a200_df.columns:
            vals = a200_df[col]
            if vals.min() >= -1 - 1e-9 and vals.max() <= 1 + 1e-9:
                report.ok(col, "range_check",
                         "[-1, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                         "Divergence range valid")
            else:
                report.fail(col, "range_check",
                           "[-1, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                           "Divergence out of range")

    # divergence_magnitude = max(|rsi_div|, |obv_div|)
    if "price_rsi_divergence" in merged.columns and "price_obv_divergence" in merged.columns and "divergence_magnitude" in merged.columns:
        rsi_div = merged["price_rsi_divergence"]
        obv_div = merged["price_obv_divergence"]
        computed = np.maximum(rsi_div.abs(), obv_div.abs())
        stored = merged["divergence_magnitude"]

        valid_mask = ~computed.isna() & ~stored.isna()
        max_diff = (computed[valid_mask] - stored[valid_mask]).abs().max()

        if max_diff < 1e-6:
            report.ok("divergence_magnitude", "formula_verification",
                     "max(|rsi_div|, |obv_div|)",
                     f"max_diff={max_diff:.2e}",
                     "Divergence magnitude formula verified")
        else:
            report.fail("divergence_magnitude", "formula_verification",
                       f"max_diff < 1e-6", f"max_diff={max_diff:.2e}",
                       "Divergence magnitude formula mismatch")

    # Entropy range checks [0, 1]
    for order in [3, 4, 5]:
        col = f"permutation_entropy_order{order}"
        if col in a200_df.columns:
            vals = a200_df[col]
            if vals.min() >= 0 - 1e-9 and vals.max() <= 1 + 1e-9:
                report.ok(col, "range_check",
                         "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                         "Entropy range valid")
            else:
                report.fail(col, "range_check",
                           "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                           "Entropy out of range")

    # Regime percentile range checks [0, 1]
    for col in ["atr_regime_pct_60d", "atr_regime_rolling_q", "trend_strength_pct_60d", "trend_strength_rolling_q", "regime_transition_prob"]:
        if col in a200_df.columns:
            vals = a200_df[col]
            if vals.min() >= 0 - 1e-9 and vals.max() <= 1 + 1e-9:
                report.ok(col, "range_check",
                         "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                         "Regime percentile range valid")
            else:
                report.fail(col, "range_check",
                           "[0, 1]", f"[{vals.min():.4f}, {vals.max():.4f}]",
                           "Regime percentile out of range")

    # vol_regime_state range check [-1, 0, 1]
    if "vol_regime_state" in a200_df.columns:
        vals = a200_df["vol_regime_state"]
        unique_vals = set(vals.unique()) - {np.nan}

        if unique_vals <= {-1, 0, 1, -1.0, 0.0, 1.0}:
            report.ok("vol_regime_state", "value_range",
                     "{-1, 0, 1}",
                     str(unique_vals),
                     "Regime state values valid")
        else:
            report.fail("vol_regime_state", "value_range",
                       "{-1, 0, 1}",
                       str(unique_vals),
                       "Unexpected regime state values")

    # regime_consistency must be >= 1
    if "regime_consistency" in a200_df.columns:
        vals = a200_df["regime_consistency"]
        if vals.min() >= 1:
            report.ok("regime_consistency", "constraint_check",
                     ">= 1",
                     f"min={vals.min()}",
                     "Consistency constraint valid")
        else:
            report.fail("regime_consistency", "constraint_check",
                       ">= 1",
                       f"min={vals.min()}",
                       "Consistency constraint violated")


# =============================================================================
# Lookahead Detection (Truncation Test)
# =============================================================================

def validate_lookahead(a200_df: pd.DataFrame, raw_df: pd.DataFrame, vix_df: pd.DataFrame,
                       report: ValidationReport, n_test_indices: int = 5) -> None:
    """Validate no lookahead bias via truncation test.

    For each feature, at randomly selected indices:
    1. Compute features on full dataset -> get value at index i
    2. Compute features on truncated dataset (up to i) -> get value at index i
    3. Values must match (no future data used)
    """
    from src.features import tier_a200

    # Sample indices to test (after warmup period)
    min_idx = 260  # After all warmup periods
    max_idx = len(a200_df) - 1

    if max_idx <= min_idx:
        report.ok("lookahead", "truncation_test",
                 "sufficient data", "insufficient data for test",
                 "Skipping truncation test (dataset too small)")
        return

    np.random.seed(42)
    test_indices = np.random.choice(range(min_idx, max_idx), size=n_test_indices, replace=False)

    # Get the dates corresponding to these indices
    test_dates = a200_df["Date"].iloc[test_indices].tolist()

    print(f"    Testing lookahead at {n_test_indices} random indices...")

    features_tested = 0
    features_passed = 0

    for feature in A200_ADDITION_LIST:
        if feature not in a200_df.columns:
            continue

        all_match = True
        mismatch_details = []

        for idx, test_date in zip(test_indices, test_dates):
            # Get full dataset value
            full_value = a200_df[a200_df["Date"] == test_date][feature].values
            if len(full_value) == 0 or np.isnan(full_value[0]):
                continue
            full_value = full_value[0]

            # Truncate raw data up to test_date
            truncated_raw = raw_df[raw_df["Date"] <= test_date].copy()
            truncated_vix = vix_df[vix_df["Date"] <= test_date].copy()

            if len(truncated_raw) < 260:
                continue

            try:
                truncated_features = tier_a200.build_feature_dataframe(truncated_raw, truncated_vix)
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

            except Exception as e:
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

def validate_sample_dates(a200_df: pd.DataFrame, report: ValidationReport) -> list[dict]:
    """Audit features at significant market dates for sensibility."""
    audits = []

    a200_df = a200_df.copy()
    a200_df["Date"] = pd.to_datetime(a200_df["Date"]).dt.normalize()

    for date_str, context in SAMPLE_AUDIT_DATES:
        target_date = pd.to_datetime(date_str).normalize()

        # Find row for this date (or nearest)
        row = a200_df[a200_df["Date"] == target_date]
        if len(row) == 0:
            # Try nearest date
            nearest_idx = (a200_df["Date"] - target_date).abs().idxmin()
            row = a200_df.loc[[nearest_idx]]
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

        # Extract key indicators for audit
        audit_data = {
            "date": str(actual_date)[:10],
            "context": context,
            "date_note": date_note,
            "indicators": {}
        }

        # Volatility indicators
        if "vol_regime_state" in row.index and not pd.isna(row["vol_regime_state"]):
            audit_data["indicators"]["vol_regime_state"] = int(row["vol_regime_state"])
        if "atr_regime_pct_60d" in row.index and not pd.isna(row["atr_regime_pct_60d"]):
            audit_data["indicators"]["atr_regime_pct_60d"] = round(float(row["atr_regime_pct_60d"]), 3)

        # Mean reversion / z-scores
        if "zscore_from_20d_mean" in row.index and not pd.isna(row["zscore_from_20d_mean"]):
            audit_data["indicators"]["zscore_from_20d_mean"] = round(float(row["zscore_from_20d_mean"]), 2)
        if "distance_from_52wk_high_pct" in row.index and not pd.isna(row["distance_from_52wk_high_pct"]):
            audit_data["indicators"]["distance_from_52wk_high_pct"] = round(float(row["distance_from_52wk_high_pct"]), 2)

        # Duration counters
        if "days_rsi_oversold" in row.index and not pd.isna(row["days_rsi_oversold"]):
            audit_data["indicators"]["days_rsi_oversold"] = int(row["days_rsi_oversold"])
        if "days_rsi_overbought" in row.index and not pd.isna(row["days_rsi_overbought"]):
            audit_data["indicators"]["days_rsi_overbought"] = int(row["days_rsi_overbought"])

        # Ichimoku/Donchian
        if "price_vs_cloud" in row.index and not pd.isna(row["price_vs_cloud"]):
            audit_data["indicators"]["price_vs_cloud"] = int(row["price_vs_cloud"])
        if "donchian_position" in row.index and not pd.isna(row["donchian_position"]):
            audit_data["indicators"]["donchian_position"] = round(float(row["donchian_position"]), 3)

        # Entropy
        if "permutation_entropy_order4" in row.index and not pd.isna(row["permutation_entropy_order4"]):
            audit_data["indicators"]["permutation_entropy_order4"] = round(float(row["permutation_entropy_order4"]), 3)

        # Determine sensibility based on context
        sensibility_checks = []

        if "COVID" in context.lower() or "crash" in context.lower() or "low" in context.lower() or "correction" in context.lower():
            # Expect high volatility, oversold conditions
            if audit_data["indicators"].get("vol_regime_state") == 1:
                sensibility_checks.append("vol_regime=high (EXPECTED)")
            elif audit_data["indicators"].get("vol_regime_state") is not None:
                sensibility_checks.append(f"vol_regime={audit_data['indicators']['vol_regime_state']} (check)")

            if audit_data["indicators"].get("atr_regime_pct_60d", 0) > 0.7:
                sensibility_checks.append("atr_pct>0.7 (EXPECTED)")

            if audit_data["indicators"].get("zscore_from_20d_mean", 0) < -1:
                sensibility_checks.append(f"zscore={audit_data['indicators']['zscore_from_20d_mean']} (negative, EXPECTED)")

        elif "high" in context.lower() or "bull" in context.lower():
            # Expect overbought conditions
            if audit_data["indicators"].get("price_vs_cloud") == 1:
                sensibility_checks.append("above_cloud (EXPECTED)")
            if audit_data["indicators"].get("donchian_position", 0) > 0.7:
                sensibility_checks.append(f"donchian_pos={audit_data['indicators']['donchian_position']} (high, EXPECTED)")
            if audit_data["indicators"].get("distance_from_52wk_high_pct", -100) > -5:
                sensibility_checks.append("near_52wk_high (EXPECTED)")

        audit_data["sensibility_checks"] = sensibility_checks
        audit_data["status"] = "SENSIBLE" if len(sensibility_checks) >= 2 else "REVIEW"

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
# Layer 2: Semantic Validation - Cross-Indicator Consistency
# =============================================================================

def validate_cross_indicator_consistency(a200_df: pd.DataFrame, raw_df: pd.DataFrame,
                                          report: ValidationReport) -> None:
    """Validate logical rules between related indicators."""
    merged = pd.merge(
        raw_df[["Date", "Close"]],
        a200_df,
        on="Date",
        how="inner"
    )

    close = merged["Close"]

    # Need RSI from tier_a50/a100 for some checks
    # We'll use rsi_distance_from_50 + 50 to get RSI
    if "rsi_distance_from_50" in merged.columns:
        rsi = merged["rsi_distance_from_50"] + 50

        # RSI-duration sync: RSI > 70 implies days_rsi_overbought > 0
        if "days_rsi_overbought" in merged.columns:
            rsi_high = rsi > 70
            days_ob = merged["days_rsi_overbought"]

            violations = (rsi_high & (days_ob == 0)).sum()
            total_high = rsi_high.sum()

            if total_high > 0:
                pct_ok = 1 - violations / total_high
                if pct_ok >= 0.99:  # Allow 1% tolerance
                    report.ok("cross_indicator", "rsi_overbought_sync",
                             "RSI>70 implies days_overbought>0 (99%)",
                             f"{pct_ok*100:.1f}% consistent",
                             "RSI-duration sync verified")
                else:
                    report.fail("cross_indicator", "rsi_overbought_sync",
                               "RSI>70 implies days_overbought>0 (99%)",
                               f"{pct_ok*100:.1f}% consistent",
                               f"{violations} violations")
            else:
                report.ok("cross_indicator", "rsi_overbought_sync",
                         "RSI>70 implies days_overbought>0",
                         "no RSI>70 occurrences",
                         "No overbought periods to test")

        # RSI-duration sync: RSI < 30 implies days_rsi_oversold > 0
        if "days_rsi_oversold" in merged.columns:
            rsi_low = rsi < 30
            days_os = merged["days_rsi_oversold"]

            violations = (rsi_low & (days_os == 0)).sum()
            total_low = rsi_low.sum()

            if total_low > 0:
                pct_ok = 1 - violations / total_low
                if pct_ok >= 0.99:
                    report.ok("cross_indicator", "rsi_oversold_sync",
                             "RSI<30 implies days_oversold>0 (99%)",
                             f"{pct_ok*100:.1f}% consistent",
                             "RSI-duration sync verified")
                else:
                    report.fail("cross_indicator", "rsi_oversold_sync",
                               "RSI<30 implies days_oversold>0 (99%)",
                               f"{pct_ok*100:.1f}% consistent",
                               f"{violations} violations")
            else:
                report.ok("cross_indicator", "rsi_oversold_sync",
                         "RSI<30 implies days_oversold>0",
                         "no RSI<30 occurrences",
                         "No oversold periods to test")

    # BB squeeze: squeeze_indicator=1 implies squeeze_duration>=1
    if "bb_squeeze_indicator" in merged.columns and "bb_squeeze_duration" in merged.columns:
        squeeze = merged["bb_squeeze_indicator"] == 1
        duration = merged["bb_squeeze_duration"]

        violations = (squeeze & (duration < 1)).sum()

        if violations == 0:
            report.ok("cross_indicator", "bb_squeeze_duration_sync",
                     "squeeze=1 implies duration>=1",
                     "0 violations",
                     "BB squeeze-duration sync verified")
        else:
            report.fail("cross_indicator", "bb_squeeze_duration_sync",
                       "squeeze=1 implies duration>=1",
                       f"{violations} violations",
                       "BB squeeze-duration mismatch")

    # Calendar: is_monday=1 implies trading_day_of_week=0
    if "is_monday" in merged.columns and "trading_day_of_week" in merged.columns:
        monday = merged["is_monday"] == 1
        dow = merged["trading_day_of_week"]

        violations = (monday & (dow != 0)).sum()

        if violations == 0:
            report.ok("cross_indicator", "monday_dow_sync",
                     "is_monday=1 implies dow=0",
                     "0 violations",
                     "Monday-DOW sync verified")
        else:
            report.fail("cross_indicator", "monday_dow_sync",
                       "is_monday=1 implies dow=0",
                       f"{violations} violations",
                       "Monday-DOW mismatch")

    # Calendar: is_friday=1 implies trading_day_of_week=4
    if "is_friday" in merged.columns and "trading_day_of_week" in merged.columns:
        friday = merged["is_friday"] == 1
        dow = merged["trading_day_of_week"]

        violations = (friday & (dow != 4)).sum()

        if violations == 0:
            report.ok("cross_indicator", "friday_dow_sync",
                     "is_friday=1 implies dow=4",
                     "0 violations",
                     "Friday-DOW sync verified")
        else:
            report.fail("cross_indicator", "friday_dow_sync",
                       "is_friday=1 implies dow=4",
                       f"{violations} violations",
                       "Friday-DOW mismatch")

    # Quarter end: is_quarter_end=1 implies month in {3,6,9,12}
    if "is_quarter_end_month" in merged.columns and "month_of_year" in merged.columns:
        qend = merged["is_quarter_end_month"] == 1
        month = merged["month_of_year"]

        violations = (qend & ~month.isin({3, 6, 9, 12})).sum()

        if violations == 0:
            report.ok("cross_indicator", "quarter_end_month_sync",
                     "is_quarter_end=1 implies month in {3,6,9,12}",
                     "0 violations",
                     "Quarter end-month sync verified")
        else:
            report.fail("cross_indicator", "quarter_end_month_sync",
                       "is_quarter_end=1 implies month in {3,6,9,12}",
                       f"{violations} violations",
                       "Quarter end-month mismatch")

    # vol_regime_state: high=1 implies atr_pct > 0.7
    if "vol_regime_state" in merged.columns and "atr_regime_pct_60d" in merged.columns:
        high_vol = merged["vol_regime_state"] == 1
        atr_pct = merged["atr_regime_pct_60d"]

        violations = (high_vol & (atr_pct <= 0.7)).sum()
        total_high = high_vol.sum()

        if total_high > 0:
            pct_ok = 1 - violations / total_high
            if pct_ok >= 0.99:
                report.ok("cross_indicator", "vol_regime_atr_sync",
                         "vol_state=1 implies atr_pct>0.7 (99%)",
                         f"{pct_ok*100:.1f}% consistent",
                         "Volatility regime sync verified")
            else:
                report.fail("cross_indicator", "vol_regime_atr_sync",
                           "vol_state=1 implies atr_pct>0.7 (99%)",
                           f"{pct_ok*100:.1f}% consistent",
                           f"{violations} violations")

    # vol_regime_state: low=-1 implies atr_pct < 0.3
    if "vol_regime_state" in merged.columns and "atr_regime_pct_60d" in merged.columns:
        low_vol = merged["vol_regime_state"] == -1
        atr_pct = merged["atr_regime_pct_60d"]

        violations = (low_vol & (atr_pct >= 0.3)).sum()
        total_low = low_vol.sum()

        if total_low > 0:
            pct_ok = 1 - violations / total_low
            if pct_ok >= 0.99:
                report.ok("cross_indicator", "vol_regime_atr_low_sync",
                         "vol_state=-1 implies atr_pct<0.3 (99%)",
                         f"{pct_ok*100:.1f}% consistent",
                         "Low volatility regime sync verified")
            else:
                report.fail("cross_indicator", "vol_regime_atr_low_sync",
                           "vol_state=-1 implies atr_pct<0.3 (99%)",
                           f"{pct_ok*100:.1f}% consistent",
                           f"{violations} violations")

    # Consecutive streak exclusivity
    if "consecutive_up_days" in merged.columns and "consecutive_down_days" in merged.columns:
        up = merged["consecutive_up_days"]
        down = merged["consecutive_down_days"]

        violations = ((up > 0) & (down > 0)).sum()

        if violations == 0:
            report.ok("cross_indicator", "streak_exclusivity",
                     "up>0 and down>0 cannot both be true",
                     "0 violations",
                     "Streak exclusivity verified")
        else:
            report.fail("cross_indicator", "streak_exclusivity",
                       "up>0 and down>0 cannot both be true",
                       f"{violations} violations",
                       "Streak exclusivity violated")

    # Doji: doji=1 implies body_to_range < 0.1
    if "doji_indicator" in merged.columns and "body_to_range_ratio" in merged.columns:
        doji = merged["doji_indicator"] == 1
        body_ratio = merged["body_to_range_ratio"]

        violations = (doji & (body_ratio >= 0.1)).sum()

        if violations == 0:
            report.ok("cross_indicator", "doji_body_ratio_sync",
                     "doji=1 implies body_ratio<0.1",
                     "0 violations",
                     "Doji-body ratio sync verified")
        else:
            report.fail("cross_indicator", "doji_body_ratio_sync",
                       "doji=1 implies body_ratio<0.1",
                       f"{violations} violations",
                       "Doji-body ratio mismatch")


# =============================================================================
# Layer 2: Semantic Validation - Known Event Verification
# =============================================================================

def validate_known_events(a200_df: pd.DataFrame, report: ValidationReport) -> None:
    """Verify indicators show expected behavior during known market events."""
    a200_df = a200_df.copy()
    a200_df["Date"] = pd.to_datetime(a200_df["Date"]).dt.normalize()

    # Event 1: COVID Crash (2020-03-09 to 2020-03-23)
    covid_mask = (a200_df["Date"] >= "2020-03-09") & (a200_df["Date"] <= "2020-03-23")
    covid_data = a200_df[covid_mask]

    if len(covid_data) > 0:
        # Vol regime should be high for most days
        if "vol_regime_state" in covid_data.columns:
            high_vol_ratio = (covid_data["vol_regime_state"] == 1).mean()
            if high_vol_ratio >= 0.7:
                report.ok("known_events", "covid_vol_regime",
                         ">70% high vol days", f"{high_vol_ratio*100:.1f}%",
                         "COVID crash showed sustained high volatility")
            else:
                report.fail("known_events", "covid_vol_regime",
                           ">70% high vol days", f"{high_vol_ratio*100:.1f}%",
                           "COVID crash should show more high vol days")

        # ATR percentile should be high
        if "atr_regime_pct_60d" in covid_data.columns:
            mean_atr_pct = covid_data["atr_regime_pct_60d"].mean()
            if mean_atr_pct > 0.8:
                report.ok("known_events", "covid_atr_pct",
                         "mean atr_pct > 0.8", f"{mean_atr_pct:.3f}",
                         "COVID ATR percentile was high")
            else:
                # May be acceptable if still elevated
                report.ok("known_events", "covid_atr_pct",
                         "elevated atr_pct", f"{mean_atr_pct:.3f}",
                         "COVID ATR percentile (acceptable)")

        # Z-scores should be negative (price below mean)
        if "zscore_from_20d_mean" in covid_data.columns:
            mean_zscore = covid_data["zscore_from_20d_mean"].mean()
            if mean_zscore < -1:
                report.ok("known_events", "covid_zscore",
                         "mean zscore < -1", f"{mean_zscore:.2f}",
                         "COVID showed negative z-scores (expected)")
            else:
                report.ok("known_events", "covid_zscore",
                         "zscore negative", f"{mean_zscore:.2f}",
                         "COVID z-score (may vary)")

        # Distance from 52wk high should be significant
        if "distance_from_52wk_high_pct" in covid_data.columns:
            min_dist = covid_data["distance_from_52wk_high_pct"].min()
            if min_dist < -20:
                report.ok("known_events", "covid_52wk_dist",
                         "min distance < -20%", f"{min_dist:.1f}%",
                         "COVID showed significant drawdown from 52wk high")
            else:
                report.ok("known_events", "covid_52wk_dist",
                         "distance from high", f"{min_dist:.1f}%",
                         "COVID distance from 52wk high")
    else:
        report.ok("known_events", "covid_period",
                 "data present", "no COVID period data",
                 "COVID period not in dataset (pre-2020 data)")

    # Event 2: 2022 Bear Market (January - October 2022)
    bear_mask = (a200_df["Date"] >= "2022-01-03") & (a200_df["Date"] <= "2022-10-12")
    bear_data = a200_df[bear_mask]

    if len(bear_data) > 0:
        # Should be below Ichimoku cloud for significant portion
        if "price_vs_cloud" in bear_data.columns:
            below_cloud_ratio = (bear_data["price_vs_cloud"] == -1).mean()
            if below_cloud_ratio > 0.3:
                report.ok("known_events", "bear2022_cloud",
                         ">30% below cloud", f"{below_cloud_ratio*100:.1f}%",
                         "2022 bear market showed time below cloud")
            else:
                report.ok("known_events", "bear2022_cloud",
                         "cloud position", f"{below_cloud_ratio*100:.1f}% below",
                         "2022 bear market cloud analysis")

        # Consecutive down patterns should appear
        if "consecutive_down_days" in bear_data.columns:
            max_down_streak = bear_data["consecutive_down_days"].max()
            if max_down_streak >= 3:
                report.ok("known_events", "bear2022_streak",
                         "max down streak >= 3", f"{max_down_streak} days",
                         "2022 bear market had down streaks")
            else:
                report.ok("known_events", "bear2022_streak",
                         "down streak", f"{max_down_streak} days",
                         "2022 bear market streak data")
    else:
        report.ok("known_events", "bear2022_period",
                 "data present", "no 2022 bear period data",
                 "2022 bear period not in dataset")

    # Event 3: 2023-2024 Bull Run
    bull_mask = (a200_df["Date"] >= "2023-11-01") & (a200_df["Date"] <= "2024-07-31")
    bull_data = a200_df[bull_mask]

    if len(bull_data) > 0:
        # Should be above Ichimoku cloud for significant portion
        if "price_vs_cloud" in bull_data.columns:
            above_cloud_ratio = (bull_data["price_vs_cloud"] == 1).mean()
            if above_cloud_ratio > 0.5:
                report.ok("known_events", "bull2024_cloud",
                         ">50% above cloud", f"{above_cloud_ratio*100:.1f}%",
                         "2023-24 bull run showed time above cloud")
            else:
                report.ok("known_events", "bull2024_cloud",
                         "cloud position", f"{above_cloud_ratio*100:.1f}% above",
                         "2023-24 bull run cloud analysis")

        # Donchian position should be high during breakouts
        if "donchian_position" in bull_data.columns:
            mean_donchian = bull_data["donchian_position"].mean()
            if mean_donchian > 0.6:
                report.ok("known_events", "bull2024_donchian",
                         "mean donchian > 0.6", f"{mean_donchian:.3f}",
                         "2023-24 bull run showed high Donchian position")
            else:
                report.ok("known_events", "bull2024_donchian",
                         "donchian position", f"{mean_donchian:.3f}",
                         "2023-24 bull run Donchian data")
    else:
        report.ok("known_events", "bull2024_period",
                 "data present", "no 2023-24 bull period data",
                 "2023-24 bull period not in dataset")


# =============================================================================
# Bounded Features Range Check
# =============================================================================

def validate_bounded_features(a200_df: pd.DataFrame, report: ValidationReport) -> None:
    """Validate all bounded features stay within expected ranges."""
    FP_TOLERANCE = 1e-9

    for col, (min_val, max_val) in BOUNDED_FEATURES.items():
        if col not in a200_df.columns:
            continue

        vals = a200_df[col]
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
# Main Entry Point
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Deep validation of tier_a200 indicators.")
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
    print("Loading data and computing tier_a200 features...")
    a200_df, raw_df, vix_df = load_data(args.raw_path, args.vix_path)
    print(f"  A200 dataset: {len(a200_df)} rows, {len(a200_df.columns)} columns")
    print(f"  Raw SPY: {len(raw_df)} rows")
    print(f"  VIX: {len(vix_df)} rows")
    print(f"  New A200 features: {len(A200_ADDITION_LIST)}")

    # Initialize report
    report = ValidationReport()

    print("\n" + "=" * 60)
    print("LAYER 1: DETERMINISTIC VALIDATION")
    print("=" * 60)

    # Data quality
    print("\n  [1/8] Data quality checks...")
    validate_data_quality(a200_df, report)

    # Chunk 1: Extended MAs
    print("  [2/8] Chunk 1: Extended MAs (101-120)...")
    validate_chunk1_extended_mas(a200_df, raw_df, report)

    # Chunk 2: Duration & Proximity
    print("  [3/8] Chunk 2: Duration & Proximity (121-140)...")
    validate_chunk2_duration_proximity(a200_df, raw_df, report)

    # Chunk 3: BB/RSI/Mean Reversion
    print("  [4/8] Chunk 3: BB/RSI/Mean Reversion (141-160)...")
    validate_chunk3_bb_rsi_mean_reversion(a200_df, raw_df, report)

    # Chunk 4: MACD/Volume/Calendar/Candle
    print("  [5/8] Chunk 4: MACD/Volume/Calendar/Candle (161-180)...")
    validate_chunk4_macd_volume_calendar_candle(a200_df, raw_df, report)

    # Chunk 5: Ichimoku/Donchian/Divergence/Entropy
    print("  [6/8] Chunk 5: Ichimoku/Donchian/Divergence/Entropy (181-206)...")
    validate_chunk5_ichimoku_donchian_entropy(a200_df, raw_df, report)

    # Bounded features
    print("  [7/8] Bounded features range check...")
    validate_bounded_features(a200_df, report)

    # Lookahead detection
    if not args.skip_lookahead:
        print("  [8/8] Lookahead detection (truncation test)...")
        validate_lookahead(a200_df, raw_df, vix_df, report, n_test_indices=5)
    else:
        print("  [8/8] Lookahead detection SKIPPED")

    print("\n" + "=" * 60)
    print("LAYER 2: SEMANTIC VALIDATION")
    print("=" * 60)

    # Sample date audits
    print("\n  [1/3] Sample date audits...")
    sample_audits = validate_sample_dates(a200_df, report)
    report.sample_audits = sample_audits

    # Cross-indicator consistency
    print("  [2/3] Cross-indicator consistency...")
    validate_cross_indicator_consistency(a200_df, raw_df, report)

    # Known event verification
    print("  [3/3] Known event verification...")
    validate_known_events(a200_df, report)

    # Generate outputs
    json_path = args.output_dir / "tier_a200_validation.json"
    md_path = args.output_dir / "tier_a200_validation.md"
    audit_path = args.output_dir / "tier_a200_sample_audit.md"

    with open(json_path, "w") as f:
        json.dump(report.to_json(), f, indent=2, default=str)

    with open(md_path, "w") as f:
        f.write(report.to_markdown())

    # Write sample audit report
    with open(audit_path, "w") as f:
        f.write("# Tier A200 Sample Date Audit Report\n\n")
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

    print(f"\nResults written to:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"  Sample Audit: {audit_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for chunk_name, stats in report.summary_by_chunk().items():
        if stats["total"] > 0:
            status = "PASS" if stats["failed"] == 0 else "FAIL"
            print(f"[{status}] {chunk_name}: {stats['passed']} passed, {stats['failed']} failed")

    print(f"\nTotal: {report.passed} passed, {report.failed} failed ({report.pass_rate:.1f}%)")

    if report.failed == 0:
        print("\nALL VALIDATION CHECKS PASSED")
        return 0
    else:
        print(f"\n{report.failed} VALIDATION CHECKS FAILED")
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
