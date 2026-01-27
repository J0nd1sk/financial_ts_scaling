#!/usr/bin/env python3
"""Generic parquet file validation utility.

Provides reusable validation checks for parquet files:
- File integrity (existence, readability, MD5)
- Schema validation (columns, types)
- Data quality (NaN, Inf, monotonicity)
- Range/bound validation
- Audit sample generation

Run: ./venv/bin/python scripts/validate_parquet_file.py <parquet_path>
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


def compute_md5(path: Path) -> str:
    """Compute MD5 checksum of a file.

    Args:
        path: Path to the file.

    Returns:
        MD5 hex digest string.
    """
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


@dataclass
class ValidationCheck:
    """Single validation check result."""

    category: str
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

    def ok(
        self,
        category: str,
        check_name: str,
        expected: Any,
        actual: Any,
        evidence: str,
    ) -> None:
        """Record a passing check."""
        self.add(ValidationCheck(category, check_name, True, expected, actual, evidence))

    def fail(
        self,
        category: str,
        check_name: str,
        expected: Any,
        actual: Any,
        evidence: str,
    ) -> None:
        """Record a failing check."""
        self.add(ValidationCheck(category, check_name, False, expected, actual, evidence))

    @property
    def passed(self) -> int:
        """Count of passed checks."""
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed(self) -> int:
        """Count of failed checks."""
        return sum(1 for c in self.checks if not c.passed)

    @property
    def pass_rate(self) -> float:
        """Pass rate as percentage."""
        total = len(self.checks)
        return self.passed / total * 100 if total > 0 else 0.0

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
            "checks": [asdict(c) for c in self.checks],
        }

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Parquet File Validation Report",
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
            "## Results",
            "",
            "| Status | Category | Check | Expected | Actual |",
            "|--------|----------|-------|----------|--------|",
        ]

        for check in self.checks:
            status = "✅" if check.passed else "❌"
            lines.append(
                f"| {status} | {check.category} | {check.check_name} | "
                f"`{check.expected}` | `{check.actual}` |"
            )

        # Failed checks summary
        failed_checks = [c for c in self.checks if not c.passed]
        if failed_checks:
            lines.extend([
                "",
                "## Failed Checks",
                "",
            ])
            for check in failed_checks:
                lines.append(f"- **{check.category}**: {check.check_name}")
                lines.append(f"  - Expected: `{check.expected}`")
                lines.append(f"  - Actual: `{check.actual}`")
                lines.append(f"  - Evidence: {check.evidence}")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# File Integrity Checks
# =============================================================================


def check_file_exists(path: Path, report: ValidationReport) -> None:
    """Check if file exists.

    Args:
        path: Path to check.
        report: ValidationReport to record result.
    """
    if path.exists():
        report.ok(
            "file_integrity",
            "file_exists",
            "file exists",
            "file exists",
            f"Path: {path}",
        )
    else:
        report.fail(
            "file_integrity",
            "file_exists",
            "file exists",
            "file not found",
            f"Path: {path}",
        )


def check_file_readable(path: Path, report: ValidationReport) -> Optional[pd.DataFrame]:
    """Check if file is readable as parquet.

    Args:
        path: Path to parquet file.
        report: ValidationReport to record result.

    Returns:
        DataFrame if readable, None otherwise.
    """
    try:
        df = pd.read_parquet(path)
        report.ok(
            "file_integrity",
            "file_readable",
            "parquet readable",
            f"read {len(df)} rows",
            f"Columns: {len(df.columns)}",
        )
        return df
    except Exception as e:
        report.fail(
            "file_integrity",
            "file_readable",
            "parquet readable",
            f"read error: {type(e).__name__}",
            str(e)[:100],
        )
        return None


# =============================================================================
# Schema Validation Checks
# =============================================================================


def check_date_column(df: pd.DataFrame, report: ValidationReport) -> None:
    """Check if Date column exists.

    Args:
        df: DataFrame to check.
        report: ValidationReport to record result.
    """
    if "Date" in df.columns:
        report.ok(
            "schema",
            "date_column_present",
            "Date column exists",
            "Date column exists",
            f"Date dtype: {df['Date'].dtype}",
        )
    else:
        report.fail(
            "schema",
            "date_column_present",
            "Date column exists",
            "Date column missing",
            f"Columns: {list(df.columns)[:5]}...",
        )


def check_column_count(
    df: pd.DataFrame, expected: int, report: ValidationReport
) -> None:
    """Check if column count matches expected.

    Args:
        df: DataFrame to check.
        expected: Expected number of columns.
        report: ValidationReport to record result.
    """
    actual = len(df.columns)
    if actual == expected:
        report.ok(
            "schema",
            "column_count",
            f"{expected} columns",
            f"{actual} columns",
            "Column count matches",
        )
    else:
        report.fail(
            "schema",
            "column_count",
            f"{expected} columns",
            f"{actual} columns",
            f"Difference: {actual - expected}",
        )


# =============================================================================
# Data Quality Checks
# =============================================================================


def check_no_nan(df: pd.DataFrame, report: ValidationReport) -> None:
    """Check for NaN values in DataFrame.

    Args:
        df: DataFrame to check.
        report: ValidationReport to record result.
    """
    nan_mask = df.isna()
    has_nan = nan_mask.any().any()

    if not has_nan:
        report.ok(
            "data_quality",
            "no_nan_values",
            "no NaN values",
            "no NaN values",
            "All values present",
        )
    else:
        nan_cols = [col for col in df.columns if df[col].isna().any()]
        nan_count = nan_mask.sum().sum()
        report.fail(
            "data_quality",
            "no_nan_values",
            "no NaN values",
            f"{nan_count} NaN values",
            f"Columns with NaN: {nan_cols[:5]}",
        )


def check_no_inf(df: pd.DataFrame, report: ValidationReport) -> None:
    """Check for Inf values in numeric columns.

    Args:
        df: DataFrame to check.
        report: ValidationReport to record result.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    has_inf = np.isinf(numeric_df).any().any()

    if not has_inf:
        report.ok(
            "data_quality",
            "no_inf_values",
            "no Inf values",
            "no Inf values",
            f"Checked {len(numeric_df.columns)} numeric columns",
        )
    else:
        inf_cols = [
            col for col in numeric_df.columns if np.isinf(numeric_df[col]).any()
        ]
        report.fail(
            "data_quality",
            "no_inf_values",
            "no Inf values",
            "Inf values found",
            f"Columns with Inf: {inf_cols[:5]}",
        )


def check_date_monotonic(df: pd.DataFrame, report: ValidationReport) -> None:
    """Check if Date column is monotonically increasing.

    Args:
        df: DataFrame to check.
        report: ValidationReport to record result.
    """
    if "Date" not in df.columns:
        report.fail(
            "data_quality",
            "date_monotonic",
            "dates monotonically increasing",
            "Date column missing",
            "Cannot check monotonicity without Date column",
        )
        return

    dates = pd.to_datetime(df["Date"])
    is_monotonic = dates.is_monotonic_increasing

    if is_monotonic:
        report.ok(
            "data_quality",
            "date_monotonic",
            "dates monotonically increasing",
            "dates monotonically increasing",
            f"Date range: {dates.min()} to {dates.max()}",
        )
    else:
        # Find first violation
        diffs = dates.diff()
        violations = (diffs < pd.Timedelta(0)).sum()
        report.fail(
            "data_quality",
            "date_monotonic",
            "dates monotonically increasing",
            f"{violations} ordering violations",
            "Dates are not sorted",
        )


def check_bounded_feature(
    df: pd.DataFrame,
    feature: str,
    min_val: float,
    max_val: float,
    report: ValidationReport,
) -> None:
    """Check if feature values are within specified bounds.

    Args:
        df: DataFrame to check.
        feature: Column name to check.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        report: ValidationReport to record result.
    """
    if feature not in df.columns:
        report.fail(
            "range_validation",
            f"{feature}_bounded",
            f"[{min_val}, {max_val}]",
            "column missing",
            f"Feature {feature} not in DataFrame",
        )
        return

    values = df[feature]
    actual_min = values.min()
    actual_max = values.max()

    in_range = (values >= min_val).all() and (values <= max_val).all()

    if in_range:
        report.ok(
            "range_validation",
            f"{feature}_bounded",
            f"[{min_val}, {max_val}]",
            f"[{actual_min:.4f}, {actual_max:.4f}]",
            "All values in range",
        )
    else:
        violations = ((values < min_val) | (values > max_val)).sum()
        report.fail(
            "range_validation",
            f"{feature}_bounded",
            f"[{min_val}, {max_val}]",
            f"[{actual_min:.4f}, {actual_max:.4f}]",
            f"{violations} values out of range",
        )


# =============================================================================
# Audit Sample Generation
# =============================================================================


def generate_audit_samples(
    df: pd.DataFrame, audit_dates: list[tuple[str, str]]
) -> str:
    """Generate markdown audit samples for specified dates.

    Args:
        df: DataFrame with Date column.
        audit_dates: List of (date_str, description) tuples.

    Returns:
        Markdown-formatted string with audit samples.
    """
    lines = [
        "# Manual Audit Samples",
        "",
    ]

    if "Date" not in df.columns:
        lines.append("**Error**: No Date column in DataFrame")
        return "\n".join(lines)

    # Normalize dates for comparison
    df_dates = pd.to_datetime(df["Date"]).dt.normalize()

    for date_str, description in audit_dates:
        target_date = pd.to_datetime(date_str).normalize()
        lines.append(f"## {date_str} - {description}")
        lines.append("")

        # Find matching row
        mask = df_dates == target_date
        if mask.sum() == 0:
            lines.append(f"**No data for {date_str}**")
            lines.append("")
            continue

        row = df[mask].iloc[0]

        # Generate table
        lines.append("| Feature | Value |")
        lines.append("|---------|-------|")

        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                lines.append(f"| {col} | {val:.4f} |")
            else:
                lines.append(f"| {col} | {val} |")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Main Entry Point
# =============================================================================


def main() -> int:
    """Main entry point for CLI usage."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Validate a parquet file for data quality."
    )
    parser.add_argument("path", type=Path, help="Path to parquet file.")
    parser.add_argument(
        "--expected-columns",
        type=int,
        default=None,
        help="Expected number of columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files.",
    )
    args = parser.parse_args()

    report = ValidationReport()

    # File integrity
    check_file_exists(args.path, report)
    if report.failed > 0:
        print(f"❌ File not found: {args.path}")
        return 1

    df = check_file_readable(args.path, report)
    if df is None:
        print(f"❌ Cannot read parquet: {args.path}")
        return 1

    # MD5
    md5 = compute_md5(args.path)
    print(f"MD5: {md5}")

    # Schema
    check_date_column(df, report)
    if args.expected_columns:
        check_column_count(df, args.expected_columns, report)

    # Data quality
    check_no_nan(df, report)
    check_no_inf(df, report)
    check_date_monotonic(df, report)

    # Output
    print(f"\n{report.to_markdown()}")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "validation.json"
        md_path = args.output_dir / "validation.md"

        import json

        with open(json_path, "w") as f:
            json.dump(report.to_json(), f, indent=2)
        with open(md_path, "w") as f:
            f.write(report.to_markdown())

        print(f"\nOutput written to: {args.output_dir}")

    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
