"""Tests for validate_parquet_file.py script."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestMD5Computation:
    """Test MD5 checksum computation."""

    def test_compute_md5_returns_correct_hash(self, tmp_path: Path) -> None:
        """Test that MD5 computation produces correct hash."""
        # Create a test file with known content
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        # Expected MD5 of "hello world"
        expected_md5 = hashlib.md5(b"hello world").hexdigest()

        # Import and test the function
        from scripts.validate_parquet_file import compute_md5

        actual_md5 = compute_md5(test_file)
        assert actual_md5 == expected_md5

    def test_compute_md5_handles_binary_file(self, tmp_path: Path) -> None:
        """Test MD5 computation on binary files."""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"\x00\x01\x02\x03\xff")

        expected_md5 = hashlib.md5(b"\x00\x01\x02\x03\xff").hexdigest()

        from scripts.validate_parquet_file import compute_md5

        actual_md5 = compute_md5(test_file)
        assert actual_md5 == expected_md5


class TestFileIntegrityChecks:
    """Test file integrity validation."""

    def test_check_file_exists_passes_for_existing_file(self, tmp_path: Path) -> None:
        """Test file existence check passes for existing files."""
        test_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_parquet(test_file)

        from scripts.validate_parquet_file import ValidationReport, check_file_exists

        report = ValidationReport()
        check_file_exists(test_file, report)

        assert report.passed == 1
        assert report.failed == 0

    def test_check_file_exists_fails_for_missing_file(self, tmp_path: Path) -> None:
        """Test file existence check fails for missing files."""
        missing_file = tmp_path / "nonexistent.parquet"

        from scripts.validate_parquet_file import ValidationReport, check_file_exists

        report = ValidationReport()
        check_file_exists(missing_file, report)

        assert report.passed == 0
        assert report.failed == 1

    def test_check_file_readable_passes_for_valid_parquet(self, tmp_path: Path) -> None:
        """Test file readability check passes for valid parquet."""
        test_file = tmp_path / "test.parquet"
        df = pd.DataFrame({"a": [1, 2, 3]})
        df.to_parquet(test_file)

        from scripts.validate_parquet_file import ValidationReport, check_file_readable

        report = ValidationReport()
        result = check_file_readable(test_file, report)

        assert result is not None
        assert report.passed == 1

    def test_check_file_readable_fails_for_corrupt_file(self, tmp_path: Path) -> None:
        """Test file readability check fails for corrupt files."""
        test_file = tmp_path / "corrupt.parquet"
        test_file.write_text("not a parquet file")

        from scripts.validate_parquet_file import ValidationReport, check_file_readable

        report = ValidationReport()
        result = check_file_readable(test_file, report)

        assert result is None
        assert report.failed == 1


class TestSchemaValidation:
    """Test schema validation checks."""

    def test_check_date_column_passes_when_present(self, tmp_path: Path) -> None:
        """Test Date column check passes when Date is present."""
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=5), "value": [1, 2, 3, 4, 5]})

        from scripts.validate_parquet_file import ValidationReport, check_date_column

        report = ValidationReport()
        check_date_column(df, report)

        assert report.passed == 1
        assert report.failed == 0

    def test_check_date_column_fails_when_missing(self) -> None:
        """Test Date column check fails when Date is missing."""
        df = pd.DataFrame({"value": [1, 2, 3]})

        from scripts.validate_parquet_file import ValidationReport, check_date_column

        report = ValidationReport()
        check_date_column(df, report)

        assert report.passed == 0
        assert report.failed == 1

    def test_check_column_count_passes_when_correct(self) -> None:
        """Test column count check passes when count matches."""
        df = pd.DataFrame({"Date": [1], "a": [2], "b": [3]})  # 3 columns

        from scripts.validate_parquet_file import ValidationReport, check_column_count

        report = ValidationReport()
        check_column_count(df, expected=3, report=report)

        assert report.passed == 1
        assert report.failed == 0

    def test_check_column_count_fails_when_wrong(self) -> None:
        """Test column count check fails when count differs."""
        df = pd.DataFrame({"Date": [1], "a": [2]})  # 2 columns

        from scripts.validate_parquet_file import ValidationReport, check_column_count

        report = ValidationReport()
        check_column_count(df, expected=3, report=report)

        assert report.passed == 0
        assert report.failed == 1


class TestDataQualityChecks:
    """Test data quality validation."""

    def test_check_no_nan_passes_for_clean_data(self) -> None:
        """Test NaN check passes when no NaN values present."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

        from scripts.validate_parquet_file import ValidationReport, check_no_nan

        report = ValidationReport()
        check_no_nan(df, report)

        assert report.passed == 1
        assert report.failed == 0

    def test_check_no_nan_fails_for_nan_data(self) -> None:
        """Test NaN check fails when NaN values present."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})

        from scripts.validate_parquet_file import ValidationReport, check_no_nan

        report = ValidationReport()
        check_no_nan(df, report)

        assert report.passed == 0
        assert report.failed == 1

    def test_check_no_inf_passes_for_clean_data(self) -> None:
        """Test Inf check passes when no Inf values present."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

        from scripts.validate_parquet_file import ValidationReport, check_no_inf

        report = ValidationReport()
        check_no_inf(df, report)

        assert report.passed == 1
        assert report.failed == 0

    def test_check_no_inf_fails_for_inf_data(self) -> None:
        """Test Inf check fails when Inf values present."""
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0], "b": [4.0, 5.0, 6.0]})

        from scripts.validate_parquet_file import ValidationReport, check_no_inf

        report = ValidationReport()
        check_no_inf(df, report)

        assert report.passed == 0
        assert report.failed == 1

    def test_check_date_monotonic_passes_for_sorted_dates(self) -> None:
        """Test date monotonicity check passes for sorted dates."""
        df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=5)})

        from scripts.validate_parquet_file import ValidationReport, check_date_monotonic

        report = ValidationReport()
        check_date_monotonic(df, report)

        assert report.passed == 1
        assert report.failed == 0

    def test_check_date_monotonic_fails_for_unsorted_dates(self) -> None:
        """Test date monotonicity check fails for unsorted dates."""
        df = pd.DataFrame(
            {"Date": pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02"])}
        )

        from scripts.validate_parquet_file import ValidationReport, check_date_monotonic

        report = ValidationReport()
        check_date_monotonic(df, report)

        assert report.passed == 0
        assert report.failed == 1


class TestBoundedFeatureValidation:
    """Test bounded feature range validation."""

    def test_check_bounded_feature_passes_in_range(self) -> None:
        """Test bounded feature check passes when values in range."""
        df = pd.DataFrame({"rsi_percentile_60d": [0.0, 0.5, 1.0]})

        from scripts.validate_parquet_file import ValidationReport, check_bounded_feature

        report = ValidationReport()
        check_bounded_feature(df, "rsi_percentile_60d", min_val=0, max_val=1, report=report)

        assert report.passed == 1
        assert report.failed == 0

    def test_check_bounded_feature_fails_out_of_range(self) -> None:
        """Test bounded feature check fails when values out of range."""
        df = pd.DataFrame({"rsi_percentile_60d": [0.0, 1.5, 0.5]})  # 1.5 is out of [0,1]

        from scripts.validate_parquet_file import ValidationReport, check_bounded_feature

        report = ValidationReport()
        check_bounded_feature(df, "rsi_percentile_60d", min_val=0, max_val=1, report=report)

        assert report.passed == 0
        assert report.failed == 1


class TestValidationReport:
    """Test ValidationReport class."""

    def test_validation_report_counts_correctly(self) -> None:
        """Test that ValidationReport counts passed/failed correctly."""
        from scripts.validate_parquet_file import ValidationReport

        report = ValidationReport()
        report.ok("test", "check1", "exp", "act", "evidence")
        report.ok("test", "check2", "exp", "act", "evidence")
        report.fail("test", "check3", "exp", "act", "evidence")

        assert report.passed == 2
        assert report.failed == 1
        assert len(report.checks) == 3

    def test_validation_report_to_json(self) -> None:
        """Test JSON export of validation report."""
        from scripts.validate_parquet_file import ValidationReport

        report = ValidationReport()
        report.ok("test", "check1", "expected", "actual", "evidence")

        json_output = report.to_json()

        assert "summary" in json_output
        assert "checks" in json_output
        assert json_output["summary"]["passed"] == 1
        assert json_output["summary"]["failed"] == 0

    def test_validation_report_to_markdown(self) -> None:
        """Test markdown export of validation report."""
        from scripts.validate_parquet_file import ValidationReport

        report = ValidationReport()
        report.ok("test", "check1", "expected", "actual", "evidence")

        md_output = report.to_markdown()

        assert "# Parquet File Validation Report" in md_output
        assert "Passed" in md_output


class TestManualAuditGeneration:
    """Test manual audit file generation."""

    def test_generate_audit_samples_creates_markdown(self) -> None:
        """Test that audit sample generation produces markdown."""
        # Create test data with the required Date column
        dates = pd.to_datetime(["2020-03-16", "2020-03-23", "2021-11-19"])
        df = pd.DataFrame({
            "Date": dates,
            "close": [250.0, 220.0, 470.0],
            "rsi_14": [20.0, 30.0, 70.0],
        })

        from scripts.validate_parquet_file import generate_audit_samples

        audit_dates = [("2020-03-16", "COVID crash")]
        md_output = generate_audit_samples(df, audit_dates)

        assert "COVID crash" in md_output
        assert "2020-03-16" in md_output
