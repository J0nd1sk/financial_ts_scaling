"""Tests for SPY OHLCV data download functionality.

These tests verify that the download script:
- Downloads SPY data successfully
- Saves data in correct format
- Produces valid data with expected columns and types
- Covers expected date range
- Supports proper train/val/test splits
"""

from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import manage_data_versions as dv


class TestDownloadSpyBasic:
    """Tests for basic download functionality."""

    def test_download_spy_basic(self, tmp_path):
        """Test that download_spy creates a parquet file with data."""
        from scripts.download_ohlcv import download_spy

        output_path = tmp_path / "SPY.parquet"
        df = download_spy(output_path=str(output_path), register_manifest=False)

        # File should exist
        assert output_path.exists(), "Parquet file was not created"

        # File should have content
        assert output_path.stat().st_size > 0, "Parquet file is empty"

        # Function should return a DataFrame
        assert isinstance(df, pd.DataFrame), "download_spy should return a DataFrame"
        assert len(df) > 0, "DataFrame should not be empty"


class TestSpyDataColumns:
    """Tests for data column structure and types."""

    def test_spy_data_columns(self, tmp_path):
        """Test that SPY data has correct columns."""
        from scripts.download_ohlcv import download_spy

        output_path = tmp_path / "SPY.parquet"
        df = download_spy(output_path=str(output_path), register_manifest=False)

        expected_columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        assert list(df.columns) == expected_columns, (
            f"Expected columns {expected_columns}, got {list(df.columns)}"
        )

    def test_spy_data_types(self, tmp_path):
        """Test that SPY data has correct column types."""
        from scripts.download_ohlcv import download_spy

        output_path = tmp_path / "SPY.parquet"
        df = download_spy(output_path=str(output_path), register_manifest=False)

        # Date should be datetime
        assert pd.api.types.is_datetime64_any_dtype(df["Date"]), (
            "Date column should be datetime type"
        )

        # OHLC should be numeric (float)
        for col in ["Open", "High", "Low", "Close"]:
            assert pd.api.types.is_float_dtype(df[col]), (
                f"{col} column should be float type"
            )

        # Volume should be numeric (int or float)
        assert pd.api.types.is_numeric_dtype(df["Volume"]), (
            "Volume column should be numeric type"
        )


class TestSpyDataCompleteness:
    """Tests for data completeness and coverage."""

    def test_spy_data_date_range(self, tmp_path):
        """Test that SPY data covers expected date range."""
        from scripts.download_ohlcv import download_spy

        output_path = tmp_path / "SPY.parquet"
        df = download_spy(output_path=str(output_path), register_manifest=False)

        # Data should start before 2000
        min_date = df["Date"].min()
        assert min_date.year < 2000, (
            f"Data should start before 2000, starts at {min_date}"
        )

        # Data should end within last 7 days
        max_date = df["Date"].max()
        seven_days_ago = datetime.now() - timedelta(days=7)
        assert max_date >= pd.Timestamp(seven_days_ago), (
            f"Data should end within last 7 days, ends at {max_date}"
        )

    def test_spy_data_no_nulls(self, tmp_path):
        """Test that SPY OHLCV data has no null values."""
        from scripts.download_ohlcv import download_spy

        output_path = tmp_path / "SPY.parquet"
        df = download_spy(output_path=str(output_path), register_manifest=False)

        # Check for nulls in OHLCV columns
        ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
        for col in ohlcv_cols:
            null_count = df[col].isnull().sum()
            assert null_count == 0, (
                f"Column {col} has {null_count} null values"
            )


class TestSpyDataSplits:
    """Tests for train/val/test data splits."""

    def test_spy_data_splits(self, tmp_path):
        """Test that data can be split into train/val/test by date."""
        from scripts.download_ohlcv import download_spy

        output_path = tmp_path / "SPY.parquet"
        df = download_spy(output_path=str(output_path), register_manifest=False)

        # Define split boundaries (from CLAUDE.md)
        train_end = pd.Timestamp("2020-12-31")
        val_start = pd.Timestamp("2021-01-01")
        val_end = pd.Timestamp("2022-12-31")
        test_start = pd.Timestamp("2023-01-01")

        # Split data
        train = df[df["Date"] <= train_end]
        val = df[(df["Date"] >= val_start) & (df["Date"] <= val_end)]
        test = df[df["Date"] >= test_start]

        # Each split should have data
        assert len(train) > 0, "Train split should have data"
        assert len(val) > 0, "Validation split should have data"
        assert len(test) > 0, "Test split should have data"

        # Train should be largest (most historical data)
        assert len(train) > len(val), "Train should have more data than validation"
        assert len(train) > len(test), "Train should have more data than test"

        # Validate split boundaries
        assert train["Date"].max() <= train_end, (
            f"Train data extends past {train_end}"
        )
        assert val["Date"].min() >= val_start, (
            f"Validation data starts before {val_start}"
        )
        assert val["Date"].max() <= val_end, (
            f"Validation data extends past {val_end}"
        )
        assert test["Date"].min() >= test_start, (
            f"Test data starts before {test_start}"
        )


class TestDownloadIdempotent:
    """Tests for idempotent download behavior."""

    def test_download_idempotent(self, tmp_path):
        """Test that download can be run multiple times safely."""
        from scripts.download_ohlcv import download_spy

        output_path = tmp_path / "SPY.parquet"

        # First download
        df1 = download_spy(output_path=str(output_path), register_manifest=False)

        # Second download (should not raise)
        df2 = download_spy(output_path=str(output_path), register_manifest=False)

        # Both should return valid DataFrames
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)

        # Data should be the same (or very similar - new data may have arrived)
        assert len(df1) > 0
        assert len(df2) > 0


class TestManifestRegistration:
    """Tests for manifest auto-registration."""

    def test_download_registers_manifest_entry(self, tmp_path, monkeypatch):
        """download_spy should append to manifest when register_manifest=True."""
        from scripts.download_ohlcv import download_spy

        manifest = tmp_path / "manifest.json"
        monkeypatch.setattr(dv, "RAW_MANIFEST", manifest)

        output_path = tmp_path / "SPY.parquet"
        download_spy(output_path=str(output_path), manifest_path=manifest)

        manifest_data = json.loads(manifest.read_text())
        entry = manifest_data["entries"][-1]
        assert entry["dataset"] == "SPY.OHLCV.daily"
        assert entry["path"] == str(output_path)
        assert entry["md5"]
