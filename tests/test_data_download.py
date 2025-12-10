"""Tests for OHLCV data download functionality.

These tests verify that the download script:
- Downloads SPY data successfully
- Saves data in correct format
- Produces valid data with expected columns and types
- Covers expected date range
- Supports proper train/val/test splits
- Supports downloading any ticker (DIA, QQQ, etc.)
- Implements retry logic for transient failures
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock
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


# =============================================================================
# Tests for download_ticker() - Multi-ticker support (Phase 5 Task 1)
# All tests use mocked yfinance - no live API calls
# =============================================================================


def _create_mock_ohlcv_dataframe(rows: int = 10) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame matching yfinance format.

    yfinance returns DataFrame with Date as INDEX, not a column.
    """
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")  # Business days
    df = pd.DataFrame({
        "Open": [100.0 + i * 0.1 for i in range(rows)],
        "High": [101.0 + i * 0.1 for i in range(rows)],
        "Low": [99.0 + i * 0.1 for i in range(rows)],
        "Close": [100.5 + i * 0.1 for i in range(rows)],
        "Volume": [1000000 + i * 1000 for i in range(rows)],
    }, index=dates)
    df.index.name = "Date"
    return df


class TestDownloadTicker:
    """Tests for download_ticker() - multi-ticker support with mocked yfinance."""

    @patch("scripts.download_ohlcv.yf.Ticker")
    def test_download_ticker_dia_basic(self, mock_ticker_class, tmp_path):
        """Test that download_ticker creates DIA parquet file with correct data."""
        from scripts.download_ohlcv import download_ticker

        # Arrange: mock yfinance to return synthetic data
        mock_df = _create_mock_ohlcv_dataframe(rows=15)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker_instance

        # Act
        df = download_ticker("DIA", str(tmp_path), register_manifest=False)

        # Assert: file created with correct name
        output_file = tmp_path / "DIA.parquet"
        assert output_file.exists(), "DIA.parquet was not created"
        assert output_file.stat().st_size > 0, "DIA.parquet is empty"

        # Assert: DataFrame has correct structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 15
        assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]

        # Assert: yfinance was called with correct ticker
        mock_ticker_class.assert_called_once_with("DIA")
        mock_ticker_instance.history.assert_called_once_with(period="max")

    @patch("scripts.download_ohlcv.yf.Ticker")
    def test_download_ticker_qqq_basic(self, mock_ticker_class, tmp_path):
        """Test that download_ticker creates QQQ parquet file with correct data."""
        from scripts.download_ohlcv import download_ticker

        # Arrange
        mock_df = _create_mock_ohlcv_dataframe(rows=20)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker_instance

        # Act
        df = download_ticker("QQQ", str(tmp_path), register_manifest=False)

        # Assert
        output_file = tmp_path / "QQQ.parquet"
        assert output_file.exists(), "QQQ.parquet was not created"
        assert len(df) == 20
        mock_ticker_class.assert_called_once_with("QQQ")

    @patch("scripts.download_ohlcv.yf.Ticker")
    def test_download_ticker_registers_manifest(self, mock_ticker_class, tmp_path):
        """Test that download_ticker registers manifest entry as {TICKER}.OHLCV.daily."""
        from scripts.download_ohlcv import download_ticker

        # Arrange
        mock_df = _create_mock_ohlcv_dataframe(rows=10)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker_instance

        manifest = tmp_path / "manifest.json"

        # Act
        download_ticker("DIA", str(tmp_path), register_manifest=True, manifest_path=manifest)

        # Assert: manifest entry created with correct dataset name
        manifest_data = json.loads(manifest.read_text())
        entry = manifest_data["entries"][-1]
        assert entry["dataset"] == "DIA.OHLCV.daily", f"Expected DIA.OHLCV.daily, got {entry['dataset']}"
        assert "DIA.parquet" in entry["path"]
        assert entry["md5"]  # MD5 should be present

    @patch("scripts.download_ohlcv.yf.Ticker")
    def test_download_ticker_invalid_raises(self, mock_ticker_class, tmp_path):
        """Test that download_ticker raises ValueError when yfinance returns empty data."""
        from scripts.download_ohlcv import download_ticker

        # Arrange: mock yfinance to return empty DataFrame
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()  # Empty
        mock_ticker_class.return_value = mock_ticker_instance

        # Act & Assert
        with pytest.raises(ValueError, match="No data returned|Failed to download"):
            download_ticker("INVALID_TICKER_XYZ", str(tmp_path), register_manifest=False)

    @patch("scripts.download_ohlcv.time.sleep")
    @patch("scripts.download_ohlcv.yf.Ticker")
    def test_download_ticker_retries_on_failure(self, mock_ticker_class, mock_sleep, tmp_path):
        """Test that download_ticker retries on transient failures with backoff."""
        from scripts.download_ohlcv import download_ticker

        # Arrange: first 2 calls fail, third succeeds
        mock_df = _create_mock_ohlcv_dataframe(rows=10)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = [
            Exception("Network error"),  # Attempt 1: fail
            Exception("Timeout"),         # Attempt 2: fail
            mock_df,                       # Attempt 3: success
        ]
        mock_ticker_class.return_value = mock_ticker_instance

        # Act
        df = download_ticker("DIA", str(tmp_path), register_manifest=False)

        # Assert: succeeded after retries
        assert len(df) == 10
        assert mock_ticker_instance.history.call_count == 3

        # Assert: sleep was called for backoff (2 times, before attempts 2 and 3)
        assert mock_sleep.call_count == 2
        # Verify backoff delays are in expected range (base + jitter)
        first_delay = mock_sleep.call_args_list[0][0][0]
        second_delay = mock_sleep.call_args_list[1][0][0]
        assert 1.0 <= first_delay <= 1.5, f"First delay {first_delay} not in [1.0, 1.5]"
        assert 2.0 <= second_delay <= 3.0, f"Second delay {second_delay} not in [2.0, 3.0]"


# =============================================================================
# Tests for CLI argument parsing (Phase 5 Task 2)
# =============================================================================


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing in download_ohlcv.py."""

    def test_cli_parse_ticker_dia(self):
        """Test that --ticker DIA parses correctly."""
        from scripts.download_ohlcv import parse_args

        args = parse_args(["--ticker", "DIA"])
        assert args.ticker == "DIA"

    def test_cli_parse_ticker_qqq(self):
        """Test that --ticker QQQ parses correctly."""
        from scripts.download_ohlcv import parse_args

        args = parse_args(["--ticker", "QQQ"])
        assert args.ticker == "QQQ"

    def test_cli_default_ticker_spy(self):
        """Test that no --ticker defaults to SPY."""
        from scripts.download_ohlcv import parse_args

        args = parse_args([])
        assert args.ticker == "SPY"

    def test_cli_parse_output_dir(self):
        """Test that --output-dir parses correctly."""
        from scripts.download_ohlcv import parse_args

        args = parse_args(["--output-dir", "/custom/path"])
        assert args.output_dir == "/custom/path"

    def test_cli_default_output_dir(self):
        """Test that no --output-dir defaults to data/raw."""
        from scripts.download_ohlcv import parse_args

        args = parse_args([])
        assert args.output_dir == "data/raw"


# =============================================================================
# Tests for index ticker handling (sanitize ^ from filenames)
# =============================================================================


class TestIndexTickerHandling:
    """Tests for index ticker filename sanitization."""

    @patch("scripts.download_ohlcv.yf.Ticker")
    def test_download_index_ticker_sanitizes_filename(self, mock_ticker_class, tmp_path):
        """Test that ^DJI creates DJI.parquet (not ^DJI.parquet)."""
        from scripts.download_ohlcv import download_ticker

        # Arrange
        mock_df = _create_mock_ohlcv_dataframe(rows=10)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker_instance

        # Act
        download_ticker("^DJI", str(tmp_path), register_manifest=False)

        # Assert: file created without ^ in name
        assert (tmp_path / "DJI.parquet").exists(), "Should create DJI.parquet"
        assert not (tmp_path / "^DJI.parquet").exists(), "Should NOT create ^DJI.parquet"

        # Assert: yfinance called with original ticker (including ^)
        mock_ticker_class.assert_called_once_with("^DJI")

    @patch("scripts.download_ohlcv.yf.Ticker")
    def test_download_index_ticker_manifest_uses_sanitized_name(self, mock_ticker_class, tmp_path):
        """Test that manifest uses sanitized ticker name (DJI.OHLCV.daily, not ^DJI.OHLCV.daily)."""
        from scripts.download_ohlcv import download_ticker

        # Arrange
        mock_df = _create_mock_ohlcv_dataframe(rows=10)
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker_instance

        manifest = tmp_path / "manifest.json"

        # Act
        download_ticker("^IXIC", str(tmp_path), register_manifest=True, manifest_path=manifest)

        # Assert: manifest entry uses sanitized name
        manifest_data = json.loads(manifest.read_text())
        entry = manifest_data["entries"][-1]
        assert entry["dataset"] == "IXIC.OHLCV.daily", f"Expected IXIC.OHLCV.daily, got {entry['dataset']}"
        assert "IXIC.parquet" in entry["path"]
        assert "^" not in entry["dataset"]
        assert "^" not in entry["path"]
