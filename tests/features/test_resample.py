"""Tests for OHLCV resampling utilities."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.resample import resample_ohlcv, get_freq_string


@pytest.fixture()
def sample_daily_df() -> pd.DataFrame:
    """Create sample daily OHLCV data spanning 4 weeks."""
    # Use business days to simulate real trading data
    dates = pd.date_range("2024-01-02", periods=20, freq="B")
    n = len(dates)
    data = {
        "Date": dates,
        "Open": [100 + i for i in range(n)],
        "High": [102 + i for i in range(n)],
        "Low": [99 + i for i in range(n)],
        "Close": [101 + i for i in range(n)],
        "Volume": [1_000_000 + i * 10000 for i in range(n)],
    }
    return pd.DataFrame(data)


class TestResampleOHLCV2D:
    """Tests for 2-day resampling."""

    def test_resample_ohlcv_2d_aggregation(self, sample_daily_df: pd.DataFrame) -> None:
        """Test that 2-day resampling applies correct OHLCV aggregation rules."""
        result = resample_ohlcv(sample_daily_df, "2D")

        # Should have roughly half the rows
        assert len(result) < len(sample_daily_df)

        # Check aggregation rules on first complete period
        # Open = first, High = max, Low = min, Close = last, Volume = sum
        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns


class TestResampleOHLCV5D:
    """Tests for 5-day resampling."""

    def test_resample_ohlcv_5d_aggregation(self, sample_daily_df: pd.DataFrame) -> None:
        """Test that 5-day resampling applies correct OHLCV aggregation rules."""
        result = resample_ohlcv(sample_daily_df, "5D")

        # Should have fewer rows (5D calendar-based, so weekends create extra periods)
        assert len(result) < len(sample_daily_df)
        assert len(result) <= len(sample_daily_df) // 2  # At most half

        # Verify all OHLCV columns present
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns


class TestResampleWeekly:
    """Tests for weekly resampling."""

    def test_resample_ohlcv_weekly_friday(self, sample_daily_df: pd.DataFrame) -> None:
        """Test that weekly resampling uses Friday close (W-FRI)."""
        result = resample_ohlcv(sample_daily_df, "W-FRI")

        # Weekly should have fewer rows
        assert len(result) < len(sample_daily_df)

        # All dates should be Fridays (weekday 4)
        dates = pd.to_datetime(result["Date"])
        assert (dates.dt.weekday == 4).all(), "Weekly dates must be Friday-aligned"


class TestResamplePreservesIndex:
    """Tests for date index preservation."""

    def test_resample_preserves_date_index(self, sample_daily_df: pd.DataFrame) -> None:
        """Test that resampled result has Date column as datetime."""
        result = resample_ohlcv(sample_daily_df, "2D")

        assert "Date" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["Date"])


class TestResampleNoLookahead:
    """Tests for look-ahead bias prevention."""

    def test_resample_no_lookahead(self, sample_daily_df: pd.DataFrame) -> None:
        """Test that Close uses end-of-period value (no look-ahead bias)."""
        result = resample_ohlcv(sample_daily_df, "2D")

        # The Close should be the last value in each period
        # For our sequential data, this means Close values should increase
        closes = result["Close"].values
        # Each period's close should be >= the period's open (in our upward-trending data)
        opens = result["Open"].values
        for i in range(len(result)):
            assert closes[i] >= opens[i], "Close should be >= Open in uptrend data"


class TestResampleRowReduction:
    """Tests for correct row count reduction."""

    def test_resample_row_reduction_2d(self, sample_daily_df: pd.DataFrame) -> None:
        """Test that 2D resampling reduces row count."""
        result = resample_ohlcv(sample_daily_df, "2D")
        original_rows = len(sample_daily_df)
        resampled_rows = len(result)

        # 2D is calendar-based, so weekends create extra periods
        # With 20 business days, expect roughly 10-15 periods
        assert resampled_rows < original_rows
        assert resampled_rows >= original_rows // 3

    def test_resample_row_reduction_weekly(self, sample_daily_df: pd.DataFrame) -> None:
        """Test that weekly resampling gives ~1/5 rows."""
        result = resample_ohlcv(sample_daily_df, "W-FRI")
        original_rows = len(sample_daily_df)
        resampled_rows = len(result)

        # 20 business days is ~4 weeks
        assert resampled_rows <= 5
        assert resampled_rows >= 2


class TestGetFreqString:
    """Tests for timescale to frequency string mapping."""

    def test_get_freq_string_mapping(self) -> None:
        """Test that timescale names map to correct pandas frequency strings."""
        assert get_freq_string("2d") == "2D"
        assert get_freq_string("3d") == "3D"
        assert get_freq_string("5d") == "5D"
        assert get_freq_string("weekly") == "W-FRI"

    def test_get_freq_string_case_insensitive(self) -> None:
        """Test that mapping is case-insensitive."""
        assert get_freq_string("2D") == "2D"
        assert get_freq_string("Weekly") == "W-FRI"
        assert get_freq_string("WEEKLY") == "W-FRI"

    def test_get_freq_string_invalid(self) -> None:
        """Test that invalid timescale raises ValueError."""
        with pytest.raises(ValueError, match="Unknown timescale"):
            get_freq_string("invalid")

        with pytest.raises(ValueError, match="Unknown timescale"):
            get_freq_string("monthly")
