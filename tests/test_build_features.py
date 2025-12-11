"""Tests for build_features_a20.py CLI parameterization."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_features_a20


class TestCLITickerParsing:
    """Tests for --ticker CLI argument parsing."""

    def test_cli_parse_ticker_dia(self) -> None:
        """Test that --ticker DIA is parsed correctly."""
        with patch("sys.argv", ["build_features_a20.py", "--ticker", "DIA"]):
            args = build_features_a20.parse_args()
        assert args.ticker == "DIA"

    def test_cli_parse_ticker_qqq(self) -> None:
        """Test that --ticker QQQ is parsed correctly."""
        with patch("sys.argv", ["build_features_a20.py", "--ticker", "QQQ"]):
            args = build_features_a20.parse_args()
        assert args.ticker == "QQQ"

    def test_cli_default_ticker_spy(self) -> None:
        """Test that default ticker is SPY when no --ticker provided."""
        with patch("sys.argv", ["build_features_a20.py"]):
            args = build_features_a20.parse_args()
        assert args.ticker == "SPY"


class TestPathConstruction:
    """Tests for dynamic path and dataset name construction."""

    def test_raw_path_includes_ticker(self) -> None:
        """Test that raw path is constructed from ticker."""
        with patch("sys.argv", ["build_features_a20.py", "--ticker", "DIA"]):
            args = build_features_a20.parse_args()
        expected = Path("data/raw/DIA.parquet")
        assert args.raw_path == expected

    def test_output_path_includes_ticker(self) -> None:
        """Test that output path includes ticker name."""
        with patch("sys.argv", ["build_features_a20.py", "--ticker", "QQQ"]):
            args = build_features_a20.parse_args()
        expected = Path("data/processed/v1/QQQ_features_a20.parquet")
        assert args.output_path == expected


class TestDatasetNaming:
    """Tests for manifest dataset name construction."""

    def test_get_dataset_names_for_ticker(self) -> None:
        """Test that dataset names follow {TICKER}.*.a20 pattern."""
        raw_name, processed_name = build_features_a20.get_dataset_names("DIA")
        assert raw_name == "DIA.OHLCV.daily"
        assert processed_name == "DIA.features.a20"

    def test_get_dataset_names_default_spy(self) -> None:
        """Test that SPY dataset names are correct."""
        raw_name, processed_name = build_features_a20.get_dataset_names("SPY")
        assert raw_name == "SPY.OHLCV.daily"
        assert processed_name == "SPY.features.a20"


class TestDIAFeatureOutput:
    """Tests for DIA feature file output validation."""

    DIA_FEATURES_PATH = Path("data/processed/v1/DIA_features_a20.parquet")
    SPY_FEATURES_PATH = Path("data/processed/v1/SPY_features_a20.parquet")
    DIA_RAW_PATH = Path("data/raw/DIA.parquet")

    def test_dia_features_file_exists(self) -> None:
        """Test that DIA features file exists at expected path."""
        assert self.DIA_FEATURES_PATH.exists(), f"DIA features not found at {self.DIA_FEATURES_PATH}"

    def test_dia_features_has_expected_columns(self) -> None:
        """Test that DIA features have same columns as SPY."""
        import pandas as pd

        dia_df = pd.read_parquet(self.DIA_FEATURES_PATH)
        spy_df = pd.read_parquet(self.SPY_FEATURES_PATH)
        assert list(dia_df.columns) == list(spy_df.columns), "DIA columns must match SPY columns"

    def test_dia_features_row_count_reasonable(self) -> None:
        """Test that DIA features row count is reasonable (raw - warmup)."""
        import pandas as pd

        dia_features = pd.read_parquet(self.DIA_FEATURES_PATH)
        dia_raw = pd.read_parquet(self.DIA_RAW_PATH)
        # Warmup period drops ~200 rows (for SMA_200)
        min_expected = len(dia_raw) - 250
        max_expected = len(dia_raw)
        assert min_expected <= len(dia_features) <= max_expected, (
            f"DIA features {len(dia_features)} rows outside expected range [{min_expected}, {max_expected}]"
        )


class TestQQQFeatureOutput:
    """Tests for QQQ feature file output validation."""

    QQQ_FEATURES_PATH = Path("data/processed/v1/QQQ_features_a20.parquet")
    SPY_FEATURES_PATH = Path("data/processed/v1/SPY_features_a20.parquet")
    QQQ_RAW_PATH = Path("data/raw/QQQ.parquet")

    def test_qqq_features_file_exists(self) -> None:
        """Test that QQQ features file exists at expected path."""
        assert self.QQQ_FEATURES_PATH.exists(), f"QQQ features not found at {self.QQQ_FEATURES_PATH}"

    def test_qqq_features_has_expected_columns(self) -> None:
        """Test that QQQ features have same columns as SPY."""
        import pandas as pd

        qqq_df = pd.read_parquet(self.QQQ_FEATURES_PATH)
        spy_df = pd.read_parquet(self.SPY_FEATURES_PATH)
        assert list(qqq_df.columns) == list(spy_df.columns), "QQQ columns must match SPY columns"

    def test_qqq_features_row_count_reasonable(self) -> None:
        """Test that QQQ features row count is reasonable (raw - warmup)."""
        import pandas as pd

        qqq_features = pd.read_parquet(self.QQQ_FEATURES_PATH)
        qqq_raw = pd.read_parquet(self.QQQ_RAW_PATH)
        # Warmup period drops ~200 rows (for SMA_200)
        min_expected = len(qqq_raw) - 250
        max_expected = len(qqq_raw)
        assert min_expected <= len(qqq_features) <= max_expected, (
            f"QQQ features {len(qqq_features)} rows outside expected range [{min_expected}, {max_expected}]"
        )
