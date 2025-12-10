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
