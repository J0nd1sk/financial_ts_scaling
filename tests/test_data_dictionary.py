"""Tests for data dictionary generator.

TDD tests for scripts/generate_data_dictionary.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


class TestGenerateCreatesFile:
    """Test that generator creates output file."""

    def test_generate_creates_file(self) -> None:
        """Test that generate_data_dictionary creates markdown file."""
        from scripts.generate_data_dictionary import generate_data_dictionary

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "data_dictionary.md"
            generate_data_dictionary(str(output_path))
            assert output_path.exists(), "Output file should be created"
            assert output_path.stat().st_size > 0, "Output file should not be empty"


class TestGenerateContent:
    """Test that generated content has required sections."""

    @pytest.fixture
    def generated_content(self) -> str:
        """Generate data dictionary and return content."""
        from scripts.generate_data_dictionary import generate_data_dictionary

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "data_dictionary.md"
            generate_data_dictionary(str(output_path))
            return output_path.read_text()

    def test_has_summary_section(self, generated_content: str) -> None:
        """Test that output contains Summary section."""
        assert "## Summary" in generated_content

    def test_documents_all_raw_files(self, generated_content: str) -> None:
        """Test that all 6 raw files are documented."""
        raw_files = ["SPY.parquet", "DIA.parquet", "QQQ.parquet", "DJI.parquet", "IXIC.parquet", "VIX.parquet"]
        for filename in raw_files:
            assert filename in generated_content, f"Raw file {filename} should be documented"

    def test_documents_all_processed_files(self, generated_content: str) -> None:
        """Test that processed files are documented."""
        # Check for key processed datasets
        processed_indicators = ["SPY_features_a20", "DIA_features_a20", "QQQ_features_a20", "VIX_features_c"]
        for dataset in processed_indicators:
            assert dataset in generated_content, f"Processed file {dataset} should be documented"

    def test_includes_column_schema(self, generated_content: str) -> None:
        """Test that column schema tables are included."""
        # Should have column schema headers
        assert "Column" in generated_content and "Dtype" in generated_content

    def test_includes_statistics(self, generated_content: str) -> None:
        """Test that statistics tables are included."""
        # Should have statistics from describe()
        assert "Mean" in generated_content or "mean" in generated_content
        assert "Std" in generated_content or "std" in generated_content


class TestGenerateIdempotent:
    """Test that generator produces consistent output."""

    def test_generate_idempotent(self) -> None:
        """Test that running generator twice produces identical output."""
        from scripts.generate_data_dictionary import generate_data_dictionary

        with tempfile.TemporaryDirectory() as tmpdir:
            output1 = Path(tmpdir) / "dict1.md"
            output2 = Path(tmpdir) / "dict2.md"

            generate_data_dictionary(str(output1))
            generate_data_dictionary(str(output2))

            content1 = output1.read_text()
            content2 = output2.read_text()

            # Remove timestamp line for comparison (it will differ)
            lines1 = [ln for ln in content1.splitlines() if not ln.startswith("Generated:")]
            lines2 = [ln for ln in content2.splitlines() if not ln.startswith("Generated:")]

            assert lines1 == lines2, "Generator should produce consistent output"


class TestColumnDescriptions:
    """Test column description lookup functions."""

    def test_get_column_descriptions_ohlcv(self) -> None:
        """Test that OHLCV columns have descriptions."""
        from scripts.generate_data_dictionary import get_column_descriptions

        descriptions = get_column_descriptions(["Open", "High", "Low", "Close", "Volume"], "ohlcv")

        assert "Open" in descriptions
        assert "opening" in descriptions["Open"].lower() or "price" in descriptions["Open"].lower()
        assert "Close" in descriptions
        assert "Volume" in descriptions

    def test_get_column_descriptions_indicators(self) -> None:
        """Test that tier_a20 indicators have descriptions."""
        from scripts.generate_data_dictionary import get_column_descriptions

        descriptions = get_column_descriptions(["rsi_daily", "sma_50", "macd_line"], "indicators")

        assert "rsi_daily" in descriptions
        assert "sma_50" in descriptions
        assert "macd_line" in descriptions
        # Check descriptions are meaningful
        assert len(descriptions["rsi_daily"]) > 10, "Description should be meaningful"
