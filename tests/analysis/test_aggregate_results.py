"""Tests for result aggregation module.

Uses synthetic JSON fixtures to verify HPO/training result aggregation.
No actual experiment results required - all tests use fixtures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.aggregate_results import (
    aggregate_hpo_results,
    summarize_experiment,
    export_results_csv,
    generate_experiment_summary_report,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_hpo_json():
    """Sample HPO result JSON matching save_best_params() format."""
    return {
        "experiment": "test_exp",
        "budget": "2M",
        "best_params": {"learning_rate": 0.001, "epochs": 50},
        "best_value": 0.52,
        "n_trials_completed": 10,
        "n_trials_pruned": 2,
        "timestamp": "2025-12-11T12:00:00+00:00",
        "study_name": "test_exp_2M",
    }


@pytest.fixture
def hpo_dir_with_results(tmp_path):
    """Create temporary HPO directory with multiple experiment results."""
    hpo_dir = tmp_path / "hpo"

    # Experiment 1: test_exp_a with 3 budgets
    exp1_dir = hpo_dir / "test_exp_a"
    exp1_dir.mkdir(parents=True)

    for budget, val in [("2M", 0.55), ("20M", 0.48), ("200M", 0.42)]:
        result = {
            "experiment": "test_exp_a",
            "budget": budget,
            "best_params": {"learning_rate": 0.001},
            "best_value": val,
            "n_trials_completed": 10,
            "n_trials_pruned": 0,
            "timestamp": "2025-12-11T12:00:00+00:00",
            "study_name": f"test_exp_a_{budget}",
        }
        with open(exp1_dir / f"test_exp_a_{budget}_best.json", "w") as f:
            json.dump(result, f)

    # Experiment 2: test_exp_b with 2 budgets
    exp2_dir = hpo_dir / "test_exp_b"
    exp2_dir.mkdir(parents=True)

    for budget, val in [("2M", 0.60), ("20M", 0.52)]:
        result = {
            "experiment": "test_exp_b",
            "budget": budget,
            "best_params": {"learning_rate": 0.002},
            "best_value": val,
            "n_trials_completed": 8,
            "n_trials_pruned": 1,
            "timestamp": "2025-12-11T13:00:00+00:00",
            "study_name": f"test_exp_b_{budget}",
        }
        with open(exp2_dir / f"test_exp_b_{budget}_best.json", "w") as f:
            json.dump(result, f)

    return hpo_dir


# =============================================================================
# Test: aggregate_hpo_results - Basic
# =============================================================================


class TestAggregateHpoResultsBasic:
    """Tests for basic aggregate_hpo_results functionality."""

    def test_aggregate_hpo_results_returns_dataframe(self, hpo_dir_with_results):
        """Test that aggregate_hpo_results returns a pandas DataFrame."""
        result = aggregate_hpo_results(hpo_dir=str(hpo_dir_with_results))
        assert isinstance(result, pd.DataFrame)

    def test_aggregate_hpo_results_expected_columns(self, hpo_dir_with_results):
        """Test that result DataFrame has expected columns."""
        result = aggregate_hpo_results(hpo_dir=str(hpo_dir_with_results))

        expected_columns = [
            "experiment",
            "budget",
            "best_value",
            "n_trials_completed",
            "timestamp",
        ]
        for col in expected_columns:
            assert col in result.columns, f"Missing column: {col}"


# =============================================================================
# Test: aggregate_hpo_results - Empty Directory
# =============================================================================


class TestAggregateHpoResultsEmpty:
    """Tests for aggregate_hpo_results with empty/missing directories."""

    def test_aggregate_hpo_results_empty_dir(self, tmp_path):
        """Test that empty directory returns empty DataFrame (not error)."""
        empty_dir = tmp_path / "empty_hpo"
        empty_dir.mkdir()

        result = aggregate_hpo_results(hpo_dir=str(empty_dir))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# =============================================================================
# Test: aggregate_hpo_results - Filter by Experiment
# =============================================================================


class TestAggregateHpoResultsFilter:
    """Tests for filtering aggregate_hpo_results by experiment name."""

    def test_aggregate_hpo_results_filters_by_experiment(self, hpo_dir_with_results):
        """Test that experiment_name filter returns only matching results."""
        result = aggregate_hpo_results(
            experiment_name="test_exp_a",
            hpo_dir=str(hpo_dir_with_results),
        )

        assert len(result) == 3  # test_exp_a has 3 budgets
        assert all(result["experiment"] == "test_exp_a")


# =============================================================================
# Test: summarize_experiment
# =============================================================================


class TestSummarizeExperiment:
    """Tests for summarize_experiment function."""

    def test_summarize_experiment_returns_dict(self, hpo_dir_with_results):
        """Test that summarize_experiment returns dict with expected keys."""
        result = summarize_experiment(
            experiment_name="test_exp_a",
            hpo_dir=str(hpo_dir_with_results),
        )

        assert isinstance(result, dict)
        assert "best_budget" in result
        assert "scaling_factor" in result
        assert "hpo_summary" in result


# =============================================================================
# Test: export_results_csv
# =============================================================================


class TestExportResultsCsv:
    """Tests for CSV export functionality."""

    def test_export_results_csv_creates_file(self, hpo_dir_with_results, tmp_path):
        """Test that export_results_csv creates a CSV file."""
        output_path = tmp_path / "results" / "all_results.csv"

        result_path = export_results_csv(
            hpo_dir=str(hpo_dir_with_results),
            output_path=str(output_path),
        )

        assert Path(result_path).exists()
        assert str(result_path).endswith(".csv")

    def test_export_results_csv_correct_content(self, hpo_dir_with_results, tmp_path):
        """Test that CSV content matches aggregated DataFrame."""
        output_path = tmp_path / "results" / "all_results.csv"

        export_results_csv(
            hpo_dir=str(hpo_dir_with_results),
            output_path=str(output_path),
        )

        # Read back and verify
        df = pd.read_csv(output_path)
        assert len(df) == 5  # 3 from exp_a + 2 from exp_b
        assert "experiment" in df.columns
        assert "best_value" in df.columns


# =============================================================================
# Test: generate_experiment_summary_report
# =============================================================================


class TestGenerateSummaryReport:
    """Tests for markdown summary report generation."""

    def test_generate_summary_report_creates_md(self, hpo_dir_with_results, tmp_path):
        """Test that generate_experiment_summary_report creates markdown file."""
        output_path = tmp_path / "reports" / "summary.md"

        result_path = generate_experiment_summary_report(
            hpo_dir=str(hpo_dir_with_results),
            output_path=str(output_path),
        )

        assert Path(result_path).exists()

        # Verify content has sections
        content = Path(result_path).read_text()
        assert "# Experiment Summary" in content or "## " in content
