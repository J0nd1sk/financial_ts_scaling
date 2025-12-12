"""Tests for experiment runner module.

Tests for CSV logging, markdown report generation, and experiment execution.
Uses fixtures and mocking - no actual training runs.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.runner import (
    update_experiment_log,
    regenerate_results_report,
    run_hpo_experiment,
    run_training_experiment,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_experiment_result():
    """Sample experiment result matching CSV schema."""
    return {
        "timestamp": "2025-12-11T12:00:00",
        "experiment": "phase6a_2M_threshold_1pct",
        "phase": "phase6a",
        "budget": "2M",
        "task": "threshold_1pct",
        "horizon": 1,
        "timescale": "daily",
        "script_path": "experiments/phase6a/hpo_2M_threshold_1pct.py",
        "run_type": "hpo",
        "status": "success",
        "duration_seconds": 3600.5,
        "val_loss": 0.485,
        "test_accuracy": 0.52,
        "hyperparameters": {"learning_rate": 0.001, "epochs": 50},
        "error_message": None,
        "thermal_max_temp": 72.5,
        "data_md5": "abc123def456",
        # Architecture columns (new for architectural HPO)
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 512,
        "param_count": 2_100_000,
    }


@pytest.fixture
def sample_failed_result(sample_experiment_result):
    """Sample failed experiment result."""
    result = sample_experiment_result.copy()
    result["status"] = "failed"
    result["val_loss"] = None
    result["test_accuracy"] = None
    result["error_message"] = "CUDA out of memory"
    return result


@pytest.fixture
def csv_log_with_entries(tmp_path, sample_experiment_result):
    """Create CSV log file with existing entries."""
    log_path = tmp_path / "results" / "experiment_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Create initial entry
    row = sample_experiment_result.copy()
    row["hyperparameters"] = json.dumps(row["hyperparameters"])

    df = pd.DataFrame([row])
    df.to_csv(log_path, index=False)

    return log_path


# =============================================================================
# Test: update_experiment_log - Basic
# =============================================================================


class TestUpdateExperimentLogBasic:
    """Tests for basic update_experiment_log functionality."""

    def test_update_experiment_log_creates_file(self, tmp_path, sample_experiment_result):
        """Test that update_experiment_log creates CSV file if missing."""
        log_path = tmp_path / "results" / "experiment_log.csv"

        update_experiment_log(sample_experiment_result, log_path)

        assert log_path.exists()

    def test_update_experiment_log_returns_path(self, tmp_path, sample_experiment_result):
        """Test that update_experiment_log returns the log path."""
        log_path = tmp_path / "results" / "experiment_log.csv"

        result = update_experiment_log(sample_experiment_result, log_path)

        assert result == log_path


class TestUpdateExperimentLogSchema:
    """Tests for CSV schema compliance."""

    def test_update_experiment_log_has_expected_columns(
        self, tmp_path, sample_experiment_result
    ):
        """Test that CSV has all 16 expected columns."""
        log_path = tmp_path / "results" / "experiment_log.csv"

        update_experiment_log(sample_experiment_result, log_path)

        df = pd.read_csv(log_path)
        expected_columns = [
            "timestamp",
            "experiment",
            "phase",
            "budget",
            "task",
            "horizon",
            "timescale",
            "script_path",
            "run_type",
            "status",
            "duration_seconds",
            "val_loss",
            "test_accuracy",
            "hyperparameters",
            "error_message",
            "thermal_max_temp",
            "data_md5",
            # Architecture columns (new for architectural HPO)
            "d_model",
            "n_layers",
            "n_heads",
            "d_ff",
            "param_count",
        ]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_update_experiment_log_serializes_hyperparameters(
        self, tmp_path, sample_experiment_result
    ):
        """Test that hyperparameters dict is serialized to JSON string."""
        log_path = tmp_path / "results" / "experiment_log.csv"

        update_experiment_log(sample_experiment_result, log_path)

        df = pd.read_csv(log_path)
        hp_value = df["hyperparameters"].iloc[0]

        # Should be valid JSON string
        parsed = json.loads(hp_value)
        assert parsed == {"learning_rate": 0.001, "epochs": 50}


class TestUpdateExperimentLogAppend:
    """Tests for appending to existing log."""

    def test_update_experiment_log_appends_to_existing(
        self, csv_log_with_entries, sample_experiment_result
    ):
        """Test that new entries are appended to existing CSV."""
        # Modify result to be distinguishable
        new_result = sample_experiment_result.copy()
        new_result["experiment"] = "phase6a_20M_threshold_1pct"
        new_result["budget"] = "20M"

        update_experiment_log(new_result, csv_log_with_entries)

        df = pd.read_csv(csv_log_with_entries)
        assert len(df) == 2

    def test_update_experiment_log_preserves_existing_data(
        self, csv_log_with_entries, sample_experiment_result
    ):
        """Test that appending preserves existing data."""
        new_result = sample_experiment_result.copy()
        new_result["experiment"] = "phase6a_20M_threshold_1pct"

        update_experiment_log(new_result, csv_log_with_entries)

        df = pd.read_csv(csv_log_with_entries)
        # First row should still have original experiment name
        assert df.iloc[0]["experiment"] == "phase6a_2M_threshold_1pct"


class TestUpdateExperimentLogFailedRuns:
    """Tests for logging failed experiments."""

    def test_update_experiment_log_handles_failed_run(
        self, tmp_path, sample_failed_result
    ):
        """Test that failed runs are logged with error message."""
        log_path = tmp_path / "results" / "experiment_log.csv"

        update_experiment_log(sample_failed_result, log_path)

        df = pd.read_csv(log_path)
        assert df.iloc[0]["status"] == "failed"
        assert df.iloc[0]["error_message"] == "CUDA out of memory"
        assert pd.isna(df.iloc[0]["val_loss"])


class TestUpdateExperimentLogArchitecture:
    """Tests for architecture column handling in CSV logging."""

    def test_update_experiment_log_includes_architecture_columns(
        self, tmp_path, sample_experiment_result
    ):
        """Test that CSV includes the 5 architecture columns."""
        log_path = tmp_path / "results" / "experiment_log.csv"

        update_experiment_log(sample_experiment_result, log_path)

        df = pd.read_csv(log_path)
        arch_columns = ["d_model", "n_layers", "n_heads", "d_ff", "param_count"]
        for col in arch_columns:
            assert col in df.columns, f"Missing architecture column: {col}"

    def test_update_experiment_log_architecture_values_correct(
        self, tmp_path, sample_experiment_result
    ):
        """Test that architecture values are correctly written to CSV."""
        log_path = tmp_path / "results" / "experiment_log.csv"

        update_experiment_log(sample_experiment_result, log_path)

        df = pd.read_csv(log_path)
        assert df.iloc[0]["d_model"] == 128
        assert df.iloc[0]["n_layers"] == 4
        assert df.iloc[0]["n_heads"] == 4
        assert df.iloc[0]["d_ff"] == 512
        assert df.iloc[0]["param_count"] == 2_100_000

    def test_update_experiment_log_without_architecture(self, tmp_path):
        """Test backwards compatibility - result without architecture fields logs with None."""
        # Result dict without architecture columns (legacy format)
        legacy_result = {
            "timestamp": "2025-12-11T12:00:00",
            "experiment": "legacy_experiment",
            "phase": "phase6a",
            "budget": "2M",
            "task": "threshold_1pct",
            "horizon": 1,
            "timescale": "daily",
            "script_path": "experiments/legacy.py",
            "run_type": "hpo",
            "status": "success",
            "duration_seconds": 1800.0,
            "val_loss": 0.49,
            "test_accuracy": 0.51,
            "hyperparameters": {"learning_rate": 0.001},
            "error_message": None,
            "thermal_max_temp": 70.0,
            "data_md5": "legacy123",
            # No architecture columns - simulating legacy result
        }
        log_path = tmp_path / "results" / "experiment_log.csv"

        update_experiment_log(legacy_result, log_path)

        df = pd.read_csv(log_path)
        # Architecture columns should exist but have null values
        assert "d_model" in df.columns
        assert pd.isna(df.iloc[0]["d_model"])
        assert pd.isna(df.iloc[0]["param_count"])


# =============================================================================
# Test: regenerate_results_report
# =============================================================================


class TestRegenerateResultsReportBasic:
    """Tests for basic regenerate_results_report functionality."""

    def test_regenerate_results_report_creates_file(
        self, csv_log_with_entries, tmp_path
    ):
        """Test that regenerate_results_report creates markdown file."""
        output_path = tmp_path / "docs" / "experiment_results.md"

        regenerate_results_report(csv_log_with_entries, output_path)

        assert output_path.exists()

    def test_regenerate_results_report_returns_path(
        self, csv_log_with_entries, tmp_path
    ):
        """Test that regenerate_results_report returns the output path."""
        output_path = tmp_path / "docs" / "experiment_results.md"

        result = regenerate_results_report(csv_log_with_entries, output_path)

        assert result == output_path


class TestRegenerateResultsReportContent:
    """Tests for markdown report content."""

    def test_regenerate_results_report_has_header(
        self, csv_log_with_entries, tmp_path
    ):
        """Test that markdown has expected header."""
        output_path = tmp_path / "docs" / "experiment_results.md"

        regenerate_results_report(csv_log_with_entries, output_path)

        content = output_path.read_text()
        assert "# Experiment Results" in content

    def test_regenerate_results_report_includes_summary(
        self, csv_log_with_entries, tmp_path
    ):
        """Test that report includes experiment summary statistics."""
        output_path = tmp_path / "docs" / "experiment_results.md"

        regenerate_results_report(csv_log_with_entries, output_path)

        content = output_path.read_text()
        # Should have total count and success rate
        assert "Total" in content or "Experiments" in content

    def test_regenerate_results_report_empty_log_creates_stub(self, tmp_path):
        """Test that empty CSV creates stub report (not error)."""
        # Create empty CSV with headers only
        log_path = tmp_path / "results" / "experiment_log.csv"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["timestamp", "experiment", "status"]).to_csv(
            log_path, index=False
        )

        output_path = tmp_path / "docs" / "experiment_results.md"

        regenerate_results_report(log_path, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "No experiments" in content or "0" in content


# =============================================================================
# Test: run_hpo_experiment
# =============================================================================


class TestRunHpoExperimentBasic:
    """Tests for run_hpo_experiment with mocked dependencies."""

    @patch("src.experiments.runner.run_hpo")
    @patch("src.experiments.runner.ThermalCallback")
    def test_run_hpo_experiment_calls_hpo(
        self, mock_thermal, mock_run_hpo, tmp_path
    ):
        """Test that run_hpo_experiment calls underlying HPO function."""
        mock_thermal.return_value.check.return_value = MagicMock(
            status="normal", temperature=65.0
        )
        mock_run_hpo.return_value = {"best_params": {"lr": 0.001}, "best_value": 0.5}

        run_hpo_experiment(
            experiment="test_exp",
            budget="2M",
            task="threshold_1pct",
            data_path=tmp_path / "data.parquet",
            output_dir=tmp_path / "outputs",
        )

        mock_run_hpo.assert_called_once()

    @patch("src.experiments.runner.run_hpo")
    @patch("src.experiments.runner.ThermalCallback")
    def test_run_hpo_experiment_returns_result_dict(
        self, mock_thermal, mock_run_hpo, tmp_path
    ):
        """Test that run_hpo_experiment returns dict with expected keys."""
        mock_thermal.return_value.check.return_value = MagicMock(
            status="normal", temperature=65.0
        )
        mock_run_hpo.return_value = {"best_params": {"lr": 0.001}, "best_value": 0.5}

        result = run_hpo_experiment(
            experiment="test_exp",
            budget="2M",
            task="threshold_1pct",
            data_path=tmp_path / "data.parquet",
            output_dir=tmp_path / "outputs",
        )

        assert isinstance(result, dict)
        assert "status" in result
        assert "val_loss" in result


class TestRunHpoExperimentThermal:
    """Tests for thermal monitoring during HPO."""

    @patch("src.experiments.runner.run_hpo")
    @patch("src.experiments.runner.ThermalCallback")
    def test_run_hpo_experiment_checks_thermal_preflight(
        self, mock_thermal, mock_run_hpo, tmp_path
    ):
        """Test that thermal status is checked before starting."""
        mock_thermal.return_value.check.return_value = MagicMock(
            status="critical", temperature=96.0
        )

        result = run_hpo_experiment(
            experiment="test_exp",
            budget="2M",
            task="threshold_1pct",
            data_path=tmp_path / "data.parquet",
            output_dir=tmp_path / "outputs",
        )

        # Should abort without running HPO
        mock_run_hpo.assert_not_called()
        assert result["status"] == "thermal_abort"

    @patch("src.experiments.runner.run_hpo")
    @patch("src.experiments.runner.ThermalCallback")
    def test_run_hpo_experiment_records_max_temp(
        self, mock_thermal, mock_run_hpo, tmp_path
    ):
        """Test that max temperature is recorded in result."""
        mock_thermal.return_value.check.return_value = MagicMock(
            status="normal", temperature=72.5
        )
        mock_run_hpo.return_value = {"best_params": {"lr": 0.001}, "best_value": 0.5}

        result = run_hpo_experiment(
            experiment="test_exp",
            budget="2M",
            task="threshold_1pct",
            data_path=tmp_path / "data.parquet",
            output_dir=tmp_path / "outputs",
        )

        assert "thermal_max_temp" in result


# =============================================================================
# Test: run_training_experiment
# =============================================================================


class TestRunTrainingExperimentBasic:
    """Tests for run_training_experiment with mocked dependencies."""

    @patch("src.experiments.runner.Trainer")
    @patch("src.experiments.runner.ThermalCallback")
    def test_run_training_experiment_calls_trainer(
        self, mock_thermal, mock_trainer, tmp_path
    ):
        """Test that run_training_experiment uses Trainer class."""
        mock_thermal.return_value.check.return_value = MagicMock(
            status="normal", temperature=65.0
        )
        mock_trainer.return_value.train.return_value = {"val_loss": 0.45}

        run_training_experiment(
            experiment="test_exp",
            budget="2M",
            task="threshold_1pct",
            data_path=tmp_path / "data.parquet",
            hyperparameters={"learning_rate": 0.001},
            output_dir=tmp_path / "outputs",
        )

        mock_trainer.assert_called_once()

    @patch("src.experiments.runner.Trainer")
    @patch("src.experiments.runner.ThermalCallback")
    def test_run_training_experiment_returns_result_dict(
        self, mock_thermal, mock_trainer, tmp_path
    ):
        """Test that run_training_experiment returns dict with expected keys."""
        mock_thermal.return_value.check.return_value = MagicMock(
            status="normal", temperature=65.0
        )
        mock_trainer.return_value.train.return_value = {"val_loss": 0.45}

        result = run_training_experiment(
            experiment="test_exp",
            budget="2M",
            task="threshold_1pct",
            data_path=tmp_path / "data.parquet",
            hyperparameters={"learning_rate": 0.001},
            output_dir=tmp_path / "outputs",
        )

        assert isinstance(result, dict)
        assert "status" in result
        assert "val_loss" in result


class TestRunTrainingExperimentThermal:
    """Tests for thermal monitoring during training."""

    @patch("src.experiments.runner.Trainer")
    @patch("src.experiments.runner.ThermalCallback")
    def test_run_training_experiment_aborts_on_critical(
        self, mock_thermal, mock_trainer, tmp_path
    ):
        """Test that training aborts on critical thermal status."""
        mock_thermal.return_value.check.return_value = MagicMock(
            status="critical", temperature=96.0
        )

        result = run_training_experiment(
            experiment="test_exp",
            budget="2M",
            task="threshold_1pct",
            data_path=tmp_path / "data.parquet",
            hyperparameters={"learning_rate": 0.001},
            output_dir=tmp_path / "outputs",
        )

        mock_trainer.assert_not_called()
        assert result["status"] == "thermal_abort"
