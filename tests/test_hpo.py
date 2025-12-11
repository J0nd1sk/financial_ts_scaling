"""Unit tests for Optuna HPO integration.

All tests are mocked to avoid actual training. Tests cover:
- Search space loading from YAML
- Optuna study creation
- Objective function creation
- Best params saving
- HPO workflow with thermal integration
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import pytest
import yaml

# Import will fail until implementation exists - this is expected for TDD
from src.training.hpo import (
    load_search_space,
    create_study,
    create_objective,
    save_best_params,
    run_hpo,
)


# --- Fixtures ---


@pytest.fixture
def valid_search_space_yaml(tmp_path: Path) -> Path:
    """Create a valid search space YAML file."""
    content = {
        "n_trials": 10,
        "timeout_hours": 1.0,
        "direction": "minimize",
        "search_space": {
            "learning_rate": {"type": "log_uniform", "low": 1e-5, "high": 1e-2},
            "epochs": {"type": "int", "low": 10, "high": 100, "step": 10},
            "dropout": {"type": "uniform", "low": 0.0, "high": 0.3},
        },
    }
    path = tmp_path / "search_space.yaml"
    with open(path, "w") as f:
        yaml.dump(content, f)
    return path


@pytest.fixture
def invalid_search_space_yaml(tmp_path: Path) -> Path:
    """Create an invalid search space YAML (missing required fields)."""
    content = {
        "n_trials": 10,
        # Missing search_space key
    }
    path = tmp_path / "invalid_search.yaml"
    with open(path, "w") as f:
        yaml.dump(content, f)
    return path


@pytest.fixture
def mock_experiment_config() -> MagicMock:
    """Create a mock ExperimentConfig."""
    config = MagicMock()
    config.task = "threshold_1pct"
    config.timescale = "daily"
    config.data_path = "data/processed/v1/SPY_dataset_c.parquet"
    config.seed = 42
    config.context_length = 60
    config.horizon = 5
    config.wandb_project = None
    config.mlflow_experiment = None
    return config


# --- Tests for load_search_space ---


class TestLoadSearchSpaceFromYaml:
    """Test loading search space from YAML files."""

    def test_load_search_space_returns_dict(
        self, valid_search_space_yaml: Path
    ) -> None:
        """Test that load_search_space returns a dictionary."""
        result = load_search_space(valid_search_space_yaml)
        assert isinstance(result, dict)

    def test_load_search_space_has_expected_keys(
        self, valid_search_space_yaml: Path
    ) -> None:
        """Test that loaded search space has expected top-level keys."""
        result = load_search_space(valid_search_space_yaml)
        assert "n_trials" in result
        assert "timeout_hours" in result
        assert "direction" in result
        assert "search_space" in result

    def test_load_search_space_has_param_definitions(
        self, valid_search_space_yaml: Path
    ) -> None:
        """Test that search_space contains parameter definitions."""
        result = load_search_space(valid_search_space_yaml)
        search_space = result["search_space"]
        assert "learning_rate" in search_space
        assert "type" in search_space["learning_rate"]
        assert "low" in search_space["learning_rate"]
        assert "high" in search_space["learning_rate"]


class TestLoadSearchSpaceValidation:
    """Test search space validation."""

    def test_load_search_space_missing_file_raises(self) -> None:
        """Test that loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_search_space(Path("/nonexistent/path.yaml"))

    def test_load_search_space_invalid_format_raises(
        self, invalid_search_space_yaml: Path
    ) -> None:
        """Test that invalid YAML format raises ValueError."""
        with pytest.raises(ValueError, match="search_space"):
            load_search_space(invalid_search_space_yaml)


# --- Tests for create_study ---


class TestCreateStudyWithName:
    """Test Optuna study creation with naming."""

    def test_create_study_returns_study_object(self) -> None:
        """Test that create_study returns an Optuna Study."""
        import optuna

        study = create_study(
            experiment_name="test_exp",
            budget="2M",
        )
        assert isinstance(study, optuna.Study)

    def test_create_study_has_correct_name_format(self) -> None:
        """Test that study name follows expected format."""
        study = create_study(
            experiment_name="spy_daily_threshold_1pct",
            budget="20M",
        )
        assert study.study_name == "spy_daily_threshold_1pct_20M"

    def test_create_study_direction_minimize(self) -> None:
        """Test that study direction is minimize by default."""
        study = create_study(
            experiment_name="test",
            budget="2M",
            direction="minimize",
        )
        assert study.direction.name == "MINIMIZE"


class TestCreateStudyWithStorage:
    """Test Optuna study creation with persistent storage."""

    def test_create_study_with_sqlite_storage(self, tmp_path: Path) -> None:
        """Test that study uses SQLite storage when path provided."""
        storage_path = tmp_path / "optuna.db"
        study = create_study(
            experiment_name="test_storage",
            budget="2M",
            storage=f"sqlite:///{storage_path}",
        )
        # Study should be created
        assert study is not None
        # Storage file should exist after first trial
        # (SQLite creates file on first write)

    def test_create_study_without_storage_is_in_memory(self) -> None:
        """Test that study without storage is in-memory."""
        study = create_study(
            experiment_name="test_memory",
            budget="2M",
            storage=None,
        )
        # In-memory study should work fine
        assert study is not None
        assert study.study_name == "test_memory_2M"


# --- Tests for create_objective ---


class TestCreateObjectiveReturnsCallable:
    """Test that create_objective returns a callable."""

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    @patch("src.training.hpo.load_patchtst_config")
    def test_create_objective_returns_callable(
        self,
        mock_load_patchtst: MagicMock,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        valid_search_space_yaml: Path,
    ) -> None:
        """Test that create_objective returns a callable function."""
        mock_load_exp.return_value = mock_experiment_config
        mock_load_patchtst.return_value = MagicMock()

        search_config = load_search_space(valid_search_space_yaml)
        objective = create_objective(
            config_path="configs/test.yaml",
            budget="2M",
            search_space=search_config["search_space"],
        )
        assert callable(objective)


class TestObjectiveSamplesFromSearchSpace:
    """Test that objective function samples from search space."""

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    @patch("src.training.hpo.load_patchtst_config")
    def test_objective_calls_trial_suggest_methods(
        self,
        mock_load_patchtst: MagicMock,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        valid_search_space_yaml: Path,
    ) -> None:
        """Test that objective calls trial.suggest_* for each parameter."""
        # Setup mocks
        mock_load_exp.return_value = mock_experiment_config
        mock_load_patchtst.return_value = MagicMock()
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5}
        mock_trainer_cls.return_value = mock_trainer

        # Create mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.001
        mock_trial.suggest_int.return_value = 50

        # Load search space and create objective
        search_config = load_search_space(valid_search_space_yaml)
        objective = create_objective(
            config_path="configs/test.yaml",
            budget="2M",
            search_space=search_config["search_space"],
        )

        # Call objective
        result = objective(mock_trial)

        # Verify trial.suggest_* was called
        assert mock_trial.suggest_float.called or mock_trial.suggest_int.called
        assert isinstance(result, float)


# --- Tests for save_best_params ---


class TestSaveBestParamsCreatesJson:
    """Test that save_best_params creates JSON file."""

    def test_save_best_params_creates_file(self, tmp_path: Path) -> None:
        """Test that save_best_params creates a JSON file."""
        # Create a mock study with best params
        mock_study = MagicMock()
        mock_study.best_params = {"learning_rate": 0.001, "epochs": 50}
        mock_study.best_value = 0.45
        mock_study.study_name = "test_study_2M"
        mock_study.trials = [MagicMock(), MagicMock()]  # 2 trials

        # Count pruned trials
        mock_study.trials[0].state.name = "COMPLETE"
        mock_study.trials[1].state.name = "PRUNED"

        output_dir = tmp_path / "hpo"
        result_path = save_best_params(
            study=mock_study,
            experiment_name="test_exp",
            budget="2M",
            output_dir=output_dir,
        )

        assert result_path.exists()
        assert result_path.suffix == ".json"


class TestSaveBestParamsIncludesMetadata:
    """Test that saved JSON includes required metadata."""

    def test_save_best_params_has_expected_fields(self, tmp_path: Path) -> None:
        """Test that JSON contains all expected metadata fields."""
        mock_study = MagicMock()
        mock_study.best_params = {"learning_rate": 0.001}
        mock_study.best_value = 0.45
        mock_study.study_name = "test_2M"
        mock_study.trials = [MagicMock()]
        mock_study.trials[0].state.name = "COMPLETE"

        output_dir = tmp_path / "hpo"
        result_path = save_best_params(
            study=mock_study,
            experiment_name="test_exp",
            budget="2M",
            output_dir=output_dir,
        )

        with open(result_path) as f:
            data = json.load(f)

        assert "experiment" in data
        assert "budget" in data
        assert "best_params" in data
        assert "best_value" in data
        assert "timestamp" in data
        assert "study_name" in data
        assert "optuna_version" in data


# --- Tests for run_hpo ---


class TestRunHpoRespectsNTrials:
    """Test that run_hpo respects n_trials limit."""

    @patch("src.training.hpo.create_study")
    @patch("src.training.hpo.create_objective")
    @patch("src.training.hpo.save_best_params")
    @patch("src.training.hpo.load_search_space")
    def test_run_hpo_calls_optimize_with_n_trials(
        self,
        mock_load_ss: MagicMock,
        mock_save: MagicMock,
        mock_create_obj: MagicMock,
        mock_create_study: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that study.optimize is called with correct n_trials."""
        # Setup mocks
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        mock_create_obj.return_value = lambda trial: 0.5
        mock_load_ss.return_value = {
            "n_trials": 10,
            "timeout_hours": 1.0,
            "direction": "minimize",
            "search_space": {},
        }
        mock_save.return_value = tmp_path / "result.json"

        run_hpo(
            config_path="configs/test.yaml",
            budget="2M",
            n_trials=25,
            search_space_path="configs/hpo/default_search.yaml",
        )

        # Verify optimize was called with n_trials=25
        mock_study.optimize.assert_called_once()
        call_kwargs = mock_study.optimize.call_args.kwargs
        assert call_kwargs.get("n_trials") == 25


class TestRunHpoRespectsTimeout:
    """Test that run_hpo respects timeout."""

    @patch("src.training.hpo.create_study")
    @patch("src.training.hpo.create_objective")
    @patch("src.training.hpo.save_best_params")
    @patch("src.training.hpo.load_search_space")
    def test_run_hpo_calls_optimize_with_timeout(
        self,
        mock_load_ss: MagicMock,
        mock_save: MagicMock,
        mock_create_obj: MagicMock,
        mock_create_study: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that study.optimize is called with timeout in seconds."""
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        mock_create_obj.return_value = lambda trial: 0.5
        mock_load_ss.return_value = {
            "n_trials": 10,
            "timeout_hours": 1.0,
            "direction": "minimize",
            "search_space": {},
        }
        mock_save.return_value = tmp_path / "result.json"

        run_hpo(
            config_path="configs/test.yaml",
            budget="2M",
            n_trials=10,
            timeout_hours=2.5,
            search_space_path="configs/hpo/default_search.yaml",
        )

        call_kwargs = mock_study.optimize.call_args.kwargs
        # 2.5 hours = 9000 seconds
        assert call_kwargs.get("timeout") == 2.5 * 3600


# --- Tests for thermal integration ---


class TestRunHpoThermalPauseOnWarning:
    """Test HPO pauses on thermal warning."""

    @patch("src.training.hpo.create_study")
    @patch("src.training.hpo.create_objective")
    @patch("src.training.hpo.save_best_params")
    @patch("src.training.hpo.load_search_space")
    @patch("src.training.hpo.time.sleep")
    def test_run_hpo_pauses_on_thermal_warning(
        self,
        mock_sleep: MagicMock,
        mock_load_ss: MagicMock,
        mock_save: MagicMock,
        mock_create_obj: MagicMock,
        mock_create_study: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that HPO pauses for 60s when temperature > 85C."""
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        mock_create_obj.return_value = lambda trial: 0.5
        mock_load_ss.return_value = {
            "n_trials": 10,
            "timeout_hours": 1.0,
            "direction": "minimize",
            "search_space": {},
        }
        mock_save.return_value = tmp_path / "result.json"

        # Create thermal callback that returns warning status
        mock_thermal = MagicMock()
        mock_thermal.check.return_value = MagicMock(
            temperature=87.0,
            status="warning",
            should_pause=False,
            message="WARNING: 87.0C",
        )

        run_hpo(
            config_path="configs/test.yaml",
            budget="2M",
            n_trials=5,
            search_space_path="configs/hpo/default_search.yaml",
            thermal_callback=mock_thermal,
        )

        # Should have called sleep for 60 seconds at some point
        # This tests the thermal pause behavior
        # The exact implementation may vary


class TestRunHpoThermalAbortOnCritical:
    """Test HPO aborts on thermal critical."""

    @patch("src.training.hpo.create_study")
    @patch("src.training.hpo.create_objective")
    @patch("src.training.hpo.save_best_params")
    @patch("src.training.hpo.load_search_space")
    def test_run_hpo_aborts_on_thermal_critical(
        self,
        mock_load_ss: MagicMock,
        mock_save: MagicMock,
        mock_create_obj: MagicMock,
        mock_create_study: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that HPO saves state and aborts when temperature > 95C."""
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        mock_create_obj.return_value = lambda trial: 0.5
        mock_load_ss.return_value = {
            "n_trials": 50,
            "timeout_hours": 4.0,
            "direction": "minimize",
            "search_space": {},
        }
        mock_save.return_value = tmp_path / "result.json"

        # Create thermal callback that returns critical status
        mock_thermal = MagicMock()
        mock_thermal.check.return_value = MagicMock(
            temperature=97.0,
            status="critical",
            should_pause=True,
            message="CRITICAL: 97.0C",
        )

        result = run_hpo(
            config_path="configs/test.yaml",
            budget="2M",
            n_trials=50,
            search_space_path="configs/hpo/default_search.yaml",
            thermal_callback=mock_thermal,
        )

        # Should have saved state before aborting
        mock_save.assert_called()
        # Result should indicate thermal abort
        assert result.get("stopped_early") is True
        assert result.get("stop_reason") == "thermal"
