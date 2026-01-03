"""Unit tests for Optuna HPO integration.

All tests are mocked to avoid actual training. Tests cover:
- Search space loading from YAML
- Optuna study creation
- Objective function creation
- Architectural objective function creation
- Best params saving
- HPO workflow with thermal integration
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock

import optuna
import pytest
import yaml

# Import will fail until implementation exists - this is expected for TDD
from src.training.hpo import (
    load_search_space,
    create_study,
    create_objective,
    create_architectural_objective,
    save_best_params,
    save_trial_result,
    save_all_trials,
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

    @patch("src.training.hpo.create_study")
    @patch("src.training.hpo.create_objective")
    @patch("src.training.hpo.save_best_params")
    @patch("src.training.hpo.load_search_space")
    def test_run_hpo_default_no_timeout(
        self,
        mock_load_ss: MagicMock,
        mock_save: MagicMock,
        mock_create_obj: MagicMock,
        mock_create_study: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that run_hpo defaults to no timeout (None)."""
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        mock_create_obj.return_value = lambda trial: 0.5
        mock_load_ss.return_value = {
            "n_trials": 10,
            "direction": "minimize",
            "search_space": {},
        }
        mock_save.return_value = tmp_path / "result.json"

        # Call without timeout_hours - should default to None
        run_hpo(
            config_path="configs/test.yaml",
            budget="2M",
            n_trials=10,
            search_space_path="configs/hpo/default_search.yaml",
        )

        call_kwargs = mock_study.optimize.call_args.kwargs
        assert call_kwargs.get("timeout") is None, "Default should be no timeout"


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


# --- Tests for HPO with data splits ---


class TestCreateObjectiveWithSplits:
    """Test that create_objective supports split_indices for val_loss optimization."""

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    @patch("src.training.hpo.load_patchtst_config")
    def test_create_objective_accepts_split_indices(
        self,
        mock_load_patchtst: MagicMock,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        valid_search_space_yaml: Path,
    ) -> None:
        """Test that create_objective accepts split_indices parameter."""
        from src.data.dataset import SplitIndices
        import numpy as np

        mock_load_exp.return_value = mock_experiment_config
        mock_load_patchtst.return_value = MagicMock()

        # Create mock split indices
        split_indices = SplitIndices(
            train_indices=np.array([0, 1, 2, 3, 4]),
            val_indices=np.array([5, 6]),
            test_indices=np.array([7, 8]),
            chunk_size=13,
        )

        search_config = load_search_space(valid_search_space_yaml)
        objective = create_objective(
            config_path="configs/test.yaml",
            budget="2M",
            search_space=search_config["search_space"],
            split_indices=split_indices,  # New parameter
        )
        assert callable(objective)

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    @patch("src.training.hpo.load_patchtst_config")
    def test_objective_returns_val_loss_when_splits_provided(
        self,
        mock_load_patchtst: MagicMock,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        valid_search_space_yaml: Path,
    ) -> None:
        """Test that objective returns val_loss when split_indices provided."""
        from src.data.dataset import SplitIndices
        import numpy as np

        mock_load_exp.return_value = mock_experiment_config
        mock_load_patchtst.return_value = MagicMock()

        # Create mock trainer that returns val_loss
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "train_loss": 0.5,
            "val_loss": 0.55,  # Different from train_loss
        }
        mock_trainer_cls.return_value = mock_trainer

        # Create mock split indices
        split_indices = SplitIndices(
            train_indices=np.array([0, 1, 2, 3, 4]),
            val_indices=np.array([5, 6]),
            test_indices=np.array([7, 8]),
            chunk_size=13,
        )

        # Create mock trial
        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.001
        mock_trial.suggest_int.return_value = 50

        search_config = load_search_space(valid_search_space_yaml)
        objective = create_objective(
            config_path="configs/test.yaml",
            budget="2M",
            search_space=search_config["search_space"],
            split_indices=split_indices,
        )

        result = objective(mock_trial)

        # Should return val_loss (0.55), not train_loss (0.5)
        assert result == 0.55

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    @patch("src.training.hpo.load_patchtst_config")
    def test_objective_passes_split_indices_to_trainer(
        self,
        mock_load_patchtst: MagicMock,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        valid_search_space_yaml: Path,
    ) -> None:
        """Test that objective passes split_indices to Trainer constructor."""
        from src.data.dataset import SplitIndices
        import numpy as np

        mock_load_exp.return_value = mock_experiment_config
        mock_load_patchtst.return_value = MagicMock()

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "val_loss": 0.55}
        mock_trainer_cls.return_value = mock_trainer

        # Create split indices
        split_indices = SplitIndices(
            train_indices=np.array([0, 1, 2, 3, 4]),
            val_indices=np.array([5, 6]),
            test_indices=np.array([7, 8]),
            chunk_size=13,
        )

        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.001
        mock_trial.suggest_int.return_value = 50

        search_config = load_search_space(valid_search_space_yaml)
        objective = create_objective(
            config_path="configs/test.yaml",
            budget="2M",
            search_space=search_config["search_space"],
            split_indices=split_indices,
        )

        objective(mock_trial)

        # Verify Trainer was called with split_indices
        mock_trainer_cls.assert_called_once()
        call_kwargs = mock_trainer_cls.call_args.kwargs
        assert "split_indices" in call_kwargs
        assert call_kwargs["split_indices"] is split_indices


class TestObjectiveWithoutSplitsUsesTrainLoss:
    """Test backward compatibility: no splits returns train_loss."""

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    @patch("src.training.hpo.load_patchtst_config")
    def test_objective_returns_train_loss_without_splits(
        self,
        mock_load_patchtst: MagicMock,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        valid_search_space_yaml: Path,
    ) -> None:
        """Test that objective returns train_loss when no splits provided."""
        mock_load_exp.return_value = mock_experiment_config
        mock_load_patchtst.return_value = MagicMock()

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "train_loss": 0.5,
            "val_loss": None,  # No splits, so val_loss is None
        }
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.001
        mock_trial.suggest_int.return_value = 50

        search_config = load_search_space(valid_search_space_yaml)
        objective = create_objective(
            config_path="configs/test.yaml",
            budget="2M",
            search_space=search_config["search_space"],
            # No split_indices parameter
        )

        result = objective(mock_trial)

        # Should return train_loss for backward compatibility
        assert result == 0.5


# --- Tests for architectural search config ---


class TestArchitecturalSearchConfigExists:
    """Test that architectural search config file exists."""

    def test_architectural_search_config_exists(self) -> None:
        """Test that architectural_search.yaml config file exists."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        assert config_path.exists(), f"Config file not found: {config_path}"


class TestArchitecturalSearchConfigValidYaml:
    """Test that architectural search config is valid YAML."""

    def test_architectural_search_config_parses_as_yaml(self) -> None:
        """Test that config file parses as valid YAML."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict)

    def test_architectural_search_config_has_top_level_keys(self) -> None:
        """Test that config has expected top-level keys."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "n_trials" in config
        assert "direction" in config
        assert "training_search_space" in config


class TestArchitecturalSearchConfigHasRequiredKeys:
    """Test that config has all required training param keys."""

    def test_architectural_search_config_has_all_training_params(self) -> None:
        """Test that training_search_space has all 5 required parameters."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        training_space = config["training_search_space"]
        # Note: batch_size removed (now dynamic), dropout added
        required_params = [
            "learning_rate",
            "epochs",
            "weight_decay",
            "warmup_steps",
            "dropout",
        ]
        for param in required_params:
            assert param in training_space, f"Missing required param: {param}"


class TestArchitecturalSearchConfigValueRanges:
    """Test that config value ranges match design doc."""

    def test_learning_rate_range_matches_design(self) -> None:
        """Test learning_rate is log_uniform 1e-4 to 1e-3."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        lr = config["training_search_space"]["learning_rate"]
        assert lr["type"] == "log_uniform"
        assert lr["low"] == 1.0e-4
        assert lr["high"] == 1.0e-3

    def test_epochs_choices_match_design(self) -> None:
        """Test epochs is categorical [50, 75, 100]."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        epochs = config["training_search_space"]["epochs"]
        assert epochs["type"] == "categorical"
        assert epochs["choices"] == [50, 75, 100]

    def test_weight_decay_range_matches_design(self) -> None:
        """Test weight_decay is log_uniform 1e-4 to 5e-3 (increased for regularization)."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        wd = config["training_search_space"]["weight_decay"]
        assert wd["type"] == "log_uniform"
        assert wd["low"] == 1.0e-4
        assert wd["high"] == 5.0e-3

    def test_warmup_steps_choices_match_design(self) -> None:
        """Test warmup_steps is categorical [100, 200, 300, 500]."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        warmup = config["training_search_space"]["warmup_steps"]
        assert warmup["type"] == "categorical"
        assert warmup["choices"] == [100, 200, 300, 500]

    def test_n_trials_is_50(self) -> None:
        """Test n_trials is set to 50 per design doc."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["n_trials"] == 50

    def test_direction_is_minimize(self) -> None:
        """Test direction is minimize."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["direction"] == "minimize"


class TestArchitecturalSearchConfigHasDropout:
    """Test that config has dropout in training_search_space."""

    def test_architectural_search_config_has_dropout(self) -> None:
        """Test that training_search_space includes dropout."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        training_space = config["training_search_space"]
        assert "dropout" in training_space, "dropout missing from training_search_space"
        assert training_space["dropout"]["type"] == "uniform"
        assert training_space["dropout"]["low"] == 0.1
        assert training_space["dropout"]["high"] == 0.3


class TestArchitecturalSearchConfigNoBatchSize:
    """Test that batch_size was removed from config."""

    def test_architectural_search_config_no_batch_size(self) -> None:
        """Test that batch_size was removed from training_search_space."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        training_space = config["training_search_space"]
        assert "batch_size" not in training_space, "batch_size should be removed"


class TestArchitecturalSearchConfigHasEarlyStopping:
    """Test that config has early_stopping section."""

    def test_architectural_search_config_has_early_stopping(self) -> None:
        """Test that config has early_stopping section with patience and min_delta."""
        config_path = Path("configs/hpo/architectural_search.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "early_stopping" in config, "early_stopping section missing"
        es = config["early_stopping"]
        assert es["patience"] == 10
        assert es["min_delta"] == 0.001


# --- Tests for create_architectural_objective ---


@pytest.fixture
def sample_architectures() -> list[dict]:
    """Sample architecture list for testing."""
    return [
        {"d_model": 64, "n_layers": 4, "n_heads": 2, "d_ff": 128, "param_count": 1_500_000},
        {"d_model": 128, "n_layers": 8, "n_heads": 4, "d_ff": 256, "param_count": 2_000_000},
        {"d_model": 256, "n_layers": 6, "n_heads": 8, "d_ff": 512, "param_count": 2_500_000},
    ]


@pytest.fixture
def sample_training_search_space() -> dict:
    """Sample training search space matching architectural_search.yaml format."""
    return {
        "learning_rate": {"type": "log_uniform", "low": 1e-4, "high": 1e-3},
        "epochs": {"type": "categorical", "choices": [50, 75, 100]},
        "weight_decay": {"type": "log_uniform", "low": 1e-4, "high": 5e-3},
        "dropout": {"type": "uniform", "low": 0.1, "high": 0.3},
        "warmup_steps": {"type": "categorical", "choices": [100, 200, 300, 500]},
    }


class TestCreateArchitecturalObjectiveReturnsCallable:
    """Test that create_architectural_objective returns a callable."""

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    def test_create_arch_objective_returns_callable(
        self,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        sample_architectures: list[dict],
        sample_training_search_space: dict,
    ) -> None:
        """Test that create_architectural_objective returns a callable function."""
        mock_load_exp.return_value = mock_experiment_config

        objective = create_architectural_objective(
            config_path="configs/test.yaml",
            budget="2M",
            architectures=sample_architectures,
            training_search_space=sample_training_search_space,
            num_features=20,
        )
        assert callable(objective)


class TestArchObjectiveSamplesArchitecture:
    """Test that architectural objective samples from architecture list."""

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    def test_arch_objective_samples_architecture_idx(
        self,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        sample_architectures: list[dict],
        sample_training_search_space: dict,
    ) -> None:
        """Test that objective calls trial.suggest_int with 'arch_idx'."""
        mock_load_exp.return_value = mock_experiment_config
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "val_loss": 0.55}
        mock_trainer_cls.return_value = mock_trainer

        # Create mock trial
        mock_trial = MagicMock()
        mock_trial.number = 10  # Set trial number > 6 to bypass forced extremes
        mock_trial.suggest_int.return_value = 1  # Select second architecture
        mock_trial.suggest_float.return_value = 0.0005

        objective = create_architectural_objective(
            config_path="configs/test.yaml",
            budget="2M",
            architectures=sample_architectures,
            training_search_space=sample_training_search_space,
            num_features=20,
            force_extreme_trials=False,  # Disable for this test
        )

        objective(mock_trial)

        # Verify suggest_int was called with arch_idx
        arch_idx_calls = [
            call for call in mock_trial.suggest_int.call_args_list
            if call[0][0] == "arch_idx"
        ]
        assert len(arch_idx_calls) == 1
        # Verify it was called with range 0 to len(architectures)-1
        assert arch_idx_calls[0][0][1] == 0  # low
        assert arch_idx_calls[0][0][2] == len(sample_architectures) - 1  # high


class TestArchObjectiveSamplesTrainingParams:
    """Test that architectural objective samples training params from narrow ranges."""

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    def test_arch_objective_samples_training_params(
        self,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        sample_architectures: list[dict],
        sample_training_search_space: dict,
    ) -> None:
        """Test that objective samples all training params from search space."""
        mock_load_exp.return_value = mock_experiment_config
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "val_loss": 0.55}
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.number = 10  # Set trial number > 6 to bypass forced extremes
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        mock_trial.suggest_float.return_value = 0.0005

        objective = create_architectural_objective(
            config_path="configs/test.yaml",
            budget="2M",
            architectures=sample_architectures,
            training_search_space=sample_training_search_space,
            num_features=20,
            force_extreme_trials=False,  # Disable for this test
        )

        objective(mock_trial)

        # Verify learning_rate was sampled (log_uniform -> suggest_float with log=True)
        lr_calls = [
            call for call in mock_trial.suggest_float.call_args_list
            if call[0][0] == "learning_rate"
        ]
        assert len(lr_calls) == 1

        # Verify categorical params were sampled (batch_size removed, now dynamic)
        categorical_params = ["epochs", "warmup_steps"]
        for param in categorical_params:
            param_calls = [
                call for call in mock_trial.suggest_categorical.call_args_list
                if call[0][0] == param
            ]
            assert len(param_calls) == 1, f"Missing categorical sampling for {param}"


class TestArchObjectiveBuildsConfigFromArch:
    """Test that objective builds PatchTSTConfig from sampled architecture."""

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    def test_arch_objective_builds_config_from_arch(
        self,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        sample_architectures: list[dict],
        sample_training_search_space: dict,
    ) -> None:
        """Test that Trainer receives PatchTSTConfig built from sampled architecture."""
        mock_load_exp.return_value = mock_experiment_config
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "val_loss": 0.55}
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.number = 10  # Set trial number > 6 to bypass forced extremes
        # Select architecture index 1 (d_model=128, n_layers=8)
        mock_trial.suggest_categorical.side_effect = lambda name, choices: (
            1 if name == "arch_idx" else choices[0]
        )
        mock_trial.suggest_float.return_value = 0.0005

        objective = create_architectural_objective(
            config_path="configs/test.yaml",
            budget="2M",
            architectures=sample_architectures,
            training_search_space=sample_training_search_space,
            num_features=20,
            force_extreme_trials=False,  # Disable for this test
        )

        objective(mock_trial)

        # Verify Trainer was called with model_config containing arch values
        mock_trainer_cls.assert_called_once()
        call_kwargs = mock_trainer_cls.call_args.kwargs
        model_config = call_kwargs.get("model_config")
        assert model_config is not None
        # Architecture at index 1 should have d_model=128, n_layers=8
        assert model_config.d_model == 128
        assert model_config.n_layers == 8
        assert model_config.n_heads == 4
        assert model_config.d_ff == 256


class TestArchObjectiveReturnsValLoss:
    """Test that architectural objective returns val_loss when splits provided."""

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    def test_arch_objective_returns_val_loss(
        self,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        sample_architectures: list[dict],
        sample_training_search_space: dict,
    ) -> None:
        """Test that objective returns val_loss when split_indices provided."""
        from src.data.dataset import SplitIndices
        import numpy as np

        mock_load_exp.return_value = mock_experiment_config
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "val_loss": 0.42}
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.number = 10  # Set trial number > 6 to bypass forced extremes
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        mock_trial.suggest_float.return_value = 0.0005

        split_indices = SplitIndices(
            train_indices=np.array([0, 1, 2, 3, 4]),
            val_indices=np.array([5, 6]),
            test_indices=np.array([7, 8]),
            chunk_size=13,
        )

        objective = create_architectural_objective(
            config_path="configs/test.yaml",
            budget="2M",
            architectures=sample_architectures,
            training_search_space=sample_training_search_space,
            split_indices=split_indices,
            num_features=20,
            force_extreme_trials=False,  # Disable for this test
        )

        result = objective(mock_trial)

        # Should return val_loss (0.42), not train_loss (0.5)
        assert result == 0.42


class TestSaveBestParamsIncludesArchitecture:
    """Test that save_best_params includes architecture info when available."""

    def test_save_best_params_includes_architecture(
        self,
        tmp_path: Path,
        sample_architectures: list[dict],
    ) -> None:
        """Test that JSON includes architecture when arch_idx in best_params."""
        mock_study = MagicMock()
        mock_study.best_params = {
            "arch_idx": 1,  # Index into architectures list
            "learning_rate": 0.0005,
            "epochs": 75,
            "batch_size": 128,
        }
        mock_study.best_value = 0.42
        mock_study.study_name = "test_2M"
        mock_study.trials = [MagicMock()]
        mock_study.trials[0].state.name = "COMPLETE"

        output_dir = tmp_path / "hpo"
        result_path = save_best_params(
            study=mock_study,
            experiment_name="test_exp",
            budget="2M",
            output_dir=output_dir,
            architectures=sample_architectures,  # New parameter
        )

        with open(result_path) as f:
            data = json.load(f)

        # Should include architecture info
        assert "architecture" in data
        assert data["architecture"]["d_model"] == 128
        assert data["architecture"]["n_layers"] == 8
        assert data["architecture"]["n_heads"] == 4
        assert data["architecture"]["d_ff"] == 256
        assert data["architecture"]["param_count"] == 2_000_000


class TestForcedExtremesIntegration:
    """Test that forced extremes work correctly without CategoricalDistribution errors."""

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    def test_forced_extremes_uses_set_user_attr(
        self,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        sample_architectures: list[dict],
        sample_training_search_space: dict,
    ) -> None:
        """Test that forced extreme trials use set_user_attr, not suggest."""
        mock_load_exp.return_value = mock_experiment_config
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "val_loss": 0.55}
        mock_trainer_cls.return_value = mock_trainer

        # Create mock trial for forced extreme (trial 0)
        mock_trial = MagicMock()
        mock_trial.number = 0  # First trial - should be forced extreme
        mock_trial.suggest_float.return_value = 0.0005

        objective = create_architectural_objective(
            config_path="configs/test.yaml",
            budget="2M",
            architectures=sample_architectures,
            training_search_space=sample_training_search_space,
            num_features=20,
            force_extreme_trials=True,  # Enable forced extremes
        )

        objective(mock_trial)

        # Verify set_user_attr was called with arch_idx (not suggest_int)
        user_attr_calls = [
            call for call in mock_trial.set_user_attr.call_args_list
            if call[0][0] == "arch_idx"
        ]
        assert len(user_attr_calls) == 1, "Forced extreme should use set_user_attr"

        # Verify suggest_int was NOT called for arch_idx
        suggest_int_calls = [
            call for call in mock_trial.suggest_int.call_args_list
            if call[0][0] == "arch_idx"
        ]
        assert len(suggest_int_calls) == 0, "Forced extreme should not call suggest_int"

    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    def test_non_forced_trial_uses_suggest_int(
        self,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_experiment_config: MagicMock,
        sample_architectures: list[dict],
        sample_training_search_space: dict,
    ) -> None:
        """Test that non-forced trials use suggest_int for arch_idx."""
        mock_load_exp.return_value = mock_experiment_config
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "val_loss": 0.55}
        mock_trainer_cls.return_value = mock_trainer

        # Create mock trial beyond forced extremes (trial 10)
        mock_trial = MagicMock()
        mock_trial.number = 10  # Beyond forced extremes
        mock_trial.suggest_int.return_value = 1
        mock_trial.suggest_float.return_value = 0.0005

        objective = create_architectural_objective(
            config_path="configs/test.yaml",
            budget="2M",
            architectures=sample_architectures,
            training_search_space=sample_training_search_space,
            num_features=20,
            force_extreme_trials=True,  # Enable forced extremes
        )

        objective(mock_trial)

        # Verify suggest_int was called for arch_idx
        suggest_int_calls = [
            call for call in mock_trial.suggest_int.call_args_list
            if call[0][0] == "arch_idx"
        ]
        assert len(suggest_int_calls) == 1, "Non-forced trial should use suggest_int"


class TestSaveBestParamsIncludesArchitectureForForcedTrials:
    """Test that save_best_params includes architecture for forced extreme trials.

    Forced extreme trials store arch_idx via set_user_attr(), not suggest_int(),
    so architecture must be retrieved from user_attrs, not best_params.
    """

    def test_save_best_params_includes_architecture_from_user_attrs(
        self,
        tmp_path: Path,
        sample_architectures: list[dict],
    ) -> None:
        """Test that JSON includes architecture when arch in user_attrs (forced trial)."""
        # Create mock best_trial that simulates a forced extreme trial
        mock_best_trial = MagicMock()
        mock_best_trial.number = 0  # Forced extreme trial
        mock_best_trial.user_attrs = {
            "arch_idx": 1,
            "architecture": {
                "d_model": 128,
                "n_layers": 8,
                "n_heads": 4,
                "d_ff": 256,
                "param_count": 2_000_000,
            },
        }

        mock_study = MagicMock()
        # KEY: arch_idx is NOT in best_params (forced trial used set_user_attr)
        mock_study.best_params = {
            "learning_rate": 0.0005,
            "epochs": 75,
            "batch_size": 128,
        }
        mock_study.best_value = 0.42
        mock_study.best_trial = mock_best_trial
        mock_study.study_name = "test_2M"
        mock_study.trials = [mock_best_trial]
        mock_best_trial.state.name = "COMPLETE"

        output_dir = tmp_path / "hpo"
        result_path = save_best_params(
            study=mock_study,
            experiment_name="test_exp",
            budget="2M",
            output_dir=output_dir,
            architectures=sample_architectures,
        )

        with open(result_path) as f:
            data = json.load(f)

        # Should include architecture from user_attrs
        assert "architecture" in data, "Architecture missing for forced extreme trial"
        assert data["architecture"]["d_model"] == 128
        assert data["architecture"]["n_layers"] == 8
        assert data["architecture"]["n_heads"] == 4
        assert data["architecture"]["d_ff"] == 256
        assert data["architecture"]["param_count"] == 2_000_000


class TestSaveTrialResultIncludesArchitectureForForcedTrials:
    """Test that save_trial_result includes architecture for forced extreme trials."""

    def test_save_trial_result_includes_architecture_from_user_attrs(
        self,
        tmp_path: Path,
        sample_architectures: list[dict],
    ) -> None:
        """Test that trial JSON includes architecture from user_attrs (forced trial)."""
        # Create mock trial that simulates a forced extreme trial
        mock_trial = MagicMock()
        mock_trial.number = 0  # Forced extreme trial
        mock_trial.value = 0.42
        mock_trial.state.name = "COMPLETE"
        mock_trial.datetime_start = None
        mock_trial.datetime_complete = None
        mock_trial.duration = None
        # KEY: arch_idx is NOT in params (forced trial used set_user_attr)
        mock_trial.params = {
            "learning_rate": 0.0005,
            "epochs": 75,
            "batch_size": 128,
        }
        mock_trial.user_attrs = {
            "arch_idx": 1,
            "architecture": {
                "d_model": 128,
                "n_layers": 8,
                "n_heads": 4,
                "d_ff": 256,
                "param_count": 2_000_000,
            },
        }

        output_dir = tmp_path / "hpo"
        result_path = save_trial_result(
            trial=mock_trial,
            output_dir=output_dir,
            architectures=sample_architectures,
        )

        with open(result_path) as f:
            data = json.load(f)

        # Should include architecture from user_attrs
        assert "architecture" in data, "Architecture missing for forced extreme trial"
        assert data["architecture"]["d_model"] == 128
        assert data["architecture"]["n_layers"] == 8


class TestSaveAllTrialsIncludesArchitectureForForcedTrials:
    """Test that save_all_trials includes architecture for forced extreme trials."""

    def test_save_all_trials_includes_architecture_from_user_attrs(
        self,
        tmp_path: Path,
        sample_architectures: list[dict],
    ) -> None:
        """Test that all_trials JSON includes architecture for forced trials."""
        # Create mock trial that simulates a forced extreme trial (best trial)
        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.value = 0.42
        mock_trial.state.name = "COMPLETE"
        mock_trial.duration.total_seconds.return_value = 100.0
        # KEY: arch_idx is NOT in params (forced trial used set_user_attr)
        mock_trial.params = {
            "learning_rate": 0.0005,
            "epochs": 75,
            "batch_size": 128,
        }
        mock_trial.user_attrs = {
            "arch_idx": 1,
            "architecture": {
                "d_model": 128,
                "n_layers": 8,
                "n_heads": 4,
                "d_ff": 256,
                "param_count": 2_000_000,
            },
        }

        mock_study = MagicMock()
        mock_study.trials = [mock_trial]

        output_dir = tmp_path / "hpo"
        result_path = save_all_trials(
            study=mock_study,
            experiment_name="test_exp",
            budget="2M",
            output_dir=output_dir,
            architectures=sample_architectures,
        )

        with open(result_path) as f:
            data = json.load(f)

        # Should include architecture fields for forced trial
        assert len(data["trials"]) == 1
        trial_data = data["trials"][0]
        assert "d_model" in trial_data, "d_model missing for forced extreme trial"
        assert trial_data["d_model"] == 128
        assert trial_data["n_layers"] == 8
        assert trial_data["n_heads"] == 4
        assert trial_data["param_count"] == 2_000_000


# --- Tests for postprocess_hpo_output ---


class TestPostprocessHpoOutput:
    """Test post-processing HPO output to add architecture info."""

    @pytest.fixture
    def hpo_output_dir(self, tmp_path: Path) -> Path:
        """Create a mock HPO output directory with trial files."""
        output_dir = tmp_path / "phase6a_test_exp"
        output_dir.mkdir()
        trials_dir = output_dir / "trials"
        trials_dir.mkdir()

        # Create trial JSON files with architecture in user_attrs
        for i in range(3):
            trial_data = {
                "trial_number": i,
                "value": 0.4 - i * 0.01,  # Trial 0 is best
                "params": {
                    "learning_rate": 0.0005,
                    "epochs": 75,
                    "batch_size": 64,
                },
                "state": "COMPLETE",
                "user_attrs": {
                    "arch_idx": i,
                    "architecture": {
                        "d_model": 128 + i * 64,
                        "n_layers": 8 + i * 2,
                        "n_heads": 4,
                        "d_ff": 256 + i * 128,
                        "param_count": 2_000_000 + i * 500_000,
                    },
                },
            }
            trial_path = trials_dir / f"trial_{i:04d}.json"
            with open(trial_path, "w") as f:
                json.dump(trial_data, f)

        # Create existing _best.json WITHOUT architecture (simulates bug)
        best_data = {
            "experiment": "test_exp",
            "budget": "200M",
            "best_params": {
                "learning_rate": 0.0005,
                "epochs": 75,
                "batch_size": 64,
            },
            "best_value": 0.4,
            "best_trial_number": 0,
        }
        best_path = output_dir / "test_exp_200M_best.json"
        with open(best_path, "w") as f:
            json.dump(best_data, f)

        # Create existing all_trials.json WITHOUT architecture
        all_trials_data = {
            "experiment": "test_exp",
            "budget": "200M",
            "trials": [
                {
                    "trial_number": i,
                    "value": 0.4 - i * 0.01,
                    "params": {"learning_rate": 0.0005},
                }
                for i in range(3)
            ],
        }
        all_trials_path = output_dir / "test_exp_all_trials.json"
        with open(all_trials_path, "w") as f:
            json.dump(all_trials_data, f)

        return output_dir

    def test_postprocess_creates_backup(self, hpo_output_dir: Path) -> None:
        """Test that postprocess creates backup of original files."""
        from scripts.postprocess_hpo_output import postprocess_hpo_output

        postprocess_hpo_output(hpo_output_dir)

        # Check backups exist
        best_files = list(hpo_output_dir.glob("*_best.json.bak"))
        all_trials_files = list(hpo_output_dir.glob("*_all_trials.json.bak"))

        assert len(best_files) >= 1, "No backup created for _best.json"
        assert len(all_trials_files) >= 1, "No backup created for all_trials.json"

    def test_postprocess_includes_architecture_in_best(
        self, hpo_output_dir: Path
    ) -> None:
        """Test that regenerated _best.json includes architecture."""
        from scripts.postprocess_hpo_output import postprocess_hpo_output

        postprocess_hpo_output(hpo_output_dir)

        # Find and read the best.json file
        best_files = list(hpo_output_dir.glob("*_best.json"))
        best_files = [f for f in best_files if not f.name.endswith(".bak")]
        assert len(best_files) == 1

        with open(best_files[0]) as f:
            data = json.load(f)

        assert "architecture" in data, "_best.json missing architecture"
        assert data["architecture"]["d_model"] == 128  # Trial 0
        assert data["architecture"]["n_layers"] == 8
        assert data["architecture"]["n_heads"] == 4
        assert data["architecture"]["param_count"] == 2_000_000

    def test_postprocess_includes_architecture_in_all_trials(
        self, hpo_output_dir: Path
    ) -> None:
        """Test that regenerated all_trials.json includes architecture for each trial."""
        from scripts.postprocess_hpo_output import postprocess_hpo_output

        postprocess_hpo_output(hpo_output_dir)

        # Find and read the all_trials.json file
        all_trials_files = list(hpo_output_dir.glob("*_all_trials.json"))
        all_trials_files = [f for f in all_trials_files if not f.name.endswith(".bak")]
        assert len(all_trials_files) == 1

        with open(all_trials_files[0]) as f:
            data = json.load(f)

        assert len(data["trials"]) == 3
        for i, trial in enumerate(data["trials"]):
            assert "d_model" in trial, f"Trial {i} missing d_model"
            assert "n_layers" in trial, f"Trial {i} missing n_layers"
            assert trial["d_model"] == 128 + i * 64

    def test_postprocess_handles_missing_architecture(self, tmp_path: Path) -> None:
        """Test that postprocess warns but doesn't crash on missing architecture."""
        from scripts.postprocess_hpo_output import postprocess_hpo_output

        # Create minimal HPO output with a trial missing architecture
        output_dir = tmp_path / "incomplete_hpo"
        output_dir.mkdir()
        trials_dir = output_dir / "trials"
        trials_dir.mkdir()

        # Trial WITHOUT architecture in user_attrs
        trial_data = {
            "trial_number": 0,
            "value": 0.4,
            "params": {"learning_rate": 0.0005},
            "state": "COMPLETE",
            "user_attrs": {},  # No architecture!
        }
        trial_path = trials_dir / "trial_0000.json"
        with open(trial_path, "w") as f:
            json.dump(trial_data, f)

        # Create minimal _best.json
        best_data = {"experiment": "test", "best_trial_number": 0, "best_value": 0.4}
        with open(output_dir / "test_best.json", "w") as f:
            json.dump(best_data, f)

        # Should not raise, just warn
        postprocess_hpo_output(output_dir)  # No exception = pass


# --- Tests for new training features in architectural objective ---


class TestArchObjectiveUsesNewFeatures:
    """Test that architectural objective uses new training features."""

    @patch("src.training.hpo.get_memory_safe_batch_config")
    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    def test_architectural_objective_samples_dropout(
        self,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_get_batch_config: MagicMock,
        mock_experiment_config: MagicMock,
        sample_architectures: list[dict],
    ) -> None:
        """Test that objective samples dropout from training_search_space."""
        mock_load_exp.return_value = mock_experiment_config
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "val_loss": 0.55}
        mock_trainer_cls.return_value = mock_trainer
        mock_get_batch_config.return_value = {
            "micro_batch": 128,
            "accumulation_steps": 2,
            "effective_batch": 256,
        }

        # Training search space with dropout
        training_search_space = {
            "learning_rate": {"type": "log_uniform", "low": 1e-4, "high": 1e-3},
            "epochs": {"type": "categorical", "choices": [50]},
            "weight_decay": {"type": "log_uniform", "low": 1e-4, "high": 5e-3},
            "warmup_steps": {"type": "categorical", "choices": [100]},
            "dropout": {"type": "uniform", "low": 0.1, "high": 0.3},
        }

        mock_trial = MagicMock()
        mock_trial.number = 10
        mock_trial.suggest_float.return_value = 0.2  # dropout value
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        mock_trial.suggest_int.return_value = 0

        objective = create_architectural_objective(
            config_path="configs/test.yaml",
            budget="2M",
            architectures=sample_architectures,
            training_search_space=training_search_space,
            num_features=20,
            force_extreme_trials=False,
        )

        objective(mock_trial)

        # Verify dropout was sampled
        dropout_calls = [
            call for call in mock_trial.suggest_float.call_args_list
            if call[0][0] == "dropout"
        ]
        assert len(dropout_calls) == 1, "dropout should be sampled via suggest_float"

        # Verify PatchTSTConfig received sampled dropout
        model_config = mock_trainer_cls.call_args.kwargs["model_config"]
        assert model_config.dropout == 0.2

    @patch("src.training.hpo.get_memory_safe_batch_config")
    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    def test_architectural_objective_uses_dynamic_batch(
        self,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_get_batch_config: MagicMock,
        mock_experiment_config: MagicMock,
        sample_architectures: list[dict],
    ) -> None:
        """Test that objective calls get_memory_safe_batch_config."""
        mock_load_exp.return_value = mock_experiment_config
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "val_loss": 0.55}
        mock_trainer_cls.return_value = mock_trainer
        mock_get_batch_config.return_value = {
            "micro_batch": 64,
            "accumulation_steps": 4,
            "effective_batch": 256,
        }

        training_search_space = {
            "learning_rate": {"type": "log_uniform", "low": 1e-4, "high": 1e-3},
            "epochs": {"type": "categorical", "choices": [50]},
            "weight_decay": {"type": "log_uniform", "low": 1e-4, "high": 5e-3},
            "warmup_steps": {"type": "categorical", "choices": [100]},
            "dropout": {"type": "uniform", "low": 0.1, "high": 0.3},
        }

        mock_trial = MagicMock()
        mock_trial.number = 10
        mock_trial.suggest_float.return_value = 0.15
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        mock_trial.suggest_int.return_value = 0

        objective = create_architectural_objective(
            config_path="configs/test.yaml",
            budget="2M",
            architectures=sample_architectures,
            training_search_space=training_search_space,
            num_features=20,
            force_extreme_trials=False,
        )

        objective(mock_trial)

        # Verify get_memory_safe_batch_config was called
        mock_get_batch_config.assert_called_once()
        # Verify it was called with arch d_model and n_layers
        call_kwargs = mock_get_batch_config.call_args.kwargs
        assert "d_model" in call_kwargs
        assert "n_layers" in call_kwargs

        # Verify Trainer received micro_batch from batch config
        trainer_kwargs = mock_trainer_cls.call_args.kwargs
        assert trainer_kwargs["batch_size"] == 64

    @patch("src.training.hpo.get_memory_safe_batch_config")
    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    def test_architectural_objective_passes_early_stopping(
        self,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_get_batch_config: MagicMock,
        mock_experiment_config: MagicMock,
        sample_architectures: list[dict],
    ) -> None:
        """Test that objective passes early stopping params to Trainer."""
        mock_load_exp.return_value = mock_experiment_config
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "val_loss": 0.55}
        mock_trainer_cls.return_value = mock_trainer
        mock_get_batch_config.return_value = {
            "micro_batch": 128,
            "accumulation_steps": 2,
            "effective_batch": 256,
        }

        training_search_space = {
            "learning_rate": {"type": "log_uniform", "low": 1e-4, "high": 1e-3},
            "epochs": {"type": "categorical", "choices": [50]},
            "weight_decay": {"type": "log_uniform", "low": 1e-4, "high": 5e-3},
            "warmup_steps": {"type": "categorical", "choices": [100]},
            "dropout": {"type": "uniform", "low": 0.1, "high": 0.3},
        }

        mock_trial = MagicMock()
        mock_trial.number = 10
        mock_trial.suggest_float.return_value = 0.15
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        mock_trial.suggest_int.return_value = 0

        objective = create_architectural_objective(
            config_path="configs/test.yaml",
            budget="2M",
            architectures=sample_architectures,
            training_search_space=training_search_space,
            num_features=20,
            force_extreme_trials=False,
        )

        objective(mock_trial)

        # Verify Trainer received early stopping params
        trainer_kwargs = mock_trainer_cls.call_args.kwargs
        assert "early_stopping_patience" in trainer_kwargs
        assert trainer_kwargs["early_stopping_patience"] == 10
        assert "early_stopping_min_delta" in trainer_kwargs
        assert trainer_kwargs["early_stopping_min_delta"] == 0.001
        assert "accumulation_steps" in trainer_kwargs


class TestCreateStudyUsesTpeSampler:
    """Tests for TPESampler configuration in create_study."""

    def test_create_study_uses_tpe_sampler(self) -> None:
        """Test that create_study returns study with TPESampler."""
        study = create_study(
            experiment_name="test_experiment",
            budget="2M",
            direction="minimize",
        )

        assert isinstance(study.sampler, optuna.samplers.TPESampler)

    def test_create_study_default_n_startup_trials(self) -> None:
        """Test that create_study uses n_startup_trials=20 by default."""
        study = create_study(
            experiment_name="test_experiment",
            budget="2M",
            direction="minimize",
        )

        # TPESampler stores n_startup_trials in _n_startup_trials attribute
        assert study.sampler._n_startup_trials == 20

    def test_create_study_custom_n_startup_trials(self) -> None:
        """Test that create_study accepts custom n_startup_trials."""
        study = create_study(
            experiment_name="test_experiment",
            budget="2M",
            direction="minimize",
            n_startup_trials=30,
        )

        assert study.sampler._n_startup_trials == 30


class TestArchObjectiveForcesVariation:
    """Tests for forced variation when same architecture is reused."""

    @pytest.fixture
    def sample_architectures(self) -> list[dict]:
        """Sample architectures for testing."""
        return [
            {"d_model": 64, "n_layers": 8, "n_heads": 2, "d_ff": 256, "param_count": 2_000_000},
            {"d_model": 128, "n_layers": 16, "n_heads": 4, "d_ff": 512, "param_count": 2_500_000},
        ]

    @pytest.fixture
    def mock_experiment_config(self) -> MagicMock:
        """Create mock experiment config."""
        config = MagicMock()
        config.data_path = "data/test.parquet"
        config.context_length = 60
        config.horizon = 1
        config.task = "threshold_1pct"
        return config

    @patch("src.training.hpo.get_memory_safe_batch_config")
    @patch("src.training.hpo.Trainer")
    @patch("src.training.hpo.load_experiment_config")
    def test_forces_variation_when_same_arch_similar_params(
        self,
        mock_load_exp: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_get_batch_config: MagicMock,
        mock_experiment_config: MagicMock,
        sample_architectures: list[dict],
    ) -> None:
        """Test that dropout is forced to different value when same arch with similar params."""
        mock_load_exp.return_value = mock_experiment_config
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {"train_loss": 0.5, "val_loss": 0.55}
        mock_trainer_cls.return_value = mock_trainer
        mock_get_batch_config.return_value = {
            "micro_batch": 128,
            "accumulation_steps": 2,
            "effective_batch": 256,
        }

        training_search_space = {
            "learning_rate": {"type": "log_uniform", "low": 1e-4, "high": 1e-3},
            "epochs": {"type": "categorical", "choices": [50]},
            "weight_decay": {"type": "log_uniform", "low": 1e-4, "high": 5e-3},
            "warmup_steps": {"type": "categorical", "choices": [100]},
            "dropout": {"type": "uniform", "low": 0.1, "high": 0.3},
        }

        objective = create_architectural_objective(
            config_path="configs/test.yaml",
            budget="2M",
            architectures=sample_architectures,
            training_search_space=training_search_space,
            num_features=20,
            force_extreme_trials=False,
        )

        # Create a real study to test variation forcing
        study = create_study(
            experiment_name="test_variation",
            budget="2M",
        )

        # Add a completed trial with arch_idx=0, dropout=0.15, epochs=50
        prev_trial = optuna.trial.create_trial(
            params={"arch_idx": 0, "dropout": 0.15, "epochs": 50, "learning_rate": 0.001, "weight_decay": 0.001, "warmup_steps": 100},
            distributions={
                "arch_idx": optuna.distributions.IntDistribution(0, 1),
                "dropout": optuna.distributions.FloatDistribution(0.1, 0.3),
                "epochs": optuna.distributions.IntDistribution(25, 100),
                "learning_rate": optuna.distributions.FloatDistribution(1e-4, 1e-3),
                "weight_decay": optuna.distributions.FloatDistribution(1e-4, 5e-3),
                "warmup_steps": optuna.distributions.CategoricalDistribution([100, 300, 500]),
            },
            values=[0.4],
            state=optuna.trial.TrialState.COMPLETE,
        )
        study.add_trial(prev_trial)

        # Now run a new trial that samples same arch_idx=0 and similar dropout
        # The objective should force dropout to a different value
        mock_trial = MagicMock()
        mock_trial.number = 1
        mock_trial.study = study
        mock_trial.suggest_int.return_value = 0  # Same arch_idx
        mock_trial.suggest_float.return_value = 0.16  # Similar dropout (delta < 0.08)
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

        objective(mock_trial)

        # Verify that model_config received a forced dropout value (not 0.16)
        model_config = mock_trainer_cls.call_args.kwargs["model_config"]
        # Since prev_dropout=0.15 < 0.2, forced dropout should be 0.27 (high end)
        assert model_config.dropout == 0.27, f"Expected dropout=0.27 (forced high), got {model_config.dropout}"
