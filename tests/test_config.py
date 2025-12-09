"""Tests for experiment configuration loading and validation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    pass


FIXTURES_DIR = Path(__file__).parent / "fixtures"
VALID_CONFIG_PATH = FIXTURES_DIR / "valid_config.yaml"
SAMPLE_FEATURES_PATH = FIXTURES_DIR / "sample_features.parquet"


class TestLoadValidConfig:
    """Test loading valid configuration files."""

    def test_load_valid_config_returns_dataclass(self, tmp_path: Path) -> None:
        """Valid YAML returns ExperimentConfig with all fields populated."""
        from src.config import load_experiment_config
        from src.config.experiment import ExperimentConfig

        # Create a valid config with actual path to fixture
        config_data = {
            "seed": 42,
            "data_path": str(SAMPLE_FEATURES_PATH.resolve()),
            "task": "threshold_1pct",
            "timescale": "daily",
            "context_length": 60,
            "horizon": 5,
            "wandb_project": "test-project",
            "mlflow_experiment": "test-experiment",
        }
        config_path = tmp_path / "valid_config.yaml"
        config_path.write_text(yaml.dump(config_data))

        config = load_experiment_config(config_path)

        assert isinstance(config, ExperimentConfig)
        assert config.seed == 42
        assert config.data_path == str(SAMPLE_FEATURES_PATH.resolve())
        assert config.task == "threshold_1pct"
        assert config.timescale == "daily"
        assert config.context_length == 60
        assert config.horizon == 5
        assert config.wandb_project == "test-project"
        assert config.mlflow_experiment == "test-experiment"


class TestConfigValidation:
    """Test configuration validation rules."""

    def test_load_config_missing_required_field_raises(self, tmp_path: Path) -> None:
        """Missing required field raises ValueError with field name."""
        from src.config import load_experiment_config

        config_data = {
            "seed": 42,
            "task": "threshold_1pct",
            "timescale": "daily",
            # data_path is missing
        }
        config_path = tmp_path / "missing_field.yaml"
        config_path.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError, match="data_path"):
            load_experiment_config(config_path)

    def test_load_config_invalid_task_raises(self, tmp_path: Path) -> None:
        """Invalid task value raises ValueError."""
        from src.config import load_experiment_config

        config_data = {
            "seed": 42,
            "data_path": str(SAMPLE_FEATURES_PATH),
            "task": "predict",  # Invalid - not in allowed list
            "timescale": "daily",
        }
        config_path = tmp_path / "invalid_task.yaml"
        config_path.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError, match="task"):
            load_experiment_config(config_path)

    def test_load_config_invalid_timescale_raises(self, tmp_path: Path) -> None:
        """Invalid timescale value raises ValueError."""
        from src.config import load_experiment_config

        config_data = {
            "seed": 42,
            "data_path": str(SAMPLE_FEATURES_PATH),
            "task": "threshold_1pct",
            "timescale": "hourly",  # Invalid - not in allowed list
        }
        config_path = tmp_path / "invalid_timescale.yaml"
        config_path.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError, match="timescale"):
            load_experiment_config(config_path)

    def test_load_config_validates_paths_exist(self, tmp_path: Path) -> None:
        """Non-existent data_path raises ValueError."""
        from src.config import load_experiment_config

        config_data = {
            "seed": 42,
            "data_path": "/nonexistent/path/to/features.parquet",
            "task": "threshold_1pct",
            "timescale": "daily",
        }
        config_path = tmp_path / "bad_path.yaml"
        config_path.write_text(yaml.dump(config_data))

        with pytest.raises(ValueError, match="data_path"):
            load_experiment_config(config_path)
