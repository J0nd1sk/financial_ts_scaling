"""Integration tests for training pipeline.

Tests the Trainer class which integrates:
- PatchTST model
- FinancialDataset
- ThermalCallback
- TrackingManager
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.config.experiment import ExperimentConfig
from src.data.dataset import FinancialDataset
from src.models.patchtst import PatchTST, PatchTSTConfig
from src.training.thermal import ThermalCallback, ThermalStatus
from src.training.tracking import TrackingConfig, TrackingManager
from src.training.trainer import Trainer  # To be implemented


# --- Fixtures ---


@pytest.fixture
def micro_dataset_df() -> pd.DataFrame:
    """Create a micro dataset (100 rows) for fast testing."""
    np.random.seed(42)
    n_rows = 100

    # Create OHLCV data
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="D")
    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)

    df = pd.DataFrame({
        "Date": dates,
        "Open": close + np.random.randn(n_rows) * 0.1,
        "High": close + np.abs(np.random.randn(n_rows) * 0.5),
        "Low": close - np.abs(np.random.randn(n_rows) * 0.5),
        "Close": close,
        "Volume": np.random.randint(1000000, 10000000, n_rows),
    })

    # Add 20 fake feature columns
    for i in range(20):
        df[f"feature_{i}"] = np.random.randn(n_rows)

    return df


@pytest.fixture
def micro_dataset_path(micro_dataset_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Save micro dataset to parquet and return path."""
    path = tmp_path / "micro_dataset.parquet"
    micro_dataset_df.to_parquet(path, index=False)
    return path


@pytest.fixture
def experiment_config(micro_dataset_path: Path) -> ExperimentConfig:
    """Create an ExperimentConfig for testing."""
    return ExperimentConfig(
        data_path=str(micro_dataset_path),
        task="threshold_1pct",
        timescale="daily",
        seed=42,
        context_length=10,  # Small for fast tests
        horizon=3,
        wandb_project=None,  # Disabled for tests
        mlflow_experiment=None,  # Disabled for tests
    )


@pytest.fixture
def model_config() -> PatchTSTConfig:
    """Create a tiny PatchTSTConfig for fast testing.

    Note: num_features=25 matches micro_dataset (OHLCV + 20 features, excluding Date).
    """
    return PatchTSTConfig(
        num_features=25,  # OHLCV (5) + 20 fake features, excluding Date
        context_length=10,
        patch_length=5,
        stride=5,
        d_model=32,
        n_heads=2,
        n_layers=1,
        d_ff=64,
        dropout=0.0,
        head_dropout=0.0,
        num_classes=1,
    )


@pytest.fixture
def mock_thermal_normal() -> ThermalCallback:
    """Create a ThermalCallback that always returns normal status."""
    return ThermalCallback(temp_provider=lambda: 50.0)


@pytest.fixture
def mock_thermal_critical() -> ThermalCallback:
    """Create a ThermalCallback that always returns critical status."""
    return ThermalCallback(temp_provider=lambda: 98.0)


# --- Tests ---


class TestTrainOneEpoch:
    """Test basic training loop completion."""

    def test_train_one_epoch_completes(
        self,
        experiment_config: ExperimentConfig,
        model_config: PatchTSTConfig,
        mock_thermal_normal: ThermalCallback,
        tmp_path: Path,
    ) -> None:
        """Test that training one epoch completes without error."""
        # Arrange
        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=8,
            learning_rate=0.001,
            epochs=1,
            device="cpu",  # Use CPU for test reliability
            checkpoint_dir=tmp_path,
            thermal_callback=mock_thermal_normal,
            tracking_manager=None,  # Disabled
        )

        # Act
        result = trainer.train()

        # Assert
        assert result is not None
        assert "train_loss" in result
        assert isinstance(result["train_loss"], float)
        assert result["train_loss"] > 0  # Loss should be positive


class TestTrainLogsMetrics:
    """Test that metrics are logged to trackers."""

    @patch("src.training.tracking.wandb")
    @patch("src.training.tracking.mlflow")
    def test_train_logs_metrics(
        self,
        mock_mlflow: MagicMock,
        mock_wandb: MagicMock,
        experiment_config: ExperimentConfig,
        model_config: PatchTSTConfig,
        mock_thermal_normal: ThermalCallback,
        tmp_path: Path,
    ) -> None:
        """Test that loss is logged to trackers during training."""
        # Arrange
        tracking_config = TrackingConfig(
            wandb_project="test-project",
            mlflow_experiment="test-experiment",
        )
        tracking_manager = TrackingManager(tracking_config)

        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=8,
            learning_rate=0.001,
            epochs=1,
            device="cpu",
            checkpoint_dir=tmp_path,
            thermal_callback=mock_thermal_normal,
            tracking_manager=tracking_manager,
        )

        # Act
        trainer.train()

        # Assert - wandb.log should have been called with train_loss
        mock_wandb.log.assert_called()
        calls = mock_wandb.log.call_args_list
        logged_keys = set()
        for call in calls:
            if call.args:
                logged_keys.update(call.args[0].keys())
            elif call.kwargs:
                logged_keys.update(call.kwargs.keys())
        assert "train_loss" in logged_keys or any(
            "train_loss" in str(c) for c in calls
        )


class TestTrainSavesCheckpoint:
    """Test that checkpoints are saved."""

    def test_train_saves_checkpoint(
        self,
        experiment_config: ExperimentConfig,
        model_config: PatchTSTConfig,
        mock_thermal_normal: ThermalCallback,
        tmp_path: Path,
    ) -> None:
        """Test that a checkpoint file is created after training."""
        # Arrange
        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=8,
            learning_rate=0.001,
            epochs=1,
            device="cpu",
            checkpoint_dir=tmp_path,
            thermal_callback=mock_thermal_normal,
            tracking_manager=None,
        )

        # Act
        trainer.train()

        # Assert - checkpoint file should exist
        checkpoint_files = list(tmp_path.glob("*.pt"))
        assert len(checkpoint_files) >= 1, "No checkpoint file created"

        # Verify checkpoint contents
        checkpoint = torch.load(checkpoint_files[0], weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "epoch" in checkpoint


class TestTrainRespectsThermalStop:
    """Test that training stops on thermal critical."""

    def test_train_respects_thermal_stop(
        self,
        experiment_config: ExperimentConfig,
        model_config: PatchTSTConfig,
        mock_thermal_critical: ThermalCallback,
        tmp_path: Path,
    ) -> None:
        """Test that training stops when thermal callback returns critical."""
        # Arrange
        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=8,
            learning_rate=0.001,
            epochs=10,  # Would take longer if not stopped
            device="cpu",
            checkpoint_dir=tmp_path,
            thermal_callback=mock_thermal_critical,
            tracking_manager=None,
        )

        # Act
        result = trainer.train()

        # Assert - training should have stopped early
        assert result is not None
        assert result.get("stopped_early", False) is True
        assert result.get("stop_reason") == "thermal"


class TestTrainVerifiesManifest:
    """Test that manifest is verified before training."""

    def test_training_verifies_manifest_before_start(
        self,
        model_config: PatchTSTConfig,
        mock_thermal_normal: ThermalCallback,
        tmp_path: Path,
    ) -> None:
        """Test that training fails if data file doesn't exist.

        Verification happens during Trainer initialization, before train() is called.
        """
        # Arrange - create config with non-existent data path
        # We need to create a file first so ExperimentConfig doesn't fail,
        # then delete it to simulate missing data
        fake_data_path = tmp_path / "fake_data.parquet"

        # Create minimal valid parquet
        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=10),
            "Close": [100.0] * 10,
            **{f"feature_{i}": [0.0] * 10 for i in range(20)},
        })
        df.to_parquet(fake_data_path, index=False)

        config = ExperimentConfig(
            data_path=str(fake_data_path),
            task="threshold_1pct",
            timescale="daily",
            seed=42,
            context_length=5,
            horizon=2,
        )

        # Now delete the file to simulate missing/invalid data
        fake_data_path.unlink()

        # Act & Assert - should raise during Trainer initialization
        with pytest.raises(FileNotFoundError):
            Trainer(
                experiment_config=config,
                model_config=model_config,
                batch_size=8,
                learning_rate=0.001,
                epochs=1,
                device="cpu",
                checkpoint_dir=tmp_path,
                thermal_callback=mock_thermal_normal,
                tracking_manager=None,
            )


class TestTrainLogsDataVersion:
    """Test that data version is logged for reproducibility."""

    @patch("src.training.tracking.wandb")
    @patch("src.training.tracking.mlflow")
    def test_training_logs_data_version(
        self,
        mock_mlflow: MagicMock,
        mock_wandb: MagicMock,
        experiment_config: ExperimentConfig,
        model_config: PatchTSTConfig,
        mock_thermal_normal: ThermalCallback,
        tmp_path: Path,
    ) -> None:
        """Test that data MD5 hash is logged to trackers."""
        # Arrange
        tracking_config = TrackingConfig(
            wandb_project="test-project",
            mlflow_experiment="test-experiment",
        )
        tracking_manager = TrackingManager(tracking_config)

        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=8,
            learning_rate=0.001,
            epochs=1,
            device="cpu",
            checkpoint_dir=tmp_path,
            thermal_callback=mock_thermal_normal,
            tracking_manager=tracking_manager,
        )

        # Act
        trainer.train()

        # Assert - config should include data_md5
        mock_wandb.config.update.assert_called()
        config_calls = mock_wandb.config.update.call_args_list
        logged_config = {}
        for call in config_calls:
            if call.args:
                logged_config.update(call.args[0])
        assert "data_md5" in logged_config or "data_path" in logged_config


class TestReproducibleBatch:
    """Test that training is reproducible with fixed seed."""

    def test_reproducible_batch_with_fixed_seed(
        self,
        experiment_config: ExperimentConfig,
        model_config: PatchTSTConfig,
        mock_thermal_normal: ThermalCallback,
        tmp_path: Path,
    ) -> None:
        """Test that same seed produces identical first batch tensors."""
        # Arrange - create two trainers with same config
        trainer1 = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=8,
            learning_rate=0.001,
            epochs=1,
            device="cpu",
            checkpoint_dir=tmp_path / "run1",
            thermal_callback=mock_thermal_normal,
            tracking_manager=None,
        )

        trainer2 = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=8,
            learning_rate=0.001,
            epochs=1,
            device="cpu",
            checkpoint_dir=tmp_path / "run2",
            thermal_callback=mock_thermal_normal,
            tracking_manager=None,
        )

        # Act - get first batch from each trainer's dataloader
        batch1_x, batch1_y = trainer1.get_first_batch()
        batch2_x, batch2_y = trainer2.get_first_batch()

        # Assert - batches should be identical
        assert torch.allclose(batch1_x, batch2_x), "Input batches differ"
        assert torch.allclose(batch1_y, batch2_y), "Target batches differ"
