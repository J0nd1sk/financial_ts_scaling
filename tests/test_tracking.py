"""Tests for experiment tracking integration (W&B + MLflow)."""

from unittest.mock import MagicMock, patch

import pytest


class TestTrackingManagerInit:
    """Tests for TrackingManager initialization."""

    def test_tracking_manager_initializes_wandb(self) -> None:
        """Test that W&B run starts when wandb_project is provided."""
        from src.training.tracking import TrackingConfig, TrackingManager

        config = TrackingConfig(
            wandb_project="test-project",
            wandb_run_name="test-run",
        )

        with patch("src.training.tracking.wandb") as mock_wandb:
            manager = TrackingManager(config)
            manager.start()

            mock_wandb.init.assert_called_once()
            call_kwargs = mock_wandb.init.call_args[1]
            assert call_kwargs["project"] == "test-project"
            assert call_kwargs["name"] == "test-run"

    def test_tracking_manager_initializes_mlflow(self) -> None:
        """Test that MLflow run starts when mlflow_experiment is provided."""
        from src.training.tracking import TrackingConfig, TrackingManager

        config = TrackingConfig(
            mlflow_experiment="test-experiment",
            mlflow_run_name="test-run",
        )

        with patch("src.training.tracking.mlflow") as mock_mlflow:
            manager = TrackingManager(config)
            manager.start()

            mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
            mock_mlflow.start_run.assert_called_once()
            call_kwargs = mock_mlflow.start_run.call_args[1]
            assert call_kwargs["run_name"] == "test-run"


class TestTrackingManagerLogMetrics:
    """Tests for metric logging."""

    def test_tracking_logs_metric_to_wandb(self) -> None:
        """Test that log_metric calls wandb.log with correct dict."""
        from src.training.tracking import TrackingConfig, TrackingManager

        config = TrackingConfig(wandb_project="test-project")

        with patch("src.training.tracking.wandb") as mock_wandb:
            manager = TrackingManager(config)
            manager.start()
            manager.log_metric("loss", 0.5, step=10)

            mock_wandb.log.assert_called_with({"loss": 0.5}, step=10)

    def test_tracking_logs_metric_to_mlflow(self) -> None:
        """Test that log_metric calls mlflow.log_metric with name/value."""
        from src.training.tracking import TrackingConfig, TrackingManager

        config = TrackingConfig(mlflow_experiment="test-experiment")

        with patch("src.training.tracking.mlflow") as mock_mlflow:
            manager = TrackingManager(config)
            manager.start()
            manager.log_metric("accuracy", 0.95, step=20)

            mock_mlflow.log_metric.assert_called_with("accuracy", 0.95, step=20)

    def test_tracking_logs_metrics_batch_to_wandb(self) -> None:
        """Test that log_metrics logs multiple metrics to wandb."""
        from src.training.tracking import TrackingConfig, TrackingManager

        config = TrackingConfig(wandb_project="test-project")

        with patch("src.training.tracking.wandb") as mock_wandb:
            manager = TrackingManager(config)
            manager.start()
            manager.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=5)

            mock_wandb.log.assert_called_with({"loss": 0.5, "accuracy": 0.9}, step=5)


class TestTrackingManagerLogConfig:
    """Tests for config logging."""

    def test_tracking_logs_config_to_wandb(self) -> None:
        """Test that log_config calls wandb.config.update."""
        from src.training.tracking import TrackingConfig, TrackingManager

        config = TrackingConfig(wandb_project="test-project")
        experiment_config = {"learning_rate": 0.001, "batch_size": 32}

        with patch("src.training.tracking.wandb") as mock_wandb:
            manager = TrackingManager(config)
            manager.start()
            manager.log_config(experiment_config)

            mock_wandb.config.update.assert_called_once_with(experiment_config)

    def test_tracking_logs_config_to_mlflow(self) -> None:
        """Test that log_config calls mlflow.log_params."""
        from src.training.tracking import TrackingConfig, TrackingManager

        config = TrackingConfig(mlflow_experiment="test-experiment")
        experiment_config = {"learning_rate": 0.001, "batch_size": 32}

        with patch("src.training.tracking.mlflow") as mock_mlflow:
            manager = TrackingManager(config)
            manager.start()
            manager.log_config(experiment_config)

            mock_mlflow.log_params.assert_called_once_with(experiment_config)


class TestTrackingManagerDisabled:
    """Tests for disabled tracker handling."""

    def test_tracking_handles_disabled_wandb(self) -> None:
        """Test that no wandb calls when wandb_project is None."""
        from src.training.tracking import TrackingConfig, TrackingManager

        config = TrackingConfig(
            wandb_project=None,
            mlflow_experiment="test-experiment",
        )

        with patch("src.training.tracking.wandb") as mock_wandb:
            with patch("src.training.tracking.mlflow"):
                manager = TrackingManager(config)
                manager.start()
                manager.log_metric("loss", 0.5)
                manager.log_config({"lr": 0.001})
                manager.finish()

                # wandb should not be called at all
                mock_wandb.init.assert_not_called()
                mock_wandb.log.assert_not_called()
                mock_wandb.config.update.assert_not_called()
                mock_wandb.finish.assert_not_called()

    def test_tracking_handles_disabled_mlflow(self) -> None:
        """Test that no mlflow calls when mlflow_experiment is None."""
        from src.training.tracking import TrackingConfig, TrackingManager

        config = TrackingConfig(
            wandb_project="test-project",
            mlflow_experiment=None,
        )

        with patch("src.training.tracking.wandb"):
            with patch("src.training.tracking.mlflow") as mock_mlflow:
                manager = TrackingManager(config)
                manager.start()
                manager.log_metric("loss", 0.5)
                manager.log_config({"lr": 0.001})
                manager.finish()

                # mlflow should not be called at all
                mock_mlflow.set_experiment.assert_not_called()
                mock_mlflow.start_run.assert_not_called()
                mock_mlflow.log_metric.assert_not_called()
                mock_mlflow.log_params.assert_not_called()
                mock_mlflow.end_run.assert_not_called()

    def test_tracking_handles_both_disabled(self) -> None:
        """Test that no errors when both trackers disabled."""
        from src.training.tracking import TrackingConfig, TrackingManager

        config = TrackingConfig()  # Both None by default

        with patch("src.training.tracking.wandb") as mock_wandb:
            with patch("src.training.tracking.mlflow") as mock_mlflow:
                manager = TrackingManager(config)
                manager.start()
                manager.log_metric("loss", 0.5)
                manager.log_metrics({"a": 1, "b": 2})
                manager.log_config({"lr": 0.001})
                manager.finish()

                # Neither should be called
                mock_wandb.init.assert_not_called()
                mock_mlflow.set_experiment.assert_not_called()


class TestTrackingManagerFinish:
    """Tests for run cleanup."""

    def test_tracking_manager_finish_closes_runs(self) -> None:
        """Test that finish() closes both W&B and MLflow runs."""
        from src.training.tracking import TrackingConfig, TrackingManager

        config = TrackingConfig(
            wandb_project="test-project",
            mlflow_experiment="test-experiment",
        )

        with patch("src.training.tracking.wandb") as mock_wandb:
            with patch("src.training.tracking.mlflow") as mock_mlflow:
                manager = TrackingManager(config)
                manager.start()
                manager.finish()

                mock_wandb.finish.assert_called_once()
                mock_mlflow.end_run.assert_called_once()
