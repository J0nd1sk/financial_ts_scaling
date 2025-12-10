"""Experiment tracking integration for W&B and MLflow."""

from dataclasses import dataclass

import mlflow
import wandb


@dataclass
class TrackingConfig:
    """Configuration for experiment tracking.

    Attributes:
        wandb_project: W&B project name. If None, W&B tracking is disabled.
        wandb_run_name: Optional name for the W&B run.
        mlflow_experiment: MLflow experiment name. If None, MLflow tracking is disabled.
        mlflow_run_name: Optional name for the MLflow run.
    """

    wandb_project: str | None = None
    wandb_run_name: str | None = None
    mlflow_experiment: str | None = None
    mlflow_run_name: str | None = None


class TrackingManager:
    """Unified interface for W&B and MLflow experiment tracking.

    Provides a single API for logging metrics, configs, and managing
    experiment runs across both tracking platforms. Either or both
    trackers can be disabled by setting their config fields to None.

    Example:
        config = TrackingConfig(
            wandb_project="my-project",
            mlflow_experiment="my-experiment",
        )
        manager = TrackingManager(config)
        manager.start()
        manager.log_config({"learning_rate": 0.001})
        manager.log_metric("loss", 0.5, step=1)
        manager.finish()
    """

    def __init__(self, config: TrackingConfig) -> None:
        """Initialize the tracking manager.

        Args:
            config: Tracking configuration specifying which trackers to use.
        """
        self._config = config
        self._wandb_enabled = config.wandb_project is not None
        self._mlflow_enabled = config.mlflow_experiment is not None

    def start(self) -> None:
        """Start tracking runs for enabled trackers."""
        if self._wandb_enabled:
            wandb.init(
                project=self._config.wandb_project,
                name=self._config.wandb_run_name,
            )

        if self._mlflow_enabled:
            mlflow.set_experiment(self._config.mlflow_experiment)
            mlflow.start_run(run_name=self._config.mlflow_run_name)

    def log_metric(
        self, name: str, value: float, step: int | None = None
    ) -> None:
        """Log a single metric to enabled trackers.

        Args:
            name: Metric name.
            value: Metric value.
            step: Optional step/epoch number.
        """
        if self._wandb_enabled:
            wandb.log({name: value}, step=step)

        if self._mlflow_enabled:
            mlflow.log_metric(name, value, step=step)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log multiple metrics to enabled trackers.

        Args:
            metrics: Dictionary of metric names to values.
            step: Optional step/epoch number.
        """
        if self._wandb_enabled:
            wandb.log(metrics, step=step)

        if self._mlflow_enabled:
            for name, value in metrics.items():
                mlflow.log_metric(name, value, step=step)

    def log_config(self, config: dict) -> None:
        """Log experiment configuration to enabled trackers.

        Args:
            config: Dictionary of configuration parameters.
        """
        if self._wandb_enabled:
            wandb.config.update(config)

        if self._mlflow_enabled:
            mlflow.log_params(config)

    def finish(self) -> None:
        """Close tracking runs for enabled trackers."""
        if self._wandb_enabled:
            wandb.finish()

        if self._mlflow_enabled:
            mlflow.end_run()
