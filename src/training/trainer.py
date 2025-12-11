"""Training loop for financial time-series models.

Integrates PatchTST model with FinancialDataset, thermal monitoring,
and experiment tracking (W&B + MLflow).
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config.experiment import ExperimentConfig
from src.data.dataset import FinancialDataset, SplitIndices
from src.models.patchtst import PatchTST, PatchTSTConfig
from src.training.thermal import ThermalCallback
from src.training.tracking import TrackingManager

# Columns that are not features (only Date - OHLCV are valid features)
NON_FEATURE_COLUMNS = {"Date"}


# Task name to threshold mapping
TASK_THRESHOLDS = {
    "direction": 0.0,  # Any positive return
    "threshold_1pct": 0.01,
    "threshold_2pct": 0.02,
    "threshold_3pct": 0.03,
    "threshold_5pct": 0.05,
    "regression": None,  # Not used for regression
}


def _compute_file_md5(file_path: Path) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class Trainer:
    """Trainer for PatchTST models on financial time-series data.

    Integrates:
    - PatchTST model
    - FinancialDataset with binary threshold targets
    - ThermalCallback for temperature monitoring
    - TrackingManager for W&B + MLflow logging

    Example:
        >>> trainer = Trainer(
        ...     experiment_config=exp_config,
        ...     model_config=model_config,
        ...     batch_size=32,
        ...     learning_rate=0.001,
        ...     epochs=10,
        ...     device="mps",
        ...     checkpoint_dir=Path("checkpoints"),
        ... )
        >>> result = trainer.train()
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        model_config: PatchTSTConfig,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        device: str,
        checkpoint_dir: Path,
        thermal_callback: ThermalCallback | None = None,
        tracking_manager: TrackingManager | None = None,
        split_indices: SplitIndices | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            experiment_config: Experiment configuration (task, data, etc.)
            model_config: PatchTST model configuration
            batch_size: Training batch size
            learning_rate: Optimizer learning rate
            epochs: Number of training epochs
            device: Device to train on ("cpu", "mps", "cuda")
            checkpoint_dir: Directory to save checkpoints
            thermal_callback: Optional thermal monitoring callback
            tracking_manager: Optional experiment tracking manager
            split_indices: Optional SplitIndices for train/val/test splits.
                If provided, creates separate train and val dataloaders.
        """
        self.experiment_config = experiment_config
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.thermal_callback = thermal_callback
        self.tracking_manager = tracking_manager
        self.split_indices = split_indices

        # Set random seeds for reproducibility
        self._set_seeds(experiment_config.seed)

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Verify data exists
        self._verify_data()

        # Compute data MD5 for reproducibility tracking
        self.data_md5 = _compute_file_md5(Path(experiment_config.data_path))

        # Detect actual feature count and update model config if needed
        self.model_config = self._adjust_model_config(model_config)

        # Create model
        self.model = PatchTST(self.model_config).to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )

        # Loss function (BCE for binary classification)
        self.criterion = nn.BCELoss()

        # Create dataloaders (with seeded generator for reproducibility)
        if split_indices is not None:
            self.train_dataloader, self.val_dataloader = self._create_split_dataloaders()
            # Backward compatibility: dataloader points to train_dataloader
            self.dataloader = self.train_dataloader
        else:
            self.dataloader = self._create_dataloader()
            self.train_dataloader = self.dataloader
            self.val_dataloader = None

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    def _verify_data(self) -> None:
        """Verify that data file exists before training.

        Raises:
            FileNotFoundError: If data file doesn't exist.
        """
        data_path = Path(self.experiment_config.data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

    def _adjust_model_config(self, model_config: PatchTSTConfig) -> PatchTSTConfig:
        """Adjust model config to match actual data feature count.

        Detects the number of feature columns in the data and updates
        model_config.num_features if it doesn't match.

        Args:
            model_config: Original model configuration.

        Returns:
            Updated model configuration with correct num_features.
        """
        data_path = Path(self.experiment_config.data_path)
        df = pd.read_parquet(data_path)

        # Count actual features (exclude non-feature columns)
        feature_columns = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
        actual_num_features = len(feature_columns)

        if model_config.num_features != actual_num_features:
            # Create new config with corrected num_features
            return replace(model_config, num_features=actual_num_features)

        return model_config

    def _create_dataloader(self) -> DataLoader:
        """Create DataLoader from experiment config.

        Returns:
            DataLoader with FinancialDataset.
        """
        # Load data
        data_path = Path(self.experiment_config.data_path)
        df = pd.read_parquet(data_path)

        # Extract close prices
        close_prices = df["Close"].values

        # Get threshold for task
        threshold = TASK_THRESHOLDS.get(self.experiment_config.task)
        if threshold is None and self.experiment_config.task != "regression":
            raise ValueError(f"Unknown task: {self.experiment_config.task}")

        # For regression, use threshold=0.0 (labels won't be used as-is)
        if threshold is None:
            threshold = 0.0

        # Create dataset
        dataset = FinancialDataset(
            features_df=df,
            close_prices=close_prices,
            context_length=self.experiment_config.context_length,
            horizon=self.experiment_config.horizon,
            threshold=threshold,
        )

        # Create dataloader with seeded generator
        generator = torch.Generator()
        generator.manual_seed(self.experiment_config.seed)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )

        return dataloader

    def _create_split_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Create train and validation DataLoaders using split indices.

        Uses SplitIndices to create separate dataloaders for train and val.
        Train samples are specified by train_indices (sliding window).
        Val samples are specified by val_indices (non-overlapping chunks).

        Returns:
            Tuple of (train_dataloader, val_dataloader).
        """
        assert self.split_indices is not None, "split_indices must be set"

        # Load data
        data_path = Path(self.experiment_config.data_path)
        df = pd.read_parquet(data_path)

        # Extract close prices
        close_prices = df["Close"].values

        # Get threshold for task
        threshold = TASK_THRESHOLDS.get(self.experiment_config.task)
        if threshold is None and self.experiment_config.task != "regression":
            raise ValueError(f"Unknown task: {self.experiment_config.task}")

        # For regression, use threshold=0.0 (labels won't be used as-is)
        if threshold is None:
            threshold = 0.0

        # Create full dataset
        full_dataset = FinancialDataset(
            features_df=df,
            close_prices=close_prices,
            context_length=self.experiment_config.context_length,
            horizon=self.experiment_config.horizon,
            threshold=threshold,
        )

        # Create train subset using train_indices
        train_subset = Subset(full_dataset, self.split_indices.train_indices.tolist())

        # Create val subset using val_indices
        val_subset = Subset(full_dataset, self.split_indices.val_indices.tolist())

        # Create dataloaders with seeded generator
        generator = torch.Generator()
        generator.manual_seed(self.experiment_config.seed)

        train_dataloader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )

        # Val dataloader: no shuffle needed
        val_dataloader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        return train_dataloader, val_dataloader

    def get_first_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the first batch from the dataloader.

        Used for reproducibility testing.

        Returns:
            Tuple of (input_tensor, target_tensor) for first batch.
        """
        # Reset dataloader with fresh generator
        self._set_seeds(self.experiment_config.seed)
        self.dataloader = self._create_dataloader()

        for batch_x, batch_y in self.dataloader:
            return batch_x, batch_y

        raise RuntimeError("Dataloader is empty")

    def train(self) -> dict[str, Any]:
        """Run training loop.

        Returns:
            Dictionary with training results:
            - train_loss: Final training loss
            - val_loss: Final validation loss (None if no splits provided)
            - stopped_early: Whether training stopped early
            - stop_reason: Reason for early stop (if applicable)
        """
        # Log config to trackers
        if self.tracking_manager:
            self.tracking_manager.start()
            self.tracking_manager.log_config({
                "seed": self.experiment_config.seed,
                "data_path": self.experiment_config.data_path,
                "data_md5": self.data_md5,
                "task": self.experiment_config.task,
                "timescale": self.experiment_config.timescale,
                "context_length": self.experiment_config.context_length,
                "horizon": self.experiment_config.horizon,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
            })

        stopped_early = False
        stop_reason = None
        epoch_loss = 0.0
        val_loss: float | None = None

        try:
            for epoch in range(self.epochs):
                epoch_loss = self._train_epoch(epoch)

                # Log train epoch metrics
                if self.tracking_manager:
                    self.tracking_manager.log_metric(
                        "train_loss", epoch_loss, step=epoch
                    )

                # Compute validation loss if splits are provided
                if self.val_dataloader is not None:
                    val_loss = self._evaluate_val()
                    if self.tracking_manager:
                        self.tracking_manager.log_metric(
                            "val_loss", val_loss, step=epoch
                        )

                # Check thermal status
                if self.thermal_callback:
                    status = self.thermal_callback.check()
                    if status.should_pause:
                        stopped_early = True
                        stop_reason = "thermal"
                        break

        finally:
            # Always save checkpoint and finish tracking
            self._save_checkpoint(self.epochs if not stopped_early else epoch)

            if self.tracking_manager:
                self.tracking_manager.finish()

        return {
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
        }

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in self.dataloader:
            # Move to device
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Check thermal status each batch (for responsive stopping)
            if self.thermal_callback:
                status = self.thermal_callback.check()
                if status.should_pause:
                    break

        return total_loss / max(num_batches, 1)

    def _evaluate_val(self) -> float:
        """Evaluate model on validation set.

        Returns:
            Average validation loss.
        """
        assert self.val_dataloader is not None, "val_dataloader must be set"

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in self.val_dataloader:
                # Move to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                num_batches += 1

        # Switch back to training mode
        self.model.train()

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "experiment_config": {
                    "seed": self.experiment_config.seed,
                    "data_path": self.experiment_config.data_path,
                    "task": self.experiment_config.task,
                    "timescale": self.experiment_config.timescale,
                },
                "data_md5": self.data_md5,
            },
            checkpoint_path,
        )
