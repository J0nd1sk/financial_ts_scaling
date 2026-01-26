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
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.config.experiment import ExperimentConfig
from src.data.dataset import FinancialDataset, SplitIndices, normalize_dataframe
from src.models.patchtst import PatchTST, PatchTSTConfig
from src.training.thermal import ThermalCallback
from src.training.tracking import TrackingManager

# Columns that are not features (only Date - OHLCV are valid features)
NON_FEATURE_COLUMNS = {"Date"}


# Task name to threshold mapping
TASK_THRESHOLDS = {
    "direction": 0.0,  # Any positive return
    "threshold_0.5pct": 0.005,
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
        accumulation_steps: int = 1,
        early_stopping_patience: int | None = None,
        early_stopping_min_delta: float = 0.001,
        early_stopping_metric: str = "val_loss",
        criterion: nn.Module | None = None,
        norm_params: dict[str, tuple[float, float]] | None = None,
        use_revin: bool = False,
        high_prices: np.ndarray | None = None,
        weight_decay: float = 0.0,
    ) -> None:
        """Initialize the trainer.

        Args:
            experiment_config: Experiment configuration (task, data, etc.)
            model_config: PatchTST model configuration
            batch_size: Training batch size (micro-batch for gradient accumulation)
            learning_rate: Optimizer learning rate
            epochs: Number of training epochs
            device: Device to train on ("cpu", "mps", "cuda")
            checkpoint_dir: Directory to save checkpoints
            thermal_callback: Optional thermal monitoring callback
            tracking_manager: Optional experiment tracking manager
            split_indices: Optional SplitIndices for train/val/test splits.
                If provided, creates separate train and val dataloaders.
            accumulation_steps: Number of gradient accumulation steps (default 1).
                Effective batch size = batch_size × accumulation_steps.
            early_stopping_patience: Number of epochs without improvement before
                stopping. None (default) disables early stopping.
            early_stopping_min_delta: Minimum improvement in metric to count
                as an improvement. Default 0.001.
            early_stopping_metric: Metric for early stopping ("val_loss" or "val_auc").
                Default "val_loss". Use "val_auc" when optimizing for ranking (e.g., SoftAUCLoss).
            criterion: Loss function to use. If None (default), uses BCELoss.
                Can pass custom loss like SoftAUCLoss for better calibration.
            norm_params: Optional normalization parameters from compute_normalization_params.
                If provided, applies Z-score normalization to features before training.
                Format: {feature_name: (mean, std), ...}
            use_revin: If True, enables RevIN (Reversible Instance Normalization) in
                the PatchTST model. RevIN normalizes per-instance at input. Default False.
            high_prices: Optional array of high prices for target calculation.
                When provided, threshold targets use max(HIGH) instead of max(CLOSE).
                This is the correct formulation: "Will HIGH reach X% above current CLOSE?"
            weight_decay: L2 regularization weight decay for optimizer. Default 0.0.
        """
        # Validate early_stopping_metric
        valid_metrics = ("val_loss", "val_auc")
        if early_stopping_metric not in valid_metrics:
            raise ValueError(
                f"early_stopping_metric must be one of {valid_metrics}, "
                f"got '{early_stopping_metric}'"
            )
        self.experiment_config = experiment_config
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.thermal_callback = thermal_callback
        self.tracking_manager = tracking_manager
        self.split_indices = split_indices
        self.accumulation_steps = accumulation_steps
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_metric = early_stopping_metric
        self.norm_params = norm_params
        self.use_revin = use_revin
        self.high_prices = high_prices
        self.weight_decay = weight_decay

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
        self.model = PatchTST(self.model_config, use_revin=self.use_revin).to(self.device)

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Loss function (default BCE for binary classification)
        self.criterion = criterion if criterion is not None else nn.BCELoss()

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

        # Apply normalization if norm_params provided
        if self.norm_params is not None:
            df = normalize_dataframe(df, self.norm_params)

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
            high_prices=self.high_prices,
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

        # Apply normalization if norm_params provided
        if self.norm_params is not None:
            df = normalize_dataframe(df, self.norm_params)

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
            high_prices=self.high_prices,
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

    def _check_early_stopping(
        self, val_loss: float, val_auc: float | None, epoch: int
    ) -> bool:
        """Check if training should stop early and save best checkpoint.

        Uses self.early_stopping_metric to determine which metric to monitor:
        - "val_loss": lower is better (default)
        - "val_auc": higher is better

        When the metric improves, saves a checkpoint (best_checkpoint.pt).

        Args:
            val_loss: Current epoch's validation loss.
            val_auc: Current epoch's validation AUC (None if single class).
            epoch: Current epoch number.

        Returns:
            True if training should stop, False otherwise.
        """
        # Determine which metric to use and whether improvement is up or down
        if self.early_stopping_metric == "val_auc":
            # AUC: higher is better
            if val_auc is None:
                # Fallback to loss if AUC undefined (single class in val set)
                current_metric = -val_loss  # Negate so "higher is better" logic works
            else:
                current_metric = val_auc
            # Check if current > best + delta (improvement = increase)
            improved = current_metric > self._best_metric + self.early_stopping_min_delta
        else:
            # Loss: lower is better
            current_metric = val_loss
            # Check if current < best - delta (improvement = decrease)
            improved = current_metric < self._best_metric - self.early_stopping_min_delta

        if improved:
            # Meaningful improvement - save best checkpoint and reset counter
            self._best_metric = current_metric
            self._best_epoch = epoch
            self._save_best_checkpoint(val_loss, epoch)
            self._epochs_without_improvement = 0
            return False

        # No meaningful improvement
        if self.early_stopping_patience is not None:
            self._epochs_without_improvement += 1
            return self._epochs_without_improvement >= self.early_stopping_patience

        return False

    def train(self, verbose: bool = False) -> dict[str, Any]:
        """Run training loop.

        Args:
            verbose: If True, capture per-epoch learning curves.

        Returns:
            Dictionary with training results:
            - train_loss: Final training loss
            - val_loss: Final validation loss (None if no splits provided)
            - stopped_early: Whether training stopped early
            - stop_reason: Reason for early stop (if applicable)
            - learning_curve: List of (epoch, train_loss, val_loss) if verbose
            - final_metrics: Detailed metrics with confusion matrix if verbose
            - split_stats: Data split statistics if splits provided
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
        val_auc: float | None = None
        learning_curve: list[dict[str, Any]] = []

        # Initialize early stopping and best checkpoint state
        # For val_auc (higher-is-better), start at -inf; for val_loss, start at +inf
        if self.early_stopping_metric == "val_auc":
            self._best_metric = float("-inf")
        else:
            self._best_metric = float("inf")
        self._best_epoch = -1
        self._epochs_without_improvement = 0

        try:
            for epoch in range(self.epochs):
                epoch_loss = self._train_epoch(epoch)

                # Log train epoch metrics
                if self.tracking_manager:
                    self.tracking_manager.log_metric(
                        "train_loss", epoch_loss, step=epoch
                    )

                # Compute validation metrics if splits are provided
                if self.val_dataloader is not None:
                    val_metrics = self._evaluate_val()
                    val_loss = val_metrics["loss"]
                    val_auc = val_metrics["auc"]

                    if self.tracking_manager:
                        self.tracking_manager.log_metric(
                            "val_loss", val_loss, step=epoch
                        )
                        if val_auc is not None:
                            self.tracking_manager.log_metric(
                                "val_auc", val_auc, step=epoch
                            )

                    # Check early stopping and save best checkpoint (only when validation set exists)
                    if self._check_early_stopping(val_loss, val_auc, epoch):
                        stopped_early = True
                        stop_reason = "early_stopping"
                        break

                # Capture learning curve if verbose
                if verbose:
                    learning_curve.append({
                        "epoch": epoch,
                        "train_loss": epoch_loss,
                        "val_loss": val_loss,
                        "val_auc": val_auc,
                    })

                # Check thermal status
                if self.thermal_callback:
                    status = self.thermal_callback.check()
                    if status.should_pause:
                        stopped_early = True
                        stop_reason = "thermal"
                        break

        finally:
            # Save final checkpoint only if no validation set (legacy behavior)
            # When validation exists, best checkpoint is saved during training
            if self.val_dataloader is None:
                self._save_checkpoint(self.epochs if not stopped_early else epoch)

            if self.tracking_manager:
                self.tracking_manager.finish()

        result: dict[str, Any] = {
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
        }

        # Add verbose outputs
        if verbose:
            result["learning_curve"] = learning_curve

            # Compute detailed final metrics with confusion matrix
            train_metrics = self._evaluate_detailed(self.train_dataloader)
            result["train_accuracy"] = train_metrics.get("accuracy")
            result["train_recall"] = train_metrics.get("recall")
            result["train_precision"] = train_metrics.get("precision")
            result["train_pred_range"] = train_metrics.get("pred_range")
            result["train_confusion"] = train_metrics.get("confusion_matrix")

            if self.val_dataloader is not None:
                val_metrics = self._evaluate_detailed(self.val_dataloader)
                result["val_accuracy"] = val_metrics.get("accuracy")
                result["val_recall"] = val_metrics.get("recall")
                result["val_precision"] = val_metrics.get("precision")
                result["val_pred_range"] = val_metrics.get("pred_range")
                result["val_confusion"] = val_metrics.get("confusion_matrix")

            # Add split statistics
            if self.split_indices is not None:
                result["split_stats"] = {
                    "n_train": len(self.split_indices.train_indices),
                    "n_val": len(self.split_indices.val_indices),
                    "n_test": len(self.split_indices.test_indices),
                }

        return result

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch with gradient accumulation.

        Args:
            epoch: Current epoch number.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Zero gradients once at epoch start
        self.optimizer.zero_grad()

        for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader):
            # Move to device
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)

            # Scale loss by accumulation steps for proper averaging
            scaled_loss = loss / self.accumulation_steps
            scaled_loss.backward()

            # Step optimizer every accumulation_steps batches
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Track unscaled loss for reporting
            total_loss += loss.item()
            num_batches += 1

            # Check thermal status each batch (for responsive stopping)
            if self.thermal_callback:
                status = self.thermal_callback.check()
                if status.should_pause:
                    break

        # Handle leftover batches (if total not divisible by accumulation_steps)
        if num_batches % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss / max(num_batches, 1)

    def _evaluate_val(self) -> dict[str, float | None]:
        """Evaluate model on validation set.

        Returns:
            Dict with 'loss' (float) and 'auc' (float or None if single class).
        """
        metrics = self._evaluate_detailed(self.val_dataloader)
        return {"loss": metrics["loss"], "auc": metrics.get("auc")}

    def _evaluate_detailed(
        self,
        dataloader: DataLoader | None,
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Evaluate model with detailed metrics.

        Args:
            dataloader: DataLoader to evaluate on.
            threshold: Classification threshold for binary predictions.

        Returns:
            Dictionary with loss, accuracy, and confusion matrix components.
        """
        if dataloader is None:
            return {"loss": 0.0, "accuracy": 0.0}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                # Move to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)

                total_loss += loss.item()
                num_batches += 1

                # Collect predictions for confusion matrix
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        # Switch back to training mode
        self.model.train()

        # Compute metrics
        avg_loss = total_loss / max(num_batches, 1)

        if all_preds:
            preds = np.concatenate(all_preds).flatten()
            targets = np.concatenate(all_targets).flatten()

            # Binary predictions
            pred_binary = (preds >= threshold).astype(int)
            target_binary = targets.astype(int)

            # Confusion matrix components
            tp = int(np.sum((pred_binary == 1) & (target_binary == 1)))
            tn = int(np.sum((pred_binary == 0) & (target_binary == 0)))
            fp = int(np.sum((pred_binary == 1) & (target_binary == 0)))
            fn = int(np.sum((pred_binary == 0) & (target_binary == 1)))

            accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

            # Compute recall (sensitivity) and precision
            # Recall = TP / (TP + FN) - what fraction of actual positives were detected
            # Precision = TP / (TP + FP) - what fraction of positive predictions were correct
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Compute AUC-ROC (requires both classes present)
            auc: float | None = None
            unique_classes = np.unique(target_binary)
            if len(unique_classes) == 2:
                try:
                    auc = float(roc_auc_score(target_binary, preds))
                except ValueError:
                    # Edge case: sklearn may still fail
                    auc = None

            # Prediction range for detecting probability collapse
            pred_min = float(np.min(preds))
            pred_max = float(np.max(preds))

            return {
                "loss": avg_loss,
                "accuracy": accuracy,
                "auc": auc,
                "recall": recall,
                "precision": precision,
                "pred_range": (pred_min, pred_max),
                "confusion_matrix": {
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                },
                "n_samples": len(preds),
            }

        return {"loss": avg_loss, "accuracy": 0.0, "auc": None}

    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint (legacy - used when no validation set).

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
                "norm_params": self.norm_params,
            },
            checkpoint_path,
        )

    def _save_best_checkpoint(self, val_loss: float, epoch: int) -> None:
        """Save best model checkpoint when val_loss improves.

        Saves to best_checkpoint.pt, overwriting any previous best.

        Args:
            val_loss: Validation loss at this checkpoint.
            epoch: Epoch number when this checkpoint was saved.
        """
        checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"
        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "experiment_config": {
                    "seed": self.experiment_config.seed,
                    "data_path": self.experiment_config.data_path,
                    "task": self.experiment_config.task,
                    "timescale": self.experiment_config.timescale,
                },
                "data_md5": self.data_md5,
                "norm_params": self.norm_params,
            },
            checkpoint_path,
        )
