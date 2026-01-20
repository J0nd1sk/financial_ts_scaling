"""Tests for custom loss functions.

Tests SoftAUCLoss which optimizes ranking directly to avoid prior collapse.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.training.losses import SoftAUCLoss


class TestSoftAUCLossBasic:
    """Basic functionality tests for SoftAUCLoss."""

    def test_soft_auc_balanced_data_returns_moderate_loss(self):
        """Test SoftAUCLoss returns value in [0.3, 0.7] for balanced random data."""
        torch.manual_seed(42)
        loss_fn = SoftAUCLoss(gamma=2.0)

        # 50/50 positive/negative, random predictions
        predictions = torch.rand(100)
        targets = torch.cat([torch.ones(50), torch.zeros(50)])

        loss = loss_fn(predictions, targets)

        assert 0.3 <= loss.item() <= 0.7, f"Expected moderate loss, got {loss.item()}"

    def test_soft_auc_perfect_separation_near_zero(self):
        """Test loss approaches 0 when all positive predictions > all negative."""
        loss_fn = SoftAUCLoss(gamma=2.0)

        # Positives get high scores, negatives get low scores
        predictions = torch.tensor([0.9, 0.8, 0.85, 0.2, 0.1, 0.15])
        targets = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)

        # With gamma=2.0, sigmoid(2*(0.15-0.8)) ≈ 0.21, so expect < 0.25
        assert loss.item() < 0.25, f"Expected loss < 0.25 for good separation, got {loss.item()}"

    def test_soft_auc_reversed_separation_near_one(self):
        """Test loss approaches 1 when all negative predictions > all positive."""
        loss_fn = SoftAUCLoss(gamma=2.0)

        # Negatives get high scores (wrong!), positives get low scores
        predictions = torch.tensor([0.1, 0.2, 0.15, 0.9, 0.8, 0.85])
        targets = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)

        # With gamma=2.0, sigmoid(2*(0.8-0.15)) ≈ 0.79, so expect > 0.75
        assert loss.item() > 0.75, f"Expected loss > 0.75 for reversed separation, got {loss.item()}"


class TestSoftAUCLossEdgeCases:
    """Edge case tests for SoftAUCLoss."""

    def test_soft_auc_single_positive_computes(self):
        """Test loss computes correctly with just 1 positive sample."""
        loss_fn = SoftAUCLoss(gamma=2.0)

        predictions = torch.tensor([0.9, 0.1, 0.2, 0.3])
        targets = torch.tensor([1.0, 0.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert 0.0 <= loss.item() <= 1.0, f"Loss should be in [0, 1], got {loss.item()}"

    def test_soft_auc_single_negative_computes(self):
        """Test loss computes correctly with just 1 negative sample."""
        loss_fn = SoftAUCLoss(gamma=2.0)

        predictions = torch.tensor([0.9, 0.8, 0.7, 0.1])
        targets = torch.tensor([1.0, 1.0, 1.0, 0.0])

        loss = loss_fn(predictions, targets)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert 0.0 <= loss.item() <= 1.0, f"Loss should be in [0, 1], got {loss.item()}"

    def test_soft_auc_no_positive_returns_half(self):
        """Test loss returns 0.5 when no positive samples (degenerate case)."""
        loss_fn = SoftAUCLoss(gamma=2.0)

        predictions = torch.tensor([0.5, 0.6, 0.7])
        targets = torch.tensor([0.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)

        assert loss.item() == pytest.approx(0.5, abs=0.01), f"Expected 0.5 for no positives, got {loss.item()}"

    def test_soft_auc_no_negative_returns_half(self):
        """Test loss returns 0.5 when no negative samples (degenerate case)."""
        loss_fn = SoftAUCLoss(gamma=2.0)

        predictions = torch.tensor([0.5, 0.6, 0.7])
        targets = torch.tensor([1.0, 1.0, 1.0])

        loss = loss_fn(predictions, targets)

        assert loss.item() == pytest.approx(0.5, abs=0.01), f"Expected 0.5 for no negatives, got {loss.item()}"


class TestSoftAUCLossGradient:
    """Gradient flow tests for SoftAUCLoss."""

    def test_soft_auc_gradient_flows(self):
        """Test gradients propagate through SoftAUCLoss."""
        loss_fn = SoftAUCLoss(gamma=2.0)

        predictions = torch.tensor([0.7, 0.6, 0.3, 0.2], requires_grad=True)
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None, "Gradients should flow to predictions"
        assert not torch.all(predictions.grad == 0), "Gradients should be non-zero"


class TestSoftAUCLossGamma:
    """Tests for gamma parameter behavior."""

    def test_soft_auc_higher_gamma_sharper_gradient(self):
        """Test that higher gamma produces larger gradients (sharper sigmoid)."""
        predictions = torch.tensor([0.6, 0.4], requires_grad=True)
        targets = torch.tensor([1.0, 0.0])

        # Low gamma
        loss_fn_low = SoftAUCLoss(gamma=1.0)
        loss_low = loss_fn_low(predictions, targets)
        loss_low.backward()
        grad_low = predictions.grad.clone()

        # Reset gradients
        predictions.grad.zero_()

        # High gamma
        loss_fn_high = SoftAUCLoss(gamma=5.0)
        loss_high = loss_fn_high(predictions, targets)
        loss_high.backward()
        grad_high = predictions.grad.clone()

        # Higher gamma should produce larger magnitude gradients
        assert torch.norm(grad_high) > torch.norm(grad_low), \
            "Higher gamma should produce larger gradients"


# --- Fixtures for Trainer tests ---


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
def experiment_config(micro_dataset_path: Path):
    """Create an ExperimentConfig for testing."""
    from src.config.experiment import ExperimentConfig

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


class TestTrainerWithCustomCriterion:
    """Tests for Trainer accepting custom criterion parameter."""

    def test_trainer_accepts_criterion_parameter(
        self, experiment_config, tmp_path
    ):
        """Test Trainer can be initialized with custom criterion."""
        from src.models.patchtst import PatchTSTConfig
        from src.training.trainer import Trainer

        model_config = PatchTSTConfig(
            num_features=25,
            context_length=10,
            patch_length=4,
            stride=2,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            dropout=0.1,
            head_dropout=0.0,
        )

        custom_criterion = SoftAUCLoss(gamma=2.0)

        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=16,
            learning_rate=0.001,
            epochs=1,
            device="cpu",
            checkpoint_dir=tmp_path,
            criterion=custom_criterion,
        )

        assert trainer.criterion is custom_criterion

    def test_trainer_defaults_to_bce_when_no_criterion(
        self, experiment_config, tmp_path
    ):
        """Test Trainer uses BCELoss when criterion=None."""
        from src.models.patchtst import PatchTSTConfig
        from src.training.trainer import Trainer

        model_config = PatchTSTConfig(
            num_features=25,
            context_length=10,
            patch_length=4,
            stride=2,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            dropout=0.1,
            head_dropout=0.0,
        )

        trainer = Trainer(
            experiment_config=experiment_config,
            model_config=model_config,
            batch_size=16,
            learning_rate=0.001,
            epochs=1,
            device="cpu",
            checkpoint_dir=tmp_path,
            # No criterion parameter
        )

        assert isinstance(trainer.criterion, nn.BCELoss)
