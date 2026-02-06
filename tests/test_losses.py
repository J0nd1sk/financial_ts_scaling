"""Tests for custom loss functions and evaluation utilities.

Tests SoftAUCLoss which optimizes ranking directly to avoid prior collapse.
Also tests evaluation function is_classification parameter.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

# Add project root to path for experiments module
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


# =============================================================================
# FocalLoss Tests
# =============================================================================


class TestFocalLossBasic:
    """Basic functionality tests for FocalLoss."""

    def test_focal_loss_balanced_data_returns_moderate_loss(self):
        """Test FocalLoss returns reasonable value for balanced random data."""
        from src.training.losses import FocalLoss

        torch.manual_seed(42)
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

        # 50/50 positive/negative, random predictions
        predictions = torch.rand(100)
        targets = torch.cat([torch.ones(50), torch.zeros(50)])

        loss = loss_fn(predictions, targets)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() > 0, "Loss should be positive"

    def test_focal_loss_perfect_separation_near_zero(self):
        """Test loss is low when predictions are confident and correct."""
        from src.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

        # High confidence correct predictions
        predictions = torch.tensor([0.95, 0.90, 0.92, 0.05, 0.08, 0.03])
        targets = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)

        # Focal loss should be very small for confident correct predictions
        assert loss.item() < 0.1, f"Expected loss < 0.1 for perfect separation, got {loss.item()}"

    def test_focal_loss_reversed_separation_high_loss(self):
        """Test loss is high when predictions are confidently wrong."""
        from src.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

        # High confidence WRONG predictions
        predictions = torch.tensor([0.05, 0.08, 0.03, 0.95, 0.90, 0.92])
        targets = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)

        # Should be much higher than perfect separation
        assert loss.item() > 0.5, f"Expected loss > 0.5 for wrong predictions, got {loss.item()}"

    def test_focal_loss_focuses_on_hard_examples(self):
        """Test that hard examples contribute more to loss than easy ones."""
        from src.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, alpha=0.5)  # alpha=0.5 for balanced weighting

        # Easy example: high confidence correct
        easy_pred = torch.tensor([0.95])
        easy_target = torch.tensor([1.0])
        easy_loss = loss_fn(easy_pred, easy_target)

        # Hard example: low confidence (near 0.5)
        hard_pred = torch.tensor([0.55])
        hard_target = torch.tensor([1.0])
        hard_loss = loss_fn(hard_pred, hard_target)

        # Hard example should have higher loss
        assert hard_loss.item() > easy_loss.item(), \
            f"Hard example loss ({hard_loss.item()}) should exceed easy ({easy_loss.item()})"


class TestFocalLossGamma:
    """Tests for gamma parameter behavior."""

    def test_focal_loss_gamma_zero_proportional_to_bce(self):
        """Test that gamma=0 removes focal modulation (proportional to BCE)."""
        from src.training.losses import FocalLoss

        torch.manual_seed(42)
        predictions = torch.rand(50)
        targets = torch.cat([torch.ones(25), torch.zeros(25)])

        # Focal loss with gamma=0, alpha=0.5 (balanced class weighting)
        # With alpha=0.5, FL = 0.5 * (-log(p_t)), so FL = BCE * 0.5
        focal_loss_fn = FocalLoss(gamma=0.0, alpha=0.5)
        focal_loss = focal_loss_fn(predictions, targets)

        # Standard BCE
        bce_loss_fn = nn.BCELoss()
        bce_loss = bce_loss_fn(predictions, targets)

        # With alpha=0.5, focal loss should be exactly half of BCE
        expected_ratio = 0.5
        actual_ratio = focal_loss.item() / bce_loss.item()

        assert abs(actual_ratio - expected_ratio) < 0.01, \
            f"gamma=0, alpha=0.5 should give FL/BCE = 0.5, got {actual_ratio:.4f}"


class TestFocalLossEdgeCases:
    """Edge case tests for FocalLoss."""

    def test_focal_loss_no_positive_handles_gracefully(self):
        """Test loss handles batch with no positive samples."""
        from src.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

        predictions = torch.tensor([0.3, 0.4, 0.5])
        targets = torch.tensor([0.0, 0.0, 0.0])  # All negative

        loss = loss_fn(predictions, targets)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_focal_loss_no_negative_handles_gracefully(self):
        """Test loss handles batch with no negative samples."""
        from src.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

        predictions = torch.tensor([0.7, 0.8, 0.9])
        targets = torch.tensor([1.0, 1.0, 1.0])  # All positive

        loss = loss_fn(predictions, targets)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"


class TestFocalLossGradient:
    """Gradient flow tests for FocalLoss."""

    def test_focal_loss_gradient_flows(self):
        """Test gradients propagate through FocalLoss."""
        from src.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

        predictions = torch.tensor([0.7, 0.6, 0.3, 0.2], requires_grad=True)
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None, "Gradients should flow to predictions"
        assert not torch.all(predictions.grad == 0), "Gradients should be non-zero"


# =============================================================================
# LabelSmoothingBCELoss Tests
# =============================================================================


class TestLabelSmoothingBCELossBasic:
    """Basic functionality tests for LabelSmoothingBCELoss."""

    def test_label_smoothing_returns_tensor(self):
        """Test LabelSmoothingBCELoss returns a scalar tensor."""
        from src.training.losses import LabelSmoothingBCELoss

        torch.manual_seed(42)
        loss_fn = LabelSmoothingBCELoss(epsilon=0.1)

        predictions = torch.rand(100)
        targets = torch.cat([torch.ones(50), torch.zeros(50)])

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_label_smoothing_epsilon_zero_equals_bce(self):
        """Test that epsilon=0 produces standard BCE."""
        from src.training.losses import LabelSmoothingBCELoss

        torch.manual_seed(42)
        predictions = torch.rand(50)
        targets = torch.cat([torch.ones(25), torch.zeros(25)])

        # Label smoothing with epsilon=0 (no smoothing)
        smooth_loss_fn = LabelSmoothingBCELoss(epsilon=0.0)
        smooth_loss = smooth_loss_fn(predictions, targets)

        # Standard BCE
        bce_loss_fn = nn.BCELoss()
        bce_loss = bce_loss_fn(predictions, targets)

        assert abs(smooth_loss.item() - bce_loss.item()) < 1e-6, \
            f"epsilon=0 should match BCE: smooth={smooth_loss.item()}, bce={bce_loss.item()}"

    def test_label_smoothing_reduces_confidence_penalty(self):
        """Test that label smoothing reduces penalty for moderate confidence."""
        from src.training.losses import LabelSmoothingBCELoss

        # Confident prediction that matches target
        predictions = torch.tensor([0.99])
        targets = torch.tensor([1.0])

        # Without smoothing: BCE penalizes being "too confident" less
        bce_loss = nn.BCELoss()(predictions, targets)

        # With smoothing: target becomes 0.9, so p=0.99 is slightly overconfident
        smooth_loss = LabelSmoothingBCELoss(epsilon=0.1)(predictions, targets)

        # Smoothed loss should be higher for very confident predictions
        # because the smoothed target is 0.9, not 1.0
        assert smooth_loss.item() > bce_loss.item(), \
            f"Label smoothing should penalize overconfidence: smooth={smooth_loss.item()}, bce={bce_loss.item()}"


class TestLabelSmoothingBCELossEdgeCases:
    """Edge case tests for LabelSmoothingBCELoss."""

    def test_label_smoothing_no_positive_handles_gracefully(self):
        """Test loss handles batch with no positive samples."""
        from src.training.losses import LabelSmoothingBCELoss

        loss_fn = LabelSmoothingBCELoss(epsilon=0.1)

        predictions = torch.tensor([0.3, 0.4, 0.5])
        targets = torch.tensor([0.0, 0.0, 0.0])  # All negative

        loss = loss_fn(predictions, targets)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_label_smoothing_no_negative_handles_gracefully(self):
        """Test loss handles batch with no negative samples."""
        from src.training.losses import LabelSmoothingBCELoss

        loss_fn = LabelSmoothingBCELoss(epsilon=0.1)

        predictions = torch.tensor([0.7, 0.8, 0.9])
        targets = torch.tensor([1.0, 1.0, 1.0])  # All positive

        loss = loss_fn(predictions, targets)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"


class TestLabelSmoothingBCELossGradient:
    """Gradient flow tests for LabelSmoothingBCELoss."""

    def test_label_smoothing_gradient_flows(self):
        """Test gradients propagate through LabelSmoothingBCELoss."""
        from src.training.losses import LabelSmoothingBCELoss

        loss_fn = LabelSmoothingBCELoss(epsilon=0.1)

        predictions = torch.tensor([0.7, 0.6, 0.3, 0.2], requires_grad=True)
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None, "Gradients should flow to predictions"
        assert not torch.all(predictions.grad == 0), "Gradients should be non-zero"


# =============================================================================
# WeightedSumLoss Tests
# =============================================================================


class TestWeightedSumLossBasic:
    """Basic functionality tests for WeightedSumLoss."""

    def test_weighted_sum_loss_forward_computes_weighted_sum(self):
        """Test WeightedSumLoss computes α*BCE + (1-α)*SoftAUC."""
        from src.training.losses import WeightedSumLoss

        torch.manual_seed(42)
        predictions = torch.rand(100)
        targets = torch.cat([torch.ones(50), torch.zeros(50)])

        alpha = 0.6
        loss_fn = WeightedSumLoss(alpha=alpha)

        # Compute weighted sum loss
        weighted_loss = loss_fn(predictions, targets)

        # Compute individual losses manually
        bce_loss = nn.BCELoss()(predictions, targets)
        softauc_loss = SoftAUCLoss(gamma=2.0)(predictions, targets)
        expected = alpha * bce_loss + (1 - alpha) * softauc_loss

        assert abs(weighted_loss.item() - expected.item()) < 1e-5, \
            f"Expected {expected.item():.6f}, got {weighted_loss.item():.6f}"

    def test_weighted_sum_loss_default_alpha_is_half(self):
        """Test WeightedSumLoss defaults to alpha=0.5."""
        from src.training.losses import WeightedSumLoss

        loss_fn = WeightedSumLoss()
        assert loss_fn.alpha == 0.5, f"Default alpha should be 0.5, got {loss_fn.alpha}"

    def test_weighted_sum_loss_alpha_one_equals_bce(self):
        """Test alpha=1.0 produces pure BCE loss."""
        from src.training.losses import WeightedSumLoss

        torch.manual_seed(42)
        predictions = torch.rand(50)
        targets = torch.cat([torch.ones(25), torch.zeros(25)])

        weighted_loss = WeightedSumLoss(alpha=1.0)(predictions, targets)
        bce_loss = nn.BCELoss()(predictions, targets)

        assert abs(weighted_loss.item() - bce_loss.item()) < 1e-6, \
            f"alpha=1.0 should equal BCE: weighted={weighted_loss.item()}, bce={bce_loss.item()}"

    def test_weighted_sum_loss_alpha_zero_equals_softauc(self):
        """Test alpha=0.0 produces pure SoftAUC loss."""
        from src.training.losses import WeightedSumLoss

        torch.manual_seed(42)
        predictions = torch.rand(50)
        targets = torch.cat([torch.ones(25), torch.zeros(25)])

        weighted_loss = WeightedSumLoss(alpha=0.0)(predictions, targets)
        softauc_loss = SoftAUCLoss(gamma=2.0)(predictions, targets)

        assert abs(weighted_loss.item() - softauc_loss.item()) < 1e-6, \
            f"alpha=0.0 should equal SoftAUC: weighted={weighted_loss.item()}, softauc={softauc_loss.item()}"


class TestWeightedSumLossValidation:
    """Validation tests for WeightedSumLoss."""

    def test_weighted_sum_loss_invalid_alpha_below_zero_raises(self):
        """Test alpha < 0 raises ValueError."""
        from src.training.losses import WeightedSumLoss

        with pytest.raises(ValueError, match="alpha must be in"):
            WeightedSumLoss(alpha=-0.1)

    def test_weighted_sum_loss_invalid_alpha_above_one_raises(self):
        """Test alpha > 1 raises ValueError."""
        from src.training.losses import WeightedSumLoss

        with pytest.raises(ValueError, match="alpha must be in"):
            WeightedSumLoss(alpha=1.1)


class TestWeightedSumLossGradient:
    """Gradient flow tests for WeightedSumLoss."""

    def test_weighted_sum_loss_gradient_flows(self):
        """Test gradients propagate through WeightedSumLoss."""
        from src.training.losses import WeightedSumLoss

        loss_fn = WeightedSumLoss(alpha=0.5)

        predictions = torch.tensor([0.7, 0.6, 0.3, 0.2], requires_grad=True)
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None, "Gradients should flow to predictions"
        assert not torch.all(predictions.grad == 0), "Gradients should be non-zero"


# =============================================================================
# WeightedBCELoss Tests
# =============================================================================


class TestWeightedBCELossBasic:
    """Basic functionality tests for WeightedBCELoss."""

    def test_weighted_bce_returns_scalar_tensor(self):
        """Test WeightedBCELoss returns a scalar tensor."""
        from src.training.losses import WeightedBCELoss

        torch.manual_seed(42)
        loss_fn = WeightedBCELoss(pos_weight=4.0)

        predictions = torch.rand(100)
        targets = torch.cat([torch.ones(25), torch.zeros(75)])  # 25% positive

        loss = loss_fn(predictions, targets)

        assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() > 0, "Loss should be positive"

    def test_weighted_bce_pos_weight_one_equals_standard_bce(self):
        """Test that pos_weight=1.0 produces standard BCE."""
        from src.training.losses import WeightedBCELoss

        torch.manual_seed(42)
        predictions = torch.rand(50)
        targets = torch.cat([torch.ones(25), torch.zeros(25)])

        # WeightedBCE with pos_weight=1.0 (no weighting)
        weighted_loss_fn = WeightedBCELoss(pos_weight=1.0)
        weighted_loss = weighted_loss_fn(predictions, targets)

        # Standard BCE
        bce_loss_fn = nn.BCELoss()
        bce_loss = bce_loss_fn(predictions, targets)

        assert abs(weighted_loss.item() - bce_loss.item()) < 1e-5, \
            f"pos_weight=1.0 should match BCE: weighted={weighted_loss.item()}, bce={bce_loss.item()}"

    def test_weighted_bce_higher_weight_increases_fn_penalty(self):
        """Test that higher pos_weight increases penalty for false negatives."""
        from src.training.losses import WeightedBCELoss

        # False negative: positive target, low prediction
        predictions = torch.tensor([0.2])  # Low prob for positive
        targets = torch.tensor([1.0])  # Positive target

        # Low weight
        loss_low = WeightedBCELoss(pos_weight=1.0)(predictions, targets)

        # High weight
        loss_high = WeightedBCELoss(pos_weight=4.0)(predictions, targets)

        assert loss_high.item() > loss_low.item(), \
            f"Higher pos_weight should increase FN penalty: low={loss_low.item()}, high={loss_high.item()}"

    def test_weighted_bce_fp_penalty_unchanged(self):
        """Test that pos_weight does not affect false positive penalty."""
        from src.training.losses import WeightedBCELoss

        # False positive: negative target, high prediction
        predictions = torch.tensor([0.8])  # High prob for negative
        targets = torch.tensor([0.0])  # Negative target

        # Low weight
        loss_low = WeightedBCELoss(pos_weight=1.0)(predictions, targets)

        # High weight
        loss_high = WeightedBCELoss(pos_weight=4.0)(predictions, targets)

        # FP penalty should be the same regardless of pos_weight
        assert abs(loss_high.item() - loss_low.item()) < 1e-5, \
            f"FP penalty should be unchanged: low={loss_low.item()}, high={loss_high.item()}"


class TestWeightedBCELossEdgeCases:
    """Edge case tests for WeightedBCELoss."""

    def test_weighted_bce_no_positive_handles_gracefully(self):
        """Test loss handles batch with no positive samples."""
        from src.training.losses import WeightedBCELoss

        loss_fn = WeightedBCELoss(pos_weight=4.0)

        predictions = torch.tensor([0.3, 0.4, 0.5])
        targets = torch.tensor([0.0, 0.0, 0.0])  # All negative

        loss = loss_fn(predictions, targets)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_weighted_bce_no_negative_handles_gracefully(self):
        """Test loss handles batch with no negative samples."""
        from src.training.losses import WeightedBCELoss

        loss_fn = WeightedBCELoss(pos_weight=4.0)

        predictions = torch.tensor([0.7, 0.8, 0.9])
        targets = torch.tensor([1.0, 1.0, 1.0])  # All positive

        loss = loss_fn(predictions, targets)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"

    def test_weighted_bce_extreme_predictions_handled(self):
        """Test loss handles predictions at 0 and 1 boundaries."""
        from src.training.losses import WeightedBCELoss

        loss_fn = WeightedBCELoss(pos_weight=4.0)

        # Extreme predictions (should be clamped by eps)
        predictions = torch.tensor([0.0, 1.0, 0.5, 0.5])
        targets = torch.tensor([0.0, 1.0, 0.0, 1.0])

        loss = loss_fn(predictions, targets)

        assert not torch.isnan(loss), "Loss should not be NaN"
        assert not torch.isinf(loss), "Loss should not be infinite"


class TestWeightedBCELossGradient:
    """Gradient flow tests for WeightedBCELoss."""

    def test_weighted_bce_gradient_flows(self):
        """Test gradients propagate through WeightedBCELoss."""
        from src.training.losses import WeightedBCELoss

        loss_fn = WeightedBCELoss(pos_weight=4.0)

        predictions = torch.tensor([0.7, 0.6, 0.3, 0.2], requires_grad=True)
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)
        loss.backward()

        assert predictions.grad is not None, "Gradients should flow to predictions"
        assert not torch.all(predictions.grad == 0), "Gradients should be non-zero"


# =============================================================================
# evaluate_forecasting_model is_classification Tests
# =============================================================================


class TestEvaluateForecastingModelClassification:
    """Tests for is_classification parameter in evaluate_forecasting_model."""

    def test_is_classification_uses_half_threshold(self):
        """Test that is_classification=True uses 0.5 threshold."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "architectures"))
        from common import evaluate_forecasting_model

        # Probability predictions: 3 above 0.5, 2 below
        predictions = np.array([0.7, 0.6, 0.55, 0.4, 0.3])
        actual_returns = np.array([0.01, -0.01, 0.005, -0.02, 0.01])
        threshold_targets = np.array([1, 0, 1, 0, 1])

        metrics = evaluate_forecasting_model(
            predicted_returns=predictions,
            actual_returns=actual_returns,
            threshold_targets=threshold_targets,
            is_classification=True,
        )

        # With 0.5 threshold: predictions [0.7, 0.6, 0.55] > 0.5 → 3 positive preds
        assert metrics["n_positive_preds"] == 3, \
            f"Expected 3 positive predictions with 0.5 threshold, got {metrics['n_positive_preds']}"

    def test_default_uses_return_threshold(self):
        """Test that default (is_classification=False) uses return_threshold."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "architectures"))
        from common import evaluate_forecasting_model

        # Return predictions: all well above 0.01 return threshold
        predictions = np.array([0.05, 0.03, 0.02, 0.015, -0.01])
        actual_returns = np.array([0.01, -0.01, 0.005, -0.02, 0.01])
        threshold_targets = np.array([1, 0, 1, 0, 1])

        metrics = evaluate_forecasting_model(
            predicted_returns=predictions,
            actual_returns=actual_returns,
            threshold_targets=threshold_targets,
            return_threshold=0.01,
            is_classification=False,
        )

        # With 0.01 return threshold: 4 predictions > 0.01
        assert metrics["n_positive_preds"] == 4, \
            f"Expected 4 positive predictions with 0.01 threshold, got {metrics['n_positive_preds']}"

    def test_classification_different_from_return_threshold(self):
        """Test that is_classification gives different results than default for probs."""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "architectures"))
        from common import evaluate_forecasting_model

        # Probability predictions in [0, 1]
        predictions = np.array([0.7, 0.4, 0.6, 0.3, 0.8])
        actual_returns = np.array([0.01, -0.01, 0.005, -0.02, 0.01])
        threshold_targets = np.array([1, 0, 1, 0, 1])

        # With is_classification=True: threshold=0.5, predictions > 0.5: [0.7, 0.6, 0.8] → 3
        metrics_cls = evaluate_forecasting_model(
            predicted_returns=predictions,
            actual_returns=actual_returns,
            threshold_targets=threshold_targets,
            is_classification=True,
        )

        # With is_classification=False (default): threshold=0.01, all > 0.01 → 5
        metrics_return = evaluate_forecasting_model(
            predicted_returns=predictions,
            actual_returns=actual_returns,
            threshold_targets=threshold_targets,
            return_threshold=0.01,
            is_classification=False,
        )

        assert metrics_cls["n_positive_preds"] == 3
        assert metrics_return["n_positive_preds"] == 5
        assert metrics_cls["n_positive_preds"] != metrics_return["n_positive_preds"]
