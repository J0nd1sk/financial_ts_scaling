"""Tests for loss function parameter sweep script."""

import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TestLossSweepConfig:
    """Test sweep configuration."""

    def test_gamma_values_defined(self):
        """Test that gamma values are defined."""
        from run_loss_sweep import GAMMA_VALUES
        assert len(GAMMA_VALUES) == 4
        assert 0.0 in GAMMA_VALUES  # weighted BCE
        assert 2.0 in GAMMA_VALUES  # standard focal

    def test_alpha_values_defined(self):
        """Test that alpha values are defined."""
        from run_loss_sweep import ALPHA_VALUES
        assert len(ALPHA_VALUES) == 10
        assert all(0 < a < 1 for a in ALPHA_VALUES)

    def test_dropout_values_defined(self):
        """Test that dropout values are defined."""
        from run_loss_sweep import DROPOUT_VALUES
        assert len(DROPOUT_VALUES) == 3
        assert all(0 < d < 1 for d in DROPOUT_VALUES)

    def test_best_architectures_defined(self):
        """Test that best architectures are defined for each tier."""
        from run_loss_sweep import BEST_ARCHITECTURES
        assert "a50" in BEST_ARCHITECTURES
        assert "a100" in BEST_ARCHITECTURES

    def test_a50_architecture_has_required_keys(self):
        """Test a50 architecture has all required keys."""
        from run_loss_sweep import BEST_ARCHITECTURES
        required = {"d_model", "n_layers", "n_heads", "d_ff_ratio",
                    "dropout", "learning_rate", "weight_decay"}
        assert required <= set(BEST_ARCHITECTURES["a50"].keys())

    def test_a100_architecture_has_required_keys(self):
        """Test a100 architecture has all required keys."""
        from run_loss_sweep import BEST_ARCHITECTURES
        required = {"d_model", "n_layers", "n_heads", "d_ff_ratio",
                    "dropout", "learning_rate", "weight_decay"}
        assert required <= set(BEST_ARCHITECTURES["a100"].keys())

    def test_n_heads_divides_d_model(self):
        """Test that n_heads divides d_model for all architectures."""
        from run_loss_sweep import BEST_ARCHITECTURES
        for tier, arch in BEST_ARCHITECTURES.items():
            assert arch["d_model"] % arch["n_heads"] == 0, \
                f"{tier}: d_model not divisible by n_heads"

    def test_total_combinations(self):
        """Test total number of sweep combinations."""
        from run_loss_sweep import GAMMA_VALUES, ALPHA_VALUES, DROPOUT_VALUES
        total = len(GAMMA_VALUES) * len(ALPHA_VALUES) * len(DROPOUT_VALUES)
        assert total == 120  # 4 gamma × 10 alpha × 3 dropout


class TestFocalLossInterpretation:
    """Test that we understand what gamma/alpha mean."""

    def test_gamma_zero_is_weighted_bce(self):
        """Test that gamma=0 gives weighted BCE (no focal effect)."""
        import torch
        from src.training.losses import FocalLoss

        # With gamma=0, focal weight = (1-p_t)^0 = 1 for all examples
        loss_fn = FocalLoss(gamma=0.0, alpha=0.5)

        predictions = torch.tensor([0.9, 0.1])
        targets = torch.tensor([1.0, 0.0])

        loss = loss_fn(predictions, targets)
        assert loss.item() > 0  # Should compute something

    def test_higher_alpha_weights_positives_more(self):
        """Test that higher alpha gives more weight to positive class."""
        import torch
        from src.training.losses import FocalLoss

        predictions = torch.tensor([0.3])  # Wrong prediction for positive
        targets = torch.tensor([1.0])

        loss_low_alpha = FocalLoss(gamma=0, alpha=0.25)(predictions, targets)
        loss_high_alpha = FocalLoss(gamma=0, alpha=0.75)(predictions, targets)

        # Higher alpha = more penalty for missing positives
        assert loss_high_alpha.item() > loss_low_alpha.item()
