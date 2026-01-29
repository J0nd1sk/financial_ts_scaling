"""Tests for Focal Loss HPO integration."""

import pytest
from pathlib import Path
import sys

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Add scripts to path for import
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))


class TestFocalLossIntegration:
    """Test that Focal Loss can be integrated with Trainer."""

    def test_focal_loss_import(self):
        """Test that FocalLoss can be imported."""
        from src.training.losses import FocalLoss

        # Verify class exists and can be instantiated
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        assert loss_fn is not None
        assert loss_fn.gamma == 2.0
        assert loss_fn.alpha == 0.25

    def test_focal_loss_forward_pass(self):
        """Test that FocalLoss computes correctly."""
        import torch
        from src.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)

        # Test with simple inputs
        predictions = torch.tensor([0.9, 0.8, 0.2, 0.1])
        targets = torch.tensor([1.0, 1.0, 0.0, 0.0])

        loss = loss_fn(predictions, targets)

        # Loss should be a scalar tensor
        assert loss.ndim == 0
        # Loss should be positive
        assert loss.item() > 0
        # Loss should be finite
        assert torch.isfinite(loss)

    def test_focal_loss_gamma_effect(self):
        """Test that higher gamma reduces loss for easy examples."""
        import torch
        from src.training.losses import FocalLoss

        # Easy example: high confidence correct prediction
        predictions = torch.tensor([0.95])
        targets = torch.tensor([1.0])

        # Lower gamma = less focus on hard examples
        loss_low_gamma = FocalLoss(gamma=0.0, alpha=0.5)(predictions, targets)
        # Higher gamma = more focus on hard examples, less penalty for easy
        loss_high_gamma = FocalLoss(gamma=2.0, alpha=0.5)(predictions, targets)

        # Higher gamma should give lower loss for easy (correct) examples
        assert loss_high_gamma.item() < loss_low_gamma.item()

    def test_trainer_accepts_criterion_parameter(self):
        """Test that Trainer.__init__ accepts criterion parameter."""
        import inspect
        from src.training.trainer import Trainer

        # Check that criterion is a valid parameter
        sig = inspect.signature(Trainer.__init__)
        params = list(sig.parameters.keys())
        assert "criterion" in params, "Trainer should accept 'criterion' parameter"


class TestFocalHpoScript:
    """Test the run_focal_hpo.py script."""

    def test_get_focal_hpo_configs_a50(self):
        """Test getting configs for a50 tier."""
        from run_focal_hpo import get_focal_hpo_configs

        configs = get_focal_hpo_configs("a50")
        assert len(configs) == 25, f"Expected 25 a50 configs, got {len(configs)}"

    def test_get_focal_hpo_configs_a100(self):
        """Test getting configs for a100 tier."""
        from run_focal_hpo import get_focal_hpo_configs

        configs = get_focal_hpo_configs("a100")
        assert len(configs) >= 5, f"Expected at least 5 a100 configs, got {len(configs)}"

    def test_get_focal_hpo_configs_invalid_tier_raises(self):
        """Test that invalid tier raises ValueError."""
        from run_focal_hpo import get_focal_hpo_configs

        with pytest.raises(ValueError, match="Unknown tier"):
            get_focal_hpo_configs("invalid_tier")

    def test_a50_configs_have_required_keys(self):
        """Test that all a50 configs have required hyperparameter keys."""
        from run_focal_hpo import get_focal_hpo_configs

        required_keys = {"d_model", "n_layers", "n_heads", "d_ff_ratio",
                         "dropout", "learning_rate", "weight_decay"}
        configs = get_focal_hpo_configs("a50")

        for name, config in configs.items():
            missing = required_keys - set(config.keys())
            assert not missing, f"Config {name} missing keys: {missing}"

    def test_a50_configs_are_unique(self):
        """Test that all a50 configs are unique (no duplicates)."""
        from run_focal_hpo import get_focal_hpo_configs

        configs = get_focal_hpo_configs("a50")

        # Convert configs to tuples for comparison
        seen = {}
        for name, config in configs.items():
            config_tuple = tuple(sorted(config.items()))
            if config_tuple in seen:
                pytest.fail(f"Duplicate config: {name} is same as {seen[config_tuple]}")
            seen[config_tuple] = name

    def test_a50_n_heads_divides_d_model(self):
        """Test that n_heads evenly divides d_model for all a50 configs."""
        from run_focal_hpo import get_focal_hpo_configs

        configs = get_focal_hpo_configs("a50")

        for name, config in configs.items():
            d_model = config["d_model"]
            n_heads = config["n_heads"]
            assert d_model % n_heads == 0, \
                f"Config {name}: d_model={d_model} not divisible by n_heads={n_heads}"

    def test_config_values_in_valid_ranges(self):
        """Test that config values are in reasonable ranges."""
        from run_focal_hpo import get_focal_hpo_configs

        for tier in ["a50", "a100"]:
            configs = get_focal_hpo_configs(tier)

            for name, config in configs.items():
                assert 32 <= config["d_model"] <= 256, f"{name}: d_model out of range"
                assert 2 <= config["n_layers"] <= 10, f"{name}: n_layers out of range"
                assert config["n_heads"] in [2, 4, 8, 16], f"{name}: n_heads not valid"
                assert config["d_ff_ratio"] in [2, 4], f"{name}: d_ff_ratio not valid"
                assert 0.0 <= config["dropout"] <= 0.95, f"{name}: dropout out of range"
                assert 1e-6 <= config["learning_rate"] <= 1e-2, f"{name}: learning_rate out of range"
                assert 0.0 <= config["weight_decay"] <= 0.1, f"{name}: weight_decay out of range"

    def test_config_names_follow_convention(self):
        """Test that config names follow FL## convention."""
        from run_focal_hpo import get_focal_hpo_configs

        for tier in ["a50", "a100"]:
            configs = get_focal_hpo_configs(tier)

            for name in configs.keys():
                assert name.startswith("FL"), f"Config name {name} should start with 'FL'"
                assert len(name) >= 3, f"Config name {name} too short"

    def test_default_focal_parameters(self):
        """Test that default focal parameters are reasonable."""
        from run_focal_hpo import DEFAULT_GAMMA, DEFAULT_ALPHA

        # Standard values from original Focal Loss paper
        assert DEFAULT_GAMMA == 2.0, "Default gamma should be 2.0"
        assert 0.0 < DEFAULT_ALPHA < 1.0, "Default alpha should be between 0 and 1"


class TestHpoTemplateIntegration:
    """Test that hpo_template.py correctly integrates loss_type."""

    def test_focal_loss_import_in_template(self):
        """Test that FocalLoss is imported in hpo_template."""
        template_path = Path(__file__).parent.parent / "experiments/templates/hpo_template.py"
        content = template_path.read_text()

        assert "from src.training.losses import FocalLoss" in content, \
            "hpo_template.py should import FocalLoss"

    def test_loss_type_argument_in_template(self):
        """Test that --loss-type argument is defined in hpo_template."""
        template_path = Path(__file__).parent.parent / "experiments/templates/hpo_template.py"
        content = template_path.read_text()

        assert "--loss-type" in content, \
            "hpo_template.py should have --loss-type argument"
        assert '"focal"' in content or "'focal'" in content, \
            "hpo_template.py should support 'focal' as a loss type"

    def test_criterion_passed_to_trainer_in_template(self):
        """Test that criterion is passed to Trainer in hpo_template."""
        template_path = Path(__file__).parent.parent / "experiments/templates/hpo_template.py"
        content = template_path.read_text()

        assert "criterion=criterion" in content, \
            "hpo_template.py should pass criterion to Trainer"

    def test_focal_loss_creation_in_template(self):
        """Test that FocalLoss is created when loss_type is focal."""
        template_path = Path(__file__).parent.parent / "experiments/templates/hpo_template.py"
        content = template_path.read_text()

        assert 'loss_type == "focal"' in content, \
            "hpo_template.py should check for focal loss type"
        assert "FocalLoss(gamma=" in content, \
            "hpo_template.py should create FocalLoss with gamma"
