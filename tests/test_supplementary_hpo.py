"""Tests for supplementary HPO trials script."""

import pytest
from pathlib import Path
import sys

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


class TestSupplementaryConfigs:
    """Test that supplementary configs are valid and complete."""

    def test_a50_configs_count(self):
        """Test that a50 has expected number of configs."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A50
        assert len(SUPPLEMENTARY_CONFIGS_A50) == 27, f"Expected 27 a50 configs, got {len(SUPPLEMENTARY_CONFIGS_A50)}"

    def test_followup_a50_configs_count(self):
        """Test that follow-up a50 has expected 13 configs (G, H, I series)."""
        from run_supplementary_hpo import FOLLOWUP_CONFIGS_A50
        assert len(FOLLOWUP_CONFIGS_A50) == 13, f"Expected 13 follow-up a50 configs, got {len(FOLLOWUP_CONFIGS_A50)}"

    def test_followup_configs_have_required_keys(self):
        """Test that all follow-up configs have required hyperparameter keys."""
        from run_supplementary_hpo import FOLLOWUP_CONFIGS_A50
        required_keys = {"d_model", "n_layers", "n_heads", "d_ff_ratio", "dropout", "learning_rate", "weight_decay"}

        for name, config in FOLLOWUP_CONFIGS_A50.items():
            missing = required_keys - set(config.keys())
            assert not missing, f"Follow-up config {name} missing keys: {missing}"

    def test_followup_configs_unique_from_original(self):
        """Test that follow-up configs don't duplicate original A-F configs."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A50, FOLLOWUP_CONFIGS_A50

        # Convert to tuples for comparison
        original_tuples = {tuple(sorted(c.items())) for c in SUPPLEMENTARY_CONFIGS_A50.values()}
        followup_tuples = {tuple(sorted(c.items())) for c in FOLLOWUP_CONFIGS_A50.values()}

        duplicates = original_tuples & followup_tuples
        assert not duplicates, f"Follow-up configs duplicate original configs"

    def test_followup_configs_are_unique(self):
        """Test that all follow-up configs are unique (no duplicates within G, H, I)."""
        from run_supplementary_hpo import FOLLOWUP_CONFIGS_A50

        config_tuples = []
        for name, config in FOLLOWUP_CONFIGS_A50.items():
            config_tuple = tuple(sorted(config.items()))
            config_tuples.append((name, config_tuple))

        seen = {}
        for name, config_tuple in config_tuples:
            if config_tuple in seen:
                pytest.fail(f"Duplicate follow-up config: {name} is same as {seen[config_tuple]}")
            seen[config_tuple] = name

    def test_followup_config_names_follow_convention(self):
        """Test that follow-up config names use G, H, I series."""
        from run_supplementary_hpo import FOLLOWUP_CONFIGS_A50

        for name in FOLLOWUP_CONFIGS_A50.keys():
            assert len(name) >= 2, f"Config name {name} too short"
            assert name[0] in ("G", "H", "I"), f"Follow-up config name {name} should start with G, H, or I"

    def test_followup_n_heads_divides_d_model(self):
        """Test that n_heads evenly divides d_model for all follow-up configs."""
        from run_supplementary_hpo import FOLLOWUP_CONFIGS_A50

        for name, config in FOLLOWUP_CONFIGS_A50.items():
            d_model = config["d_model"]
            n_heads = config["n_heads"]
            assert d_model % n_heads == 0, f"Follow-up config {name}: d_model={d_model} not divisible by n_heads={n_heads}"

    def test_followup_config_values_in_valid_ranges(self):
        """Test that follow-up config values are in reasonable ranges."""
        from run_supplementary_hpo import FOLLOWUP_CONFIGS_A50

        for name, config in FOLLOWUP_CONFIGS_A50.items():
            assert 32 <= config["d_model"] <= 256, f"{name}: d_model out of range"
            assert 2 <= config["n_layers"] <= 10, f"{name}: n_layers out of range"
            assert config["n_heads"] in [2, 4, 8, 16], f"{name}: n_heads not valid"
            assert config["d_ff_ratio"] in [2, 4], f"{name}: d_ff_ratio not valid"
            assert 0.0 <= config["dropout"] <= 0.95, f"{name}: dropout out of range"
            assert 1e-6 <= config["learning_rate"] <= 1e-2, f"{name}: learning_rate out of range"
            assert 0.0 <= config["weight_decay"] <= 0.1, f"{name}: weight_decay out of range"

    def test_a100_configs_count(self):
        """Test that a100 has expected number of configs."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A100
        assert len(SUPPLEMENTARY_CONFIGS_A100) == 24, f"Expected 24 a100 configs, got {len(SUPPLEMENTARY_CONFIGS_A100)}"

    def test_a50_configs_have_required_keys(self):
        """Test that all a50 configs have required hyperparameter keys."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A50
        required_keys = {"d_model", "n_layers", "n_heads", "d_ff_ratio", "dropout", "learning_rate", "weight_decay"}

        for name, config in SUPPLEMENTARY_CONFIGS_A50.items():
            missing = required_keys - set(config.keys())
            assert not missing, f"Config {name} missing keys: {missing}"

    def test_a100_configs_have_required_keys(self):
        """Test that all a100 configs have required hyperparameter keys."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A100
        required_keys = {"d_model", "n_layers", "n_heads", "d_ff_ratio", "dropout", "learning_rate", "weight_decay"}

        for name, config in SUPPLEMENTARY_CONFIGS_A100.items():
            missing = required_keys - set(config.keys())
            assert not missing, f"Config {name} missing keys: {missing}"

    def test_a50_configs_are_unique(self):
        """Test that all a50 configs are unique (no duplicates)."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A50

        # Convert configs to tuples for comparison
        config_tuples = []
        for name, config in SUPPLEMENTARY_CONFIGS_A50.items():
            config_tuple = tuple(sorted(config.items()))
            config_tuples.append((name, config_tuple))

        # Check for duplicates
        seen = {}
        for name, config_tuple in config_tuples:
            if config_tuple in seen:
                pytest.fail(f"Duplicate config: {name} is same as {seen[config_tuple]}")
            seen[config_tuple] = name

    def test_a100_configs_are_unique(self):
        """Test that all a100 configs are unique (no duplicates)."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A100

        # Convert configs to tuples for comparison
        config_tuples = []
        for name, config in SUPPLEMENTARY_CONFIGS_A100.items():
            config_tuple = tuple(sorted(config.items()))
            config_tuples.append((name, config_tuple))

        # Check for duplicates
        seen = {}
        for name, config_tuple in config_tuples:
            if config_tuple in seen:
                pytest.fail(f"Duplicate config: {name} is same as {seen[config_tuple]}")
            seen[config_tuple] = name

    def test_a50_n_heads_divides_d_model(self):
        """Test that n_heads evenly divides d_model for all a50 configs."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A50

        for name, config in SUPPLEMENTARY_CONFIGS_A50.items():
            d_model = config["d_model"]
            n_heads = config["n_heads"]
            assert d_model % n_heads == 0, f"Config {name}: d_model={d_model} not divisible by n_heads={n_heads}"

    def test_a100_n_heads_divides_d_model(self):
        """Test that n_heads evenly divides d_model for all a100 configs."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A100

        for name, config in SUPPLEMENTARY_CONFIGS_A100.items():
            d_model = config["d_model"]
            n_heads = config["n_heads"]
            assert d_model % n_heads == 0, f"Config {name}: d_model={d_model} not divisible by n_heads={n_heads}"

    def test_config_values_in_valid_ranges(self):
        """Test that config values are in reasonable ranges."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A50, SUPPLEMENTARY_CONFIGS_A100

        all_configs = {**SUPPLEMENTARY_CONFIGS_A50, **SUPPLEMENTARY_CONFIGS_A100}

        for name, config in all_configs.items():
            assert 32 <= config["d_model"] <= 256, f"{name}: d_model out of range"
            assert 2 <= config["n_layers"] <= 10, f"{name}: n_layers out of range"
            assert config["n_heads"] in [2, 4, 8, 16], f"{name}: n_heads not valid"
            assert config["d_ff_ratio"] in [2, 4], f"{name}: d_ff_ratio not valid"
            assert 0.0 <= config["dropout"] <= 0.95, f"{name}: dropout out of range"
            assert 1e-6 <= config["learning_rate"] <= 1e-2, f"{name}: learning_rate out of range"
            assert 0.0 <= config["weight_decay"] <= 0.1, f"{name}: weight_decay out of range"


class TestConfigNaming:
    """Test config naming conventions."""

    def test_a50_config_names_follow_convention(self):
        """Test that a50 config names follow naming convention."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A50

        for name in SUPPLEMENTARY_CONFIGS_A50.keys():
            # Names should be like A1, A2, B1, etc.
            assert len(name) >= 2, f"Config name {name} too short"
            assert name[0].isupper(), f"Config name {name} should start with uppercase letter"

    def test_a100_config_names_follow_convention(self):
        """Test that a100 config names follow naming convention."""
        from run_supplementary_hpo import SUPPLEMENTARY_CONFIGS_A100

        for name in SUPPLEMENTARY_CONFIGS_A100.keys():
            # Names should be like A1, A2, B1, etc.
            assert len(name) >= 2, f"Config name {name} too short"
            assert name[0].isupper(), f"Config name {name} should start with uppercase letter"


class TestHelperFunctions:
    """Test helper functions in the script."""

    def test_get_configs_for_tier_a50(self):
        """Test getting configs for a50 tier."""
        from run_supplementary_hpo import get_configs_for_tier

        configs = get_configs_for_tier("a50")
        assert len(configs) == 27

    def test_get_configs_for_tier_a50_round2(self):
        """Test getting follow-up configs for a50 tier round 2."""
        from run_supplementary_hpo import get_configs_for_tier

        configs = get_configs_for_tier("a50", round_num=2)
        assert len(configs) == 13

    def test_get_configs_for_tier_a100(self):
        """Test getting configs for a100 tier."""
        from run_supplementary_hpo import get_configs_for_tier

        configs = get_configs_for_tier("a100")
        assert len(configs) == 24

    def test_get_configs_for_tier_a100_round2_raises(self):
        """Test that a100 round 2 raises ValueError (not implemented)."""
        from run_supplementary_hpo import get_configs_for_tier

        with pytest.raises(ValueError, match="Round 2 only available for a50"):
            get_configs_for_tier("a100", round_num=2)

    def test_get_configs_for_invalid_tier_raises(self):
        """Test that invalid tier raises ValueError."""
        from run_supplementary_hpo import get_configs_for_tier

        with pytest.raises(ValueError, match="Unknown tier"):
            get_configs_for_tier("a999")
