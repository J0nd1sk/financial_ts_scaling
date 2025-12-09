"""Tests for parameter counting and budget validation."""

import pytest

from src.models.patchtst import PatchTST, PatchTSTConfig
from src.models.utils import count_parameters


class TestCountParameters:
    """Test the count_parameters utility function."""

    def test_count_parameters_returns_integer(self):
        """count_parameters should return an integer."""
        config = PatchTSTConfig(
            num_features=20,
            context_length=60,
            patch_length=10,
            stride=10,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            dropout=0.1,
            head_dropout=0.0,
        )
        model = PatchTST(config)
        param_count = count_parameters(model)
        assert isinstance(param_count, int)

    def test_count_parameters_matches_pytorch_builtin(self):
        """count_parameters should match sum of p.numel() for trainable params."""
        config = PatchTSTConfig(
            num_features=20,
            context_length=60,
            patch_length=10,
            stride=10,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            dropout=0.1,
            head_dropout=0.0,
        )
        model = PatchTST(config)
        expected = sum(p.numel() for p in model.parameters() if p.requires_grad)
        actual = count_parameters(model)
        assert actual == expected


class TestParameterBudget2M:
    """Test 2M parameter budget configuration."""

    def test_2m_config_within_budget(self):
        """2M config should have 1.5M <= params <= 2.5M."""
        from src.models.configs import load_patchtst_config

        config = load_patchtst_config("2m")
        model = PatchTST(config)
        param_count = count_parameters(model)
        assert 1_500_000 <= param_count <= 2_500_000, (
            f"2M config has {param_count:,} params, expected 1.5M-2.5M"
        )


class TestParameterBudget20M:
    """Test 20M parameter budget configuration."""

    def test_20m_config_within_budget(self):
        """20M config should have 15M <= params <= 25M."""
        from src.models.configs import load_patchtst_config

        config = load_patchtst_config("20m")
        model = PatchTST(config)
        param_count = count_parameters(model)
        assert 15_000_000 <= param_count <= 25_000_000, (
            f"20M config has {param_count:,} params, expected 15M-25M"
        )


class TestParameterBudget200M:
    """Test 200M parameter budget configuration."""

    def test_200m_config_within_budget(self):
        """200M config should have 150M <= params <= 250M."""
        from src.models.configs import load_patchtst_config

        config = load_patchtst_config("200m")
        model = PatchTST(config)
        param_count = count_parameters(model)
        assert 150_000_000 <= param_count <= 250_000_000, (
            f"200M config has {param_count:,} params, expected 150M-250M"
        )
