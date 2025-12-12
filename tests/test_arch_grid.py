"""Tests for architecture grid generation and parameter estimation.

TDD: These tests define the specification for src/models/arch_grid.py
"""

import pytest

from src.models.patchtst import PatchTST, PatchTSTConfig
from src.models.utils import count_parameters


class TestEstimateParamCount:
    """Test parameter count estimation accuracy."""

    def test_estimate_matches_actual_2m(self):
        """Estimation should match actual model params for 2M config."""
        from src.models.arch_grid import estimate_param_count

        # 2M config from patchtst_2m.yaml
        config = PatchTSTConfig(
            num_features=20,
            context_length=60,
            patch_length=10,
            stride=10,
            d_model=192,
            n_heads=6,
            n_layers=4,
            d_ff=768,
            dropout=0.1,
            head_dropout=0.0,
            num_classes=1,
        )
        model = PatchTST(config)
        actual = count_parameters(model)

        estimated = estimate_param_count(
            d_model=192,
            n_layers=4,
            n_heads=6,
            d_ff=768,
            num_features=20,
            context_length=60,
            patch_len=10,
            stride=10,
            num_classes=1,
        )

        # Allow 0.1% tolerance for rounding
        assert abs(estimated - actual) / actual < 0.001, (
            f"Estimated {estimated:,} but actual is {actual:,}"
        )

    def test_estimate_matches_actual_20m(self):
        """Estimation should match actual model params for 20M config."""
        from src.models.arch_grid import estimate_param_count

        # 20M config from patchtst_20m.yaml
        config = PatchTSTConfig(
            num_features=20,
            context_length=60,
            patch_length=10,
            stride=10,
            d_model=512,
            n_heads=8,
            n_layers=8,
            d_ff=2048,
            dropout=0.1,
            head_dropout=0.0,
            num_classes=1,
        )
        model = PatchTST(config)
        actual = count_parameters(model)

        estimated = estimate_param_count(
            d_model=512,
            n_layers=8,
            n_heads=8,
            d_ff=2048,
            num_features=20,
            context_length=60,
            patch_len=10,
            stride=10,
            num_classes=1,
        )

        assert abs(estimated - actual) / actual < 0.001, (
            f"Estimated {estimated:,} but actual is {actual:,}"
        )

    def test_estimate_matches_actual_200m(self):
        """Estimation should match actual model params for 200M config."""
        from src.models.arch_grid import estimate_param_count

        # 200M config from patchtst_200m.yaml
        config = PatchTSTConfig(
            num_features=20,
            context_length=60,
            patch_length=10,
            stride=10,
            d_model=1536,
            n_heads=16,
            n_layers=16,
            d_ff=6144,
            dropout=0.1,
            head_dropout=0.0,
            num_classes=1,
        )
        model = PatchTST(config)
        actual = count_parameters(model)

        estimated = estimate_param_count(
            d_model=1536,
            n_layers=16,
            n_heads=16,
            d_ff=6144,
            num_features=20,
            context_length=60,
            patch_len=10,
            stride=10,
            num_classes=1,
        )

        assert abs(estimated - actual) / actual < 0.001, (
            f"Estimated {estimated:,} but actual is {actual:,}"
        )

    def test_estimate_returns_int(self):
        """estimate_param_count should return an integer."""
        from src.models.arch_grid import estimate_param_count

        result = estimate_param_count(
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            num_features=25,
        )
        assert isinstance(result, int)

    def test_estimate_with_different_num_features(self):
        """Estimation should work with different num_features values."""
        from src.models.arch_grid import estimate_param_count

        # Test with num_features=25 (tier_a25)
        result_25 = estimate_param_count(
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            num_features=25,
        )

        # Test with num_features=20 (tier_a20)
        result_20 = estimate_param_count(
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            num_features=20,
        )

        # More features = more params (in patch embedding)
        assert result_25 > result_20


class TestArchSearchSpace:
    """Test ARCH_SEARCH_SPACE constant matches design doc."""

    def test_search_space_has_required_keys(self):
        """ARCH_SEARCH_SPACE should have all required keys."""
        from src.models.arch_grid import ARCH_SEARCH_SPACE

        required_keys = {"d_model", "n_layers", "n_heads", "d_ff_ratio"}
        assert required_keys == set(ARCH_SEARCH_SPACE.keys())

    def test_d_model_values_match_design(self):
        """d_model values should match design doc exactly."""
        from src.models.arch_grid import ARCH_SEARCH_SPACE

        expected = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
        assert ARCH_SEARCH_SPACE["d_model"] == expected

    def test_n_layers_values_match_design(self):
        """n_layers values should match design doc exactly."""
        from src.models.arch_grid import ARCH_SEARCH_SPACE

        expected = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48]
        assert ARCH_SEARCH_SPACE["n_layers"] == expected

    def test_n_heads_values_match_design(self):
        """n_heads values should match design doc exactly."""
        from src.models.arch_grid import ARCH_SEARCH_SPACE

        expected = [2, 4, 8, 16, 32]
        assert ARCH_SEARCH_SPACE["n_heads"] == expected

    def test_d_ff_ratio_values_match_design(self):
        """d_ff_ratio values should be 2 and 4 only."""
        from src.models.arch_grid import ARCH_SEARCH_SPACE

        expected = [2, 4]
        assert ARCH_SEARCH_SPACE["d_ff_ratio"] == expected


class TestGenerateArchitectureGrid:
    """Test architecture grid generation."""

    def test_grid_filters_invalid_heads(self):
        """Grid should filter out configs where n_heads doesn't divide d_model."""
        from src.models.arch_grid import generate_architecture_grid

        grid = generate_architecture_grid(num_features=25)

        for arch in grid:
            assert arch["d_model"] % arch["n_heads"] == 0, (
                f"Invalid arch: d_model={arch['d_model']} not divisible by n_heads={arch['n_heads']}"
            )

    def test_grid_includes_valid_combinations(self):
        """Grid should include valid head/model combinations."""
        from src.models.arch_grid import generate_architecture_grid

        grid = generate_architecture_grid(num_features=25)

        # d_model=128 with n_heads=4 should be valid
        valid_combos = [
            (a["d_model"], a["n_heads"])
            for a in grid
            if a["d_model"] == 128 and a["n_heads"] == 4
        ]
        assert len(valid_combos) > 0, "Expected d_model=128, n_heads=4 to be valid"

    def test_grid_computes_d_ff_correctly(self):
        """d_ff should be computed as d_model * ratio."""
        from src.models.arch_grid import generate_architecture_grid

        grid = generate_architecture_grid(num_features=25)

        for arch in grid:
            # d_ff should be 2x or 4x d_model
            ratio = arch["d_ff"] / arch["d_model"]
            assert ratio in [2, 4], (
                f"Invalid d_ff ratio: {ratio} for d_model={arch['d_model']}, d_ff={arch['d_ff']}"
            )

    def test_grid_returns_list_of_dicts(self):
        """Grid should return list of dicts with expected keys."""
        from src.models.arch_grid import generate_architecture_grid

        grid = generate_architecture_grid(num_features=25)

        assert isinstance(grid, list)
        assert len(grid) > 0

        required_keys = {"d_model", "n_layers", "n_heads", "d_ff", "param_count"}
        for arch in grid:
            assert isinstance(arch, dict)
            assert required_keys.issubset(set(arch.keys())), (
                f"Missing keys: {required_keys - set(arch.keys())}"
            )

    def test_grid_includes_param_count(self):
        """Each architecture dict should include estimated param_count."""
        from src.models.arch_grid import generate_architecture_grid

        grid = generate_architecture_grid(num_features=25)

        for arch in grid:
            assert "param_count" in arch
            assert isinstance(arch["param_count"], int)
            assert arch["param_count"] > 0


class TestFilterByBudget:
    """Test budget filtering with ±25% tolerance."""

    def test_filter_respects_25pct_tolerance(self):
        """All filtered architectures should be within ±25% of budget."""
        from src.models.arch_grid import filter_by_budget, generate_architecture_grid

        grid = generate_architecture_grid(num_features=25)
        budget = 2_000_000
        filtered = filter_by_budget(grid, budget)

        min_allowed = int(budget * 0.75)
        max_allowed = int(budget * 1.25)

        for arch in filtered:
            assert min_allowed <= arch["param_count"] <= max_allowed, (
                f"Arch with {arch['param_count']:,} params outside ±25% of {budget:,}"
            )

    def test_filter_2m_returns_reasonable_count(self):
        """2M budget should return reasonable number of architectures."""
        from src.models.arch_grid import filter_by_budget, generate_architecture_grid

        grid = generate_architecture_grid(num_features=25)
        filtered = filter_by_budget(grid, 2_000_000)

        # Should have enough architectures for meaningful HPO exploration
        # but not so many that the grid is unbounded
        assert 10 <= len(filtered) <= 150, (
            f"Expected 10-150 architectures for 2M, got {len(filtered)}"
        )

    def test_filter_20m_returns_reasonable_count(self):
        """20M budget should return reasonable number of architectures."""
        from src.models.arch_grid import filter_by_budget, generate_architecture_grid

        grid = generate_architecture_grid(num_features=25)
        filtered = filter_by_budget(grid, 20_000_000)

        assert 15 <= len(filtered) <= 50, (
            f"Expected 15-50 architectures for 20M, got {len(filtered)}"
        )

    def test_filter_200m_returns_reasonable_count(self):
        """200M budget should return reasonable number of architectures."""
        from src.models.arch_grid import filter_by_budget, generate_architecture_grid

        grid = generate_architecture_grid(num_features=25)
        filtered = filter_by_budget(grid, 200_000_000)

        # Should have enough architectures for meaningful HPO exploration
        assert 10 <= len(filtered) <= 150, (
            f"Expected 10-150 architectures for 200M, got {len(filtered)}"
        )


class TestGetArchitecturesForBudget:
    """Test main entry point function."""

    def test_2m_returns_list(self):
        """get_architectures_for_budget('2M') should return a list."""
        from src.models.arch_grid import get_architectures_for_budget

        result = get_architectures_for_budget("2M", num_features=25)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_case_insensitive(self):
        """Budget string should be case-insensitive."""
        from src.models.arch_grid import get_architectures_for_budget

        result_upper = get_architectures_for_budget("2M", num_features=25)
        result_lower = get_architectures_for_budget("2m", num_features=25)

        assert len(result_upper) == len(result_lower)

    def test_includes_extreme_d_model_min(self):
        """Result should include architecture with minimum valid d_model."""
        from src.models.arch_grid import get_architectures_for_budget

        result = get_architectures_for_budget("2M", num_features=25)
        d_models = [arch["d_model"] for arch in result]
        min_d_model = min(d_models)

        # Should have at least one arch at the minimum d_model
        min_count = sum(1 for d in d_models if d == min_d_model)
        assert min_count >= 1, f"Expected at least one arch with min d_model={min_d_model}"

    def test_includes_extreme_d_model_max(self):
        """Result should include architecture with maximum valid d_model."""
        from src.models.arch_grid import get_architectures_for_budget

        result = get_architectures_for_budget("2M", num_features=25)
        d_models = [arch["d_model"] for arch in result]
        max_d_model = max(d_models)

        # Should have at least one arch at the maximum d_model
        max_count = sum(1 for d in d_models if d == max_d_model)
        assert max_count >= 1, f"Expected at least one arch with max d_model={max_d_model}"

    def test_includes_extreme_n_layers_min(self):
        """Result should include architecture with minimum n_layers."""
        from src.models.arch_grid import get_architectures_for_budget

        result = get_architectures_for_budget("2M", num_features=25)
        n_layers_list = [arch["n_layers"] for arch in result]
        min_layers = min(n_layers_list)

        min_count = sum(1 for n in n_layers_list if n == min_layers)
        assert min_count >= 1, f"Expected at least one arch with min n_layers={min_layers}"

    def test_includes_extreme_n_layers_max(self):
        """Result should include architecture with maximum n_layers."""
        from src.models.arch_grid import get_architectures_for_budget

        result = get_architectures_for_budget("2M", num_features=25)
        n_layers_list = [arch["n_layers"] for arch in result]
        max_layers = max(n_layers_list)

        max_count = sum(1 for n in n_layers_list if n == max_layers)
        assert max_count >= 1, f"Expected at least one arch with max n_layers={max_layers}"

    def test_invalid_budget_raises_valueerror(self):
        """Invalid budget string should raise ValueError."""
        from src.models.arch_grid import get_architectures_for_budget

        with pytest.raises(ValueError, match="[Ii]nvalid budget"):
            get_architectures_for_budget("5M", num_features=25)

    def test_all_budgets_work(self):
        """All valid budgets should return results."""
        from src.models.arch_grid import get_architectures_for_budget

        for budget in ["2M", "20M", "200M", "2B"]:
            result = get_architectures_for_budget(budget, num_features=25)
            assert len(result) > 0, f"Budget {budget} returned empty result"

    def test_architectures_sorted_by_param_count(self):
        """Architectures should be sorted by param_count ascending."""
        from src.models.arch_grid import get_architectures_for_budget

        result = get_architectures_for_budget("2M", num_features=25)
        param_counts = [arch["param_count"] for arch in result]

        assert param_counts == sorted(param_counts), "Architectures should be sorted by param_count"
