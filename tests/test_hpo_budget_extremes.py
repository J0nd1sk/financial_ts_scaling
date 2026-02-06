"""Tests for budget-aware HPO with forced extremes.

Tests for:
1. BUDGET_CONFIGS dictionary structure and parameter validation
2. compute_budget_aware_extremes() function
3. generate_forced_configs() function
4. check_early_stopping_convergence() function

These tests follow the two-phase budget-aware HPO strategy for transformer
architectures across parameter scales (750k → 2M → 20M → 200M).
"""

import pytest
from unittest.mock import MagicMock, patch


class TestBudgetConfigs:
    """Tests for BUDGET_CONFIGS dictionary."""

    def test_budget_configs_has_all_budgets(self):
        """Test BUDGET_CONFIGS contains all 4 parameter budgets."""
        from src.training.hpo_budget_extremes import BUDGET_CONFIGS

        expected_budgets = ["750k", "2M", "20M", "200M"]
        assert set(BUDGET_CONFIGS.keys()) == set(expected_budgets)

    def test_budget_configs_has_shallow_and_deep(self):
        """Test each budget has both shallow-wide and deep-narrow configs."""
        from src.training.hpo_budget_extremes import BUDGET_CONFIGS

        for budget, configs in BUDGET_CONFIGS.items():
            assert "shallow" in configs, f"{budget} missing shallow config"
            assert "deep" in configs, f"{budget} missing deep config"

    def test_budget_configs_d_model_divisible_by_n_heads(self):
        """Test all configs have d_model divisible by n_heads."""
        from src.training.hpo_budget_extremes import BUDGET_CONFIGS

        for budget, styles in BUDGET_CONFIGS.items():
            for style, config in styles.items():
                d_model = config["d_model"]
                n_heads = config["n_heads"]
                assert d_model % n_heads == 0, (
                    f"{budget}/{style}: d_model={d_model} not divisible by n_heads={n_heads}"
                )

    def test_budget_configs_parameter_estimates_reasonable(self):
        """Test parameter count estimates are in expected ranges."""
        from src.training.hpo_budget_extremes import BUDGET_CONFIGS

        # Expected rough ranges (factor of 2 tolerance)
        budget_ranges = {
            "750k": (400_000, 1_500_000),
            "2M": (1_000_000, 4_000_000),
            "20M": (10_000_000, 40_000_000),
            "200M": (100_000_000, 400_000_000),
        }

        for budget, styles in BUDGET_CONFIGS.items():
            min_params, max_params = budget_ranges[budget]
            for style, config in styles.items():
                # Rough param estimate: 12 * n_layers * d_model^2
                d_model = config["d_model"]
                n_layers = config["n_layers"]
                estimated_params = 12 * n_layers * d_model * d_model
                assert min_params <= estimated_params <= max_params, (
                    f"{budget}/{style}: estimated {estimated_params:,} params "
                    f"outside range [{min_params:,}, {max_params:,}]"
                )


class TestComputeBudgetAwareExtremes:
    """Tests for compute_budget_aware_extremes() function."""

    def test_returns_valid_config_dict(self):
        """Test function returns dict with required keys."""
        from src.training.hpo_budget_extremes import compute_budget_aware_extremes

        config = compute_budget_aware_extremes("2M", "shallow")
        required_keys = {"d_model", "n_layers", "n_heads", "dropout", "learning_rate", "weight_decay"}
        assert required_keys.issubset(set(config.keys()))

    def test_shallow_has_larger_d_model_than_deep(self):
        """Test shallow configs have larger d_model than deep for same budget."""
        from src.training.hpo_budget_extremes import compute_budget_aware_extremes

        for budget in ["2M", "20M", "200M"]:
            shallow = compute_budget_aware_extremes(budget, "shallow")
            deep = compute_budget_aware_extremes(budget, "deep")
            assert shallow["d_model"] > deep["d_model"], (
                f"{budget}: shallow d_model should be > deep d_model"
            )

    def test_deep_has_more_layers_than_shallow(self):
        """Test deep configs have more layers than shallow for same budget."""
        from src.training.hpo_budget_extremes import compute_budget_aware_extremes

        for budget in ["2M", "20M", "200M"]:
            shallow = compute_budget_aware_extremes(budget, "shallow")
            deep = compute_budget_aware_extremes(budget, "deep")
            assert deep["n_layers"] > shallow["n_layers"], (
                f"{budget}: deep n_layers should be > shallow n_layers"
            )


class TestRegularizationExtremes:
    """Tests for REGULARIZATION_EXTREMES dictionary."""

    def test_regularization_extremes_defined(self):
        """Test REGULARIZATION_EXTREMES is defined with expected keys."""
        from src.training.hpo_budget_extremes import REGULARIZATION_EXTREMES

        expected_keys = {"dropout", "learning_rate", "weight_decay"}
        assert expected_keys.issubset(set(REGULARIZATION_EXTREMES.keys()))

    def test_dropout_extremes_in_valid_range(self):
        """Test dropout extremes are between 0 and 1."""
        from src.training.hpo_budget_extremes import REGULARIZATION_EXTREMES

        for value in REGULARIZATION_EXTREMES["dropout"]:
            assert 0.0 < value < 1.0, f"Invalid dropout: {value}"

    def test_learning_rate_extremes_reasonable(self):
        """Test learning rate extremes span meaningful range."""
        from src.training.hpo_budget_extremes import REGULARIZATION_EXTREMES

        lrs = REGULARIZATION_EXTREMES["learning_rate"]
        assert min(lrs) <= 1e-4, "Should include low learning rate"
        assert max(lrs) >= 1e-4, "Should include high learning rate"


class TestGenerateForcedConfigs:
    """Tests for generate_forced_configs() function."""

    def test_generates_18_forced_configs_default(self):
        """Test default generation produces 18 forced extreme configs."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs()
        assert len(configs) == 18, f"Expected 18 configs, got {len(configs)}"

    def test_all_configs_have_required_keys(self):
        """Test all generated configs have required architecture keys."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        required_keys = {"d_model", "n_layers", "n_heads", "dropout", "learning_rate", "weight_decay"}
        configs = generate_forced_configs()
        for i, config in enumerate(configs):
            assert required_keys.issubset(set(config.keys())), (
                f"Config {i} missing required keys: {required_keys - set(config.keys())}"
            )

    def test_all_configs_have_valid_d_model_n_heads(self):
        """Test all configs have d_model divisible by n_heads."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs()
        for i, config in enumerate(configs):
            d_model = config["d_model"]
            n_heads = config["n_heads"]
            assert d_model % n_heads == 0, (
                f"Config {i}: d_model={d_model} not divisible by n_heads={n_heads}"
            )

    def test_configs_cover_all_budgets(self):
        """Test generated configs cover all 4 parameter budgets."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs()
        budgets_covered = set(c.get("budget", c.get("param_budget")) for c in configs)
        expected_budgets = {"750k", "2M", "20M", "200M"}
        assert expected_budgets.issubset(budgets_covered), (
            f"Missing budgets: {expected_budgets - budgets_covered}"
        )

    def test_configs_include_dropout_extremes(self):
        """Test configs include low (0.1) and high (0.7) dropout values."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs()
        dropouts = [c["dropout"] for c in configs]
        assert 0.1 in dropouts, "Should include low dropout 0.1"
        assert 0.7 in dropouts, "Should include high dropout 0.7"

    def test_configs_include_lr_extremes(self):
        """Test configs include low (1e-5) and high (1e-3) learning rates."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs()
        lrs = [c["learning_rate"] for c in configs]
        assert 1e-5 in lrs, "Should include low LR 1e-5"
        assert 1e-3 in lrs, "Should include high LR 1e-3"

    def test_configs_include_weight_decay_extremes(self):
        """Test configs include zero and high weight decay values."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs()
        wds = [c["weight_decay"] for c in configs]
        assert 0.0 in wds, "Should include zero weight decay"
        assert 1e-2 in wds, "Should include high weight decay 1e-2"

    def test_can_filter_to_specific_budgets(self):
        """Test can generate configs for subset of budgets."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs(budgets=["2M", "20M"])
        budgets_in_configs = set(c.get("budget", c.get("param_budget")) for c in configs)
        # Should only have 2M and 20M (some configs may not have budget field explicitly)
        assert "750k" not in budgets_in_configs or configs[0].get("budget") in ["2M", "20M"]

    def test_configs_have_descriptive_names(self):
        """Test each config has a descriptive name field."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs()
        for i, config in enumerate(configs):
            assert "name" in config, f"Config {i} missing name field"
            assert len(config["name"]) > 0, f"Config {i} has empty name"


class TestCheckEarlyStoppingConvergence:
    """Tests for check_early_stopping_convergence() function."""

    def test_returns_false_with_few_trials(self):
        """Test returns False when fewer than min_trials completed."""
        from src.training.hpo_budget_extremes import check_early_stopping_convergence

        # Mock study with only 5 trials
        mock_study = MagicMock()
        mock_study.trials = [MagicMock(state=MagicMock(name="COMPLETE"), value=-0.65) for _ in range(5)]

        result = check_early_stopping_convergence(
            mock_study, top_n=5, threshold=0.02, min_trials=20
        )
        assert result is False, "Should return False with fewer than min_trials"

    def test_returns_false_when_top_n_not_converged(self):
        """Test returns False when top-N trials differ by more than threshold."""
        from src.training.hpo_budget_extremes import check_early_stopping_convergence

        # Mock study with divergent top trials
        mock_trials = []
        values = [-0.70, -0.65, -0.60, -0.55, -0.50]  # Range > 0.02
        for v in values:
            trial = MagicMock()
            trial.state = MagicMock()
            trial.state.name = "COMPLETE"
            trial.value = v
            mock_trials.append(trial)

        # Add more trials to meet min_trials
        for _ in range(20):
            trial = MagicMock()
            trial.state = MagicMock()
            trial.state.name = "COMPLETE"
            trial.value = -0.45
            mock_trials.append(trial)

        mock_study = MagicMock()
        mock_study.trials = mock_trials

        result = check_early_stopping_convergence(
            mock_study, top_n=5, threshold=0.02, min_trials=20
        )
        assert result is False, "Should return False when top-5 not converged"

    def test_returns_true_when_converged(self):
        """Test returns True when top-N trials are within threshold."""
        from src.training.hpo_budget_extremes import check_early_stopping_convergence

        # Mock study with converged top trials
        mock_trials = []
        # Top 5 trials all within 0.02 of each other
        top_values = [-0.70, -0.695, -0.69, -0.685, -0.68]  # Range = 0.02
        for v in top_values:
            trial = MagicMock()
            trial.state = MagicMock()
            trial.state.name = "COMPLETE"
            trial.value = v
            mock_trials.append(trial)

        # Add more trials to meet min_trials
        for _ in range(20):
            trial = MagicMock()
            trial.state = MagicMock()
            trial.state.name = "COMPLETE"
            trial.value = -0.50
            mock_trials.append(trial)

        mock_study = MagicMock()
        mock_study.trials = mock_trials

        result = check_early_stopping_convergence(
            mock_study, top_n=5, threshold=0.02, min_trials=20
        )
        assert result is True, "Should return True when top-5 converged within threshold"


class TestDefaultRegularization:
    """Tests for DEFAULT_REGULARIZATION dictionary."""

    def test_default_regularization_defined(self):
        """Test DEFAULT_REGULARIZATION is defined with expected values."""
        from src.training.hpo_budget_extremes import DEFAULT_REGULARIZATION

        assert DEFAULT_REGULARIZATION["dropout"] == 0.5
        assert DEFAULT_REGULARIZATION["learning_rate"] == 1e-4
        assert DEFAULT_REGULARIZATION["weight_decay"] == 1e-3


class TestForcedConfigGroups:
    """Tests for specific forced config groups from the plan."""

    def test_group1_budget_x_architecture_8_configs(self):
        """Test Group 1 produces 8 configs (4 budgets × 2 arch styles)."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs()
        # Filter to configs that use default regularization
        group1_configs = [
            c for c in configs
            if c.get("group") == "budget_architecture" or (
                c["dropout"] == 0.5 and
                c["learning_rate"] == 1e-4 and
                c["weight_decay"] == 1e-3
            )
        ]
        assert len(group1_configs) >= 8, f"Expected at least 8 Group 1 configs, got {len(group1_configs)}"

    def test_group2_dropout_extremes_configs(self):
        """Test Group 2 produces dropout extreme configs."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs()
        dropout_extremes = [c for c in configs if c["dropout"] in [0.1, 0.3, 0.7]]
        assert len(dropout_extremes) >= 3, f"Expected at least 3 dropout extreme configs"

    def test_group3_lr_extremes_configs(self):
        """Test Group 3 produces learning rate extreme configs."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs()
        lr_extremes = [c for c in configs if c["learning_rate"] in [1e-5, 1e-3]]
        assert len(lr_extremes) >= 2, f"Expected at least 2 LR extreme configs"

    def test_group4_wd_extremes_configs(self):
        """Test Group 4 produces weight decay extreme configs."""
        from src.training.hpo_budget_extremes import generate_forced_configs

        configs = generate_forced_configs()
        wd_extremes = [c for c in configs if c["weight_decay"] in [0.0, 1e-2]]
        assert len(wd_extremes) >= 2, f"Expected at least 2 weight decay extreme configs"


class TestBudgetEstimation:
    """Tests for parameter budget estimation helpers."""

    def test_estimate_params_function_exists(self):
        """Test estimate_params helper function is defined."""
        from src.training.hpo_budget_extremes import estimate_params

        # Should return approximate parameter count
        result = estimate_params(d_model=256, n_layers=4)
        assert isinstance(result, (int, float))
        assert result > 0

    def test_estimate_params_scales_with_d_model(self):
        """Test parameter estimate scales quadratically with d_model."""
        from src.training.hpo_budget_extremes import estimate_params

        params_128 = estimate_params(d_model=128, n_layers=4)
        params_256 = estimate_params(d_model=256, n_layers=4)
        # 256^2 / 128^2 = 4, so params should roughly 4x
        ratio = params_256 / params_128
        assert 3.0 < ratio < 5.0, f"Expected ~4x scaling, got {ratio:.2f}x"

    def test_estimate_params_scales_with_n_layers(self):
        """Test parameter estimate scales linearly with n_layers."""
        from src.training.hpo_budget_extremes import estimate_params

        params_2 = estimate_params(d_model=256, n_layers=2)
        params_4 = estimate_params(d_model=256, n_layers=4)
        # Should roughly 2x
        ratio = params_4 / params_2
        assert 1.5 < ratio < 2.5, f"Expected ~2x scaling, got {ratio:.2f}x"
