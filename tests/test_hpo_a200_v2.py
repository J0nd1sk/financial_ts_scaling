"""Tests for HPO a200 v2 script with forced extremes and coverage tracking.

Tests cover:
- Forced extreme configuration generation (6 configs)
- Objective function behavior (forced vs TPE trials)
- Expanded search space values
- pred_range and metric logging
- CoverageTracker integration
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is in path for experiments imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def v2_search_space() -> dict:
    """Expected v2 search space with expanded ranges."""
    return {
        # Architecture (unchanged from v1)
        "d_model": [64, 96, 128, 160, 192],
        "n_layers": [4, 5, 6, 7, 8],
        "n_heads": [4, 8],
        "d_ff_ratio": [2, 4],
        # Training - EXPANDED
        "learning_rate": [5e-5, 7e-5, 8e-5, 9e-5, 1e-4, 1.5e-4],
        "dropout": [0.3, 0.4, 0.5, 0.6, 0.7],
        "weight_decay": [1e-5, 1e-4, 3e-4, 5e-4, 1e-3],
    }


# =============================================================================
# Tests for Forced Extreme Configuration
# =============================================================================


class TestComputeForcedExtremes:
    """Test forced extreme configuration generation."""

    def test_compute_forced_extremes_returns_6_configs(
        self, v2_search_space: dict
    ) -> None:
        """Test that compute_forced_extreme_configs returns exactly 6 configs."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import (
            compute_forced_extreme_configs,
        )

        configs = compute_forced_extreme_configs(v2_search_space)
        assert len(configs) == 6, f"Expected 6 forced configs, got {len(configs)}"

    def test_forced_extremes_all_valid(self, v2_search_space: dict) -> None:
        """Test that all 6 forced configs have d_model % n_heads == 0."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import (
            compute_forced_extreme_configs,
        )

        configs = compute_forced_extreme_configs(v2_search_space)
        for i, config in enumerate(configs):
            d_model = config["d_model"]
            n_heads = config["n_heads"]
            assert d_model % n_heads == 0, (
                f"Config {i} invalid: d_model={d_model} not divisible by n_heads={n_heads}"
            )

    def test_forced_extremes_cover_bounds(self, v2_search_space: dict) -> None:
        """Test that extremes include min/max of each architecture dimension."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import (
            compute_forced_extreme_configs,
        )

        configs = compute_forced_extreme_configs(v2_search_space)

        # Extract values from configs
        d_models = [c["d_model"] for c in configs]
        n_layers_list = [c["n_layers"] for c in configs]
        n_heads_list = [c["n_heads"] for c in configs]

        # Check d_model bounds
        assert min(v2_search_space["d_model"]) in d_models, "Missing min d_model"
        assert max(v2_search_space["d_model"]) in d_models, "Missing max d_model"

        # Check n_layers bounds
        assert min(v2_search_space["n_layers"]) in n_layers_list, "Missing min n_layers"
        assert max(v2_search_space["n_layers"]) in n_layers_list, "Missing max n_layers"

        # Check n_heads bounds
        assert min(v2_search_space["n_heads"]) in n_heads_list, "Missing min n_heads"
        assert max(v2_search_space["n_heads"]) in n_heads_list, "Missing max n_heads"

    def test_forced_extremes_have_required_keys(self, v2_search_space: dict) -> None:
        """Test that each forced config has d_model, n_layers, n_heads keys."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import (
            compute_forced_extreme_configs,
        )

        configs = compute_forced_extreme_configs(v2_search_space)
        required_keys = {"d_model", "n_layers", "n_heads"}

        for i, config in enumerate(configs):
            assert required_keys <= set(config.keys()), (
                f"Config {i} missing keys: {required_keys - set(config.keys())}"
            )


# =============================================================================
# Tests for Objective Function Behavior
# =============================================================================


class TestObjectiveForcedTrials:
    """Test objective function behavior for forced extreme trials (0-5)."""

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.pd.read_parquet")
    def test_forced_trial_uses_set_user_attr(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that trials 0-5 use set_user_attr for arch params, not suggest_categorical."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import objective

        # Setup mocks
        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(values=[1.0] * 1000))
        mock_read_parquet.return_value = mock_df

        mock_splitter = MagicMock()
        mock_splitter.split.return_value = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "val_auc": 0.55,
            "val_precision": 0.6,
            "val_recall": 0.1,
            "val_pred_range": (0.3, 0.7),
        }
        mock_trainer_cls.return_value = mock_trainer

        # Create mock trial for forced extreme (trial 0)
        mock_trial = MagicMock()
        mock_trial.number = 0  # Forced extreme trial
        mock_trial.suggest_categorical.return_value = 1e-4  # For training params

        # Run objective
        objective(mock_trial)

        # Verify set_user_attr was called for architecture params
        set_user_attr_calls = {
            call[0][0] for call in mock_trial.set_user_attr.call_args_list
        }
        assert "forced_d_model" in set_user_attr_calls
        assert "forced_n_layers" in set_user_attr_calls
        assert "forced_n_heads" in set_user_attr_calls
        assert "forced_extreme" in set_user_attr_calls

        # Verify suggest_categorical was NOT called for d_model/n_layers/n_heads
        suggest_cat_params = {
            call[0][0] for call in mock_trial.suggest_categorical.call_args_list
        }
        assert "d_model" not in suggest_cat_params
        assert "n_layers" not in suggest_cat_params
        assert "n_heads" not in suggest_cat_params

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.pd.read_parquet")
    def test_forced_trial_5_is_last_forced(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that trial 5 is still a forced extreme trial."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import objective

        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(values=[1.0] * 1000))
        mock_read_parquet.return_value = mock_df

        mock_splitter = MagicMock()
        mock_splitter.split.return_value = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "val_auc": 0.55,
            "val_pred_range": (0.3, 0.7),
        }
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.number = 5  # Last forced extreme trial
        mock_trial.suggest_categorical.return_value = 1e-4

        objective(mock_trial)

        # Verify set_user_attr was called (forced trial)
        set_user_attr_calls = {
            call[0][0] for call in mock_trial.set_user_attr.call_args_list
        }
        assert "forced_extreme" in set_user_attr_calls


class TestObjectiveTPETrials:
    """Test objective function behavior for TPE trials (6+)."""

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.CoverageTracker")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.pd.read_parquet")
    def test_tpe_trial_uses_suggest_categorical(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_coverage_tracker_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that trials 6+ use suggest_categorical for arch params."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import objective

        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(values=[1.0] * 1000))
        mock_read_parquet.return_value = mock_df

        mock_splitter = MagicMock()
        mock_splitter.split.return_value = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "val_auc": 0.55,
            "val_pred_range": (0.3, 0.7),
        }
        mock_trainer_cls.return_value = mock_trainer

        # Mock CoverageTracker to return config unchanged
        mock_tracker = MagicMock()
        mock_tracker.suggest_coverage_config.side_effect = lambda x: x
        mock_coverage_tracker_cls.from_study.return_value = mock_tracker

        mock_trial = MagicMock()
        mock_trial.number = 6  # First TPE trial
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        mock_trial.study = MagicMock()

        objective(mock_trial)

        # Verify suggest_categorical was called for d_model/n_layers/n_heads
        suggest_cat_params = {
            call[0][0] for call in mock_trial.suggest_categorical.call_args_list
        }
        assert "d_model" in suggest_cat_params
        assert "n_layers" in suggest_cat_params
        assert "n_heads" in suggest_cat_params

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.CoverageTracker")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.pd.read_parquet")
    def test_coverage_redirect_on_duplicate(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_coverage_tracker_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that TPE trial is redirected if architecture already tested."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import objective

        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(values=[1.0] * 1000))
        mock_read_parquet.return_value = mock_df

        mock_splitter = MagicMock()
        mock_splitter.split.return_value = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "val_auc": 0.55,
            "val_pred_range": (0.3, 0.7),
        }
        mock_trainer_cls.return_value = mock_trainer

        # Mock CoverageTracker to redirect config
        mock_tracker = MagicMock()

        def redirect_config(proposed):
            # Simulate redirect: change d_model from 64 to 96
            result = proposed.copy()
            if result["d_model"] == 64:
                result["d_model"] = 96
            return result

        mock_tracker.suggest_coverage_config.side_effect = redirect_config
        mock_coverage_tracker_cls.from_study.return_value = mock_tracker

        mock_trial = MagicMock()
        mock_trial.number = 10  # TPE trial
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]
        mock_trial.study = MagicMock()

        objective(mock_trial)

        # Verify coverage_redirect was set in user_attrs
        set_user_attr_calls = {
            call[0][0] for call in mock_trial.set_user_attr.call_args_list
        }
        assert "coverage_redirect" in set_user_attr_calls


class TestObjectiveMetricsLogging:
    """Test that objective logs metrics correctly."""

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.pd.read_parquet")
    def test_pred_range_logged_in_user_attrs(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that trial.user_attrs contains pred_min/pred_max after training."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import objective

        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(values=[1.0] * 1000))
        mock_read_parquet.return_value = mock_df

        mock_splitter = MagicMock()
        mock_splitter.split.return_value = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "val_auc": 0.55,
            "val_precision": 0.6,
            "val_recall": 0.1,
            "val_pred_range": (0.25, 0.75),
        }
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.suggest_categorical.return_value = 1e-4

        objective(mock_trial)

        # Verify pred_range was logged
        set_user_attr_calls = dict(mock_trial.set_user_attr.call_args_list)
        call_dict = {call[0][0]: call[0][1] for call in mock_trial.set_user_attr.call_args_list}

        assert "pred_min" in call_dict, "pred_min not logged"
        assert "pred_max" in call_dict, "pred_max not logged"
        assert call_dict["pred_min"] == 0.25
        assert call_dict["pred_max"] == 0.75

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.pd.read_parquet")
    def test_precision_recall_logged(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that trial.user_attrs contains val_precision/val_recall."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import objective

        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(values=[1.0] * 1000))
        mock_read_parquet.return_value = mock_df

        mock_splitter = MagicMock()
        mock_splitter.split.return_value = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "val_auc": 0.55,
            "val_precision": 0.65,
            "val_recall": 0.12,
            "val_pred_range": (0.3, 0.7),
        }
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.suggest_categorical.return_value = 1e-4

        objective(mock_trial)

        call_dict = {call[0][0]: call[0][1] for call in mock_trial.set_user_attr.call_args_list}

        assert "val_precision" in call_dict, "val_precision not logged"
        assert "val_recall" in call_dict, "val_recall not logged"
        assert call_dict["val_precision"] == 0.65
        assert call_dict["val_recall"] == 0.12


# =============================================================================
# Tests for Expanded Search Space
# =============================================================================


class TestExpandedSearchSpace:
    """Test that search space has expanded values."""

    def test_dropout_has_five_values(self) -> None:
        """Test dropout has [0.3, 0.4, 0.5, 0.6, 0.7]."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import SEARCH_SPACE_V2

        expected = [0.3, 0.4, 0.5, 0.6, 0.7]
        assert SEARCH_SPACE_V2["dropout"] == expected, (
            f"Expected dropout={expected}, got {SEARCH_SPACE_V2['dropout']}"
        )

    def test_learning_rate_has_six_values(self) -> None:
        """Test learning_rate has [5e-5, 7e-5, 8e-5, 9e-5, 1e-4, 1.5e-4]."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import SEARCH_SPACE_V2

        expected = [5e-5, 7e-5, 8e-5, 9e-5, 1e-4, 1.5e-4]
        assert SEARCH_SPACE_V2["learning_rate"] == expected, (
            f"Expected learning_rate={expected}, got {SEARCH_SPACE_V2['learning_rate']}"
        )

    def test_weight_decay_has_five_values(self) -> None:
        """Test weight_decay has [1e-5, 1e-4, 3e-4, 5e-4, 1e-3]."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import SEARCH_SPACE_V2

        expected = [1e-5, 1e-4, 3e-4, 5e-4, 1e-3]
        assert SEARCH_SPACE_V2["weight_decay"] == expected, (
            f"Expected weight_decay={expected}, got {SEARCH_SPACE_V2['weight_decay']}"
        )

    def test_context_length_is_75(self) -> None:
        """Test CONTEXT_LENGTH = 75 (not 80, per ablation finding)."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import CONTEXT_LENGTH

        assert CONTEXT_LENGTH == 75, f"Expected CONTEXT_LENGTH=75, got {CONTEXT_LENGTH}"


# =============================================================================
# Tests for Valid Architecture Combinations
# =============================================================================


class TestValidArchCombos:
    """Test valid architecture combination counting."""

    def test_valid_arch_combos_count(self, v2_search_space: dict) -> None:
        """Test that valid (d_model, n_layers, n_heads) combos is ~45."""
        from src.training.hpo_coverage import CoverageTracker

        tracker = CoverageTracker(v2_search_space)
        valid_count = len(tracker._valid_combos)

        # d_model: 5 values, n_layers: 5 values, n_heads: 2 values
        # All d_models (64, 96, 128, 160, 192) are divisible by 4
        # Only 64, 128, 192 are divisible by 8
        # Valid combos: 5*5*1 (all d_models with n_heads=4) + 3*5*1 (some with n_heads=8)
        # = 25 + 15 = 40 (but this is approximate)
        assert 35 <= valid_count <= 50, (
            f"Expected ~40-45 valid combos, got {valid_count}"
        )

    def test_forced_plus_tpe_can_cover_space(self, v2_search_space: dict) -> None:
        """Test that 6 forced + 44 TPE trials can achieve good coverage."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import (
            compute_forced_extreme_configs,
            N_TRIALS,
        )
        from src.training.hpo_coverage import CoverageTracker

        forced_configs = compute_forced_extreme_configs(v2_search_space)
        tracker = CoverageTracker(v2_search_space)
        valid_count = len(tracker._valid_combos)

        # 6 forced + remaining TPE trials
        tpe_trials = N_TRIALS - len(forced_configs)

        # Total trials should be able to cover most valid combos
        assert tpe_trials >= valid_count - len(forced_configs), (
            f"Not enough TPE trials ({tpe_trials}) to cover remaining combos"
        )


# =============================================================================
# Tests for Script Configuration
# =============================================================================


class TestScriptConfiguration:
    """Test script-level configuration values."""

    def test_n_trials_is_50(self) -> None:
        """Test N_TRIALS = 50."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import N_TRIALS

        assert N_TRIALS == 50

    def test_horizon_is_1(self) -> None:
        """Test HORIZON = 1."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import HORIZON

        assert HORIZON == 1

    def test_budget_is_20m(self) -> None:
        """Test BUDGET = '20M'."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import BUDGET

        assert BUDGET == "20M"

    def test_num_features_correct(self) -> None:
        """Test NUM_FEATURES = 211 (206 indicators + 5 OHLCV)."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import NUM_FEATURES

        assert NUM_FEATURES == 211


# =============================================================================
# Tests for Extreme Type Labels
# =============================================================================


class TestExtremeTypeLabels:
    """Test that forced extreme trials have correct type labels."""

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v2.pd.read_parquet")
    def test_extreme_type_labels_set(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that extreme_type is set for forced trials."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v2 import objective

        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(values=[1.0] * 1000))
        mock_read_parquet.return_value = mock_df

        mock_splitter = MagicMock()
        mock_splitter.split.return_value = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "val_auc": 0.55,
            "val_pred_range": (0.3, 0.7),
        }
        mock_trainer_cls.return_value = mock_trainer

        expected_types = [
            "min_d_model",
            "max_d_model",
            "min_n_layers",
            "max_n_layers",
            "min_n_heads",
            "max_n_heads",
        ]

        for trial_num in range(6):
            mock_trial = MagicMock()
            mock_trial.number = trial_num
            mock_trial.suggest_categorical.return_value = 1e-4

            objective(mock_trial)

            call_dict = {
                call[0][0]: call[0][1]
                for call in mock_trial.set_user_attr.call_args_list
            }
            assert "extreme_type" in call_dict, f"Trial {trial_num} missing extreme_type"
            assert call_dict["extreme_type"] == expected_types[trial_num], (
                f"Trial {trial_num}: expected {expected_types[trial_num]}, "
                f"got {call_dict['extreme_type']}"
            )
