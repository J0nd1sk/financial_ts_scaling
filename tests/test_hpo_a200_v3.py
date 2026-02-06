"""Tests for HPO a200 v3 script with precision-first optimization.

Tests cover:
- Composite objective weighting (precision*2 + recall*1 + auc*0.1)
- Loss type as categorical parameter (focal vs weighted_bce)
- Conditional loss parameters (focal_alpha/gamma vs bce_pos_weight)
- Multi-threshold metrics logging (t30, t40, t50, t60, t70)
- All user_attrs are set correctly
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
def v3_search_space() -> dict:
    """Expected v3 search space with loss parameters."""
    return {
        # Architecture (unchanged)
        "d_model": [64, 96, 128, 160, 192],
        "n_layers": [4, 5, 6, 7, 8],
        "n_heads": [4, 8],
        "d_ff_ratio": [2, 4],
        # Training
        "learning_rate": [5e-5, 7e-5, 1e-4, 1.5e-4],
        "dropout": [0.3, 0.4, 0.5, 0.6],
        "weight_decay": [1e-5, 1e-4, 5e-4, 1e-3],
        # Loss function (NEW in v3)
        "loss_type": ["focal", "weighted_bce"],
        "focal_alpha": [0.3, 0.5, 0.7, 0.9],
        "focal_gamma": [0.0, 0.5, 1.0, 2.0],
        "bce_pos_weight": [1.0, 2.0, 3.0, 5.0],
    }


# =============================================================================
# Tests for Composite Objective Function
# =============================================================================


class TestCompositeObjective:
    """Test composite objective function weights."""

    def test_compute_composite_returns_float(self) -> None:
        """Test that compute_composite_score returns a float."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import compute_composite_score

        result = compute_composite_score(
            precision=0.6, recall=0.1, auc=0.55
        )
        assert isinstance(result, float)

    def test_composite_weights_precision_highest(self) -> None:
        """Test that precision has highest weight (2.0)."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import compute_composite_score

        # High precision, zero recall, zero AUC
        high_prec = compute_composite_score(precision=1.0, recall=0.0, auc=0.0)
        # Zero precision, high recall, zero AUC
        high_rec = compute_composite_score(precision=0.0, recall=1.0, auc=0.0)

        assert high_prec > high_rec, (
            f"Precision weight should be highest: "
            f"high_prec={high_prec}, high_rec={high_rec}"
        )
        # Check actual ratio is ~2:1
        assert high_prec / high_rec == pytest.approx(2.0, rel=0.01)

    def test_composite_weights_recall_second(self) -> None:
        """Test that recall has second highest weight (1.0)."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import compute_composite_score

        # High recall, zero precision, zero AUC
        high_rec = compute_composite_score(precision=0.0, recall=1.0, auc=0.0)
        # Zero recall, zero precision, high AUC
        high_auc = compute_composite_score(precision=0.0, recall=0.0, auc=1.0)

        assert high_rec > high_auc, (
            f"Recall weight should exceed AUC weight: "
            f"high_rec={high_rec}, high_auc={high_auc}"
        )
        # Check actual ratio is ~10:1 (1.0 vs 0.1)
        assert high_rec / high_auc == pytest.approx(10.0, rel=0.01)

    def test_composite_weights_auc_tertiary(self) -> None:
        """Test that AUC has lowest weight (0.1)."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import compute_composite_score

        # With equal precision and recall values
        score = compute_composite_score(precision=0.5, recall=0.5, auc=0.5)

        # Expected: 0.5*2 + 0.5*1 + 0.5*0.1 = 1.0 + 0.5 + 0.05 = 1.55
        expected = (0.5 * 2.0) + (0.5 * 1.0) + (0.5 * 0.1)
        assert score == pytest.approx(expected, rel=0.001)

    def test_composite_formula_correct(self) -> None:
        """Test that composite = precision*2 + recall*1 + auc*0.1."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import compute_composite_score

        # Specific values
        precision, recall, auc = 0.65, 0.12, 0.72
        expected = (precision * 2.0) + (recall * 1.0) + (auc * 0.1)

        result = compute_composite_score(precision=precision, recall=recall, auc=auc)
        assert result == pytest.approx(expected, rel=0.001)


# =============================================================================
# Tests for Search Space with Loss Parameters
# =============================================================================


class TestSearchSpaceV3:
    """Test v3 search space configuration."""

    def test_loss_type_categorical_in_search_space(self) -> None:
        """Test that loss_type is a categorical with focal and weighted_bce."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import SEARCH_SPACE_V3

        assert "loss_type" in SEARCH_SPACE_V3
        assert SEARCH_SPACE_V3["loss_type"] == ["focal", "weighted_bce"]

    def test_focal_alpha_in_search_space(self) -> None:
        """Test that focal_alpha has expected values."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import SEARCH_SPACE_V3

        assert "focal_alpha" in SEARCH_SPACE_V3
        assert SEARCH_SPACE_V3["focal_alpha"] == [0.3, 0.5, 0.7, 0.9]

    def test_focal_gamma_in_search_space(self) -> None:
        """Test that focal_gamma has expected values."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import SEARCH_SPACE_V3

        assert "focal_gamma" in SEARCH_SPACE_V3
        assert SEARCH_SPACE_V3["focal_gamma"] == [0.0, 0.5, 1.0, 2.0]

    def test_bce_pos_weight_in_search_space(self) -> None:
        """Test that bce_pos_weight has expected values."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import SEARCH_SPACE_V3

        assert "bce_pos_weight" in SEARCH_SPACE_V3
        assert SEARCH_SPACE_V3["bce_pos_weight"] == [1.0, 2.0, 3.0, 5.0]


# =============================================================================
# Tests for Conditional Loss Parameters
# =============================================================================


class TestConditionalLossParams:
    """Test that loss parameters are conditional on loss_type."""

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.pd.read_parquet")
    def test_focal_params_used_when_loss_type_focal(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that focal_alpha and focal_gamma are used when loss_type=focal."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import objective

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

        mock_trial = MagicMock()
        mock_trial.number = 0

        # Return values for suggest_categorical
        def suggest_side_effect(name, choices):
            if name == "loss_type":
                return "focal"
            elif name == "focal_alpha":
                return 0.5
            elif name == "focal_gamma":
                return 1.0
            return choices[0]

        mock_trial.suggest_categorical.side_effect = suggest_side_effect

        objective(mock_trial)

        # Verify focal params were suggested
        suggest_cat_params = {
            call[0][0] for call in mock_trial.suggest_categorical.call_args_list
        }
        assert "focal_alpha" in suggest_cat_params
        assert "focal_gamma" in suggest_cat_params
        assert "bce_pos_weight" not in suggest_cat_params

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.pd.read_parquet")
    def test_bce_pos_weight_used_when_loss_type_weighted_bce(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that bce_pos_weight is used when loss_type=weighted_bce."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import objective

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

        mock_trial = MagicMock()
        mock_trial.number = 0

        def suggest_side_effect(name, choices):
            if name == "loss_type":
                return "weighted_bce"
            elif name == "bce_pos_weight":
                return 2.0
            return choices[0]

        mock_trial.suggest_categorical.side_effect = suggest_side_effect

        objective(mock_trial)

        suggest_cat_params = {
            call[0][0] for call in mock_trial.suggest_categorical.call_args_list
        }
        assert "bce_pos_weight" in suggest_cat_params
        assert "focal_alpha" not in suggest_cat_params
        assert "focal_gamma" not in suggest_cat_params


# =============================================================================
# Tests for Multi-Threshold Metrics Logging
# =============================================================================


class TestMultiThresholdMetrics:
    """Test that metrics are logged at multiple thresholds."""

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.pd.read_parquet")
    def test_multi_threshold_metrics_logged(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that precision/recall at thresholds [0.3, 0.4, 0.5, 0.6, 0.7] are logged."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import objective

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
            # Multi-threshold metrics
            "val_precision_t30": 0.52,
            "val_recall_t30": 0.25,
            "val_precision_t40": 0.58,
            "val_recall_t40": 0.18,
            "val_precision_t50": 0.60,
            "val_recall_t50": 0.10,
            "val_precision_t60": 0.68,
            "val_recall_t60": 0.05,
            "val_precision_t70": 0.80,
            "val_recall_t70": 0.02,
        }
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

        objective(mock_trial)

        call_dict = {
            call[0][0]: call[0][1]
            for call in mock_trial.set_user_attr.call_args_list
        }

        # Verify threshold metrics are logged
        for t in ["t30", "t40", "t50", "t60", "t70"]:
            assert f"val_precision_{t}" in call_dict, f"Missing val_precision_{t}"
            assert f"val_recall_{t}" in call_dict, f"Missing val_recall_{t}"

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.pd.read_parquet")
    def test_all_threshold_user_attrs_set(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that all expected threshold user attrs are present."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import objective

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
            "val_precision_t30": 0.52,
            "val_recall_t30": 0.25,
            "val_precision_t40": 0.58,
            "val_recall_t40": 0.18,
            "val_precision_t50": 0.60,
            "val_recall_t50": 0.10,
            "val_precision_t60": 0.68,
            "val_recall_t60": 0.05,
            "val_precision_t70": 0.80,
            "val_recall_t70": 0.02,
        }
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

        objective(mock_trial)

        call_names = {call[0][0] for call in mock_trial.set_user_attr.call_args_list}

        expected_attrs = [
            "val_precision", "val_recall", "val_auc",
            "composite_score", "pred_min", "pred_max",
            "val_precision_t30", "val_recall_t30",
            "val_precision_t40", "val_recall_t40",
            "val_precision_t50", "val_recall_t50",
            "val_precision_t60", "val_recall_t60",
            "val_precision_t70", "val_recall_t70",
        ]

        for attr in expected_attrs:
            assert attr in call_names, f"Missing user_attr: {attr}"


# =============================================================================
# Tests for Objective Returns Composite Score
# =============================================================================


class TestObjectiveReturnsComposite:
    """Test that objective function returns composite score, not AUC."""

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.pd.read_parquet")
    def test_objective_returns_composite_not_auc(
        self,
        mock_read_parquet: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that objective returns composite score, not raw AUC."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import (
            objective,
            compute_composite_score,
        )

        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(values=[1.0] * 1000))
        mock_read_parquet.return_value = mock_df

        mock_splitter = MagicMock()
        mock_splitter.split.return_value = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        precision, recall, auc = 0.65, 0.12, 0.72
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "val_auc": auc,
            "val_precision": precision,
            "val_recall": recall,
            "val_pred_range": (0.3, 0.7),
            "val_precision_t30": 0.5,
            "val_recall_t30": 0.2,
            "val_precision_t40": 0.5,
            "val_recall_t40": 0.2,
            "val_precision_t50": precision,
            "val_recall_t50": recall,
            "val_precision_t60": 0.7,
            "val_recall_t60": 0.05,
            "val_precision_t70": 0.8,
            "val_recall_t70": 0.02,
        }
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.number = 0
        mock_trial.suggest_categorical.side_effect = lambda name, choices: choices[0]

        result = objective(mock_trial)

        expected_composite = compute_composite_score(
            precision=precision, recall=recall, auc=auc
        )

        assert result == pytest.approx(expected_composite, rel=0.001)
        assert result != auc  # Explicitly verify it's not raw AUC


# =============================================================================
# Tests for Script Configuration
# =============================================================================


class TestScriptConfigurationV3:
    """Test script-level configuration values."""

    def test_experiment_name_is_v3(self) -> None:
        """Test EXPERIMENT_NAME contains v3."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import EXPERIMENT_NAME

        assert "v3" in EXPERIMENT_NAME

    def test_n_trials_is_50(self) -> None:
        """Test N_TRIALS = 50."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import N_TRIALS

        assert N_TRIALS == 50

    def test_optimization_direction_is_maximize(self) -> None:
        """Test that study is configured to maximize (composite score)."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import DIRECTION

        assert DIRECTION == "maximize"

    def test_context_length_is_80(self) -> None:
        """Test CONTEXT_LENGTH = 80 (standard, from CLAUDE.md)."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import CONTEXT_LENGTH

        assert CONTEXT_LENGTH == 80


# =============================================================================
# Tests for Loss Function Instantiation
# =============================================================================


class TestLossFunctionInstantiation:
    """Test that correct loss function is instantiated based on loss_type."""

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.FocalLoss")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.pd.read_parquet")
    def test_focal_loss_instantiated_with_params(
        self,
        mock_read_parquet: MagicMock,
        mock_focal_loss_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that FocalLoss is instantiated with alpha and gamma when loss_type=focal."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import objective

        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(values=[1.0] * 1000))
        mock_read_parquet.return_value = mock_df

        mock_splitter = MagicMock()
        mock_splitter.split.return_value = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        mock_focal = MagicMock()
        mock_focal_loss_cls.return_value = mock_focal

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "val_auc": 0.55,
            "val_precision": 0.6,
            "val_recall": 0.1,
            "val_pred_range": (0.3, 0.7),
            "val_precision_t30": 0.5, "val_recall_t30": 0.2,
            "val_precision_t40": 0.5, "val_recall_t40": 0.2,
            "val_precision_t50": 0.6, "val_recall_t50": 0.1,
            "val_precision_t60": 0.7, "val_recall_t60": 0.05,
            "val_precision_t70": 0.8, "val_recall_t70": 0.02,
        }
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.number = 0

        def suggest_side_effect(name, choices):
            if name == "loss_type":
                return "focal"
            elif name == "focal_alpha":
                return 0.7
            elif name == "focal_gamma":
                return 1.5
            return choices[0]

        mock_trial.suggest_categorical.side_effect = suggest_side_effect

        objective(mock_trial)

        # Verify FocalLoss was instantiated with correct params
        mock_focal_loss_cls.assert_called_once()
        call_kwargs = mock_focal_loss_cls.call_args[1]
        assert call_kwargs.get("alpha") == 0.7
        assert call_kwargs.get("gamma") == 1.5

    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.SimpleSplitter")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.Trainer")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.WeightedBCELoss")
    @patch("experiments.phase6c_a200.hpo_20m_h1_a200_v3.pd.read_parquet")
    def test_weighted_bce_instantiated_with_pos_weight(
        self,
        mock_read_parquet: MagicMock,
        mock_weighted_bce_cls: MagicMock,
        mock_trainer_cls: MagicMock,
        mock_splitter_cls: MagicMock,
    ) -> None:
        """Test that WeightedBCELoss is instantiated with pos_weight when loss_type=weighted_bce."""
        from experiments.phase6c_a200.hpo_20m_h1_a200_v3 import objective

        mock_df = MagicMock()
        mock_df.__getitem__ = MagicMock(return_value=MagicMock(values=[1.0] * 1000))
        mock_read_parquet.return_value = mock_df

        mock_splitter = MagicMock()
        mock_splitter.split.return_value = MagicMock()
        mock_splitter_cls.return_value = mock_splitter

        mock_weighted = MagicMock()
        mock_weighted_bce_cls.return_value = mock_weighted

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = {
            "val_auc": 0.55,
            "val_precision": 0.6,
            "val_recall": 0.1,
            "val_pred_range": (0.3, 0.7),
            "val_precision_t30": 0.5, "val_recall_t30": 0.2,
            "val_precision_t40": 0.5, "val_recall_t40": 0.2,
            "val_precision_t50": 0.6, "val_recall_t50": 0.1,
            "val_precision_t60": 0.7, "val_recall_t60": 0.05,
            "val_precision_t70": 0.8, "val_recall_t70": 0.02,
        }
        mock_trainer_cls.return_value = mock_trainer

        mock_trial = MagicMock()
        mock_trial.number = 0

        def suggest_side_effect(name, choices):
            if name == "loss_type":
                return "weighted_bce"
            elif name == "bce_pos_weight":
                return 3.0
            return choices[0]

        mock_trial.suggest_categorical.side_effect = suggest_side_effect

        objective(mock_trial)

        # Verify WeightedBCELoss was instantiated with correct pos_weight
        mock_weighted_bce_cls.assert_called_once()
        call_kwargs = mock_weighted_bce_cls.call_args[1]
        assert call_kwargs.get("pos_weight") == 3.0
