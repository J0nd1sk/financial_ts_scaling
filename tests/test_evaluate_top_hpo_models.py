"""Tests for evaluate_top_hpo_models script."""

import json
import pytest
from pathlib import Path


class TestLoadAllTrials:
    """Tests for load_all_trials function."""

    def test_load_all_trials_current_format(self, tmp_path):
        """Test that load_all_trials finds current directory format."""
        from scripts.evaluate_top_hpo_models import load_all_trials

        # Patch PROJECT_ROOT to use tmp_path
        import scripts.evaluate_top_hpo_models as module
        original_root = module.PROJECT_ROOT
        module.PROJECT_ROOT = tmp_path

        try:
            # Create test all_trials.json with current format: hpo_2m_1_a50
            trials_data = [
                {"number": 0, "value": 0.72, "state": "TrialState.COMPLETE", "params": {"d_model": 64}},
                {"number": 1, "value": 0.73, "state": "TrialState.COMPLETE", "params": {"d_model": 96}},
            ]
            output_dir = tmp_path / "outputs" / "phase6c_a50" / "hpo_2m_1_a50"
            output_dir.mkdir(parents=True)
            with open(output_dir / "all_trials.json", "w") as f:
                json.dump(trials_data, f)

            result = load_all_trials("a50", "2M", 1)

            assert len(result) == 2
            assert result[0]["value"] == 0.72
            assert result[1]["value"] == 0.73
        finally:
            module.PROJECT_ROOT = original_root

    def test_load_all_trials_missing(self, tmp_path):
        """Test that load_all_trials returns empty list for missing file."""
        from scripts.evaluate_top_hpo_models import load_all_trials

        import scripts.evaluate_top_hpo_models as module
        original_root = module.PROJECT_ROOT
        module.PROJECT_ROOT = tmp_path

        try:
            result = load_all_trials("a50", "2M", 1)
            assert result == []
        finally:
            module.PROJECT_ROOT = original_root


class TestEvaluateTopModels:
    """Tests for evaluate_top_models function."""

    def test_evaluate_top_models_dry_run(self, tmp_path):
        """Test that evaluate_top_models returns results in dry_run mode."""
        from scripts.evaluate_top_hpo_models import evaluate_top_models

        import scripts.evaluate_top_hpo_models as module
        original_root = module.PROJECT_ROOT
        module.PROJECT_ROOT = tmp_path

        try:
            # Create test trials for one budget
            trials_data = [
                {"number": 0, "value": 0.72, "state": "TrialState.COMPLETE",
                 "params": {"d_model": 64, "n_layers": 3, "n_heads": 4, "dropout": 0.3}},
                {"number": 1, "value": 0.73, "state": "TrialState.COMPLETE",
                 "params": {"d_model": 96, "n_layers": 4, "n_heads": 8, "dropout": 0.3}},
                {"number": 2, "value": 0.71, "state": "TrialState.COMPLETE",
                 "params": {"d_model": 48, "n_layers": 2, "n_heads": 4, "dropout": 0.3}},
            ]
            output_dir = tmp_path / "outputs" / "phase6c_a50" / "hpo_2m_1_a50"
            output_dir.mkdir(parents=True)
            with open(output_dir / "all_trials.json", "w") as f:
                json.dump(trials_data, f)

            results = evaluate_top_models("a50", horizon=1, top_n=2, dry_run=True)

            assert results["tier"] == "a50"
            assert results["horizon"] == 1
            assert results["top_n"] == 2
            assert "2M" in results["results_by_budget"]
            # Should have top 2 models
            budget_results = results["results_by_budget"]["2M"]
            assert isinstance(budget_results, list)
            assert len(budget_results) == 2
            # Top model should be trial 1 (AUC 0.73)
            assert budget_results[0]["trial_number"] == 1
            assert budget_results[0]["original_auc"] == 0.73
        finally:
            module.PROJECT_ROOT = original_root


class TestGenerateSummaryReport:
    """Tests for generate_summary_report function."""

    def test_report_format(self):
        """Test that generate_summary_report produces valid markdown."""
        from scripts.evaluate_top_hpo_models import generate_summary_report

        results = {
            "tier": "a50",
            "horizon": 1,
            "top_n": 2,
            "timestamp": "2026-01-27",
            "results_by_budget": {
                "2M": [
                    {"trial_number": 1, "val_auc": 0.73, "val_precision": 0.65,
                     "val_recall": 0.10, "val_pred_range": (0.4, 0.6)},
                    {"trial_number": 0, "val_auc": 0.72, "val_precision": 0.60,
                     "val_recall": 0.05, "val_pred_range": (0.45, 0.55)},
                ],
                "20M": {"error": "No trials found"},
                "200M": [],
            },
        }

        report = generate_summary_report(results)

        assert isinstance(report, str)
        assert "# Top HPO Models" in report
        assert "a50" in report
        assert "2M" in report
        assert "|" in report  # Has tables
        assert "Key Findings" in report

    def test_report_handles_missing_metrics(self):
        """Test that report handles None/missing metrics gracefully."""
        from scripts.evaluate_top_hpo_models import generate_summary_report

        results = {
            "tier": "a100",
            "horizon": 1,
            "top_n": 1,
            "results_by_budget": {
                "2M": [
                    {"trial_number": 0, "val_auc": None, "val_precision": None,
                     "val_recall": None, "val_pred_range": None},
                ],
            },
        }

        report = generate_summary_report(results)

        assert "N/A" in report  # Missing metrics shown as N/A
