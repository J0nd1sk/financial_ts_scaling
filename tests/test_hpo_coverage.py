"""Tests for HPO coverage tracking module."""

import pytest
from unittest.mock import MagicMock, Mock


class TestCoverageTracker:
    """Tests for CoverageTracker class."""

    def test_record_config_updates_tested_set(self):
        """Test that record_config adds architecture combo to tested set."""
        from src.training.hpo_coverage import CoverageTracker

        search_space = {
            "d_model": [32, 64, 96],
            "n_layers": [2, 3, 4],
            "n_heads": [2, 4, 8],
        }
        tracker = CoverageTracker(search_space)

        config = {"d_model": 64, "n_layers": 3, "n_heads": 4}
        tracker.record_config(config)

        # Check that the combo was recorded
        untested = tracker.get_untested_arch_combos()
        assert (64, 3, 4) not in untested

    def test_get_untested_combos_correct(self):
        """Test that get_untested_arch_combos returns set difference."""
        from src.training.hpo_coverage import CoverageTracker

        search_space = {
            "d_model": [32, 64],
            "n_layers": [2, 3],
            "n_heads": [4],  # Only valid if d_model % 4 == 0
        }
        tracker = CoverageTracker(search_space)

        # Record one config
        tracker.record_config({"d_model": 32, "n_layers": 2, "n_heads": 4})

        untested = tracker.get_untested_arch_combos()

        # Should have 3 remaining valid combos: (32,3,4), (64,2,4), (64,3,4)
        assert (32, 2, 4) not in untested  # This was recorded
        assert (64, 2, 4) in untested
        assert (64, 3, 4) in untested
        assert (32, 3, 4) in untested

    def test_suggest_returns_original_if_new(self):
        """Test that suggest_coverage_config returns original for new combos."""
        from src.training.hpo_coverage import CoverageTracker

        search_space = {
            "d_model": [32, 64, 96],
            "n_layers": [2, 3, 4],
            "n_heads": [2, 4, 8],
        }
        tracker = CoverageTracker(search_space)

        # Propose a config that hasn't been tested
        proposed = {
            "d_model": 64,
            "n_layers": 3,
            "n_heads": 4,
            "learning_rate": 1e-4,
        }
        result = tracker.suggest_coverage_config(proposed)

        # Should return the same config unchanged
        assert result == proposed

    def test_suggest_redirects_duplicate(self):
        """Test that suggest_coverage_config redirects duplicate combos."""
        from src.training.hpo_coverage import CoverageTracker

        search_space = {
            "d_model": [32, 64],
            "n_layers": [2, 3],
            "n_heads": [4],
        }
        tracker = CoverageTracker(search_space)

        # Record a config
        tracker.record_config({"d_model": 32, "n_layers": 2, "n_heads": 4})

        # Propose the same architecture combo
        proposed = {
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 4,
            "learning_rate": 1e-4,
        }
        result = tracker.suggest_coverage_config(proposed)

        # Should redirect to different architecture combo
        result_combo = (result["d_model"], result["n_layers"], result["n_heads"])
        assert result_combo != (32, 2, 4)
        # Non-architecture params should be preserved
        assert result["learning_rate"] == 1e-4

    def test_suggest_validates_n_heads(self):
        """Test that redirected config has valid d_model % n_heads == 0."""
        from src.training.hpo_coverage import CoverageTracker

        search_space = {
            "d_model": [32, 48, 64],  # 48 not divisible by 8
            "n_layers": [2, 3],
            "n_heads": [4, 8],
        }
        tracker = CoverageTracker(search_space)

        # Record most configs to force redirect
        tracker.record_config({"d_model": 32, "n_layers": 2, "n_heads": 4})
        tracker.record_config({"d_model": 32, "n_layers": 3, "n_heads": 4})
        tracker.record_config({"d_model": 64, "n_layers": 2, "n_heads": 4})
        tracker.record_config({"d_model": 64, "n_layers": 3, "n_heads": 4})

        # Propose a tested combo
        proposed = {"d_model": 32, "n_layers": 2, "n_heads": 4}
        result = tracker.suggest_coverage_config(proposed)

        # Result must have valid d_model % n_heads constraint
        assert result["d_model"] % result["n_heads"] == 0

    def test_from_study_reconstructs_state(self):
        """Test that from_study classmethod reconstructs coverage state."""
        from src.training.hpo_coverage import CoverageTracker

        # Create mock study with completed trials
        mock_trial1 = Mock()
        mock_trial1.state = "COMPLETE"
        mock_trial1.params = {"d_model": 32, "n_layers": 2, "n_heads": 4}
        mock_trial1.user_attrs = {}

        mock_trial2 = Mock()
        mock_trial2.state = "COMPLETE"
        mock_trial2.params = {}
        mock_trial2.user_attrs = {
            "forced_d_model": 64,
            "forced_n_layers": 3,
            "forced_n_heads": 8,
        }

        mock_study = Mock()
        mock_study.trials = [mock_trial1, mock_trial2]

        search_space = {
            "d_model": [32, 64],
            "n_layers": [2, 3],
            "n_heads": [4, 8],
        }

        tracker = CoverageTracker.from_study(mock_study, search_space)

        # Both configs should be recorded
        untested = tracker.get_untested_arch_combos()
        assert (32, 2, 4) not in untested
        assert (64, 3, 8) not in untested

    def test_coverage_stats_percentages(self):
        """Test that coverage_stats returns correct percentages."""
        from src.training.hpo_coverage import CoverageTracker

        search_space = {
            "d_model": [32, 64],
            "n_layers": [2, 3],
            "n_heads": [4],
        }
        tracker = CoverageTracker(search_space)

        # Total valid combos: 4 (all d_model values divisible by 4)
        # Record 2 of them
        tracker.record_config({"d_model": 32, "n_layers": 2, "n_heads": 4})
        tracker.record_config({"d_model": 64, "n_layers": 3, "n_heads": 4})

        stats = tracker.coverage_stats()

        assert stats["total_valid_combos"] == 4
        assert stats["tested_combos"] == 2
        assert stats["untested_combos"] == 2
        assert stats["coverage_pct"] == pytest.approx(50.0)

    def test_suggest_returns_original_when_all_tested(self):
        """Test that suggest returns original when all combos are tested."""
        from src.training.hpo_coverage import CoverageTracker

        search_space = {
            "d_model": [32],
            "n_layers": [2],
            "n_heads": [4],
        }
        tracker = CoverageTracker(search_space)

        # Record the only valid combo
        tracker.record_config({"d_model": 32, "n_layers": 2, "n_heads": 4})

        # Propose same combo - no alternatives available
        proposed = {"d_model": 32, "n_layers": 2, "n_heads": 4, "lr": 1e-4}
        result = tracker.suggest_coverage_config(proposed)

        # Should return original since no untested combos remain
        assert result == proposed
