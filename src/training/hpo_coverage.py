"""HPO coverage tracking for architecture exploration.

Tracks tested parameter combinations to reduce duplicate trials and ensure
better exploration of the architecture search space during HPO.
"""

from itertools import product
from typing import Any


class CoverageTracker:
    """Tracks parameter combination coverage during HPO.

    Maintains a record of tested (d_model, n_layers, n_heads) architecture
    combinations and provides suggestions to redirect duplicate configurations
    to untested regions of the search space.
    """

    def __init__(self, search_space: dict):
        """Initialize coverage tracker with search space definition.

        Args:
            search_space: Dict with lists of values for d_model, n_layers, n_heads
        """
        self._search_space = search_space
        self._tested_combos: set[tuple[int, int, int]] = set()

        # Pre-compute all valid architecture combinations
        self._valid_combos = self._compute_valid_combos()

    def _compute_valid_combos(self) -> set[tuple[int, int, int]]:
        """Compute all valid (d_model, n_layers, n_heads) combinations.

        A combination is valid if d_model % n_heads == 0.
        """
        d_models = self._search_space.get("d_model", [])
        n_layers_list = self._search_space.get("n_layers", [])
        n_heads_list = self._search_space.get("n_heads", [])

        valid = set()
        for d_model, n_layers, n_heads in product(d_models, n_layers_list, n_heads_list):
            if d_model % n_heads == 0:
                valid.add((d_model, n_layers, n_heads))

        return valid

    def record_config(self, config: dict) -> None:
        """Record a tested configuration.

        Args:
            config: Configuration dict containing d_model, n_layers, n_heads
        """
        combo = (config["d_model"], config["n_layers"], config["n_heads"])
        self._tested_combos.add(combo)

    def get_untested_arch_combos(self) -> set[tuple[int, int, int]]:
        """Get all valid architecture combos that haven't been tested.

        Returns:
            Set of (d_model, n_layers, n_heads) tuples not yet tested
        """
        return self._valid_combos - self._tested_combos

    def suggest_coverage_config(self, proposed: dict) -> dict:
        """Suggest a config, redirecting duplicates to untested combos.

        If the proposed architecture combination has already been tested,
        redirects to an untested combination while preserving non-architecture
        parameters (learning_rate, dropout, etc.).

        Args:
            proposed: Proposed configuration dict

        Returns:
            Either the original config (if combo is new) or a redirected
            config with an untested architecture combination
        """
        proposed_combo = (proposed["d_model"], proposed["n_layers"], proposed["n_heads"])

        # If this combo hasn't been tested, return as-is
        if proposed_combo not in self._tested_combos:
            return proposed

        # Get untested combos
        untested = self.get_untested_arch_combos()

        # If all combos tested, return original (no alternatives)
        if not untested:
            return proposed

        # Pick an untested combo (first one for determinism)
        new_combo = sorted(untested)[0]

        # Create redirected config with new architecture, preserving other params
        result = proposed.copy()
        result["d_model"] = new_combo[0]
        result["n_layers"] = new_combo[1]
        result["n_heads"] = new_combo[2]

        return result

    def coverage_stats(self) -> dict:
        """Get coverage statistics.

        Returns:
            Dict with total_valid_combos, tested_combos, untested_combos, coverage_pct
        """
        total = len(self._valid_combos)
        tested = len(self._tested_combos)
        untested = total - tested
        pct = (tested / total * 100) if total > 0 else 0.0

        return {
            "total_valid_combos": total,
            "tested_combos": tested,
            "untested_combos": untested,
            "coverage_pct": pct,
        }

    @classmethod
    def from_study(cls, study: Any, search_space: dict) -> "CoverageTracker":
        """Reconstruct coverage tracker from an existing Optuna study.

        Rebuilds the coverage state by examining completed trials in the study,
        supporting both TPE trials (params) and forced extreme trials (user_attrs).

        Args:
            study: Optuna Study object with trials
            search_space: Search space definition

        Returns:
            CoverageTracker with state reconstructed from study trials
        """
        tracker = cls(search_space)

        for trial in study.trials:
            # Skip non-complete trials
            if str(trial.state) != "COMPLETE":
                continue

            # Try to get architecture params from regular params first
            d_model = trial.params.get("d_model")
            n_layers = trial.params.get("n_layers")
            n_heads = trial.params.get("n_heads")

            # If not in params, check user_attrs for forced extreme trials
            if d_model is None:
                d_model = trial.user_attrs.get("forced_d_model")
            if n_layers is None:
                n_layers = trial.user_attrs.get("forced_n_layers")
            if n_heads is None:
                n_heads = trial.user_attrs.get("forced_n_heads")

            # Record if we have all architecture params
            if d_model is not None and n_layers is not None and n_heads is not None:
                tracker.record_config({
                    "d_model": d_model,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                })

        return tracker
