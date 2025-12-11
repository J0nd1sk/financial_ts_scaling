#!/usr/bin/env python3
"""HPO CLI for financial time-series scaling experiments.

Usage:
    python scripts/run_hpo.py --config configs/experiments/spy_daily_threshold_1pct.yaml --budget 2M

Arguments:
    --config: Path to experiment config YAML file
    --budget: Model parameter budget (2m, 20m, 200m, 2b)
    --n-trials: Number of HPO trials (default: 50)
    --timeout: Maximum time in hours (default: 4.0)
    --search-space: Path to search space YAML (default: configs/hpo/default_search.yaml)
    --no-thermal: Disable thermal monitoring
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.hpo import run_hpo
from src.training.thermal import ThermalCallback


def create_thermal_callback() -> ThermalCallback | None:
    """Create thermal callback with a simple temperature provider.

    Returns None if temperature reading is not available.
    """
    try:
        def get_cpu_temp() -> float:
            """Get CPU temperature.

            Placeholder implementation - returns safe default.
            In production, integrate with powermetrics or osx-cpu-temp.
            """
            return 60.0

        return ThermalCallback(temp_provider=get_cpu_temp)
    except Exception:
        return None


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Optuna HPO for financial time-series scaling experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment config YAML file",
    )
    parser.add_argument(
        "--budget",
        type=str,
        required=True,
        choices=["2m", "20m", "200m", "2b", "2M", "20M", "200M", "2B"],
        help="Model parameter budget",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of HPO trials to run",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=4.0,
        help="Maximum time in hours",
    )
    parser.add_argument(
        "--search-space",
        type=Path,
        default=Path("configs/hpo/default_search.yaml"),
        help="Path to search space YAML file",
    )
    parser.add_argument(
        "--no-thermal",
        action="store_true",
        help="Disable thermal monitoring",
    )

    args = parser.parse_args()

    # Validate config exists
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        return 1

    # Validate search space exists
    if not args.search_space.exists():
        print(f"Error: Search space file not found: {args.search_space}")
        return 1

    # Normalize budget to uppercase
    budget = args.budget.upper()

    # Create thermal callback
    thermal_callback = None
    if not args.no_thermal:
        thermal_callback = create_thermal_callback()
        if thermal_callback:
            print("Thermal monitoring enabled")
        else:
            print("Thermal monitoring not available")

    # Print configuration
    print(f"\nHPO Configuration:")
    print(f"  Config: {args.config}")
    print(f"  Budget: {budget}")
    print(f"  Trials: {args.n_trials}")
    print(f"  Timeout: {args.timeout} hours")
    print(f"  Search Space: {args.search_space}")
    print()

    # Run HPO
    print("Starting HPO...")
    result = run_hpo(
        config_path=str(args.config),
        budget=budget,
        n_trials=args.n_trials,
        timeout_hours=args.timeout,
        search_space_path=str(args.search_space),
        thermal_callback=thermal_callback,
    )

    # Report results
    print(f"\nHPO Complete!")
    print(f"  Trials completed: {result['n_trials']}")

    if result.get("stopped_early"):
        print(f"  Stopped early: {result.get('stop_reason')}")

    if result.get("best_value") is not None:
        print(f"  Best loss: {result['best_value']:.6f}")
        print(f"  Best params:")
        for param, value in result.get("best_params", {}).items():
            if isinstance(value, float):
                print(f"    {param}: {value:.6g}")
            else:
                print(f"    {param}: {value}")

    print(f"  Results saved to: {result['output_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
