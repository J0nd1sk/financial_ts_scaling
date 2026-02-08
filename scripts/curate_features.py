#!/usr/bin/env python3
"""Feature curation CLI - main entry point for feature reduction and quality analysis.

Usage:
    python scripts/curate_features.py analyze --tier a200
    python scripts/curate_features.py reduce --tier a200 --dry-run
    python scripts/curate_features.py reduce --tier a200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze feature quality for a tier."""
    import pandas as pd
    from src.features.curation import importance, quality

    print(f"Analyzing features for tier: {args.tier}")
    print("=" * 60)

    # Load data (placeholder - actual implementation would load real data)
    print("Loading feature data...")

    # For now, show what would be analyzed
    if args.tier == "a200":
        from src.features.tier_a200 import FEATURE_LIST
        n_features = len(FEATURE_LIST)
    elif args.tier == "a500":
        from src.features.tier_a500 import FEATURE_LIST
        n_features = len(FEATURE_LIST)
    else:
        print(f"Unknown tier: {args.tier}")
        return 1

    print(f"Total features in {args.tier}: {n_features}")
    print()
    print("To run full analysis, processed data must be available.")
    print("Analysis would compute:")
    print("  - Target correlation (Pearson/Spearman)")
    print("  - Mutual information with target")
    print("  - Redundancy matrix between features")
    print("  - Combined importance scores")
    print()

    return 0


def cmd_reduce(args: argparse.Namespace) -> int:
    """Run iterative feature reduction."""
    from src.features.curation import reduction

    print(f"Feature reduction for tier: {args.tier}")
    print("=" * 60)

    if args.dry_run:
        print("[DRY RUN] No actual reduction will be performed.")
        print()

    # Show configuration
    config = reduction.ReductionConfig()
    print("Reduction configuration:")
    print(f"  Initial removal fraction: {config.initial_removal_fraction:.0%}")
    print(f"  Fallback removal fraction: {config.fallback_removal_fraction:.0%}")
    print(f"  Max rounds: {config.max_rounds}")
    print()

    # Show category minimums
    print("Category minimums (features preserved):")
    for cat, minimum in sorted(reduction.CATEGORY_MINIMUMS.items()):
        print(f"  {cat}: {minimum}")
    print()

    if args.tier == "a200":
        from src.features.tier_a200 import FEATURE_LIST
    elif args.tier == "a500":
        from src.features.tier_a500 import FEATURE_LIST
    else:
        print(f"Unknown tier: {args.tier}")
        return 1

    print(f"Starting features: {len(FEATURE_LIST)}")
    print()

    # Count features per category
    counts = reduction.count_features_per_category(FEATURE_LIST)
    print("Features per category:")
    for cat, count in sorted(counts.items()):
        minimum = reduction.CATEGORY_MINIMUMS.get(cat, 1)
        status = "OK" if count >= minimum else "BELOW MIN"
        print(f"  {cat}: {count} (min: {minimum}) [{status}]")
    print()

    if args.dry_run:
        print("To run actual reduction, remove --dry-run flag.")
        return 0

    print("Full reduction requires trained model and processed data.")
    print("See experiments/feature_curation/reduce_a200.py for full workflow.")
    return 0


def cmd_quality(args: argparse.Namespace) -> int:
    """Run quality test on a single feature."""
    print(f"Quality test for feature: {args.feature}")
    print(f"Tier: {args.tier}")
    print("=" * 60)
    print()
    print("This command tests a single feature for quality gates.")
    print("See scripts/test_feature_quality.py for full implementation.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Feature curation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze feature quality")
    analyze_parser.add_argument("--tier", required=True, help="Feature tier (a200, a500)")

    # reduce command
    reduce_parser = subparsers.add_parser("reduce", help="Run iterative reduction")
    reduce_parser.add_argument("--tier", required=True, help="Feature tier (a200, a500)")
    reduce_parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")

    # quality command
    quality_parser = subparsers.add_parser("quality", help="Test single feature quality")
    quality_parser.add_argument("--feature", required=True, help="Feature name to test")
    quality_parser.add_argument("--tier", required=True, help="Feature tier context")

    args = parser.parse_args()

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "reduce":
        return cmd_reduce(args)
    elif args.command == "quality":
        return cmd_quality(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
