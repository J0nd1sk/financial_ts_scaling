#!/usr/bin/env python3
"""Reduce a200 tier features through iterative importance-based removal.

This experiment:
1. Trains baseline model on all 206 a200 features
2. Computes multi-method importance scores
3. Removes bottom 10% features (respecting category minimums)
4. Validates model performance vs baseline
5. Repeats until convergence

Output:
    outputs/feature_curation/a200_reduction/
    ├── baseline/metrics.json
    ├── round_N/
    │   ├── importance.csv
    │   ├── removed.json
    │   └── metrics.json
    └── final/
        └── curated_features.json

Usage:
    python experiments/feature_curation/reduce_a200.py --dry-run
    python experiments/feature_curation/reduce_a200.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.curation import importance, quality, validation, reduction


def main() -> int:
    parser = argparse.ArgumentParser(description="Reduce a200 features")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--max-rounds", type=int, default=20, help="Maximum reduction rounds")
    args = parser.parse_args()

    print("=" * 70)
    print("Feature Reduction: tier_a200 (206 features)")
    print("=" * 70)
    print()

    # Configuration
    config = reduction.ReductionConfig(
        initial_removal_fraction=0.10,
        fallback_removal_fraction=0.05,
        max_rounds=args.max_rounds,
    )

    # Output directory
    output_dir = PROJECT_ROOT / "outputs" / "feature_curation" / "a200_reduction"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load feature list
    from src.features.tier_a200 import FEATURE_LIST
    initial_features = list(FEATURE_LIST)

    print(f"Initial features: {len(initial_features)}")
    print(f"Max rounds: {config.max_rounds}")
    print(f"Initial removal fraction: {config.initial_removal_fraction:.0%}")
    print(f"Fallback removal fraction: {config.fallback_removal_fraction:.0%}")
    print()

    # Show category counts
    counts = reduction.count_features_per_category(initial_features)
    print("Category counts:")
    for cat, count in sorted(counts.items()):
        minimum = reduction.CATEGORY_MINIMUMS.get(cat, 1)
        print(f"  {cat}: {count} (min: {minimum})")
    print()

    if args.dry_run:
        print("[DRY RUN] Reduction would proceed as follows:")
        print()
        print("Phase 1: BASELINE")
        print("  - Train PatchTST on all 206 features")
        print("  - Record precision, recall, pred_range, AUC")
        print("  - Save checkpoint")
        print()
        print("Phase 2: ITERATE (up to {} rounds)".format(config.max_rounds))
        print("  For each round:")
        print("    - Compute importance scores (correlation, MI, redundancy)")
        print("    - Remove bottom 10% features (respecting category minimums)")
        print("    - Retrain model")
        print("    - Validate vs baseline thresholds:")
        print("      * Precision: >= 95% of baseline (abort < 90%)")
        print("      * Recall: >= 90% of baseline (abort < 80%)")
        print("      * pred_range: >= 0.10 (abort < 0.05)")
        print("      * AUC: >= 95% of baseline (abort < 90%)")
        print("    - If WARNING: try 5% removal")
        print("    - If ABORT: restore previous, stop")
        print()
        print("Phase 3: OUTPUT")
        print("  - Save curated_features.json")
        print("  - Save reduction history")
        print()
        print("To run actual reduction, remove --dry-run flag.")
        print()
        print("Note: This requires:")
        print("  - Processed data at data/processed/a200/")
        print("  - GPU/MPS available for training")
        print("  - ~2-3 hours runtime")
        return 0

    # Full reduction would go here
    print("Full reduction requires processed data and training infrastructure.")
    print("This is a placeholder for the actual implementation.")
    print()
    print("Implementation steps:")
    print("1. Load processed data with a200 features")
    print("2. Train baseline model and save metrics")
    print("3. Run reduction loop with validation")
    print("4. Save curated feature list")

    return 0


if __name__ == "__main__":
    sys.exit(main())
