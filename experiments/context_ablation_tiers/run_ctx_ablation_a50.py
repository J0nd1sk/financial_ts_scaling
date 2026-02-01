#!/usr/bin/env python3
"""
Context Length Ablation for A50 Feature Tier.

Runs training at multiple context lengths (60d, 80d, 90d, 120d, 180d, 252d)
using the best architecture from a50 HPO to find optimal context for this tier.

Usage:
    # Run full ablation (all 6 context lengths)
    python experiments/context_ablation_tiers/run_ctx_ablation_a50.py

    # Run single context length
    python experiments/context_ablation_tiers/run_ctx_ablation_a50.py --context 80

    # Dry run (show config only)
    python experiments/context_ablation_tiers/run_ctx_ablation_a50.py --dry-run
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.context_ablation_tiers.train_tier_ctx import (
    train_with_context,
    CONTEXT_LENGTHS,
    TIER_BEST_ARCH,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

TIER = "a50"
OUTPUT_BASE = PROJECT_ROOT / "outputs/context_ablation_tiers" / TIER


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run context length ablation for a50 tier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--context",
        type=int,
        default=None,
        help="Single context length to run (default: run all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configuration without training",
    )

    args = parser.parse_args()

    print("=" * 70)
    print(f"CONTEXT LENGTH ABLATION: {TIER.upper()}")
    print("=" * 70)
    print()
    print(f"Architecture: {TIER_BEST_ARCH[TIER]}")
    print()

    # Determine which context lengths to run
    if args.context:
        contexts = [args.context]
    else:
        contexts = CONTEXT_LENGTHS

    print(f"Context lengths to test: {contexts}")
    print()

    if args.dry_run:
        print("DRY RUN - no training will be performed")
        return 0

    # Run ablation
    all_results = []
    for ctx in contexts:
        output_dir = OUTPUT_BASE / f"ctx{ctx}"
        print(f"\n{'='*70}")
        print(f"Running context_length={ctx}")
        print(f"{'='*70}")

        result = train_with_context(
            tier=TIER,
            context_length=ctx,
            output_dir=output_dir,
            verbose=True,
        )

        all_results.append(result)

    # Summary
    print()
    print("=" * 70)
    print("ABLATION SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Context':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'Time (min)':<10}")
    print("-" * 50)

    best_ctx = None
    best_auc = 0.0

    for r in all_results:
        if "error" in r:
            print(f"{r['context_length']:<10} ERROR: {r['error'][:40]}")
        else:
            auc = r.get("val_auc", 0)
            prec = r.get("val_precision", 0)
            recall = r.get("val_recall", 0)
            time_min = r.get("training_time_min", 0)
            print(f"{r['context_length']:<10} {auc:<10.4f} {prec:<10.4f} {recall:<10.4f} {time_min:<10.1f}")

            if auc > best_auc:
                best_auc = auc
                best_ctx = r["context_length"]

    print()
    if best_ctx:
        print(f"Best context length: {best_ctx}d (AUC: {best_auc:.4f})")

    # Save summary
    summary = {
        "tier": TIER,
        "architecture": TIER_BEST_ARCH[TIER],
        "results": all_results,
        "best_context": best_ctx,
        "best_auc": best_auc,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_BASE / "ablation_summary.json"
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
