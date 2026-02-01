#!/usr/bin/env python3
"""
Cross-Tier Context Length Ablation Comparison.

Compares context ablation results across all tiers (a50, a100, a200) to:
1. Identify optimal context length per tier
2. Determine if 80d assumption holds across feature counts
3. Visualize AUC vs context length relationship

Usage:
    # Generate comparison report
    python experiments/context_ablation_tiers/compare_all_tiers.py

    # Generate with plots (requires matplotlib)
    python experiments/context_ablation_tiers/compare_all_tiers.py --plot
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CONFIGURATION
# ============================================================================

TIERS = ["a50", "a100", "a200"]
CONTEXT_LENGTHS = [60, 80, 90, 120, 180, 252]
OUTPUT_BASE = PROJECT_ROOT / "outputs/context_ablation_tiers"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_tier_results(tier: str) -> dict | None:
    """Load ablation summary for a tier."""
    summary_path = OUTPUT_BASE / tier / "ablation_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def load_all_results() -> dict[str, dict]:
    """Load results from all tiers."""
    results = {}
    for tier in TIERS:
        data = load_tier_results(tier)
        if data:
            results[tier] = data
    return results


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_results(all_results: dict) -> dict:
    """Analyze cross-tier context ablation results."""
    analysis = {
        "tiers_completed": list(all_results.keys()),
        "tiers_missing": [t for t in TIERS if t not in all_results],
        "per_tier": {},
        "cross_tier": {},
    }

    # Per-tier analysis
    for tier, data in all_results.items():
        tier_analysis = {
            "best_context": data.get("best_context"),
            "best_auc": data.get("best_auc"),
            "architecture": data.get("architecture"),
            "context_aucs": {},
        }

        # Extract AUC per context
        for r in data.get("results", []):
            if "error" not in r:
                ctx = r.get("context_length")
                auc = r.get("val_auc")
                if ctx and auc:
                    tier_analysis["context_aucs"][ctx] = auc

        # Calculate AUC improvement over 80d baseline
        baseline_auc = tier_analysis["context_aucs"].get(80, 0)
        best_auc = tier_analysis["best_auc"] or 0
        if baseline_auc > 0:
            tier_analysis["improvement_over_80d"] = (best_auc - baseline_auc) / baseline_auc * 100
        else:
            tier_analysis["improvement_over_80d"] = None

        analysis["per_tier"][tier] = tier_analysis

    # Cross-tier analysis
    if len(all_results) > 1:
        # Check if 80d is optimal for all tiers
        optimal_contexts = [
            analysis["per_tier"][t]["best_context"]
            for t in all_results.keys()
        ]
        analysis["cross_tier"]["all_optimal_at_80d"] = all(c == 80 for c in optimal_contexts)
        analysis["cross_tier"]["optimal_contexts"] = {
            t: analysis["per_tier"][t]["best_context"]
            for t in all_results.keys()
        }

        # Compare AUC at 80d across tiers
        analysis["cross_tier"]["auc_at_80d"] = {
            t: analysis["per_tier"][t]["context_aucs"].get(80)
            for t in all_results.keys()
        }

    return analysis


# ============================================================================
# REPORTING
# ============================================================================

def print_report(analysis: dict) -> None:
    """Print analysis report to console."""
    print("=" * 70)
    print("CROSS-TIER CONTEXT LENGTH ABLATION COMPARISON")
    print("=" * 70)
    print()

    # Status
    print(f"Tiers completed: {', '.join(analysis['tiers_completed']) or 'None'}")
    if analysis["tiers_missing"]:
        print(f"Tiers missing: {', '.join(analysis['tiers_missing'])}")
    print()

    if not analysis["per_tier"]:
        print("No results to analyze. Run context ablation scripts first.")
        return

    # Per-tier summary
    print("=" * 70)
    print("PER-TIER RESULTS")
    print("=" * 70)
    print()

    for tier, data in analysis["per_tier"].items():
        print(f"--- {tier.upper()} ---")
        print(f"  Best context: {data['best_context']}d")
        print(f"  Best AUC: {data['best_auc']:.4f}" if data['best_auc'] else "  Best AUC: N/A")
        if data["improvement_over_80d"] is not None:
            imp = data["improvement_over_80d"]
            print(f"  Improvement over 80d: {imp:+.2f}%")
        print()

        # AUC by context
        print("  AUC by context length:")
        for ctx in CONTEXT_LENGTHS:
            auc = data["context_aucs"].get(ctx)
            if auc:
                marker = " <-- BEST" if ctx == data["best_context"] else ""
                print(f"    {ctx}d: {auc:.4f}{marker}")
            else:
                print(f"    {ctx}d: (not run)")
        print()

    # Cross-tier comparison
    if analysis["cross_tier"]:
        print("=" * 70)
        print("CROSS-TIER COMPARISON")
        print("=" * 70)
        print()

        # Optimal contexts
        print("Optimal context per tier:")
        for tier, ctx in analysis["cross_tier"]["optimal_contexts"].items():
            print(f"  {tier}: {ctx}d")
        print()

        if analysis["cross_tier"]["all_optimal_at_80d"]:
            print("FINDING: 80d context is optimal for ALL tiers")
        else:
            print("FINDING: Optimal context varies by tier")
        print()

        # AUC comparison at 80d
        print("AUC at 80d context (baseline comparison):")
        for tier, auc in analysis["cross_tier"]["auc_at_80d"].items():
            if auc:
                print(f"  {tier}: {auc:.4f}")
        print()


def save_report(analysis: dict) -> Path:
    """Save analysis to JSON file."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_BASE / "cross_tier_comparison.json"

    report = {
        **analysis,
        "timestamp": datetime.now().isoformat(),
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return report_path


# ============================================================================
# PLOTTING
# ============================================================================

def plot_results(analysis: dict) -> None:
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Skipping plots.")
        return

    # AUC vs Context Length for all tiers
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"a50": "blue", "a100": "green", "a200": "red"}
    markers = {"a50": "o", "a100": "s", "a200": "^"}

    for tier, data in analysis["per_tier"].items():
        contexts = []
        aucs = []
        for ctx in CONTEXT_LENGTHS:
            auc = data["context_aucs"].get(ctx)
            if auc:
                contexts.append(ctx)
                aucs.append(auc)

        if contexts:
            ax.plot(
                contexts, aucs,
                marker=markers.get(tier, "o"),
                color=colors.get(tier, "gray"),
                label=f"{tier.upper()} (best: {data['best_context']}d)",
                linewidth=2,
                markersize=8,
            )

            # Mark best
            best_ctx = data["best_context"]
            best_auc = data["best_auc"]
            if best_ctx and best_auc:
                ax.scatter([best_ctx], [best_auc], s=200, c=colors.get(tier),
                          marker="*", zorder=5, edgecolors="black")

    ax.set_xlabel("Context Length (days)", fontsize=12)
    ax.set_ylabel("Validation AUC", fontsize=12)
    ax.set_title("Context Length Ablation by Feature Tier", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add vertical line at 80d (original assumption)
    ax.axvline(x=80, color="gray", linestyle="--", alpha=0.5, label="80d baseline")

    plt.tight_layout()

    plot_path = OUTPUT_BASE / "context_ablation_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to {plot_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare context ablation results across tiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots",
    )

    args = parser.parse_args()

    # Load data
    print("Loading results...")
    all_results = load_all_results()

    if not all_results:
        print("\nNo ablation results found.")
        print("Run the per-tier ablation scripts first:")
        print("  python experiments/context_ablation_tiers/run_ctx_ablation_a50.py")
        print("  python experiments/context_ablation_tiers/run_ctx_ablation_a100.py")
        print("  python experiments/context_ablation_tiers/run_ctx_ablation_a200.py")
        return 1

    # Analyze
    analysis = analyze_results(all_results)

    # Report
    print_report(analysis)

    # Save
    report_path = save_report(analysis)
    print(f"Report saved to {report_path}")

    # Plot
    if args.plot:
        plot_results(analysis)

    return 0


if __name__ == "__main__":
    sys.exit(main())
