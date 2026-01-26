#!/usr/bin/env python3
"""
Alternative Architecture Investigation: Comparison Analysis

Compares all architecture experiments against PatchTST baseline.
Generates summary table and precision-recall comparison.

Usage:
    python compare_architectures.py           # Compare existing results
    python compare_architectures.py --run     # Run all experiments then compare
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from experiments.architectures.common import OUTPUT_BASE


# ============================================================================
# BASELINE REFERENCE (PatchTST)
# ============================================================================

PATCHTST_BASELINE = {
    "model": "PatchTST",
    "budget": "200M",
    "horizon": 1,
    "val_auc": 0.718,
    "precision_at_90_recall": 0.04,  # 4% recall at 90% precision
    "precision_at_75_recall": 0.23,  # 23% recall at 75% precision
}


# ============================================================================
# RESULT COLLECTION
# ============================================================================

def load_experiment_results(experiment_name: str) -> dict | None:
    """Load results from experiment directory."""
    results_path = OUTPUT_BASE / experiment_name / "results.json"
    if not results_path.exists():
        return None

    with open(results_path) as f:
        return json.load(f)


def collect_all_results() -> list[dict]:
    """Collect results from all architecture experiments."""
    experiments = [
        "itransformer_forecast",
        "informer_forecast",
        "informer_forecast_long_context",
    ]

    results = []
    for exp_name in experiments:
        result = load_experiment_results(exp_name)
        if result:
            results.append(result)
            print(f"  Loaded: {exp_name}")
        else:
            print(f"  Missing: {exp_name}")

    return results


# ============================================================================
# COMPARISON ANALYSIS
# ============================================================================

def compare_to_baseline(results: list[dict]) -> pd.DataFrame:
    """Create comparison table against PatchTST baseline."""
    rows = []

    # Add baseline
    rows.append({
        "Model": "PatchTST (baseline)",
        "AUC": PATCHTST_BASELINE["val_auc"],
        "Δ AUC": 0.0,
        "P@R20%": "-",
        "P@R30%": "-",
        "P@R50%": "-",
        "Direction Acc": "-",
        "Status": "BASELINE",
    })

    for result in results:
        val_metrics = result.get("val_metrics", {})
        pr = result.get("precision_recall", {})

        auc = val_metrics.get("auc")
        delta_auc = (auc - PATCHTST_BASELINE["val_auc"]) if auc else None

        # Determine status
        if auc is None:
            status = "FAILED"
        elif auc >= 0.74:
            status = "✅ BEAT"
        elif auc >= 0.70:
            status = "⚠️ VIABLE"
        else:
            status = "❌ WORSE"

        rows.append({
            "Model": result.get("model", "Unknown"),
            "AUC": auc,
            "Δ AUC": delta_auc,
            "P@R20%": pr.get("precision_at_recall_20"),
            "P@R30%": pr.get("precision_at_recall_30"),
            "P@R50%": pr.get("precision_at_recall_50"),
            "Direction Acc": val_metrics.get("direction_accuracy"),
            "Status": status,
        })

    return pd.DataFrame(rows)


def analyze_precision_recall(results: list[dict]) -> str:
    """Analyze precision-recall tradeoffs across models."""
    lines = [
        "",
        "=" * 70,
        "PRECISION-RECALL ANALYSIS",
        "=" * 70,
        "",
        "Goal: Find models with better precision at improved recall vs PatchTST",
        f"PatchTST baseline: 90% precision → 4% recall, 75% precision → 23% recall",
        "",
    ]

    for result in results:
        model = result.get("model", "Unknown")
        pr = result.get("precision_recall", {})

        if not pr:
            lines.append(f"{model}: No P-R data available")
            continue

        lines.append(f"{model}:")
        for recall_level in [10, 20, 30, 40, 50]:
            p_key = f"precision_at_recall_{recall_level}"
            r_key = f"actual_recall_{recall_level}"
            precision = pr.get(p_key, "N/A")
            actual_recall = pr.get(r_key, "N/A")
            if isinstance(precision, float):
                lines.append(f"  P@R{recall_level}%: {precision:.3f} (actual recall: {actual_recall:.3f})")
            else:
                lines.append(f"  P@R{recall_level}%: {precision}")

        lines.append("")

    return "\n".join(lines)


def generate_recommendation(results: list[dict]) -> str:
    """Generate recommendation based on results."""
    lines = [
        "",
        "=" * 70,
        "RECOMMENDATION",
        "=" * 70,
        "",
    ]

    # Find best model by AUC
    best_result = None
    best_auc = 0

    for result in results:
        auc = result.get("val_metrics", {}).get("auc", 0)
        if auc and auc > best_auc:
            best_auc = auc
            best_result = result

    if not best_result:
        lines.append("No valid results to analyze.")
        lines.append("")
        lines.append("ACTION: Run experiments first with:")
        lines.append("  python experiments/architectures/itransformer_forecast.py")
        lines.append("  python experiments/architectures/informer_forecast.py")
        return "\n".join(lines)

    baseline_auc = PATCHTST_BASELINE["val_auc"]
    improvement = (best_auc - baseline_auc) / baseline_auc * 100

    if best_auc >= 0.74:
        lines.extend([
            f"✅ SUCCESS: {best_result['model']} achieves AUC {best_auc:.4f}",
            f"   Improvement over baseline: {improvement:+.1f}%",
            "",
            "RECOMMENDATION: Proceed to Phase 2 - port architecture to codebase",
            "   - Integrate with existing Trainer class",
            "   - Implement direct classification head",
            "   - Run fair comparison with identical data pipeline",
        ])
    elif best_auc >= 0.70:
        lines.extend([
            f"⚠️ VIABLE: {best_result['model']} achieves AUC {best_auc:.4f}",
            f"   Change vs baseline: {improvement:+.1f}%",
            "",
            "RECOMMENDATION: Check precision-recall curve",
            "   - If P-R tradeoff is better, consider Phase 2",
            "   - If P-R is similar, architecture likely doesn't matter",
            "   - Focus may be better spent on feature engineering (Phase 6C)",
        ])
    else:
        lines.extend([
            f"❌ WORSE: Best model ({best_result['model']}) only achieves AUC {best_auc:.4f}",
            f"   Change vs baseline: {improvement:+.1f}%",
            "",
            "RECOMMENDATION: Abandon alternative architectures",
            "   - PatchTST remains the best option",
            "   - Focus on feature engineering (Phase 6C) instead",
            "   - Architecture doesn't seem to be the limiting factor",
        ])

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def run_all_experiments():
    """Run all architecture experiments."""
    print("Running all architecture experiments...")
    print()

    from experiments.architectures.itransformer_forecast import run_experiment as run_itransformer
    from experiments.architectures.informer_forecast import run_experiment as run_informer

    print("=" * 70)
    print("RUNNING: iTransformer")
    print("=" * 70)
    run_itransformer()
    print()

    print("=" * 70)
    print("RUNNING: Informer")
    print("=" * 70)
    run_informer()
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true",
                        help="Run all experiments before comparing")
    args = parser.parse_args()

    print("=" * 70)
    print("ALTERNATIVE ARCHITECTURE INVESTIGATION - COMPARISON")
    print("=" * 70)

    if args.run:
        run_all_experiments()

    print("\nCollecting results...")
    results = collect_all_results()

    if not results:
        print("\nNo results found. Run experiments first:")
        print("  python experiments/architectures/itransformer_forecast.py")
        print("  python experiments/architectures/informer_forecast.py")
        print("\nOr use: python compare_architectures.py --run")
        return

    # Generate comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    df = compare_to_baseline(results)
    print()
    print(df.to_string(index=False))

    # Precision-recall analysis
    print(analyze_precision_recall(results))

    # Recommendation
    print(generate_recommendation(results))

    # Save summary
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_BASE / "comparison_summary.json"

    summary = {
        "baseline": PATCHTST_BASELINE,
        "experiments": [
            {
                "model": r.get("model"),
                "auc": r.get("val_metrics", {}).get("auc"),
                "config": r.get("config"),
            }
            for r in results
        ],
        "comparison_table": df.to_dict(orient="records"),
        "timestamp": datetime.now().isoformat(),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
