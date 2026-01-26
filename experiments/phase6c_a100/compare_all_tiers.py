#!/usr/bin/env python3
"""
Phase 6C A100: Cross-Tier Comparison

Compare performance across all feature tiers:
- a20: 20 features (Phase 6A baseline)
- a50: 50 features (Phase 6C first comparison)
- a100: 100 features (current)

Key questions:
1. Does 20M > 200M pattern persist at a100?
2. Does a100 improve over a50?
3. Is there diminishing returns in feature scaling?
"""
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs/phase6c_a100"

# Results directories for each tier
TIER_RESULTS = {
    "a20": PROJECT_ROOT / "outputs/phase6a_final",  # Phase 6A experiments
    "a50": PROJECT_ROOT / "outputs/phase6c",
    "a100": PROJECT_ROOT / "outputs/phase6c_a100",
}

# Experiment naming patterns
BUDGETS = ["2M", "20M", "200M"]
HORIZONS = [1, 2, 3, 5]

# Known a20 results from Phase 6A (if results files not available)
A20_BASELINE = {
    ("2M", 1): 0.706,
    ("20M", 1): 0.710,
    ("200M", 1): 0.698,
    ("2M", 2): 0.655,
    ("20M", 2): 0.661,
    ("200M", 2): 0.648,
    ("2M", 3): 0.633,
    ("20M", 3): 0.641,
    ("200M", 3): 0.628,
    ("2M", 5): 0.618,
    ("20M", 5): 0.625,
    ("200M", 5): 0.612,
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_tier_results(tier, results_dir):
    """Load all experiment results for a tier."""
    results = {}

    for budget in BUDGETS:
        for horizon in HORIZONS:
            # Naming conventions differ by tier
            if tier == "a20":
                # Phase 6A naming: phase6a_2m_h1, phase6a_20m_h1, etc.
                exp_patterns = [f"phase6a_{budget.lower()}_h{horizon}"]
            else:
                # Phase 6C naming: s1_01_2m_h1, s1_02_20m_h1, etc.
                budget_idx = BUDGETS.index(budget)
                horizon_idx = HORIZONS.index(horizon)
                exp_num = budget_idx + horizon_idx * 3 + 1
                exp_patterns = [
                    f"s1_{exp_num:02d}_{budget.lower()}_h{horizon}",
                    # Fallback patterns for legacy naming
                    f"s1_{BUDGETS.index(budget)+1:02d}_{budget.lower()}_h{horizon}",
                ]

            for exp_name in exp_patterns:
                results_path = results_dir / exp_name / "results.json"
                if results_path.exists():
                    with open(results_path) as f:
                        data = json.load(f)
                        auc = data.get("val_metrics", {}).get("auc")
                        if auc:
                            results[(budget, horizon)] = auc
                            break

    return results


def load_all_results():
    """Load results from all tiers."""
    all_results = {}

    for tier, results_dir in TIER_RESULTS.items():
        if results_dir.exists():
            tier_results = load_tier_results(tier, results_dir)
            if tier_results:
                all_results[tier] = tier_results
                print(f"Loaded {len(tier_results)} results for {tier}")
            else:
                print(f"No results found for {tier} at {results_dir}")

                # Use baseline values for a20 if available
                if tier == "a20":
                    all_results[tier] = A20_BASELINE
                    print(f"  Using baseline values for {tier}")
        else:
            print(f"Results directory not found: {results_dir}")

            # Use baseline values for a20
            if tier == "a20":
                all_results[tier] = A20_BASELINE
                print(f"  Using baseline values for {tier}")

    return all_results


# ============================================================================
# ANALYSIS
# ============================================================================

def create_comparison_matrix(all_results):
    """Create comparison matrix: tier x budget x horizon."""
    tiers = sorted(all_results.keys())

    # Create DataFrames for each horizon
    matrices = {}
    for horizon in HORIZONS:
        data = []
        for tier in tiers:
            row = []
            for budget in BUDGETS:
                auc = all_results.get(tier, {}).get((budget, horizon))
                row.append(auc)
            data.append(row)

        df = pd.DataFrame(data, index=tiers, columns=BUDGETS)
        matrices[f"H{horizon}"] = df

    return matrices


def find_best_budget_per_tier(all_results):
    """Find optimal budget for each tier."""
    best = {}
    for tier, results in all_results.items():
        tier_aucs = {}
        for (budget, horizon), auc in results.items():
            if budget not in tier_aucs:
                tier_aucs[budget] = []
            tier_aucs[budget].append(auc)

        # Average AUC across horizons
        avg_aucs = {b: np.mean(aucs) for b, aucs in tier_aucs.items()}
        best_budget = max(avg_aucs, key=avg_aucs.get)
        best[tier] = {
            "budget": best_budget,
            "avg_auc": avg_aucs[best_budget],
            "all_budgets": avg_aucs,
        }

    return best


def compute_tier_deltas(all_results, base_tier="a20"):
    """Compute improvement from base tier to other tiers."""
    if base_tier not in all_results:
        return {}

    base = all_results[base_tier]
    deltas = {}

    for tier, results in all_results.items():
        if tier == base_tier:
            continue

        tier_deltas = {}
        for key, auc in results.items():
            if key in base:
                base_auc = base[key]
                delta = (auc - base_auc) / base_auc * 100  # Percentage change
                tier_deltas[key] = {"auc": auc, "base_auc": base_auc, "delta_pct": delta}

        deltas[tier] = tier_deltas

    return deltas


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("Phase 6C A100: Cross-Tier Comparison")
    print("=" * 70)

    # Load all results
    print("\nLoading results...")
    all_results = load_all_results()

    if len(all_results) < 2:
        print("\nInsufficient data for comparison.")
        print("Need at least 2 tiers with results.")
        return

    # Create comparison matrices
    print("\n" + "=" * 70)
    print("COMPARISON MATRIX: AUC by Tier x Budget x Horizon")
    print("=" * 70)

    matrices = create_comparison_matrix(all_results)
    for horizon_label, matrix in matrices.items():
        print(f"\n{horizon_label}:")
        print(matrix.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"))

    # Best budget per tier
    print("\n" + "=" * 70)
    print("OPTIMAL BUDGET BY TIER")
    print("=" * 70)

    best_budgets = find_best_budget_per_tier(all_results)
    for tier, info in best_budgets.items():
        print(f"\n{tier}:")
        print(f"  Best budget: {info['budget']} (avg AUC: {info['avg_auc']:.4f})")
        print("  All budgets:", {b: f"{v:.4f}" for b, v in info['all_budgets'].items()})

    # Tier deltas
    print("\n" + "=" * 70)
    print("TIER IMPROVEMENT OVER a20 BASELINE")
    print("=" * 70)

    deltas = compute_tier_deltas(all_results, "a20")
    for tier, tier_deltas in deltas.items():
        print(f"\n{tier} vs a20:")
        avg_delta = np.mean([d["delta_pct"] for d in tier_deltas.values()])
        print(f"  Average delta: {avg_delta:+.2f}%")
        for key, d in sorted(tier_deltas.items()):
            budget, horizon = key
            print(f"  {budget} H{horizon}: {d['base_auc']:.4f} -> {d['auc']:.4f} ({d['delta_pct']:+.2f}%)")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # 1. Does 20M > 200M pattern persist?
    if "a100" in all_results:
        a100 = all_results["a100"]
        h1_20m = a100.get(("20M", 1))
        h1_200m = a100.get(("200M", 1))
        if h1_20m and h1_200m:
            if h1_20m > h1_200m:
                print(f"1. 20M > 200M pattern PERSISTS at a100: {h1_20m:.4f} > {h1_200m:.4f}")
            else:
                print(f"1. 20M > 200M pattern REVERSED at a100: {h1_20m:.4f} < {h1_200m:.4f}")

    # 2. Does a100 improve over a50?
    if "a100" in all_results and "a50" in all_results:
        a100_avg = np.mean(list(all_results["a100"].values()))
        a50_avg = np.mean(list(all_results["a50"].values()))
        delta = (a100_avg - a50_avg) / a50_avg * 100
        if delta > 0:
            print(f"2. a100 IMPROVES over a50: avg AUC {a50_avg:.4f} -> {a100_avg:.4f} (+{delta:.2f}%)")
        else:
            print(f"2. a100 DOES NOT improve over a50: avg AUC {a50_avg:.4f} -> {a100_avg:.4f} ({delta:.2f}%)")

    # 3. Feature scaling curve
    if len(all_results) >= 2:
        feature_counts = {"a20": 20, "a50": 50, "a100": 100}
        avg_aucs = {}
        for tier, results in all_results.items():
            if tier in feature_counts:
                avg_aucs[feature_counts[tier]] = np.mean(list(results.values()))

        if len(avg_aucs) >= 2:
            print("3. Feature scaling curve:")
            for n_features in sorted(avg_aucs.keys()):
                print(f"   {n_features} features: avg AUC = {avg_aucs[n_features]:.4f}")

    # Save results
    output = {
        "matrices": {k: v.to_dict() for k, v in matrices.items()},
        "best_budgets": best_budgets,
        "deltas": {tier: {str(k): v for k, v in td.items()} for tier, td in deltas.items()},
        "raw_results": {tier: {str(k): v for k, v in results.items()} for tier, results in all_results.items()},
        "timestamp": datetime.now().isoformat(),
    }

    output_path = OUTPUT_DIR / "tier_comparison.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
