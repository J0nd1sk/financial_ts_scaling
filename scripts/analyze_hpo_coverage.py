#!/usr/bin/env python3
"""
HPO Coverage Analysis Script

Analyzes HPO trial data to identify:
1. Value frequency per hyperparameter
2. Convergence patterns (repeated configurations)
3. Coverage gaps (untested combinations)
4. Cross-budget trend analysis
5. Architecture pattern correlations (wide/shallow vs narrow/deep)
6. Training HP impact analysis

Usage:
    python scripts/analyze_hpo_coverage.py [--output reports/hpo_coverage.md]
"""
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from itertools import product
from typing import Any

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_trials(budget: str) -> list[dict]:
    """Load trial data for a given budget."""
    trials_path = PROJECT_ROOT / f"outputs/phase6c_a100/hpo_{budget.lower()}_h1/all_trials.json"
    if not trials_path.exists():
        print(f"Warning: {trials_path} not found")
        return []
    with open(trials_path) as f:
        return json.load(f)


def load_best_params(budget: str) -> dict:
    """Load best params for a given budget."""
    params_path = PROJECT_ROOT / f"outputs/phase6c_a100/hpo_{budget.lower()}_h1/best_params.json"
    if not params_path.exists():
        return {}
    with open(params_path) as f:
        return json.load(f)


def analyze_value_frequencies(trials: list[dict], search_space: dict) -> dict[str, Counter]:
    """Compute frequency of each hyperparameter value."""
    frequencies = defaultdict(Counter)
    for trial in trials:
        if trial["state"] != "TrialState.COMPLETE":
            continue
        for param, value in trial["params"].items():
            frequencies[param][value] += 1
    return dict(frequencies)


def analyze_convergence(trials: list[dict]) -> dict:
    """Identify convergence patterns - repeated configurations."""
    config_counts = Counter()
    config_to_trials = defaultdict(list)

    for trial in trials:
        if trial["state"] != "TrialState.COMPLETE":
            continue
        # Create hashable config tuple
        config = tuple(sorted(trial["params"].items()))
        config_counts[config] += 1
        config_to_trials[config].append({
            "number": trial["number"],
            "value": trial["value"]
        })

    # Find repeated configs
    repeated = {
        k: {
            "count": v,
            "trials": config_to_trials[k],
            "config": dict(k)
        }
        for k, v in config_counts.items() if v > 1
    }

    # Sort by count descending
    repeated = dict(sorted(repeated.items(), key=lambda x: x[1]["count"], reverse=True))

    return {
        "total_trials": len(trials),
        "unique_configs": len(config_counts),
        "repeated_configs": len(repeated),
        "wasted_trials": sum(v["count"] - 1 for v in repeated.values()),
        "top_repeated": list(repeated.values())[:5]
    }


def identify_coverage_gaps(trials: list[dict], search_space: dict) -> dict:
    """Identify combinations that were never tested."""
    # Get tested combinations for key architecture params
    tested_combos = set()
    for trial in trials:
        if trial["state"] != "TrialState.COMPLETE":
            continue
        # Focus on architecture params
        arch_params = ["d_model", "n_layers", "n_heads"]
        combo = tuple(trial["params"].get(p) for p in arch_params if p in trial["params"])
        tested_combos.add(combo)

    # Generate all possible combinations from search space
    arch_params = ["d_model", "n_layers", "n_heads"]
    available = [search_space.get(p, []) for p in arch_params]
    all_combos = set(product(*available))

    # Find untested
    untested = all_combos - tested_combos

    # Categorize gaps
    gaps = {
        "architecture": {
            "total_possible": len(all_combos),
            "tested": len(tested_combos),
            "untested_count": len(untested),
            "coverage_pct": round(100 * len(tested_combos) / len(all_combos), 1) if all_combos else 0,
            "untested_examples": list(untested)[:10]
        }
    }

    # Training HP coverage
    training_params = ["learning_rate", "dropout", "weight_decay"]
    tested_training = set()
    for trial in trials:
        if trial["state"] != "TrialState.COMPLETE":
            continue
        combo = tuple(trial["params"].get(p) for p in training_params)
        tested_training.add(combo)

    available_training = [search_space.get(p, []) for p in training_params]
    all_training = set(product(*available_training))
    untested_training = all_training - tested_training

    gaps["training"] = {
        "total_possible": len(all_training),
        "tested": len(tested_training),
        "untested_count": len(untested_training),
        "coverage_pct": round(100 * len(tested_training) / len(all_training), 1) if all_training else 0,
        "untested_examples": list(untested_training)[:10]
    }

    return gaps


def analyze_architecture_patterns(trials: list[dict]) -> dict:
    """Analyze width vs depth patterns."""
    results = []
    for trial in trials:
        if trial["state"] != "TrialState.COMPLETE":
            continue
        params = trial["params"]
        d_model = params.get("d_model", 64)
        d_ff_ratio = params.get("d_ff_ratio", 4)
        n_layers = params.get("n_layers", 4)

        width = d_model * d_ff_ratio
        depth = n_layers
        width_depth_ratio = width / depth if depth > 0 else 0

        results.append({
            "trial": trial["number"],
            "auc": trial["value"],
            "d_model": d_model,
            "n_layers": n_layers,
            "width": width,
            "depth": depth,
            "width_depth_ratio": width_depth_ratio
        })

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    # Correlation analysis
    correlations = {
        "width_auc": round(df["width"].corr(df["auc"]), 4),
        "depth_auc": round(df["depth"].corr(df["auc"]), 4),
        "ratio_auc": round(df["width_depth_ratio"].corr(df["auc"]), 4)
    }

    # Best configs by pattern
    df_sorted = df.sort_values("auc", ascending=False)
    top_5 = df_sorted.head(5)

    patterns = {
        "correlations": correlations,
        "top_5_patterns": [
            {
                "trial": int(row["trial"]),
                "auc": round(row["auc"], 4),
                "d_model": int(row["d_model"]),
                "n_layers": int(row["n_layers"]),
                "width_depth_ratio": round(row["width_depth_ratio"], 2)
            }
            for _, row in top_5.iterrows()
        ],
        "avg_by_d_model": df.groupby("d_model")["auc"].mean().to_dict(),
        "avg_by_n_layers": df.groupby("n_layers")["auc"].mean().to_dict()
    }

    return patterns


def analyze_training_hp_impact(trials: list[dict]) -> dict:
    """Analyze impact of training hyperparameters."""
    results = []
    for trial in trials:
        if trial["state"] != "TrialState.COMPLETE":
            continue
        results.append({
            "auc": trial["value"],
            **trial["params"]
        })

    df = pd.DataFrame(results)
    if df.empty:
        return {}

    impact = {}
    for param in ["learning_rate", "dropout", "weight_decay"]:
        if param in df.columns:
            grouped = df.groupby(param)["auc"].agg(["mean", "std", "count"])
            impact[param] = {
                str(idx): {
                    "mean_auc": round(row["mean"], 4),
                    "std_auc": round(row["std"], 4) if not pd.isna(row["std"]) else 0,
                    "count": int(row["count"])
                }
                for idx, row in grouped.iterrows()
            }

    return impact


def format_markdown_report(
    budget: str,
    best_params: dict,
    frequencies: dict,
    convergence: dict,
    gaps: dict,
    patterns: dict,
    training_impact: dict,
    search_space: dict
) -> str:
    """Format analysis as markdown report."""
    lines = []
    lines.append(f"## {budget} Budget Analysis\n")

    # Best configuration
    if best_params:
        lines.append("### Best Configuration")
        lines.append(f"- **Best AUC**: {best_params.get('best_value', 'N/A'):.4f}")
        lines.append(f"- **Best Trial**: {best_params.get('best_trial', 'N/A')}")
        if "best_params" in best_params:
            for k, v in best_params["best_params"].items():
                lines.append(f"  - {k}: {v}")
        lines.append("")

    # Convergence analysis
    lines.append("### Convergence Analysis")
    lines.append(f"- Total trials: {convergence.get('total_trials', 0)}")
    lines.append(f"- Unique configurations: {convergence.get('unique_configs', 0)}")
    lines.append(f"- Repeated configurations: {convergence.get('repeated_configs', 0)}")
    lines.append(f"- Wasted trials (repeats): {convergence.get('wasted_trials', 0)}")
    if convergence.get("top_repeated"):
        lines.append("\n**Most Repeated Configurations:**")
        for i, rep in enumerate(convergence["top_repeated"][:3], 1):
            config = rep["config"]
            lines.append(f"  {i}. {rep['count']}x: d={config.get('d_model')}, L={config.get('n_layers')}, h={config.get('n_heads')}")
    lines.append("")

    # Value frequencies
    lines.append("### Value Frequencies")
    for param in ["d_model", "n_layers", "n_heads", "learning_rate", "dropout", "weight_decay"]:
        if param in frequencies:
            counts = frequencies[param]
            total = sum(counts.values())
            freq_str = ", ".join(f"{v}:{c}" for v, c in sorted(counts.items()))
            lines.append(f"- **{param}**: {freq_str}")
    lines.append("")

    # Coverage gaps
    lines.append("### Coverage Gaps")
    if "architecture" in gaps:
        arch = gaps["architecture"]
        lines.append(f"- **Architecture combinations**: {arch['tested']}/{arch['total_possible']} ({arch['coverage_pct']}% coverage)")
        if arch.get("untested_examples"):
            lines.append(f"  - Untested examples (d_model, n_layers, n_heads): {arch['untested_examples'][:5]}")
    if "training" in gaps:
        train = gaps["training"]
        lines.append(f"- **Training HP combinations**: {train['tested']}/{train['total_possible']} ({train['coverage_pct']}% coverage)")
    lines.append("")

    # Architecture patterns
    if patterns:
        lines.append("### Architecture Patterns")
        if "correlations" in patterns:
            corr = patterns["correlations"]
            lines.append(f"- Width↔AUC correlation: {corr.get('width_auc', 'N/A')}")
            lines.append(f"- Depth↔AUC correlation: {corr.get('depth_auc', 'N/A')}")
            lines.append(f"- Width/Depth ratio↔AUC: {corr.get('ratio_auc', 'N/A')}")

        if "avg_by_d_model" in patterns:
            lines.append("\n**Average AUC by d_model:**")
            for d, auc in sorted(patterns["avg_by_d_model"].items()):
                lines.append(f"  - d_model={d}: {auc:.4f}")

        if "avg_by_n_layers" in patterns:
            lines.append("\n**Average AUC by n_layers:**")
            for n, auc in sorted(patterns["avg_by_n_layers"].items()):
                lines.append(f"  - n_layers={n}: {auc:.4f}")
        lines.append("")

    # Training HP impact
    if training_impact:
        lines.append("### Training HP Impact")
        for param, values in training_impact.items():
            lines.append(f"\n**{param}:**")
            for val, stats in sorted(values.items(), key=lambda x: x[1]["mean_auc"], reverse=True):
                lines.append(f"  - {val}: mean={stats['mean_auc']:.4f} (n={stats['count']})")
        lines.append("")

    return "\n".join(lines)


def generate_cross_budget_comparison(all_results: dict) -> str:
    """Generate cross-budget comparison section."""
    lines = []
    lines.append("## Cross-Budget Comparison\n")

    # Summary table
    lines.append("### Summary")
    lines.append("| Budget | Best AUC | Best d_model | Best n_layers | Best n_heads | Best LR | Best dropout |")
    lines.append("|--------|----------|--------------|---------------|--------------|---------|--------------|")

    for budget in ["2M", "20M", "200M"]:
        if budget in all_results:
            bp = all_results[budget].get("best_params", {}).get("best_params", {})
            best_val = all_results[budget].get("best_params", {}).get("best_value", 0)
            lines.append(
                f"| {budget} | {best_val:.4f} | {bp.get('d_model', 'N/A')} | "
                f"{bp.get('n_layers', 'N/A')} | {bp.get('n_heads', 'N/A')} | "
                f"{bp.get('learning_rate', 'N/A')} | {bp.get('dropout', 'N/A')} |"
            )
    lines.append("")

    # Scaling law analysis
    lines.append("### Scaling Law Analysis")
    aucs = []
    for budget in ["2M", "20M", "200M"]:
        if budget in all_results:
            auc = all_results[budget].get("best_params", {}).get("best_value", 0)
            aucs.append((budget, auc))

    if len(aucs) >= 2:
        # Check if scaling law holds
        sorted_aucs = sorted(aucs, key=lambda x: {"2M": 1, "20M": 2, "200M": 3}.get(x[0], 0))
        lines.append(f"- 2M: {dict(aucs).get('2M', 'N/A'):.4f}")
        lines.append(f"- 20M: {dict(aucs).get('20M', 'N/A'):.4f}")
        lines.append(f"- 200M: {dict(aucs).get('200M', 'N/A'):.4f}")

        if dict(aucs).get("20M", 0) > dict(aucs).get("2M", 0) > dict(aucs).get("200M", 0):
            lines.append("\n**Scaling law VIOLATED**: 20M > 2M > 200M")
            lines.append("This suggests larger models may be overfitting or require different training regimes.")
        elif dict(aucs).get("200M", 0) > dict(aucs).get("20M", 0) > dict(aucs).get("2M", 0):
            lines.append("\n**Scaling law HOLDS**: 200M > 20M > 2M")
        else:
            lines.append("\n**Scaling law PARTIAL**: Non-monotonic relationship")
    lines.append("")

    # Cross-budget patterns
    lines.append("### Cross-Budget Pattern Consistency")

    # Check dropout patterns
    dropout_best = {}
    for budget in ["2M", "20M", "200M"]:
        if budget in all_results:
            bp = all_results[budget].get("best_params", {}).get("best_params", {})
            dropout_best[budget] = bp.get("dropout")

    if len(set(dropout_best.values())) > 1:
        lines.append(f"- **Dropout**: Inconsistent across budgets {dropout_best}")
    else:
        lines.append(f"- **Dropout**: Consistent at {list(dropout_best.values())[0]}")

    # Check LR patterns
    lr_best = {}
    for budget in ["2M", "20M", "200M"]:
        if budget in all_results:
            bp = all_results[budget].get("best_params", {}).get("best_params", {})
            lr_best[budget] = bp.get("learning_rate")

    if len(set(lr_best.values())) > 1:
        lines.append(f"- **Learning rate**: Inconsistent across budgets {lr_best}")
    else:
        lines.append(f"- **Learning rate**: Consistent at {list(lr_best.values())[0]}")

    lines.append("")

    # Efficiency analysis
    lines.append("### Trial Efficiency")
    for budget in ["2M", "20M", "200M"]:
        if budget in all_results:
            conv = all_results[budget].get("convergence", {})
            wasted_pct = round(100 * conv.get("wasted_trials", 0) / conv.get("total_trials", 1), 1)
            lines.append(f"- **{budget}**: {conv.get('wasted_trials', 0)}/{conv.get('total_trials', 0)} wasted ({wasted_pct}%)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze HPO coverage")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    print("=" * 70)
    print("HPO COVERAGE ANALYSIS")
    print("=" * 70)

    all_results = {}

    for budget in ["2M", "20M", "200M"]:
        print(f"\nAnalyzing {budget} budget...")

        trials = load_trials(budget)
        if not trials:
            print(f"  No trials found for {budget}")
            continue

        best_params = load_best_params(budget)
        search_space = best_params.get("search_space", {})

        frequencies = analyze_value_frequencies(trials, search_space)
        convergence = analyze_convergence(trials)
        gaps = identify_coverage_gaps(trials, search_space)
        patterns = analyze_architecture_patterns(trials)
        training_impact = analyze_training_hp_impact(trials)

        all_results[budget] = {
            "best_params": best_params,
            "frequencies": frequencies,
            "convergence": convergence,
            "gaps": gaps,
            "patterns": patterns,
            "training_impact": training_impact,
            "search_space": search_space
        }

        # Print summary
        print(f"  Best AUC: {best_params.get('best_value', 'N/A'):.4f}")
        print(f"  Unique configs: {convergence['unique_configs']}/{convergence['total_trials']}")
        print(f"  Wasted trials: {convergence['wasted_trials']}")

    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)

    report_lines = ["# HPO Coverage Analysis Report\n"]
    report_lines.append("*Generated from Phase 6C A100 HPO experiments*\n")

    # Cross-budget comparison first
    report_lines.append(generate_cross_budget_comparison(all_results))

    # Then per-budget details
    for budget in ["2M", "20M", "200M"]:
        if budget in all_results:
            r = all_results[budget]
            report_lines.append(format_markdown_report(
                budget,
                r["best_params"],
                r["frequencies"],
                r["convergence"],
                r["gaps"],
                r["patterns"],
                r["training_impact"],
                r["search_space"]
            ))

    # Methodology recommendations
    report_lines.append("## Methodology Recommendations\n")
    report_lines.append("### Issues Identified")

    total_wasted = sum(r.get("convergence", {}).get("wasted_trials", 0) for r in all_results.values())
    report_lines.append(f"1. **High trial redundancy**: {total_wasted} total wasted trials across budgets")
    report_lines.append("2. **Missing metrics**: Precision, recall, pred_range not captured in HPO")
    report_lines.append("3. **No forced extreme trials**: Pure TPE converges to local optima quickly")
    report_lines.append("4. **Inconsistent patterns**: Different optimal HPs per budget suggests search space issues")

    report_lines.append("\n### Recommended Improvements")
    report_lines.append("1. **Two-phase HPO**: Forced extremes (6 trials) + TPE exploration (44 trials)")
    report_lines.append("2. **Capture all metrics**: Save precision, recall, pred_range per trial")
    report_lines.append("3. **Coverage-aware sampling**: Ensure parameter combinations are tested")
    report_lines.append("4. **Cross-budget validation**: Test best configs from smaller budgets on larger")

    report = "\n".join(report_lines)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    else:
        # Default output
        output_path = PROJECT_ROOT / "outputs/phase6c_a100/hpo_coverage_report.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")

    # Also print to console
    print("\n" + report)

    return all_results


if __name__ == "__main__":
    main()
