#!/usr/bin/env python3
"""
Comprehensive HPO Audit Script for Alternative Architecture Investigation.

Analyzes stored HPO trial data to:
1. Detect probability collapse (narrow pred_range)
2. Analyze metric distributions across trials
3. Identify parameter importance via correlation
4. Generate threshold analysis (based on stored metrics)
5. Produce comprehensive audit report

Usage:
    python scripts/audit_hpo_results.py

Output:
    outputs/hpo/architectures/audit_report.md
    outputs/hpo/architectures/audit_data.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# CONFIGURATION
# ============================================================================

HPO_BASE = PROJECT_ROOT / "outputs/hpo/architectures"
ARCHITECTURES = ["itransformer", "informer"]

# Thresholds for issue detection
COLLAPSE_THRESHOLD = 0.1  # pred_range < 0.1 indicates probability collapse
AUC_BASELINE = 0.718  # PatchTST baseline
AUC_MINIMUM = 0.55  # Below this is concerning

# Analysis configuration
RECALL_THRESHOLD_HIGH = 0.95  # Suspiciously high recall
PRECISION_THRESHOLD_LOW = 0.25  # Low precision indicates poor discrimination


# ============================================================================
# DATA LOADING
# ============================================================================

def load_trials(arch_dir: Path) -> list[dict]:
    """Load all trial JSONs from an architecture directory."""
    trials_dir = arch_dir / "trials"
    if not trials_dir.exists():
        return []

    trials = []
    for trial_file in sorted(trials_dir.glob("trial_*.json")):
        with open(trial_file) as f:
            trials.append(json.load(f))

    return trials


def load_best_params(arch_dir: Path) -> dict | None:
    """Load best_params.json if it exists."""
    best_file = arch_dir / "best_params.json"
    if best_file.exists():
        with open(best_file) as f:
            return json.load(f)
    return None


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_probability_collapse(trials: list[dict]) -> dict:
    """Detect probability collapse in trials.

    Collapse indicators:
    - pred_range < COLLAPSE_THRESHOLD
    - All predictions on one side of 0.5
    """
    collapsed_trials = []
    healthy_trials = []

    for trial in trials:
        user_attrs = trial.get("user_attrs", {})
        pred_range = user_attrs.get("pred_range", [0, 1])

        if isinstance(pred_range, list) and len(pred_range) == 2:
            range_width = pred_range[1] - pred_range[0]
            pred_min, pred_max = pred_range

            # Check for collapse
            is_collapsed = range_width < COLLAPSE_THRESHOLD
            all_high = pred_min > 0.5  # All predictions above 0.5
            all_low = pred_max < 0.5  # All predictions below 0.5

            trial_info = {
                "trial_number": trial.get("trial_number"),
                "pred_range": pred_range,
                "range_width": range_width,
                "auc": trial.get("auc", 0),
                "recall": user_attrs.get("val_recall", 0),
                "precision": user_attrs.get("val_precision", 0),
            }

            if is_collapsed or all_high or all_low:
                trial_info["collapse_type"] = (
                    "narrow_range" if is_collapsed else
                    "all_high" if all_high else "all_low"
                )
                collapsed_trials.append(trial_info)
            else:
                healthy_trials.append(trial_info)

    return {
        "total_trials": len(trials),
        "collapsed_count": len(collapsed_trials),
        "healthy_count": len(healthy_trials),
        "collapse_rate": len(collapsed_trials) / len(trials) if trials else 0,
        "collapsed_trials": collapsed_trials,
        "healthy_trials": healthy_trials,
    }


def analyze_metric_distributions(trials: list[dict]) -> dict:
    """Analyze distribution of key metrics across trials."""
    metrics = {
        "auc": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "accuracy": [],
        "pred_range_width": [],
    }

    for trial in trials:
        user_attrs = trial.get("user_attrs", {})

        metrics["auc"].append(trial.get("auc", 0) or 0)
        metrics["precision"].append(user_attrs.get("val_precision", 0) or 0)
        metrics["recall"].append(user_attrs.get("val_recall", 0) or 0)
        metrics["f1"].append(user_attrs.get("val_f1", 0) or 0)
        metrics["accuracy"].append(user_attrs.get("val_accuracy", 0) or 0)

        pred_range = user_attrs.get("pred_range", [0, 1])
        if isinstance(pred_range, list) and len(pred_range) == 2:
            metrics["pred_range_width"].append(pred_range[1] - pred_range[0])
        else:
            metrics["pred_range_width"].append(0)

    # Compute statistics
    stats = {}
    for metric_name, values in metrics.items():
        values = np.array(values)
        stats[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }

    return stats


def analyze_parameter_importance(trials: list[dict]) -> dict:
    """Analyze correlation between parameters and AUC."""
    if not trials:
        return {}

    # Extract parameters and AUC
    data = []
    for trial in trials:
        if trial.get("state") != "COMPLETE":
            continue

        params = trial.get("params", {})
        auc = trial.get("auc", 0) or 0

        row = {"auc": auc}
        row.update(params)
        data.append(row)

    if not data:
        return {}

    df = pd.DataFrame(data)

    # Compute correlations with AUC
    correlations = {}
    for col in df.columns:
        if col == "auc":
            continue

        try:
            # For numeric columns, compute Pearson correlation
            if df[col].dtype in [np.float64, np.int64]:
                corr = df["auc"].corr(df[col])
                correlations[col] = float(corr) if not np.isnan(corr) else 0
            else:
                # For categorical, compute mean AUC per category
                means = df.groupby(col)["auc"].mean()
                correlations[col] = {
                    "type": "categorical",
                    "mean_auc_by_value": means.to_dict(),
                }
        except Exception:
            continue

    # Sort by absolute correlation (for numeric params)
    numeric_corrs = {k: v for k, v in correlations.items() if isinstance(v, (int, float))}
    sorted_params = sorted(numeric_corrs.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        "correlations": correlations,
        "sorted_by_importance": sorted_params,
    }


def analyze_suspicious_metrics(trials: list[dict]) -> dict:
    """Identify trials with suspicious metric patterns."""
    suspicious = {
        "high_recall_low_precision": [],  # Recall > 0.95 with precision < 0.25
        "perfect_metrics": [],  # Any metric = 1.0 exactly
        "degenerate_auc": [],  # AUC close to 0.5 (random)
        "below_baseline": [],  # AUC < 0.55
    }

    for trial in trials:
        user_attrs = trial.get("user_attrs", {})
        trial_num = trial.get("trial_number")
        auc = trial.get("auc", 0) or 0
        recall = user_attrs.get("val_recall", 0) or 0
        precision = user_attrs.get("val_precision", 0) or 0

        # Check for high recall / low precision
        if recall > RECALL_THRESHOLD_HIGH and precision < PRECISION_THRESHOLD_LOW:
            suspicious["high_recall_low_precision"].append({
                "trial": trial_num,
                "recall": recall,
                "precision": precision,
                "auc": auc,
            })

        # Check for perfect metrics (suspicious)
        for metric_name in ["val_accuracy", "val_precision", "val_recall", "val_f1"]:
            val = user_attrs.get(metric_name, 0)
            if val == 1.0:
                suspicious["perfect_metrics"].append({
                    "trial": trial_num,
                    "metric": metric_name,
                    "value": val,
                })

        # Degenerate AUC (very close to 0.5)
        if abs(auc - 0.5) < 0.02:
            suspicious["degenerate_auc"].append({
                "trial": trial_num,
                "auc": auc,
            })

        # Below minimum threshold
        if auc < AUC_MINIMUM:
            suspicious["below_baseline"].append({
                "trial": trial_num,
                "auc": auc,
            })

    return suspicious


def generate_audit_summary(arch_name: str, trials: list[dict], best_params: dict | None) -> dict:
    """Generate complete audit summary for an architecture."""
    if not trials:
        return {"error": f"No trials found for {arch_name}"}

    collapse_analysis = analyze_probability_collapse(trials)
    metric_stats = analyze_metric_distributions(trials)
    param_importance = analyze_parameter_importance(trials)
    suspicious = analyze_suspicious_metrics(trials)

    # Determine overall validity
    issues = []

    if collapse_analysis["collapse_rate"] > 0.3:
        issues.append(f"High probability collapse rate: {collapse_analysis['collapse_rate']:.1%}")

    if len(suspicious["high_recall_low_precision"]) > len(trials) * 0.5:
        issues.append(f"Many trials with suspiciously high recall: {len(suspicious['high_recall_low_precision'])}")

    if metric_stats["auc"]["max"] < AUC_MINIMUM:
        issues.append(f"Best AUC ({metric_stats['auc']['max']:.4f}) below minimum threshold ({AUC_MINIMUM})")

    validity = "VALID" if not issues else ("PARTIAL" if len(issues) < 2 else "QUESTIONABLE")

    return {
        "architecture": arch_name,
        "total_trials": len(trials),
        "completed_trials": len([t for t in trials if t.get("state") == "COMPLETE"]),
        "best_auc": metric_stats["auc"]["max"],
        "mean_auc": metric_stats["auc"]["mean"],
        "best_params": best_params.get("best_params") if best_params else None,
        "collapse_analysis": collapse_analysis,
        "metric_distributions": metric_stats,
        "parameter_importance": param_importance,
        "suspicious_patterns": suspicious,
        "issues": issues,
        "validity_assessment": validity,
    }


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_markdown_report(audits: dict[str, dict]) -> str:
    """Generate comprehensive markdown audit report."""
    lines = [
        "# HPO Audit Report",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]

    # Summary table
    lines.extend([
        "| Architecture | Trials | Best AUC | Mean AUC | Collapse Rate | Validity |",
        "|--------------|--------|----------|----------|---------------|----------|",
    ])

    for arch_name, audit in audits.items():
        if "error" in audit:
            lines.append(f"| {arch_name} | ERROR | - | - | - | {audit['error']} |")
        else:
            lines.append(
                f"| {arch_name} | {audit['total_trials']} | "
                f"{audit['best_auc']:.4f} | {audit['mean_auc']:.4f} | "
                f"{audit['collapse_analysis']['collapse_rate']:.1%} | "
                f"**{audit['validity_assessment']}** |"
            )

    lines.extend(["", "---", ""])

    # Detailed sections per architecture
    for arch_name, audit in audits.items():
        if "error" in audit:
            lines.extend([f"## {arch_name.title()}", "", f"**Error:** {audit['error']}", ""])
            continue

        lines.extend([
            f"## {arch_name.title()}",
            "",
            f"**Trials:** {audit['total_trials']} ({audit['completed_trials']} completed)",
            f"**Best AUC:** {audit['best_auc']:.4f}",
            f"**Validity:** {audit['validity_assessment']}",
            "",
        ])

        # Issues
        if audit["issues"]:
            lines.append("### Issues Detected")
            lines.append("")
            for issue in audit["issues"]:
                lines.append(f"- {issue}")
            lines.append("")

        # Probability collapse
        collapse = audit["collapse_analysis"]
        lines.extend([
            "### Probability Collapse Analysis",
            "",
            f"- **Collapsed trials:** {collapse['collapsed_count']} / {collapse['total_trials']} ({collapse['collapse_rate']:.1%})",
            f"- **Healthy trials:** {collapse['healthy_count']}",
            "",
        ])

        if collapse["collapsed_trials"]:
            lines.extend([
                "**Collapsed Trials (sample):**",
                "",
                "| Trial | Type | Range | AUC | Recall |",
                "|-------|------|-------|-----|--------|",
            ])
            for t in collapse["collapsed_trials"][:5]:  # Show first 5
                lines.append(
                    f"| {t['trial_number']} | {t.get('collapse_type', 'unknown')} | "
                    f"[{t['pred_range'][0]:.3f}, {t['pred_range'][1]:.3f}] | "
                    f"{t['auc']:.4f} | {t['recall']:.2%} |"
                )
            lines.append("")

        # Metric distributions
        stats = audit["metric_distributions"]
        lines.extend([
            "### Metric Distributions",
            "",
            "| Metric | Mean | Std | Min | Max | Median |",
            "|--------|------|-----|-----|-----|--------|",
        ])
        for metric_name, metric_stats in stats.items():
            lines.append(
                f"| {metric_name} | {metric_stats['mean']:.4f} | "
                f"{metric_stats['std']:.4f} | {metric_stats['min']:.4f} | "
                f"{metric_stats['max']:.4f} | {metric_stats['median']:.4f} |"
            )
        lines.append("")

        # Parameter importance
        if audit["parameter_importance"].get("sorted_by_importance"):
            lines.extend([
                "### Parameter Importance (correlation with AUC)",
                "",
                "| Parameter | Correlation |",
                "|-----------|-------------|",
            ])
            for param, corr in audit["parameter_importance"]["sorted_by_importance"][:8]:
                lines.append(f"| {param} | {corr:+.3f} |")
            lines.append("")

        # Suspicious patterns
        suspicious = audit["suspicious_patterns"]
        if any(suspicious.values()):
            lines.extend([
                "### Suspicious Patterns",
                "",
            ])

            if suspicious["high_recall_low_precision"]:
                lines.append(f"- **High recall / low precision:** {len(suspicious['high_recall_low_precision'])} trials")
            if suspicious["degenerate_auc"]:
                lines.append(f"- **Degenerate AUC (~0.5):** {len(suspicious['degenerate_auc'])} trials")
            if suspicious["below_baseline"]:
                lines.append(f"- **Below minimum AUC:** {len(suspicious['below_baseline'])} trials")

            lines.append("")

        # Best configuration
        if audit["best_params"]:
            lines.extend([
                "### Best Configuration",
                "",
            ])
            for param, value in audit["best_params"].items():
                if isinstance(value, float):
                    lines.append(f"- **{param}:** {value:.6g}")
                else:
                    lines.append(f"- **{param}:** {value}")
            lines.append("")

        lines.extend(["---", ""])

    # Conclusions
    lines.extend([
        "## Conclusions",
        "",
    ])

    all_valid = all(
        audit.get("validity_assessment") == "VALID"
        for audit in audits.values()
        if "error" not in audit
    )

    if all_valid:
        lines.append("**Overall Assessment:** All architectures show valid HPO results.")
    else:
        lines.append("**Overall Assessment:** Some issues detected. See individual architecture sections for details.")
        lines.append("")
        lines.append("### Recommendations")
        lines.append("")

        for arch_name, audit in audits.items():
            if "error" in audit:
                lines.append(f"- **{arch_name}:** No data available. Run HPO first.")
            elif audit["issues"]:
                for issue in audit["issues"]:
                    lines.append(f"- **{arch_name}:** {issue}")

    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by audit_hpo_results.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run comprehensive HPO audit."""
    print("=" * 70)
    print("HPO AUDIT - Alternative Architectures")
    print("=" * 70)
    print()

    audits = {}

    for arch_name in ARCHITECTURES:
        arch_dir = HPO_BASE / arch_name
        print(f"Auditing {arch_name}...")

        if not arch_dir.exists():
            print(f"  Directory not found: {arch_dir}")
            audits[arch_name] = {"error": "Directory not found"}
            continue

        # Load data
        trials = load_trials(arch_dir)
        best_params = load_best_params(arch_dir)

        if not trials:
            print(f"  No trials found")
            audits[arch_name] = {"error": "No trials found"}
            continue

        print(f"  Found {len(trials)} trials")

        # Generate audit
        audit = generate_audit_summary(arch_name, trials, best_params)
        audits[arch_name] = audit

        # Print summary
        print(f"  Best AUC: {audit['best_auc']:.4f}")
        print(f"  Collapse rate: {audit['collapse_analysis']['collapse_rate']:.1%}")
        print(f"  Validity: {audit['validity_assessment']}")
        print()

    # Generate report
    print("Generating audit report...")
    report = generate_markdown_report(audits)

    # Save outputs
    output_dir = HPO_BASE
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "audit_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report: {report_path}")

    data_path = output_dir / "audit_data.json"
    with open(data_path, "w") as f:
        json.dump(audits, f, indent=2, default=str)
    print(f"  Data: {data_path}")

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    # Print overall summary
    print()
    for arch_name, audit in audits.items():
        if "error" in audit:
            status = f"ERROR: {audit['error']}"
        else:
            status = audit["validity_assessment"]
        print(f"  {arch_name}: {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
