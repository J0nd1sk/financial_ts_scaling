#!/usr/bin/env python3
"""Test quality of a single feature.

Usage:
    python scripts/test_feature_quality.py --feature rsi_14 --tier a200
    python scripts/test_feature_quality.py --feature random_noise --tier a200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.curation import importance, quality


def test_feature_quality(
    feature_name: str,
    feature_values: pd.Series,
    target: pd.Series,
    existing_features: pd.DataFrame | None = None,
) -> dict:
    """Test quality of a single feature.

    Args:
        feature_name: Name of the feature.
        feature_values: Feature values as a Series.
        target: Binary target Series.
        existing_features: Optional DataFrame of existing features for redundancy check.

    Returns:
        Dict with quality test results.
    """
    results = {
        "feature": feature_name,
        "data_quality": {},
        "signal_quality": {},
        "redundancy": {},
        "verdict": None,
    }

    # =========================================================================
    # Data Quality Checks
    # =========================================================================

    # Check for NaN after warmup (252 days)
    warmup_slice = feature_values.iloc[252:]
    nan_count = warmup_slice.isna().sum()
    results["data_quality"]["nan_after_warmup"] = int(nan_count)
    results["data_quality"]["nan_check"] = "PASS" if nan_count == 0 else "FAIL"

    # Check for Inf values
    inf_count = np.isinf(feature_values.replace([np.nan], 0)).sum()
    results["data_quality"]["inf_count"] = int(inf_count)
    results["data_quality"]["inf_check"] = "PASS" if inf_count == 0 else "FAIL"

    # Check for constant values
    unique_values = feature_values.dropna().nunique()
    is_constant = unique_values <= 1
    results["data_quality"]["unique_values"] = int(unique_values)
    results["data_quality"]["constant_check"] = "PASS" if not is_constant else "FAIL"

    # Value range
    min_val = feature_values.min()
    max_val = feature_values.max()
    results["data_quality"]["value_range"] = [float(min_val), float(max_val)]

    # Data quality verdict
    data_checks = [
        results["data_quality"]["nan_check"],
        results["data_quality"]["inf_check"],
        results["data_quality"]["constant_check"],
    ]
    results["data_quality"]["verdict"] = "PASS" if all(c == "PASS" for c in data_checks) else "FAIL"

    # =========================================================================
    # Signal Quality Checks
    # =========================================================================

    # Target correlation
    X = pd.DataFrame({feature_name: feature_values})
    target_corr = importance.compute_target_correlation(X, target, method="spearman")
    corr_value = abs(target_corr[feature_name])
    corr_grade = quality.grade_target_correlation(corr_value)
    results["signal_quality"]["target_correlation"] = float(corr_value)
    results["signal_quality"]["correlation_grade"] = corr_grade.value

    # Mutual information
    mi = importance.compute_mutual_information(X, target)
    mi_value = mi[feature_name]
    mi_grade = quality.grade_mutual_information(mi_value)
    results["signal_quality"]["mutual_information"] = float(mi_value)
    results["signal_quality"]["mi_grade"] = mi_grade.value

    # Signal quality verdict
    signal_grades = [corr_grade, mi_grade]
    if any(g == quality.QualityGrade.FAIL for g in signal_grades):
        results["signal_quality"]["verdict"] = "FAIL"
    elif any(g == quality.QualityGrade.MARGINAL for g in signal_grades):
        results["signal_quality"]["verdict"] = "MARGINAL"
    else:
        results["signal_quality"]["verdict"] = "PASS"

    # =========================================================================
    # Redundancy Check
    # =========================================================================

    if existing_features is not None and len(existing_features.columns) > 0:
        # Compute correlation with existing features
        all_features = existing_features.copy()
        all_features[feature_name] = feature_values

        max_redundancy = importance.get_max_redundancy(all_features)
        redundancy_value = max_redundancy[feature_name]
        redundancy_grade = quality.grade_max_redundancy(redundancy_value)

        # Find most similar feature
        corr_matrix = importance.compute_redundancy_matrix(all_features)
        other_corrs = corr_matrix[feature_name].drop(feature_name)
        most_similar_feature = other_corrs.idxmax()
        most_similar_corr = other_corrs.max()

        results["redundancy"]["max_redundancy"] = float(redundancy_value)
        results["redundancy"]["redundancy_grade"] = redundancy_grade.value
        results["redundancy"]["most_similar_feature"] = most_similar_feature
        results["redundancy"]["most_similar_correlation"] = float(most_similar_corr)
        results["redundancy"]["verdict"] = redundancy_grade.value
    else:
        results["redundancy"]["verdict"] = "PASS"
        results["redundancy"]["note"] = "No existing features for comparison"

    # =========================================================================
    # Overall Verdict
    # =========================================================================

    verdicts = [
        results["data_quality"]["verdict"],
        results["signal_quality"]["verdict"],
        results["redundancy"]["verdict"],
    ]

    if "FAIL" in verdicts:
        results["verdict"] = "FAIL"
    elif "MARGINAL" in verdicts:
        results["verdict"] = "MARGINAL"
    else:
        results["verdict"] = "PASS"

    return results


def format_quality_report(results: dict) -> str:
    """Format quality test results as markdown."""
    lines = []
    lines.append(f"## Feature Quality Test: {results['feature']}")
    lines.append(f"### Summary: {results['verdict']}")
    lines.append("")

    # Data Quality
    lines.append("### Data Quality")
    dq = results["data_quality"]
    status = "[PASS]" if dq["nan_check"] == "PASS" else "[FAIL]"
    lines.append(f"{status} No NaN after warmup: {dq['nan_after_warmup']} found")

    status = "[PASS]" if dq["inf_check"] == "PASS" else "[FAIL]"
    lines.append(f"{status} No Inf values: {dq['inf_count']} found")

    status = "[PASS]" if dq["constant_check"] == "PASS" else "[FAIL]"
    lines.append(f"{status} Non-constant: {dq['unique_values']} unique values")

    lines.append(f"    Value range: {dq['value_range']}")
    lines.append("")

    # Signal Quality
    lines.append("### Signal Quality")
    sq = results["signal_quality"]
    status = "[PASS]" if sq["correlation_grade"] == "pass" else f"[{sq['correlation_grade'].upper()}]"
    lines.append(f"{status} Target correlation: {sq['target_correlation']:.4f} (threshold: 0.02)")

    status = "[PASS]" if sq["mi_grade"] == "pass" else f"[{sq['mi_grade'].upper()}]"
    lines.append(f"{status} Mutual information: {sq['mutual_information']:.6f} (threshold: 0.001)")
    lines.append("")

    # Redundancy
    lines.append("### Redundancy")
    rd = results["redundancy"]
    if "max_redundancy" in rd:
        status = "[PASS]" if rd["redundancy_grade"] == "pass" else f"[{rd['redundancy_grade'].upper()}]"
        lines.append(f"{status} Max redundancy: {rd['max_redundancy']:.2f} (threshold: 0.95)")
        lines.append(f"    Most similar: {rd['most_similar_feature']} (r={rd['most_similar_correlation']:.2f})")
    else:
        lines.append(f"    {rd.get('note', 'N/A')}")
    lines.append("")

    # Recommendation
    lines.append("### Recommendation")
    if results["verdict"] == "PASS":
        lines.append(f"**{results['verdict']}** - Add to curated tier")
    elif results["verdict"] == "MARGINAL":
        lines.append(f"**{results['verdict']}** - Consider with review")
    else:
        lines.append(f"**{results['verdict']}** - Do not add")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test feature quality")
    parser.add_argument("--feature", required=True, help="Feature name to test")
    parser.add_argument("--tier", required=True, help="Feature tier context (a200, a500)")
    args = parser.parse_args()

    print(f"Testing feature: {args.feature}")
    print(f"Tier context: {args.tier}")
    print("=" * 60)
    print()

    # Create synthetic test data for demonstration
    # In production, this would load real data
    np.random.seed(42)
    n = 500

    # Create target
    base = np.random.randn(n)
    target = pd.Series((base > 0).astype(float))

    if args.feature == "random_noise":
        # Pure noise - should FAIL
        feature_values = pd.Series(np.random.randn(n))
    else:
        # Create a feature with some signal
        signal = 0.3 if "rsi" in args.feature.lower() else 0.2
        feature_values = pd.Series(base * signal + np.random.randn(n) * (1 - signal))

    # Create some existing features for redundancy check
    existing_features = pd.DataFrame({
        "sma_50": base * 0.2 + np.random.randn(n) * 0.8,
        "ema_20": base * 0.25 + np.random.randn(n) * 0.75,
        "rsi_14": np.random.uniform(30, 70, n),
    })

    # Run quality test
    results = test_feature_quality(
        feature_name=args.feature,
        feature_values=feature_values,
        target=target,
        existing_features=existing_features,
    )

    # Print formatted report
    report = format_quality_report(results)
    print(report)

    return 0 if results["verdict"] != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
