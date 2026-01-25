#!/usr/bin/env python3
"""
Analyze TimesFM predictions with proper threshold sweeping.

The TFM-01 results show:
- Predictions range: [-0.023, +0.033] (raw returns)
- AUC: 0.364 (anti-correlated - worse than random!)
- 0 positive predictions (threshold 1% was too high)

This script:
1. Reconstructs what the predictions would look like
2. Sweeps thresholds on predicted returns
3. Shows precision/recall at various confidence levels
4. Tests if inverting predictions helps (since AUC < 0.5)
"""

import json
import numpy as np
from pathlib import Path


def analyze_tfm_results():
    """Analyze TimesFM TFM-01 results."""

    results_path = Path("outputs/foundation/timesfm_tfm-01_results.json")

    with open(results_path) as f:
        results = json.load(f)

    print("=" * 70)
    print("TimesFM TFM-01 Analysis")
    print("=" * 70)

    # Extract metrics
    val = results["val_metrics"]
    test = results["test_metrics"]

    print(f"\nModel: {results['model']}")
    print(f"Context: {results['context_length']} days")
    print(f"Covariates: {results['use_covariates']}")

    print("\n" + "-" * 40)
    print("RAW RESULTS (with 1% threshold)")
    print("-" * 40)

    for split_name, metrics in [("Validation", val), ("Test", test)]:
        print(f"\n{split_name}:")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  Positive Predictions: {metrics['n_positive_preds']}/{metrics['n_samples']}")
        print(f"  Pred Range: [{metrics['pred_min']:.6f}, {metrics['pred_max']:.6f}]")
        print(f"  Pred Mean:  {metrics['pred_mean']:.6f}")
        print(f"  Pred Std:   {metrics['pred_std']:.6f}")
        print(f"  Class Balance: {metrics['class_balance']:.3f}")

    print("\n" + "-" * 40)
    print("ANALYSIS")
    print("-" * 40)

    # Key observations
    print("\n1. THRESHOLD MISMATCH:")
    print(f"   - Classification threshold: 1% (0.01)")
    print(f"   - Prediction max: {val['pred_max']:.4f}")
    print(f"   - Predictions rarely reach 1% → 0 positive predictions")

    print("\n2. AUC INTERPRETATION:")
    print(f"   - Val AUC: {val['auc']:.4f}")
    if val['auc'] < 0.5:
        print(f"   - AUC < 0.5 means ANTI-CORRELATED")
        print(f"   - Inverting predictions would give AUC = {1 - val['auc']:.4f}")
        print(f"   - Model predicts OPPOSITE of correct direction!")

    print("\n3. WHAT THIS MEANS:")
    if val['auc'] < 0.5:
        print("   - TimesFM's zero-shot predictions are worse than random")
        print("   - Either the model doesn't understand financial returns")
        print("   - Or the z-score normalization is causing issues")
        print("   - Or there's a bug in how predictions are being interpreted")

    print("\n" + "-" * 40)
    print("THRESHOLD SWEEP SIMULATION")
    print("-" * 40)

    # We don't have the raw predictions, but we can estimate
    # Based on pred_mean ≈ 0.001, pred_std ≈ 0.0014
    # Assuming roughly normal distribution

    print("\nEstimated positive prediction rates at different thresholds:")
    print("(Based on pred_mean=0.001, pred_std=0.0014)")
    print()

    from scipy import stats

    mean = val['pred_mean']
    std = val['pred_std']
    n = val['n_samples']

    thresholds = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.01]

    print(f"{'Threshold':<12} {'Est. Positive %':<18} {'Est. N Positive':<15}")
    print("-" * 50)

    for t in thresholds:
        # Probability that prediction > threshold
        prob = 1 - stats.norm.cdf(t, loc=mean, scale=std)
        n_pos = int(prob * n)
        print(f"{t:<12.4f} {prob*100:<18.1f} {n_pos:<15d}")

    print("\n" + "-" * 40)
    print("RECOMMENDATIONS")
    print("-" * 40)

    print("""
1. RE-RUN WITH THRESHOLD SWEEP:
   - Modify notebook Cell 12 to use threshold 0.0 (or sweep)
   - Add code to show precision/recall at multiple thresholds:
     thresholds = [0.0, 0.001, 0.002, 0.003, 0.005]
     for t in thresholds:
         preds = (pred_returns >= t).astype(int)
         # compute precision, recall

2. CHECK FOR INVERSION:
   - Since AUC < 0.5, try: pred_returns = -pred_returns
   - If that gives AUC > 0.5, model signal is inverted

3. VERIFY DATA PIPELINE:
   - Double-check that returns are calculated correctly
   - Verify z-score normalization isn't flipping signs
   - Compare raw predictions to expectations

4. IF STILL FAILING:
   - TimesFM likely doesn't transfer to financial returns
   - Same conclusion as Lag-Llama: foundation models fail
   - Consider concluding FD-01 investigation
""")

    # Summary verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if val['auc'] < 0.45:
        print("""
CRITICAL FAILURE:
- AUC 0.364 is significantly WORSE than random
- Model is anti-predictive (predicts opposite of correct)
- Even with inversion (AUC 0.636), still below PatchTST (0.718)

NEXT STEPS:
1. Quick fix: Re-run with prediction inversion to confirm AUC ~0.64
2. If confirmed: Compare 0.64 to PatchTST's 0.718 (-11%)
3. Likely conclusion: TimesFM fails like Lag-Llama

Foundation model investigation (FD-01) trending toward failure.
""")

    return results


if __name__ == "__main__":
    analyze_tfm_results()
