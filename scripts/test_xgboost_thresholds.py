#!/usr/bin/env python3
"""Test XGBoost on multiple thresholds for comparison with RF and PatchTST."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_targets(df: pd.DataFrame, threshold: float) -> pd.Series:
    """Create binary target: 1 if next-day return > threshold."""
    returns = df["Close"].pct_change().shift(-1)
    return (returns > threshold).astype(int)


def run_xgboost_experiment(threshold: float, threshold_name: str):
    """Run XGBoost on a single threshold."""
    print(f"\n{'='*60}")
    print(f"XGBoost - {threshold_name} threshold ({threshold*100:.1f}%)")
    print(f"{'='*60}")

    # Load data
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data/processed/v1/SPY_dataset_c.parquet"
    df = pd.read_parquet(data_path)

    # Create target
    df["target"] = create_targets(df, threshold)
    df = df.dropna(subset=["target"])

    # Get feature columns (exclude Date, OHLCV, target)
    exclude = ["Date", "Open", "High", "Low", "Close", "Volume", "target"]
    feature_cols = [c for c in df.columns if c not in exclude]

    # Split data using SimpleSplitter logic
    # Train: <2023, Val: 2023-2024, Test: 2025+
    train_mask = df["Date"] < "2023-01-01"
    val_mask = (df["Date"] >= "2023-01-01") & (df["Date"] < "2025-01-01")
    test_mask = df["Date"] >= "2025-01-01"

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, "target"].values
    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, "target"].values

    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")
    print(f"Train positive rate: {y_train.mean():.1%}")
    print(f"Val positive rate: {y_val.mean():.1%}")

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc",
        early_stopping_rounds=10,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probs)

    print(f"\nResults:")
    print(f"  Val AUC: {val_auc:.4f}")
    print(f"  Prediction range: [{val_probs.min():.3f}, {val_probs.max():.3f}]")
    print(f"  Prediction std: {val_probs.std():.4f}")

    # Feature importance (top 5)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-5:][::-1]
    print(f"\nTop 5 features:")
    for idx in top_idx:
        print(f"  {feature_cols[idx]}: {importances[idx]:.1%}")

    return {
        "threshold": threshold_name,
        "val_auc": val_auc,
        "pred_min": float(val_probs.min()),
        "pred_max": float(val_probs.max()),
        "pred_std": float(val_probs.std()),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "val_pos_rate": float(y_val.mean()),
    }


def main():
    thresholds = [
        (0.005, "0.5%"),
        (0.01, "1.0%"),
        (0.02, "2.0%"),
    ]

    results = []
    for threshold, name in thresholds:
        result = run_xgboost_experiment(threshold, name)
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: XGBoost Results")
    print(f"{'='*60}")
    print(f"{'Threshold':<12} {'Val AUC':<10} {'Pred Range':<20} {'Pos Rate':<10}")
    print("-" * 52)
    for r in results:
        pred_range = f"[{r['pred_min']:.3f}, {r['pred_max']:.3f}]"
        print(f"{r['threshold']:<12} {r['val_auc']:<10.4f} {pred_range:<20} {r['val_pos_rate']:<10.1%}")

    # Save results
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "outputs/threshold_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "xgboost_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'xgboost_results.json'}")


if __name__ == "__main__":
    main()
