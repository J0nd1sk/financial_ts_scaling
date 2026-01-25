#!/usr/bin/env python
"""Build tier a100 features for a given ticker.

Usage:
    ./venv/bin/python scripts/build_features_a100.py \
        --ticker SPY \
        --raw-path data/raw/SPY.parquet \
        --vix-path data/raw/VIX.parquet \
        --output-path data/processed/v1/SPY_dataset_a100.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import tier_a100


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tier a100 features")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g., SPY)")
    parser.add_argument("--raw-path", required=True, help="Path to raw OHLCV parquet")
    parser.add_argument("--vix-path", required=True, help="Path to VIX parquet")
    parser.add_argument("--output-path", required=True, help="Output parquet path")
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    vix_path = Path(args.vix_path)
    output_path = Path(args.output_path)

    if not raw_path.exists():
        print(f"ERROR: Raw data not found at {raw_path}")
        sys.exit(1)

    if not vix_path.exists():
        print(f"ERROR: VIX data not found at {vix_path}")
        sys.exit(1)

    # Load data
    print(f"Loading raw data from {raw_path}")
    raw_df = pd.read_parquet(raw_path)
    print(f"  Rows: {len(raw_df)}, Date range: {raw_df['Date'].min()} to {raw_df['Date'].max()}")

    print(f"Loading VIX data from {vix_path}")
    vix_df = pd.read_parquet(vix_path)
    print(f"  Rows: {len(vix_df)}, Date range: {vix_df['Date'].min()} to {vix_df['Date'].max()}")

    # Build features
    print("Building tier a100 features...")
    feature_df = tier_a100.build_feature_dataframe(raw_df, vix_df)

    # Report results
    print(f"\nFeature DataFrame:")
    print(f"  Rows: {len(feature_df)}")
    print(f"  Columns: {len(feature_df.columns)} (Date + {len(tier_a100.FEATURE_LIST)} features)")
    print(f"  Date range: {feature_df['Date'].min()} to {feature_df['Date'].max()}")

    # Check for NaN
    nan_count = feature_df.drop(columns=["Date"]).isnull().sum().sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values found!")
        # Show which columns have NaN
        nan_cols = feature_df.columns[feature_df.isnull().any()].tolist()
        print(f"  Columns with NaN: {nan_cols}")
    else:
        print("  No NaN values")

    # Verify feature count
    expected_features = len(tier_a100.FEATURE_LIST)
    actual_features = len(feature_df.columns) - 1  # Exclude Date
    if actual_features != expected_features:
        print(f"  WARNING: Expected {expected_features} features, got {actual_features}!")
    else:
        print(f"  Feature count verified: {actual_features}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(output_path, index=False)
    print(f"\nSaved to {output_path}")

    # Print feature list summary
    print("\nFeatures (a100) - first 50 from a50:")
    for i, feat in enumerate(tier_a100.FEATURE_LIST[:50], 1):
        print(f"  {i:3d}. {feat}")

    print("\nFeatures (a100) - new 50 from a100:")
    for i, feat in enumerate(tier_a100.FEATURE_LIST[50:], 51):
        print(f"  {i:3d}. {feat}")


if __name__ == "__main__":
    main()
