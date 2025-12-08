#!/usr/bin/env python3
"""Build feature dataset for SPY parameter scaling experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts import manage_data_versions as dv
from src.features import tier_a20

RAW_DATASET = "SPY.OHLCV.daily"
PROCESSED_DATASET = "SPY.features.a20"
PROCESSED_TIER = "a20"
RAW_PATH_DEFAULT = Path("data/raw/SPY.parquet")
OUTPUT_PATH_DEFAULT = Path("data/processed/v1/SPY_features_a20.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SPY feature dataset.")
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH_DEFAULT, help="Path to raw SPY parquet")
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH_DEFAULT, help="Path for feature parquet")
    parser.add_argument("--version", type=int, default=1, help="Processed dataset version")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_df = tier_a20.load_raw_data(args.raw_path)
    feature_df = tier_a20.build_feature_dataframe(raw_df)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(args.output_path, index=False)

    raw_md5 = dv._compute_md5(args.raw_path)  # type: ignore[attr-defined]
    dv.register_processed_entry(
        dataset=PROCESSED_DATASET,
        version=args.version,
        tier=PROCESSED_TIER,
        file_path=args.output_path,
        source_raw_md5s=[raw_md5],
    )
    print(f"Wrote {len(feature_df)} feature rows with columns: {tier_a20.FEATURE_LIST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

