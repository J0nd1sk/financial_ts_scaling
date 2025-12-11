#!/usr/bin/env python3
"""Build VIX feature dataset (tier c) for scaling experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import manage_data_versions as dv
from src.features import tier_c_vix

PROCESSED_TIER = "c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VIX features (tier c).")
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=Path("data/raw/VIX.parquet"),
        help="Path to raw VIX parquet",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/v1/VIX_features_c.parquet"),
        help="Path for VIX feature parquet",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Processed dataset version",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load raw VIX data
    raw_df = tier_c_vix.pd.read_parquet(args.raw_path)

    # Build VIX features
    feature_df = tier_c_vix.build_vix_features(raw_df)

    # Write output
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(args.output_path, index=False)

    # Register in manifest
    raw_md5 = dv._compute_md5(args.raw_path)  # type: ignore[attr-defined]
    dv.register_processed_entry(
        dataset="VIX.features.c",
        version=args.version,
        tier=PROCESSED_TIER,
        file_path=args.output_path,
        source_raw_md5s=[raw_md5],
    )

    print(f"Wrote {len(feature_df)} rows with {len(tier_c_vix.VIX_FEATURE_LIST)} VIX features")
    print(f"Output: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
