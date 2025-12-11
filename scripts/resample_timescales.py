#!/usr/bin/env python3
"""Resample raw OHLCV data to lower frequencies for multi-timescale experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts import manage_data_versions as dv
from src.features.resample import get_freq_string, resample_ohlcv


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Resample OHLCV data to lower frequency."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help="Ticker symbol (default: SPY)",
    )
    parser.add_argument(
        "--timescale",
        type=str,
        required=True,
        choices=["2d", "3d", "5d", "weekly"],
        help="Target timescale: 2d, 3d, 5d, weekly",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=None,
        help="Path to raw parquet (default: data/raw/{TICKER}.parquet)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path (default: data/processed/v1/{TICKER}_OHLCV_{timescale}.parquet)",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Processed dataset version (default: 1)",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Compute default paths if not provided
    if args.raw_path is None:
        args.raw_path = Path(f"data/raw/{args.ticker}.parquet")

    if args.output_path is None:
        args.output_path = Path(
            f"data/processed/v1/{args.ticker}_OHLCV_{args.timescale}.parquet"
        )

    # Validate raw file exists
    if not args.raw_path.exists():
        print(f"Error: Raw file not found: {args.raw_path}")
        return 1

    # Load raw data
    print(f"Loading raw data from {args.raw_path}")
    raw_df = pd.read_parquet(args.raw_path)

    # Get pandas frequency string
    freq = get_freq_string(args.timescale)
    print(f"Resampling to {args.timescale} (freq={freq})")

    # Resample
    resampled_df = resample_ohlcv(raw_df, freq)

    # Save output
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    resampled_df.to_parquet(args.output_path, index=False)
    print(f"Wrote {len(resampled_df)} rows to {args.output_path}")

    # Register in manifest
    dataset_name = f"{args.ticker}.OHLCV.{args.timescale}"
    raw_md5 = dv._compute_md5(args.raw_path)  # type: ignore[attr-defined]
    dv.register_processed_entry(
        dataset=dataset_name,
        version=args.version,
        tier=args.timescale,
        file_path=args.output_path,
        source_raw_md5s=[raw_md5],
    )
    print(f"Registered {dataset_name} in processed manifest")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
