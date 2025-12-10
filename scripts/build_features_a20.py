#!/usr/bin/env python3
"""Build feature dataset for parameter scaling experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import manage_data_versions as dv
from src.features import tier_a20

PROCESSED_TIER = "a20"


def get_dataset_names(ticker: str) -> tuple[str, str]:
    """Return (raw_dataset_name, processed_dataset_name) for a ticker."""
    raw_name = f"{ticker}.OHLCV.daily"
    processed_name = f"{ticker}.features.a20"
    return raw_name, processed_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature dataset.")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--version", type=int, default=1, help="Processed dataset version")
    args, _ = parser.parse_known_args()

    # Compute default paths from ticker
    default_raw = Path(f"data/raw/{args.ticker}.parquet")
    default_output = Path(f"data/processed/v1/{args.ticker}_features_a20.parquet")

    # Re-parse with computed defaults
    parser.add_argument("--raw-path", type=Path, default=default_raw, help="Path to raw parquet")
    parser.add_argument("--output-path", type=Path, default=default_output, help="Path for feature parquet")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _, processed_dataset = get_dataset_names(args.ticker)

    raw_df = tier_a20.load_raw_data(args.raw_path)
    feature_df = tier_a20.build_feature_dataframe(raw_df)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(args.output_path, index=False)

    raw_md5 = dv._compute_md5(args.raw_path)  # type: ignore[attr-defined]
    dv.register_processed_entry(
        dataset=processed_dataset,
        version=args.version,
        tier=PROCESSED_TIER,
        file_path=args.output_path,
        source_raw_md5s=[raw_md5],
    )
    print(f"Wrote {len(feature_df)} rows for {args.ticker} with {len(tier_a20.FEATURE_LIST)} features")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
