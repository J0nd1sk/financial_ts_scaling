#!/usr/bin/env python3
"""Build a combined dataset parquet with OHLCV, features, and optional labels.

Inputs:
- Raw OHLCV: data/raw/SPY.parquet
- Features: data/processed/v1/SPY_features_a20.parquet
- Optional labels: data/processed/v1/SPY_labels_thresholds.parquet

Outputs:
- Combined parquet (default: data/processed/v1/SPY_dataset_a20.parquet)
- Manifest registration as processed dataset with sources = raw + features (+ labels if provided)

Columns order:
Date, Open, High, Low, Close, Volume, [features...], [labels...]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from scripts import manage_data_versions as dv

DEFAULT_RAW_PATH = Path("data/raw/SPY.parquet")
DEFAULT_FEATURES_PATH = Path("data/processed/v1/SPY_features_a20.parquet")
DEFAULT_LABELS_PATH = Path("data/processed/v1/SPY_labels_thresholds.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/v1/SPY_dataset_a20.parquet")

DEFAULT_DATASET = "SPY.dataset.a20"
DEFAULT_TIER = "a20"

OHLCV_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def build_combined(
    raw_path: Path,
    features_path: Path,
    labels_path: Path | None,
    include_labels: bool,
) -> pd.DataFrame:
    raw = pd.read_parquet(raw_path)
    feats = pd.read_parquet(features_path)

    raw["Date"] = pd.to_datetime(raw["Date"])
    feats["Date"] = pd.to_datetime(feats["Date"])

    df = pd.merge(raw, feats, on="Date", how="inner", validate="one_to_one")

    if include_labels and labels_path and labels_path.exists():
        labels = pd.read_parquet(labels_path)
        labels["Date"] = pd.to_datetime(labels["Date"])
        df = pd.merge(df, labels, on="Date", how="inner", validate="one_to_one")

    df = df.dropna().reset_index(drop=True)
    return df


def register_combined(
    output_path: Path,
    dataset: str,
    version: int,
    tier: str,
    sources: Sequence[Path],
) -> None:
    md5s = [dv._compute_md5(p) for p in sources]  # type: ignore[attr-defined]
    dv.register_processed_entry(
        dataset=dataset,
        version=version,
        tier=tier,
        file_path=output_path,
        source_raw_md5s=md5s,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build combined dataset parquet.")
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH, help="Path to raw OHLCV parquet")
    parser.add_argument("--features-path", type=Path, default=DEFAULT_FEATURES_PATH, help="Path to features parquet")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_LABELS_PATH, help="Path to labels parquet (optional)")
    parser.add_argument("--include-labels", action="store_true", help="Include labels if labels-path exists")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to write combined parquet")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset name to register")
    parser.add_argument("--version", type=int, default=1, help="Processed dataset version")
    parser.add_argument("--tier", type=str, default=DEFAULT_TIER, help="Tier name")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    df = build_combined(
        raw_path=args.raw_path,
        features_path=args.features_path,
        labels_path=args.labels_path if args.include_labels else None,
        include_labels=args.include_labels,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output_path, index=False)

    sources = [args.raw_path, args.features_path]
    if args.include_labels and args.labels_path.exists():
        sources.append(args.labels_path)
    register_combined(
        output_path=args.output_path,
        dataset=args.dataset,
        version=args.version,
        tier=args.tier,
        sources=sources,
    )
    print(f"Wrote combined dataset: {len(df)} rows to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

