#!/usr/bin/env python3
"""Build threshold-based labels for backtesting and training sanity checks.

Labels are computed as:
    future_max_h{H} = max(Close[t+1 : t+H])
    label_h{H}_t{T}pct = 1 if future_max_h{H} >= Close[t] * (1 + T) else 0

Defaults:
- Horizons: 1, 2, 3, 5 days
- Thresholds: 1%, 2%, 5%
- Input features: data/processed/v1/SPY_features_a20.parquet (for date alignment)
- Input raw: data/raw/SPY.parquet
- Output: data/processed/v1/SPY_labels_thresholds.parquet

The output is manifest-registered as a processed artifact with tier "thresholds"
and dataset name "SPY.labels.daily".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from scripts import manage_data_versions as dv

DEFAULT_FEATURES_PATH = Path("data/processed/v1/SPY_features_a20.parquet")
DEFAULT_RAW_PATH = Path("data/raw/SPY.parquet")
DEFAULT_OUTPUT_PATH = Path("data/processed/v1/SPY_labels_thresholds.parquet")
DEFAULT_DATASET = "SPY.labels.daily"
DEFAULT_TIER = "thresholds"


def _future_max(close: pd.Series, horizon: int) -> pd.Series:
    """Compute future max over next `horizon` days (excluding current)."""
    return close.shift(-1).iloc[::-1].rolling(window=horizon, min_periods=horizon).max().iloc[::-1]


def build_labels(
    raw_df: pd.DataFrame,
    feature_dates: pd.Series,
    horizons: Sequence[int],
    thresholds: Sequence[float],
) -> pd.DataFrame:
    """Compute labels aligned to feature dates."""
    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    close = df["Close"]

    out = pd.DataFrame({"Date": df["Date"]})

    for horizon in horizons:
        future_col = f"future_max_h{horizon}"
        out[future_col] = _future_max(close, horizon)
        for thr in thresholds:
            pct = int(round(thr * 100))
            label_col = f"label_h{horizon}_t{pct}pct"
            out[label_col] = (out[future_col] >= close * (1 + thr)).astype(int)

    # Align to feature dates and drop rows with insufficient future window
    feature_dates = pd.to_datetime(feature_dates)
    merged = pd.merge(pd.DataFrame({"Date": feature_dates}), out, on="Date", how="inner")
    merged = merged.dropna().reset_index(drop=True)
    return merged


def write_labels_and_register(
    raw_path: Path,
    features_path: Path,
    output_path: Path,
    dataset: str,
    version: int,
    tier: str,
    horizons: Sequence[int],
    thresholds: Sequence[float],
) -> Path:
    raw_df = pd.read_parquet(raw_path)
    feature_dates = pd.read_parquet(features_path)["Date"]
    labels = build_labels(raw_df, feature_dates, horizons, thresholds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_path, index=False)

    raw_md5 = dv._compute_md5(raw_path)  # type: ignore[attr-defined]
    feat_md5 = dv._compute_md5(features_path)  # type: ignore[attr-defined]
    dv.register_processed_entry(
        dataset=dataset,
        version=version,
        tier=tier,
        file_path=output_path,
        source_raw_md5s=[raw_md5, feat_md5],
    )
    print(f"Wrote labels: {len(labels)} rows to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build threshold labels for SPY.")
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH, help="Path to raw SPY parquet")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help="Path to processed features parquet (for date alignment)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write labels parquet",
    )
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset name to register")
    parser.add_argument("--version", type=int, default=1, help="Processed dataset version")
    parser.add_argument("--tier", type=str, default=DEFAULT_TIER, help="Tier name (e.g., thresholds)")
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,2,3,5",
        help="Comma-separated horizons (days)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.01,0.02,0.05",
        help="Comma-separated thresholds (fractions, e.g., 0.01 for 1%%)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x]
    thresholds = [float(x) for x in args.thresholds.split(",") if x]
    write_labels_and_register(
        raw_path=args.raw_path,
        features_path=args.features_path,
        output_path=args.output_path,
        dataset=args.dataset,
        version=args.version,
        tier=args.tier,
        horizons=horizons,
        thresholds=thresholds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

