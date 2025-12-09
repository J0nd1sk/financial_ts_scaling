"""Tests for combined dataset builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import build_dataset_combined as bdc


def test_build_combined_features_only(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame(
        {
            "Date": dates,
            "Open": [1, 2, 3],
            "High": [1, 2, 3],
            "Low": [1, 2, 3],
            "Close": [1, 2, 3],
            "Volume": [10, 20, 30],
        }
    )
    feats = pd.DataFrame(
        {
            "Date": dates,
            "f1": [0.1, 0.2, 0.3],
        }
    )
    raw_path = tmp_path / "raw.parquet"
    feats_path = tmp_path / "feats.parquet"
    raw.to_parquet(raw_path, index=False)
    feats.to_parquet(feats_path, index=False)

    df = bdc.build_combined(raw_path, feats_path, labels_path=None, include_labels=False)

    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume", "f1"]
    assert len(df) == 3
    assert df["f1"].tolist() == [0.1, 0.2, 0.3]


def test_build_combined_with_labels(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    raw = pd.DataFrame(
        {
            "Date": dates,
            "Open": [1, 2, 3, 4],
            "High": [1, 2, 3, 4],
            "Low": [1, 2, 3, 4],
            "Close": [1, 2, 3, 4],
            "Volume": [10, 20, 30, 40],
        }
    )
    feats = pd.DataFrame({"Date": dates, "f1": [0.1, 0.2, 0.3, 0.4]})
    labels = pd.DataFrame({"Date": dates, "label_h1_t1pct": [0, 1, 1, 0]})
    raw_path = tmp_path / "raw.parquet"
    feats_path = tmp_path / "feats.parquet"
    labels_path = tmp_path / "labels.parquet"
    raw.to_parquet(raw_path, index=False)
    feats.to_parquet(feats_path, index=False)
    labels.to_parquet(labels_path, index=False)

    df = bdc.build_combined(raw_path, feats_path, labels_path=labels_path, include_labels=True)

    assert "label_h1_t1pct" in df.columns
    assert len(df) == 4
    assert df["label_h1_t1pct"].tolist() == [0, 1, 1, 0]

