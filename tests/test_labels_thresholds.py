"""Tests for threshold label generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts import build_labels_thresholds as blt


def test_build_labels_computes_expected(tmp_path: Path) -> None:
    # Synthetic raw close prices over 5 days
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    close = pd.Series([100, 101, 102, 103, 104], index=dates)
    raw_df = pd.DataFrame({"Date": dates, "Close": close, "Open": close, "High": close, "Low": close, "Volume": 1})

    # Features exist for first 4 days (simulate indicator warmup alignment)
    features = pd.DataFrame({"Date": dates[:4], "dummy": [0, 0, 0, 0]})
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)

    labels = blt.build_labels(
        raw_df=raw_df,
        feature_dates=features["Date"],
        horizons=[2],
        thresholds=[0.01, 0.05],
    )

    expected_cols = {"Date", "future_max_h2", "label_h2_t1pct", "label_h2_t5pct"}
    assert set(labels.columns) == expected_cols

    # Rows with sufficient future window: first 3 days after alignment/intersection
    assert len(labels) == 3

    # Compute expected labels manually
    # Day0: future max over next 2 days = 102; 1% label=1 (102>=101), 5% label=0 (102<105)
    # Day1: future max over next 2 days = 103; 1% label=1 (103>=102.01), 5% label=0 (103<106.05)
    # Day2: future max over next 2 days = 104; 1% label=1 (104>=103.02), 5% label=0 (104<107.1)
    assert labels["label_h2_t1pct"].tolist() == [1, 1, 1]
    assert labels["label_h2_t5pct"].tolist() == [0, 0, 0]


def test_write_labels_registers_manifest(tmp_path: Path, monkeypatch) -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    raw_df = pd.DataFrame(
        {
            "Date": dates,
            "Close": [100, 102, 104, 106],
            "Open": [100, 102, 104, 106],
            "High": [100, 102, 104, 106],
            "Low": [100, 102, 104, 106],
            "Volume": 1,
        }
    )
    raw_path = tmp_path / "raw.parquet"
    raw_df.to_parquet(raw_path, index=False)

    features = pd.DataFrame({"Date": dates, "dummy": 0})
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)

    output_path = tmp_path / "labels.parquet"

    calls = {}

    def fake_register(dataset, version, tier, file_path, source_raw_md5s, manifest_path=None):
        calls["dataset"] = dataset
        calls["version"] = version
        calls["tier"] = tier
        calls["file_path"] = file_path
        calls["sources"] = list(source_raw_md5s)
        return None

    monkeypatch.setattr(blt.dv, "register_processed_entry", fake_register)

    blt.write_labels_and_register(
        raw_path=raw_path,
        features_path=features_path,
        output_path=output_path,
        dataset="SPY.labels.daily",
        version=1,
        tier="thresholds",
        horizons=[1],
        thresholds=[0.01],
    )

    assert output_path.exists()
    df = pd.read_parquet(output_path)
    assert not df.empty
    assert calls["dataset"] == "SPY.labels.daily"
    assert calls["version"] == 1
    assert calls["tier"] == "thresholds"
    assert len(calls["sources"]) == 2

