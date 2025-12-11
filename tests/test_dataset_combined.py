"""Tests for combined dataset builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import build_dataset_combined as bdc
from src.features.tier_a20 import FEATURE_LIST as A20_FEATURES
from src.features.tier_c_vix import VIX_FEATURE_LIST


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


# =============================================================================
# VIX Integration Tests (Task 7)
# =============================================================================


def test_combine_spy_with_vix(tmp_path: Path) -> None:
    """Test that SPY features merge correctly with VIX features."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")

    # Raw OHLCV
    raw = pd.DataFrame({
        "Date": dates,
        "Open": [100.0, 101.0, 102.0],
        "High": [101.0, 102.0, 103.0],
        "Low": [99.0, 100.0, 101.0],
        "Close": [100.5, 101.5, 102.5],
        "Volume": [1000, 1100, 1200],
    })

    # Asset features (simplified - just one feature for test)
    feats = pd.DataFrame({"Date": dates, "indicator_1": [0.5, 0.6, 0.7]})

    # VIX features (8 columns as per tier_c_vix)
    vix_feats = pd.DataFrame({
        "Date": dates,
        "vix_close": [15.0, 16.0, 17.0],
        "vix_sma_10": [14.5, 15.5, 16.5],
        "vix_sma_20": [14.0, 15.0, 16.0],
        "vix_percentile_60d": [50.0, 55.0, 60.0],
        "vix_zscore_20d": [0.1, 0.2, 0.3],
        "vix_regime": ["normal", "normal", "normal"],
        "vix_change_1d": [1.0, 2.0, 3.0],
        "vix_change_5d": [2.0, 3.0, 4.0],
    })

    # Write to parquet
    raw_path = tmp_path / "raw.parquet"
    feats_path = tmp_path / "feats.parquet"
    vix_path = tmp_path / "vix.parquet"
    raw.to_parquet(raw_path, index=False)
    feats.to_parquet(feats_path, index=False)
    vix_feats.to_parquet(vix_path, index=False)

    # Call build_combined with VIX
    df = bdc.build_combined(
        raw_path, feats_path,
        labels_path=None, include_labels=False,
        vix_path=vix_path, include_vix=True,
    )

    # Verify all columns present
    assert "vix_close" in df.columns
    assert "vix_regime" in df.columns
    assert "indicator_1" in df.columns
    assert len(df) == 3
    # Verify VIX values preserved
    assert df["vix_close"].tolist() == [15.0, 16.0, 17.0]


def test_combine_inner_join_dates(tmp_path: Path) -> None:
    """Test that only overlapping dates appear in output (inner join)."""
    # SPY dates: Jan 1, 2, 3, 4
    spy_dates = pd.date_range("2024-01-01", periods=4, freq="D")
    # VIX dates: Jan 2, 3, 4, 5 (overlap: Jan 2, 3, 4)
    vix_dates = pd.date_range("2024-01-02", periods=4, freq="D")

    raw = pd.DataFrame({
        "Date": spy_dates,
        "Open": [1, 2, 3, 4],
        "High": [1, 2, 3, 4],
        "Low": [1, 2, 3, 4],
        "Close": [1, 2, 3, 4],
        "Volume": [10, 20, 30, 40],
    })
    feats = pd.DataFrame({"Date": spy_dates, "f1": [0.1, 0.2, 0.3, 0.4]})
    vix_feats = pd.DataFrame({
        "Date": vix_dates,
        "vix_close": [15.0, 16.0, 17.0, 18.0],
        "vix_sma_10": [14.5, 15.5, 16.5, 17.5],
        "vix_sma_20": [14.0, 15.0, 16.0, 17.0],
        "vix_percentile_60d": [50.0, 55.0, 60.0, 65.0],
        "vix_zscore_20d": [0.1, 0.2, 0.3, 0.4],
        "vix_regime": ["normal", "normal", "normal", "normal"],
        "vix_change_1d": [1.0, 2.0, 3.0, 4.0],
        "vix_change_5d": [2.0, 3.0, 4.0, 5.0],
    })

    raw_path = tmp_path / "raw.parquet"
    feats_path = tmp_path / "feats.parquet"
    vix_path = tmp_path / "vix.parquet"
    raw.to_parquet(raw_path, index=False)
    feats.to_parquet(feats_path, index=False)
    vix_feats.to_parquet(vix_path, index=False)

    df = bdc.build_combined(
        raw_path, feats_path,
        labels_path=None, include_labels=False,
        vix_path=vix_path, include_vix=True,
    )

    # Only 3 overlapping dates: Jan 2, 3, 4
    assert len(df) == 3
    expected_dates = pd.date_range("2024-01-02", periods=3, freq="D")
    assert df["Date"].tolist() == list(expected_dates)


def test_combine_no_nan_after_join(tmp_path: Path) -> None:
    """Test that no NaN values exist after joining features."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")

    raw = pd.DataFrame({
        "Date": dates,
        "Open": [100.0] * 5,
        "High": [101.0] * 5,
        "Low": [99.0] * 5,
        "Close": [100.5] * 5,
        "Volume": [1000] * 5,
    })
    feats = pd.DataFrame({"Date": dates, "f1": [0.1, 0.2, 0.3, 0.4, 0.5]})
    vix_feats = pd.DataFrame({
        "Date": dates,
        "vix_close": [15.0, 16.0, 17.0, 18.0, 19.0],
        "vix_sma_10": [14.5, 15.5, 16.5, 17.5, 18.5],
        "vix_sma_20": [14.0, 15.0, 16.0, 17.0, 18.0],
        "vix_percentile_60d": [50.0, 55.0, 60.0, 65.0, 70.0],
        "vix_zscore_20d": [0.1, 0.2, 0.3, 0.4, 0.5],
        "vix_regime": ["normal"] * 5,
        "vix_change_1d": [1.0, 2.0, 3.0, 4.0, 5.0],
        "vix_change_5d": [2.0, 3.0, 4.0, 5.0, 6.0],
    })

    raw_path = tmp_path / "raw.parquet"
    feats_path = tmp_path / "feats.parquet"
    vix_path = tmp_path / "vix.parquet"
    raw.to_parquet(raw_path, index=False)
    feats.to_parquet(feats_path, index=False)
    vix_feats.to_parquet(vix_path, index=False)

    df = bdc.build_combined(
        raw_path, feats_path,
        labels_path=None, include_labels=False,
        vix_path=vix_path, include_vix=True,
    )

    # No NaN values in any column
    assert df.isna().sum().sum() == 0


def test_combine_feature_count(tmp_path: Path) -> None:
    """Test that combined dataset has correct column count: Date + 5 OHLCV + 20 ind + 8 VIX = 34."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")

    raw = pd.DataFrame({
        "Date": dates,
        "Open": [100.0] * 3,
        "High": [101.0] * 3,
        "Low": [99.0] * 3,
        "Close": [100.5] * 3,
        "Volume": [1000] * 3,
    })

    # Create full 20 indicator features
    feats_data = {"Date": dates}
    for feat_name in A20_FEATURES:
        feats_data[feat_name] = [1.0, 2.0, 3.0]
    feats = pd.DataFrame(feats_data)

    # Create full 8 VIX features
    vix_feats = pd.DataFrame({
        "Date": dates,
        "vix_close": [15.0, 16.0, 17.0],
        "vix_sma_10": [14.5, 15.5, 16.5],
        "vix_sma_20": [14.0, 15.0, 16.0],
        "vix_percentile_60d": [50.0, 55.0, 60.0],
        "vix_zscore_20d": [0.1, 0.2, 0.3],
        "vix_regime": ["normal", "normal", "normal"],
        "vix_change_1d": [1.0, 2.0, 3.0],
        "vix_change_5d": [2.0, 3.0, 4.0],
    })

    raw_path = tmp_path / "raw.parquet"
    feats_path = tmp_path / "feats.parquet"
    vix_path = tmp_path / "vix.parquet"
    raw.to_parquet(raw_path, index=False)
    feats.to_parquet(feats_path, index=False)
    vix_feats.to_parquet(vix_path, index=False)

    df = bdc.build_combined(
        raw_path, feats_path,
        labels_path=None, include_labels=False,
        vix_path=vix_path, include_vix=True,
    )

    # Date + 5 OHLCV + 20 indicators + 8 VIX = 34 columns
    expected_cols = 1 + 5 + len(A20_FEATURES) + len(VIX_FEATURE_LIST)
    assert len(df.columns) == expected_cols, f"Expected {expected_cols}, got {len(df.columns)}"


def test_combine_rejects_no_overlap(tmp_path: Path) -> None:
    """Test that build_combined raises ValueError when no date overlap exists."""
    # SPY dates: January
    spy_dates = pd.date_range("2024-01-01", periods=3, freq="D")
    # VIX dates: February (no overlap)
    vix_dates = pd.date_range("2024-02-01", periods=3, freq="D")

    raw = pd.DataFrame({
        "Date": spy_dates,
        "Open": [1, 2, 3],
        "High": [1, 2, 3],
        "Low": [1, 2, 3],
        "Close": [1, 2, 3],
        "Volume": [10, 20, 30],
    })
    feats = pd.DataFrame({"Date": spy_dates, "f1": [0.1, 0.2, 0.3]})
    vix_feats = pd.DataFrame({
        "Date": vix_dates,
        "vix_close": [15.0, 16.0, 17.0],
        "vix_sma_10": [14.5, 15.5, 16.5],
        "vix_sma_20": [14.0, 15.0, 16.0],
        "vix_percentile_60d": [50.0, 55.0, 60.0],
        "vix_zscore_20d": [0.1, 0.2, 0.3],
        "vix_regime": ["normal", "normal", "normal"],
        "vix_change_1d": [1.0, 2.0, 3.0],
        "vix_change_5d": [2.0, 3.0, 4.0],
    })

    raw_path = tmp_path / "raw.parquet"
    feats_path = tmp_path / "feats.parquet"
    vix_path = tmp_path / "vix.parquet"
    raw.to_parquet(raw_path, index=False)
    feats.to_parquet(feats_path, index=False)
    vix_feats.to_parquet(vix_path, index=False)

    with pytest.raises(ValueError, match="no overlapping dates"):
        bdc.build_combined(
            raw_path, feats_path,
            labels_path=None, include_labels=False,
            vix_path=vix_path, include_vix=True,
        )

