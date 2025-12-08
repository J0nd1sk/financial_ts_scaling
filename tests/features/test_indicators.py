"""Tests for indicator computations."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import indicators


@pytest.fixture()
def sample_daily_df() -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=200, freq="B")
    data = {
        "Date": dates,
        "Open": np.linspace(100, 150, len(dates)),
        "High": np.linspace(101, 151, len(dates)),
        "Low": np.linspace(99, 149, len(dates)),
        "Close": np.linspace(100, 150, len(dates)),
        "Volume": np.linspace(1_000_000, 2_000_000, len(dates)),
    }
    return pd.DataFrame(data)


def test_resample_to_weekly_starts_monday(sample_daily_df: pd.DataFrame) -> None:
    weekly = indicators._resample_to_weekly(sample_daily_df)  # type: ignore[attr-defined]
    assert (weekly.index.weekday == 0).all(), "Weekly index must be Monday-aligned"


def test_build_feature_dataframe_contains_all_features(sample_daily_df: pd.DataFrame) -> None:
    feature_df = indicators.build_feature_dataframe(sample_daily_df)
    assert feature_df.columns.tolist()[1:] == indicators.FEATURE_LIST
    assert feature_df.isnull().sum().sum() == 0
    assert len(feature_df) < len(sample_daily_df), "Warm-up rows should be dropped"


def test_weekly_indicators_forward_filled(sample_daily_df: pd.DataFrame) -> None:
    weekly_features = indicators._compute_weekly_indicators(sample_daily_df)  # type: ignore[attr-defined]
    assert weekly_features.shape[0] == sample_daily_df.shape[0]
    assert weekly_features["rsi_weekly"].notnull().any()
    assert weekly_features["stochrsi_weekly"].notnull().any()

