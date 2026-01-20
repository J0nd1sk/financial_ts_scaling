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

from src.features import tier_a20


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
    weekly = tier_a20._resample_to_weekly(sample_daily_df)  # type: ignore[attr-defined]
    assert (weekly.index.weekday == 0).all(), "Weekly index must be Monday-aligned"


def test_build_feature_dataframe_contains_all_features(sample_daily_df: pd.DataFrame) -> None:
    feature_df = tier_a20.build_feature_dataframe(sample_daily_df)
    assert feature_df.columns.tolist()[1:] == tier_a20.FEATURE_LIST
    assert feature_df.isnull().sum().sum() == 0
    assert len(feature_df) < len(sample_daily_df), "Warm-up rows should be dropped"


def test_weekly_indicators_forward_filled(sample_daily_df: pd.DataFrame) -> None:
    weekly_features = tier_a20._compute_weekly_indicators(sample_daily_df)  # type: ignore[attr-defined]
    assert weekly_features.shape[0] == sample_daily_df.shape[0]
    assert weekly_features["rsi_weekly"].notnull().any()
    assert weekly_features["stochrsi_weekly"].notnull().any()


class TestWeeklyIndicatorLookaheadBias:
    """Tests to verify weekly indicators have no look-ahead bias.

    Look-ahead bias occurs when Monday's indicator uses data from the
    current week (Mon-Fri) instead of only using data available by Monday
    (i.e., the previous completed week).
    """

    @pytest.fixture
    def multi_week_df(self) -> pd.DataFrame:
        """Create 40 weeks of oscillating data for RSI calculation.

        Weekly RSI needs 14 weekly periods to compute, which is ~70 business days.
        We create 40 weeks (~200 days) with oscillating prices so each week
        has a DIFFERENT RSI value, making look-ahead bias detectable.
        """
        # Start on a Monday, use business days
        dates = pd.date_range("2024-01-01", periods=200, freq="B")
        n = len(dates)

        # Create weekly oscillation: week 1 up, week 2 down, week 3 up, etc.
        # This produces RSI values that vary week to week (~54 vs ~58)
        close = []
        for i in range(n):
            week_num = i // 5  # 5 business days per week
            if week_num % 2 == 0:  # Even weeks: go up
                daily_change = 2
            else:  # Odd weeks: go down
                daily_change = -1.5
            if i == 0:
                close.append(100.0)
            else:
                close.append(close[-1] + daily_change)

        close = np.array(close)

        return pd.DataFrame({
            "Date": dates,
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": np.ones(n) * 1_000_000,
        })

    def test_weekly_rsi_no_lookahead_monday_vs_friday(self, multi_week_df: pd.DataFrame) -> None:
        """Monday should NOT see current week's data (which includes Friday).

        With correct implementation (no look-ahead):
        - Monday of week N sees RSI from week N-1
        - Friday of week N sees RSI from week N-1 (same as Monday!)
        - Both days in the same week should have the same weekly RSI

        With look-ahead bias bug:
        - Monday of week N would see RSI from week N (computed using Mon-Fri data)
        - This means Monday "sees" Friday's data before Friday happens
        """
        weekly_features = tier_a20._compute_weekly_indicators(multi_week_df)

        # Use week 20 (index ~100) to ensure we're past RSI warmup (~70 days)
        dates = multi_week_df["Date"]

        # Find Monday of week 20
        mondays = dates[dates.dt.weekday == 0]
        monday_week_20 = mondays.iloc[19]  # 0-indexed, so 19 = week 20
        monday_idx = dates[dates == monday_week_20].index[0]

        # Friday of the SAME week (Mon + 4 business days = Fri)
        friday_idx = monday_idx + 4

        rsi_monday = weekly_features.loc[monday_idx, "rsi_weekly"]
        rsi_friday = weekly_features.loc[friday_idx, "rsi_weekly"]

        # Sanity check: both should have valid values
        assert pd.notna(rsi_monday), f"Monday RSI is NaN at index {monday_idx}"
        assert pd.notna(rsi_friday), f"Friday RSI is NaN at index {friday_idx}"

        # With NO look-ahead bias: Monday and Friday should have SAME RSI
        # (both days in week N see RSI from week N-1)
        assert rsi_monday == rsi_friday, (
            f"Look-ahead bias detected! "
            f"Monday RSI ({rsi_monday:.4f}) != Friday RSI ({rsi_friday:.4f}). "
            f"Both days in the same week should have the same weekly RSI."
        )

    def test_weekly_rsi_all_days_in_week_have_same_value(self, multi_week_df: pd.DataFrame) -> None:
        """All days within the same week should have identical weekly RSI.

        With correct implementation:
        - Monday through Friday of week N all see RSI from week N-1
        - So all days in the week should have the same weekly RSI value

        With look-ahead bias:
        - Monday sees RSI from week N (current week)
        - But RSI was calculated from data through Friday of week N
        - So Monday-Thursday might differ from Friday
        """
        weekly_features = tier_a20._compute_weekly_indicators(multi_week_df)

        dates = multi_week_df["Date"]
        mondays = dates[dates.dt.weekday == 0]

        # Check week 22 (well past RSI warmup)
        monday_week_22 = mondays.iloc[21]
        monday_idx = dates[dates == monday_week_22].index[0]

        # Get RSI for all 5 days of the week (Mon, Tue, Wed, Thu, Fri)
        week_rsi_values = []
        for offset in range(5):
            idx = monday_idx + offset
            rsi = weekly_features.loc[idx, "rsi_weekly"]
            week_rsi_values.append(rsi)

        # All values should be identical (all from previous week's RSI)
        unique_values = set(week_rsi_values)
        assert len(unique_values) == 1, (
            f"Look-ahead bias detected! Days within week 22 have different RSI values: "
            f"{week_rsi_values}. All days should have the same weekly RSI."
        )

    def test_weekly_rsi_value_matches_previous_week(self, multi_week_df: pd.DataFrame) -> None:
        """Monday's weekly RSI should match the RSI computed from PREVIOUS week only.

        This is the definitive test for look-ahead bias. We manually compute
        weekly RSI and verify that Monday of week N has RSI from week N-1,
        NOT RSI from week N.
        """
        import talib

        # Compute weekly indicators using the function under test
        weekly_features = tier_a20._compute_weekly_indicators(multi_week_df)

        # Also compute weekly RSI manually for comparison
        weekly_df = tier_a20._resample_to_weekly(multi_week_df)
        weekly_rsi_raw = talib.RSI(weekly_df["Close"], timeperiod=14)

        dates = multi_week_df["Date"]
        mondays = dates[dates.dt.weekday == 0]

        # Week 22 should have valid RSI (well past warmup)
        monday_week_22 = mondays.iloc[21]
        monday_idx = dates[dates == monday_week_22].index[0]

        # Get the RSI value on Monday of week 22 from our function
        actual_rsi = weekly_features.loc[monday_idx, "rsi_weekly"]

        # Get expected RSI: should be from week 21 (previous week), NOT week 22
        # weekly_rsi_raw is indexed by Monday of each week
        week_21_monday = mondays.iloc[20]
        week_22_monday = mondays.iloc[21]

        expected_rsi_correct = weekly_rsi_raw.loc[week_21_monday]  # Week 21 RSI (correct)
        expected_rsi_buggy = weekly_rsi_raw.loc[week_22_monday]    # Week 22 RSI (look-ahead)

        # Verify our value matches the PREVIOUS week's RSI, not current week's
        assert abs(actual_rsi - expected_rsi_correct) < 0.001, (
            f"Monday of week 22 should have RSI from week 21 ({expected_rsi_correct:.4f}), "
            f"but got {actual_rsi:.4f}"
        )
        assert abs(actual_rsi - expected_rsi_buggy) > 0.001, (
            f"Monday of week 22 has RSI from week 22 ({expected_rsi_buggy:.4f}) - "
            f"this indicates look-ahead bias!"
        )

