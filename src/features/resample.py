"""OHLCV resampling utilities for multi-timescale experiments."""

from __future__ import annotations

import pandas as pd

# Mapping from human-readable timescale names to pandas frequency strings
TIMESCALE_MAP = {
    "2d": "2D",
    "3d": "3D",
    "5d": "5D",
    "weekly": "W-FRI",
}


def get_freq_string(timescale: str) -> str:
    """Convert timescale name to pandas frequency string.

    Args:
        timescale: '2d', '3d', '5d', 'weekly' (case-insensitive)

    Returns:
        Pandas frequency: '2D', '3D', '5D', 'W-FRI'

    Raises:
        ValueError: If timescale is not recognized.
    """
    key = timescale.lower()
    if key not in TIMESCALE_MAP:
        raise ValueError(f"Unknown timescale: {timescale}. Valid options: {list(TIMESCALE_MAP.keys())}")
    return TIMESCALE_MAP[key]


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample OHLCV data to lower frequency.

    Args:
        df: DataFrame with Date column and OHLCV columns
        freq: Target frequency ('2D', '3D', '5D', 'W-FRI')

    Returns:
        Resampled DataFrame with aggregated OHLCV:
        - Open: first
        - High: max
        - Low: min
        - Close: last
        - Volume: sum

    Note: Uses end-of-period labeling to avoid look-ahead bias.
    """
    # Ensure Date is datetime and set as index
    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])
    df_indexed = df_copy.set_index("Date")

    # Define aggregation rules for OHLCV
    agg_rules = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }

    # Resample with end-of-period labeling (label='right', closed='right')
    # This ensures no look-ahead bias: the period label is the end date
    resampled = (
        df_indexed
        .resample(freq, label="right", closed="right")
        .agg(agg_rules)
        .dropna(subset=["Close"])
    )

    # Reset index to get Date back as a column
    result = resampled.reset_index()

    return result
