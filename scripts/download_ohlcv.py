#!/usr/bin/env python3
"""Download OHLCV data from Yahoo Finance.

This module provides functions to download historical OHLCV data
for financial instruments using the yfinance library.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from scripts import manage_data_versions as dv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "SPY.OHLCV.daily"


def download_spy(
    output_path: str = "data/raw/SPY.parquet",
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    register_manifest: bool = True,
    manifest_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Download SPY OHLCV data from Yahoo Finance.

    Parameters
    ----------
    output_path : str
        Path to save parquet file

    Returns
    -------
    pd.DataFrame
        OHLCV data with columns: Date, Open, High, Low, Close, Volume

    Raises
    ------
    ValueError
        If download fails or data is invalid
    """
    logger.info("Downloading SPY data from Yahoo Finance...")

    try:
        # Download full history
        ticker = yf.Ticker("SPY")
        df = ticker.history(period="max")

        if df.empty:
            raise ValueError("No data returned from Yahoo Finance")

        logger.info(f"Downloaded {len(df)} rows")

    except Exception as e:
        raise ValueError(f"Failed to download SPY data: {e}") from e

    # Reset index to make Date a column
    df = df.reset_index()

    # Rename columns to standard format
    df = df.rename(columns={"Date": "Date"})

    # Select and order columns
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]

    # Check all required columns exist
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols]

    # Ensure Date is datetime (remove timezone if present)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

    # Ensure OHLC are float
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)

    # Ensure Volume is numeric
    df["Volume"] = pd.to_numeric(df["Volume"])

    # Validate no nulls in OHLCV
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

    # Ensure output directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    if register_manifest:
        target_manifest = manifest_path or dv.RAW_MANIFEST
        dv.register_raw_entry(dataset_name, output_file, manifest_path=target_manifest)
        logger.info("Registered %s in manifest %s", dataset_name, target_manifest)

    return df


if __name__ == "__main__":
    download_spy()
