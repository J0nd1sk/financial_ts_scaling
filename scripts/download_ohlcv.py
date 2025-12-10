#!/usr/bin/env python3
"""Download OHLCV data from Yahoo Finance.

This module provides functions to download historical OHLCV data
for financial instruments using the yfinance library.
"""

import argparse
import logging
import random
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

from scripts import manage_data_versions as dv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "SPY.OHLCV.daily"
BASE_RETRY_DELAYS = [1, 2, 4]  # seconds, exponential backoff
MAX_RETRIES = 3


def _download_with_retry(ticker: str, max_retries: int = MAX_RETRIES) -> pd.DataFrame:
    """Download ticker data with exponential backoff and jitter on failure.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'SPY', 'DIA', 'QQQ')
    max_retries : int
        Maximum number of retry attempts (default 3)

    Returns
    -------
    pd.DataFrame
        OHLCV data with Date as index

    Raises
    ------
    ValueError
        If download fails after all retries or returns empty data
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            yf_ticker = yf.Ticker(ticker)
            df = yf_ticker.history(period="max")

            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            return df

        except Exception as e:
            last_exception = e

            if attempt == max_retries:
                # Final attempt failed
                raise ValueError(
                    f"Failed to download {ticker} after {max_retries + 1} attempts: {e}"
                ) from e

            # Calculate backoff delay with jitter
            base = BASE_RETRY_DELAYS[min(attempt, len(BASE_RETRY_DELAYS) - 1)]
            jitter = random.uniform(0, base * 0.5)  # 0-50% jitter
            delay = base + jitter

            logger.warning(
                f"Attempt {attempt + 1} failed for {ticker}, retrying in {delay:.1f}s: {e}"
            )
            time.sleep(delay)

    # Should never reach here, but satisfy type checker
    raise ValueError(f"Failed to download {ticker}") from last_exception


def _sanitize_ticker_for_filename(ticker: str) -> str:
    """Remove special characters from ticker for use in filenames.

    Index tickers like ^DJI, ^GSPC, ^IXIC have ^ which is problematic in filenames.

    Parameters
    ----------
    ticker : str
        Raw ticker symbol (e.g., '^DJI', 'SPY')

    Returns
    -------
    str
        Sanitized ticker suitable for filenames (e.g., 'DJI', 'SPY')
    """
    return ticker.replace("^", "")


def download_ticker(
    ticker: str,
    output_dir: str = "data/raw",
    *,
    register_manifest: bool = True,
    manifest_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Download OHLCV data for any ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'SPY', 'DIA', 'QQQ', '^DJI', '^IXIC')
    output_dir : str
        Directory to save parquet file (file will be {sanitized_ticker}.parquet)
    register_manifest : bool
        Whether to register in data manifest (default True)
    manifest_path : Optional[Path]
        Custom manifest path (default: RAW_MANIFEST)

    Returns
    -------
    pd.DataFrame
        OHLCV data with columns: Date, Open, High, Low, Close, Volume

    Raises
    ------
    ValueError
        If download fails or data is invalid

    Notes
    -----
    Index tickers (^DJI, ^IXIC, etc.) have the ^ removed from filenames
    and manifest entries for filesystem compatibility.
    """
    # Sanitize ticker for filenames (^DJI -> DJI)
    sanitized_ticker = _sanitize_ticker_for_filename(ticker)

    logger.info(f"Downloading {ticker} data from Yahoo Finance...")

    # Download with retry logic
    df = _download_with_retry(ticker)
    logger.info(f"Downloaded {len(df)} rows for {ticker}")

    # Reset index to make Date a column
    df = df.reset_index()

    # Select and order columns
    required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]

    # Check all required columns exist
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols]

    # Ensure Date is datetime (remove timezone if present)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)

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
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Save to parquet (use sanitized ticker for filename)
    output_file = output_dir_path / f"{sanitized_ticker}.parquet"
    df.to_parquet(output_file, index=False)
    logger.info(f"Saved to {output_file}")

    # Register in manifest (use sanitized ticker for dataset name)
    if register_manifest:
        dataset_name = f"{sanitized_ticker}.OHLCV.daily"
        target_manifest = manifest_path or dv.RAW_MANIFEST
        dv.register_raw_entry(dataset_name, output_file, manifest_path=target_manifest)
        logger.info(f"Registered {dataset_name} in manifest {target_manifest}")

    return df


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
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)

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


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Parameters
    ----------
    args : Optional[List[str]]
        Command line arguments. If None, uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ticker and output_dir attributes.
    """
    parser = argparse.ArgumentParser(
        description="Download OHLCV data from Yahoo Finance."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help="Ticker symbol to download (default: SPY)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for parquet file (default: data/raw)",
    )
    return parser.parse_args(args)


if __name__ == "__main__":
    cli_args = parse_args()
    download_ticker(cli_args.ticker, cli_args.output_dir)
