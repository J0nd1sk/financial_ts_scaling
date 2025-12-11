#!/usr/bin/env python3
"""Generate comprehensive data dictionary for all parquet files.

Scans data/raw/ and data/processed/v1/ and creates markdown documentation
with file metadata, column schemas, and statistics.

Usage:
    python scripts/generate_data_dictionary.py [--output PATH]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Import feature lists for descriptions
from src.features.tier_a20 import FEATURE_LIST as TIER_A20_FEATURES
from src.features.tier_c_vix import VIX_FEATURE_LIST
from src.features.resample import TIMESCALE_MAP

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "v1"

# Column descriptions for known columns
OHLCV_DESCRIPTIONS = {
    "Date": "Trading date (datetime index)",
    "Open": "Opening price (USD)",
    "High": "Intraday high price (USD)",
    "Low": "Intraday low price (USD)",
    "Close": "Closing/adjusted price (USD)",
    "Volume": "Trading volume (shares)",
}

INDICATOR_DESCRIPTIONS = {
    # Moving averages
    "dema_9": "9-period Double Exponential Moving Average",
    "dema_10": "10-period Double Exponential Moving Average",
    "sma_12": "12-period Simple Moving Average",
    "dema_20": "20-period Double Exponential Moving Average",
    "dema_25": "25-period Double Exponential Moving Average",
    "sma_50": "50-period Simple Moving Average",
    "dema_90": "90-period Double Exponential Moving Average",
    "sma_100": "100-period Simple Moving Average",
    "sma_200": "200-period Simple Moving Average",
    # Oscillators
    "rsi_daily": "14-period Relative Strength Index (daily)",
    "rsi_weekly": "14-period Relative Strength Index (weekly, forward-filled)",
    "stochrsi_daily": "Stochastic RSI %K (daily)",
    "stochrsi_weekly": "Stochastic RSI %K (weekly, forward-filled)",
    # MACD
    "macd_line": "MACD line (12/26/9 EMA)",
    # Volume
    "obv": "On-Balance Volume",
    "adosc": "Accumulation/Distribution Oscillator (3/10)",
    # Volatility
    "atr_14": "14-period Average True Range",
    "adx_14": "14-period Average Directional Index",
    "bb_percent_b": "Bollinger Bands %B (20-period, 2 std dev)",
    # VWAP
    "vwap_20": "20-period Volume-Weighted Average Price",
}

VIX_DESCRIPTIONS = {
    "vix_close": "VIX closing value",
    "vix_sma_10": "10-day VIX simple moving average",
    "vix_sma_20": "20-day VIX simple moving average",
    "vix_percentile_60d": "60-day rolling percentile rank (0-100)",
    "vix_zscore_20d": "20-day rolling z-score",
    "vix_regime": "Volatility regime: low (<15), normal (15-25), high (>=25)",
    "vix_change_1d": "1-day percent change",
    "vix_change_5d": "5-day percent change",
}


def get_column_descriptions(columns: list[str], file_type: str) -> dict[str, str]:
    """Get descriptions for columns based on file type.

    Args:
        columns: List of column names
        file_type: 'ohlcv', 'indicators', 'vix', or 'combined'

    Returns:
        Dictionary mapping column names to descriptions
    """
    descriptions = {}

    for col in columns:
        if col in OHLCV_DESCRIPTIONS:
            descriptions[col] = OHLCV_DESCRIPTIONS[col]
        elif col in INDICATOR_DESCRIPTIONS:
            descriptions[col] = INDICATOR_DESCRIPTIONS[col]
        elif col in VIX_DESCRIPTIONS:
            descriptions[col] = VIX_DESCRIPTIONS[col]
        else:
            descriptions[col] = f"Column: {col}"

    return descriptions


def get_file_stats(filepath: Path) -> dict[str, Any]:
    """Extract metadata and statistics from a parquet file.

    Args:
        filepath: Path to parquet file

    Returns:
        Dictionary with file info, schema, and statistics
    """
    df = pd.read_parquet(filepath)

    # Basic file info
    info = {
        "path": str(filepath.relative_to(PROJECT_ROOT)),
        "filename": filepath.name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }

    # Date range if Date column exists
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        info["date_min"] = df["Date"].min().strftime("%Y-%m-%d")
        info["date_max"] = df["Date"].max().strftime("%Y-%m-%d")

    # Statistics for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        stats_df = df[numeric_cols].describe()
        info["statistics"] = stats_df.round(2).to_dict()

    return info


def format_statistics_table(stats: dict[str, dict]) -> str:
    """Format statistics dictionary as markdown table.

    Args:
        stats: Dictionary from df.describe().to_dict()

    Returns:
        Markdown table string
    """
    if not stats:
        return "*No numeric columns*\n"

    # Get all columns and stat rows
    columns = list(stats.keys())
    stat_rows = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]

    # Header
    lines = ["| Statistic | " + " | ".join(columns) + " |"]
    lines.append("|" + "|".join(["---"] * (len(columns) + 1)) + "|")

    # Data rows
    for stat in stat_rows:
        row = [stat.capitalize()]
        for col in columns:
            val = stats[col].get(stat, "")
            if isinstance(val, float):
                # Format large numbers with commas, small with decimals
                if abs(val) >= 1000:
                    row.append(f"{val:,.0f}")
                else:
                    row.append(f"{val:.2f}")
            else:
                row.append(str(val))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


def format_column_schema(columns: list[str], dtypes: dict, descriptions: dict) -> str:
    """Format column schema as markdown table.

    Args:
        columns: List of column names
        dtypes: Dictionary of column dtypes
        descriptions: Dictionary of column descriptions

    Returns:
        Markdown table string
    """
    lines = ["| Column | Dtype | Description |"]
    lines.append("|--------|-------|-------------|")

    for col in columns:
        dtype = dtypes.get(col, "unknown")
        desc = descriptions.get(col, "")
        lines.append(f"| {col} | {dtype} | {desc} |")

    return "\n".join(lines) + "\n"


def format_file_section(file_info: dict[str, Any], file_type: str) -> str:
    """Format a single file's documentation section.

    Args:
        file_info: Dictionary from get_file_stats()
        file_type: Type for description lookup

    Returns:
        Markdown section string
    """
    lines = []
    filename = file_info["filename"]

    lines.append(f"### {filename}\n")

    # File properties table
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| Path | {file_info['path']} |")
    lines.append(f"| Rows | {file_info['rows']:,} |")
    lines.append(f"| Columns | {file_info['columns']} |")

    if "date_min" in file_info:
        lines.append(f"| Date Range | {file_info['date_min']} to {file_info['date_max']} |")

    lines.append("")

    # Column schema
    lines.append("#### Column Schema\n")
    descriptions = get_column_descriptions(file_info["column_names"], file_type)
    lines.append(format_column_schema(file_info["column_names"], file_info["dtypes"], descriptions))

    # Statistics
    if "statistics" in file_info:
        lines.append("#### Statistics\n")
        lines.append(format_statistics_table(file_info["statistics"]))

    lines.append("---\n")
    return "\n".join(lines)


def scan_parquet_files(directories: list[Path]) -> list[tuple[Path, str]]:
    """Scan directories for parquet files.

    Args:
        directories: List of directories to scan

    Returns:
        List of (filepath, file_type) tuples
    """
    files = []

    for directory in directories:
        if not directory.exists():
            continue
        for filepath in sorted(directory.glob("*.parquet")):
            # Determine file type based on name
            name = filepath.stem.lower()
            if "features_a20" in name:
                file_type = "indicators"
            elif "features_c" in name or "vix" in name.lower():
                file_type = "vix"
            elif "dataset" in name:
                file_type = "combined"
            else:
                file_type = "ohlcv"
            files.append((filepath, file_type))

    return files


def generate_data_dictionary(output_path: str = "docs/data_dictionary.md") -> None:
    """Generate comprehensive data dictionary for all parquet files.

    Args:
        output_path: Path to write markdown output
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    # Header
    lines.append("# Data Dictionary\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Generator: `scripts/generate_data_dictionary.py`\n")

    # Scan files
    raw_files = scan_parquet_files([RAW_DIR])
    processed_files = scan_parquet_files([PROCESSED_DIR])

    # Summary section
    lines.append("## Summary\n")

    # Calculate totals
    raw_stats = [get_file_stats(fp) for fp, _ in raw_files]
    processed_stats = [get_file_stats(fp) for fp, _ in processed_files]

    total_raw_rows = sum(s["rows"] for s in raw_stats)
    total_proc_rows = sum(s["rows"] for s in processed_stats)

    lines.append("| Location | Files | Total Rows |")
    lines.append("|----------|-------|------------|")
    lines.append(f"| data/raw/ | {len(raw_files)} | {total_raw_rows:,} |")
    lines.append(f"| data/processed/v1/ | {len(processed_files)} | {total_proc_rows:,} |")
    lines.append("")

    # Feature list summary
    lines.append("### Feature Lists\n")
    lines.append(f"- **Tier A20 Indicators**: {len(TIER_A20_FEATURES)} features")
    lines.append(f"- **VIX Features**: {len(VIX_FEATURE_LIST)} features")
    lines.append(f"- **Timescales**: {', '.join(TIMESCALE_MAP.keys())}")
    lines.append("")

    # Data lineage
    lines.append("### Data Lineage\n")
    lines.append("```")
    lines.append("Raw OHLCV (yfinance)")
    lines.append("    │")
    lines.append("    ├── SPY.parquet ──► SPY_features_a20.parquet ──► SPY_dataset_c.parquet")
    lines.append("    ├── DIA.parquet ──► DIA_features_a20.parquet")
    lines.append("    ├── QQQ.parquet ──► QQQ_features_a20.parquet")
    lines.append("    └── VIX.parquet ──► VIX_features_c.parquet ──┘")
    lines.append("```")
    lines.append("")

    # Raw data files
    lines.append("## Raw Data Files\n")
    for (filepath, file_type), stats in zip(raw_files, raw_stats):
        lines.append(format_file_section(stats, file_type))

    # Processed data files
    lines.append("## Processed Data Files\n")
    for (filepath, file_type), stats in zip(processed_files, processed_stats):
        lines.append(format_file_section(stats, file_type))

    # Indicator reference
    lines.append("## Indicator Reference\n")
    lines.append("### Tier A20 Indicators\n")
    lines.append("| Indicator | Description |")
    lines.append("|-----------|-------------|")
    for indicator in TIER_A20_FEATURES:
        desc = INDICATOR_DESCRIPTIONS.get(indicator, indicator)
        lines.append(f"| {indicator} | {desc} |")
    lines.append("")

    lines.append("### VIX Features\n")
    lines.append("| Feature | Description |")
    lines.append("|---------|-------------|")
    for feature in VIX_FEATURE_LIST:
        desc = VIX_DESCRIPTIONS.get(feature, feature)
        lines.append(f"| {feature} | {desc} |")
    lines.append("")

    lines.append("### Timescales\n")
    lines.append("| Name | Pandas Freq | Description |")
    lines.append("|------|-------------|-------------|")
    lines.append("| daily | D | Daily OHLCV (default) |")
    for name, freq in TIMESCALE_MAP.items():
        if name == "weekly":
            desc = "Weekly (Friday close)"
        else:
            desc = f"{name.upper()} calendar days"
        lines.append(f"| {name} | {freq} | {desc} |")
    lines.append("")

    # Write output
    content = "\n".join(lines)
    output.write_text(content)
    print(f"Data dictionary written to: {output}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate data dictionary")
    parser.add_argument(
        "--output",
        type=str,
        default="docs/data_dictionary.md",
        help="Output path for markdown file",
    )
    args = parser.parse_args()

    generate_data_dictionary(args.output)


if __name__ == "__main__":
    main()
