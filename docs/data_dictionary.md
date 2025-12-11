# Data Dictionary

Generated: 2025-12-10 20:55:06
Generator: `scripts/generate_data_dictionary.py`

## Summary

| Location | Files | Total Rows |
|----------|-------|------------|
| data/raw/ | 6 | 53,449 |
| data/processed/v1/ | 8 | 53,310 |

### Feature Lists

- **Tier A20 Indicators**: 20 features
- **VIX Features**: 8 features
- **Timescales**: 2d, 3d, 5d, weekly

### Data Lineage

```
Raw OHLCV (yfinance)
    │
    ├── SPY.parquet ──► SPY_features_a20.parquet ──► SPY_dataset_c.parquet
    ├── DIA.parquet ──► DIA_features_a20.parquet
    ├── QQQ.parquet ──► QQQ_features_a20.parquet
    └── VIX.parquet ──► VIX_features_c.parquet ──┘
```

## Raw Data Files

### DIA.parquet

| Property | Value |
|----------|-------|
| Path | data/raw/DIA.parquet |
| Rows | 7,018 |
| Columns | 6 |
| Date Range | 1998-01-20 to 2025-12-10 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| Open | float64 | Opening price (USD) |
| High | float64 | Intraday high price (USD) |
| Low | float64 | Intraday low price (USD) |
| Close | float64 | Closing/adjusted price (USD) |
| Volume | int64 | Trading volume (shares) |

#### Statistics

| Statistic | Open | High | Low | Close | Volume |
|---|---|---|---|---|---|
| Count | 7,018 | 7,018 | 7,018 | 7,018 | 7,018 |
| Mean | 150.93 | 151.75 | 150.05 | 150.94 | 6,456,482 |
| Std | 111.84 | 112.38 | 111.28 | 111.87 | 6,516,253 |
| Min | 42.31 | 43.29 | 41.55 | 42.11 | 71,000 |
| 25% | 64.16 | 64.58 | 63.71 | 64.15 | 2,837,375 |
| 50% | 93.28 | 93.75 | 92.87 | 93.28 | 4,772,100 |
| 75% | 220.86 | 222.16 | 219.37 | 220.83 | 7,776,700 |
| Max | 480.79 | 483.79 | 479.89 | 482.15 | 91,695,200 |

---

### DJI.parquet

| Property | Value |
|----------|-------|
| Path | data/raw/DJI.parquet |
| Rows | 8,546 |
| Columns | 6 |
| Date Range | 1992-01-02 to 2025-12-09 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| Open | float64 | Opening price (USD) |
| High | float64 | Intraday high price (USD) |
| Low | float64 | Intraday low price (USD) |
| Close | float64 | Closing/adjusted price (USD) |
| Volume | int64 | Trading volume (shares) |

#### Statistics

| Statistic | Open | High | Low | Close | Volume |
|---|---|---|---|---|---|
| Count | 8,546 | 8,546 | 8,546 | 8,546 | 8,546 |
| Mean | 15,828 | 15,920 | 15,732 | 15,832 | 208,319,781 |
| Std | 10,700 | 10,752 | 10,646 | 10,702 | 147,805,504 |
| Min | 3,137 | 3,173 | 3,096 | 3,137 | 8,410,000 |
| 25% | 9,027 | 9,088 | 8,942 | 9,028 | 84,247,500 |
| 50% | 11,346 | 11,417 | 11,262 | 11,352 | 203,150,000 |
| 75% | 21,060 | 21,188 | 20,989 | 21,107 | 293,597,500 |
| Max | 48,174 | 48,432 | 48,016 | 48,255 | 1,412,960,000 |

---

### IXIC.parquet

| Property | Value |
|----------|-------|
| Path | data/raw/IXIC.parquet |
| Rows | 13,829 |
| Columns | 6 |
| Date Range | 1971-02-05 to 2025-12-10 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| Open | float64 | Opening price (USD) |
| High | float64 | Intraday high price (USD) |
| Low | float64 | Intraday low price (USD) |
| Close | float64 | Closing/adjusted price (USD) |
| Volume | int64 | Trading volume (shares) |

#### Statistics

| Statistic | Open | High | Low | Close | Volume |
|---|---|---|---|---|---|
| Count | 13,829 | 13,829 | 13,829 | 13,829 | 13,829 |
| Mean | 3,101 | 3,123 | 3,078 | 3,102 | 1,406,572,828 |
| Std | 4,530 | 4,560 | 4,496 | 4,530 | 1,979,829,488 |
| Min | 54.87 | 54.87 | 54.87 | 54.87 | 0.00 |
| 25% | 284.47 | 284.69 | 283.80 | 284.30 | 38,840,000 |
| 50% | 1,463 | 1,485 | 1,443 | 1,465 | 765,240,000 |
| 75% | 3,212 | 3,233 | 3,177 | 3,205 | 1,983,630,000 |
| Max | 23,987 | 24,020 | 23,765 | 23,958 | 93,974,540,000 |

---

### QQQ.parquet

| Property | Value |
|----------|-------|
| Path | data/raw/QQQ.parquet |
| Rows | 6,731 |
| Columns | 6 |
| Date Range | 1999-03-10 to 2025-12-09 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| Open | float64 | Opening price (USD) |
| High | float64 | Intraday high price (USD) |
| Low | float64 | Intraday low price (USD) |
| Close | float64 | Closing/adjusted price (USD) |
| Volume | int64 | Trading volume (shares) |

#### Statistics

| Statistic | Open | High | Low | Close | Volume |
|---|---|---|---|---|---|
| Count | 6,731 | 6,731 | 6,731 | 6,731 | 6,731 |
| Mean | 134.36 | 135.40 | 133.20 | 134.37 | 65,000,853 |
| Std | 141.39 | 142.32 | 140.34 | 141.41 | 48,229,498 |
| Min | 16.86 | 17.36 | 16.71 | 16.97 | 3,302,000 |
| 25% | 37.05 | 37.40 | 36.64 | 37.02 | 30,969,450 |
| 50% | 66.12 | 66.87 | 65.44 | 66.05 | 52,195,200 |
| 75% | 174.41 | 175.79 | 172.94 | 174.48 | 85,520,050 |
| Max | 635.59 | 637.01 | 630.25 | 635.77 | 616,772,300 |

---

### SPY.parquet

| Property | Value |
|----------|-------|
| Path | data/raw/SPY.parquet |
| Rows | 8,272 |
| Columns | 6 |
| Date Range | 1993-01-29 to 2025-12-08 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| Open | float64 | Opening price (USD) |
| High | float64 | Intraday high price (USD) |
| Low | float64 | Intraday low price (USD) |
| Close | float64 | Closing/adjusted price (USD) |
| Volume | int64 | Trading volume (shares) |

#### Statistics

| Statistic | Open | High | Low | Close | Volume |
|---|---|---|---|---|---|
| Count | 8,272 | 8,272 | 8,272 | 8,272 | 8,272 |
| Mean | 164.43 | 165.37 | 163.39 | 164.45 | 83,295,276 |
| Std | 148.08 | 148.83 | 147.26 | 148.12 | 89,932,187 |
| Min | 24.02 | 24.09 | 23.69 | 24.02 | 5,200 |
| 25% | 70.82 | 71.35 | 70.09 | 70.82 | 12,059,650 |
| 50% | 95.43 | 96.03 | 94.60 | 95.35 | 62,926,500 |
| 75% | 217.31 | 219.29 | 216.49 | 218.57 | 111,094,925 |
| Max | 688.72 | 689.70 | 684.83 | 687.39 | 871,026,300 |

---

### VIX.parquet

| Property | Value |
|----------|-------|
| Path | data/raw/VIX.parquet |
| Rows | 9,053 |
| Columns | 6 |
| Date Range | 1990-01-02 to 2025-12-10 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| Open | float64 | Opening price (USD) |
| High | float64 | Intraday high price (USD) |
| Low | float64 | Intraday low price (USD) |
| Close | float64 | Closing/adjusted price (USD) |
| Volume | int64 | Trading volume (shares) |

#### Statistics

| Statistic | Open | High | Low | Close | Volume |
|---|---|---|---|---|---|
| Count | 9,053 | 9,053 | 9,053 | 9,053 | 9,053 |
| Mean | 19.56 | 20.38 | 18.79 | 19.46 | 0.00 |
| Std | 7.86 | 8.35 | 7.34 | 7.79 | 0.00 |
| Min | 9.01 | 9.31 | 8.56 | 9.14 | 0.00 |
| 25% | 13.97 | 14.59 | 13.44 | 13.93 | 0.00 |
| 50% | 17.66 | 18.33 | 17.02 | 17.61 | 0.00 |
| 75% | 22.92 | 23.76 | 22.09 | 22.76 | 0.00 |
| Max | 82.69 | 89.53 | 72.76 | 82.69 | 0.00 |

---

## Processed Data Files

### DIA_features_a20.parquet

| Property | Value |
|----------|-------|
| Path | data/processed/v1/DIA_features_a20.parquet |
| Rows | 6,819 |
| Columns | 21 |
| Date Range | 1998-11-02 to 2025-12-10 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| dema_9 | float64 | 9-period Double Exponential Moving Average |
| dema_10 | float64 | 10-period Double Exponential Moving Average |
| sma_12 | float64 | 12-period Simple Moving Average |
| dema_20 | float64 | 20-period Double Exponential Moving Average |
| dema_25 | float64 | 25-period Double Exponential Moving Average |
| sma_50 | float64 | 50-period Simple Moving Average |
| dema_90 | float64 | 90-period Double Exponential Moving Average |
| sma_100 | float64 | 100-period Simple Moving Average |
| sma_200 | float64 | 200-period Simple Moving Average |
| rsi_daily | float64 | 14-period Relative Strength Index (daily) |
| rsi_weekly | float64 | 14-period Relative Strength Index (weekly, forward-filled) |
| stochrsi_daily | float64 | Stochastic RSI %K (daily) |
| stochrsi_weekly | float64 | Stochastic RSI %K (weekly, forward-filled) |
| macd_line | float64 | MACD line (12/26/9 EMA) |
| obv | float64 | On-Balance Volume |
| adosc | float64 | Accumulation/Distribution Oscillator (3/10) |
| atr_14 | float64 | 14-period Average True Range |
| adx_14 | float64 | 14-period Average Directional Index |
| bb_percent_b | float64 | Bollinger Bands %B (20-period, 2 std dev) |
| vwap_20 | float64 | 20-period Volume-Weighted Average Price |

#### Statistics

| Statistic | dema_9 | dema_10 | sma_12 | dema_20 | dema_25 | sma_50 | dema_90 | sma_100 | sma_200 | rsi_daily | rsi_weekly | stochrsi_daily | stochrsi_weekly | macd_line | obv | adosc | atr_14 | adx_14 | bb_percent_b | vwap_20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Count | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 | 6,819 |
| Mean | 153.94 | 153.94 | 153.60 | 153.94 | 153.94 | 152.41 | 153.84 | 150.90 | 148.04 | 54.62 | 57.11 | 53.51 | 57.98 | 0.44 | 510,111,784 | 1,350,796 | 1.93 | 23.05 | 0.58 | 153.07 |
| Std | 112.08 | 112.08 | 111.71 | 112.09 | 112.09 | 110.51 | 111.95 | 109.00 | 106.30 | 11.73 | 11.06 | 41.95 | 41.28 | 1.90 | 315,149,100 | 5,185,844 | 1.63 | 8.81 | 0.33 | 111.30 |
| Min | 44.45 | 44.49 | 45.94 | 44.85 | 45.13 | 45.49 | 45.78 | 47.64 | 48.20 | 15.80 | 19.31 | 0.00 | 0.00 | -20.52 | -177,027,700 | -50,524,855 | 0.50 | 7.08 | -0.42 | 46.01 |
| 25% | 65.35 | 65.38 | 65.09 | 65.46 | 65.47 | 64.57 | 65.38 | 64.19 | 63.83 | 46.52 | 50.11 | 0.03 | 8.75 | -0.27 | 300,213,150 | -716,008 | 0.90 | 16.65 | 0.33 | 64.96 |
| 50% | 97.05 | 97.02 | 97.13 | 97.34 | 97.55 | 96.27 | 97.46 | 96.48 | 91.11 | 55.30 | 57.50 | 58.69 | 68.45 | 0.44 | 569,796,800 | 1,200,410 | 1.22 | 21.36 | 0.65 | 96.86 |
| 75% | 224.18 | 224.28 | 223.55 | 224.11 | 224.45 | 222.38 | 223.13 | 221.28 | 219.20 | 62.94 | 64.67 | 100.00 | 100.00 | 1.12 | 768,337,650 | 3,608,995 | 2.66 | 27.50 | 0.84 | 222.09 |
| Max | 479.27 | 479.05 | 475.62 | 477.38 | 477.02 | 469.51 | 477.01 | 460.09 | 438.72 | 89.43 | 94.03 | 100.00 | 100.00 | 8.03 | 1,077,247,900 | 35,561,953 | 12.77 | 67.95 | 1.33 | 472.60 |

---

### QQQ_features_a20.parquet

| Property | Value |
|----------|-------|
| Path | data/processed/v1/QQQ_features_a20.parquet |
| Rows | 6,532 |
| Columns | 21 |
| Date Range | 1999-12-21 to 2025-12-09 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| dema_9 | float64 | 9-period Double Exponential Moving Average |
| dema_10 | float64 | 10-period Double Exponential Moving Average |
| sma_12 | float64 | 12-period Simple Moving Average |
| dema_20 | float64 | 20-period Double Exponential Moving Average |
| dema_25 | float64 | 25-period Double Exponential Moving Average |
| sma_50 | float64 | 50-period Simple Moving Average |
| dema_90 | float64 | 90-period Double Exponential Moving Average |
| sma_100 | float64 | 100-period Simple Moving Average |
| sma_200 | float64 | 200-period Simple Moving Average |
| rsi_daily | float64 | 14-period Relative Strength Index (daily) |
| rsi_weekly | float64 | 14-period Relative Strength Index (weekly, forward-filled) |
| stochrsi_daily | float64 | Stochastic RSI %K (daily) |
| stochrsi_weekly | float64 | Stochastic RSI %K (weekly, forward-filled) |
| macd_line | float64 | MACD line (12/26/9 EMA) |
| obv | float64 | On-Balance Volume |
| adosc | float64 | Accumulation/Distribution Oscillator (3/10) |
| atr_14 | float64 | 14-period Average True Range |
| adx_14 | float64 | 14-period Average Directional Index |
| bb_percent_b | float64 | Bollinger Bands %B (20-period, 2 std dev) |
| vwap_20 | float64 | 20-period Volume-Weighted Average Price |

#### Statistics

| Statistic | dema_9 | dema_10 | sma_12 | dema_20 | dema_25 | sma_50 | dema_90 | sma_100 | sma_200 | rsi_daily | rsi_weekly | stochrsi_daily | stochrsi_weekly | macd_line | obv | adosc | atr_14 | adx_14 | bb_percent_b | vwap_20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Count | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 | 6,532 |
| Mean | 136.91 | 136.91 | 136.45 | 136.91 | 136.91 | 134.85 | 136.76 | 132.77 | 128.94 | 54.44 | 56.78 | 54.21 | 56.07 | 0.59 | 4,330,534,912 | 10,646,315 | 2.44 | 23.73 | 0.58 | 135.77 |
| Std | 142.78 | 142.78 | 142.05 | 142.79 | 142.80 | 139.62 | 142.46 | 136.55 | 131.27 | 11.85 | 12.05 | 42.06 | 41.78 | 2.76 | 2,621,421,017 | 48,529,867 | 2.65 | 8.43 | 0.33 | 141.19 |
| Min | 16.99 | 17.01 | 17.71 | 17.18 | 17.28 | 19.20 | 17.27 | 19.96 | 20.81 | 19.15 | 20.17 | 0.00 | 0.00 | -17.71 | -1,572,551,800 | -357,303,686 | 0.35 | 7.46 | -0.44 | 18.00 |
| 25% | 36.50 | 36.51 | 36.59 | 36.59 | 36.65 | 36.87 | 36.06 | 36.51 | 35.54 | 45.58 | 48.73 | 0.31 | 1.50 | -0.28 | 2,037,917,850 | -11,295,249 | 0.67 | 17.34 | 0.31 | 36.59 |
| 50% | 71.07 | 71.09 | 70.94 | 71.26 | 71.17 | 70.17 | 71.65 | 68.55 | 66.40 | 55.16 | 58.13 | 60.27 | 65.28 | 0.34 | 5,147,666,050 | 13,588,722 | 1.11 | 22.08 | 0.65 | 70.89 |
| 75% | 178.07 | 177.87 | 178.14 | 178.11 | 178.23 | 176.44 | 177.62 | 170.84 | 166.90 | 63.22 | 65.81 | 100.00 | 100.00 | 1.09 | 6,019,912,150 | 34,830,855 | 3.59 | 29.12 | 0.85 | 177.75 |
| Max | 632.79 | 632.44 | 624.39 | 628.53 | 627.40 | 612.27 | 625.94 | 594.14 | 547.59 | 85.59 | 84.92 | 100.00 | 100.00 | 14.66 | 9,073,770,300 | 255,400,557 | 19.77 | 56.98 | 1.32 | 618.93 |

---

### SPY_OHLCV_2d.parquet

| Property | Value |
|----------|-------|
| Path | data/processed/v1/SPY_OHLCV_2d.parquet |
| Rows | 5,030 |
| Columns | 6 |
| Date Range | 1993-01-29 to 2025-12-09 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| Open | float64 | Opening price (USD) |
| High | float64 | Intraday high price (USD) |
| Low | float64 | Intraday low price (USD) |
| Close | float64 | Closing/adjusted price (USD) |
| Volume | int64 | Trading volume (shares) |

#### Statistics

| Statistic | Open | High | Low | Close | Volume |
|---|---|---|---|---|---|
| Count | 5,030 | 5,030 | 5,030 | 5,030 | 5,030 |
| Mean | 164.99 | 166.28 | 163.62 | 165.06 | 136,981,814 |
| Std | 148.73 | 149.76 | 147.64 | 148.82 | 158,055,741 |
| Min | 24.02 | 24.11 | 23.69 | 24.02 | 5,200 |
| 25% | 70.95 | 71.63 | 70.08 | 70.89 | 19,646,750 |
| 50% | 95.63 | 96.35 | 94.80 | 95.55 | 98,267,300 |
| 75% | 218.83 | 220.09 | 216.72 | 219.31 | 180,927,825 |
| Max | 688.72 | 689.70 | 682.19 | 687.06 | 1,420,092,600 |

---

### SPY_OHLCV_weekly.parquet

| Property | Value |
|----------|-------|
| Path | data/processed/v1/SPY_OHLCV_weekly.parquet |
| Rows | 1,716 |
| Columns | 6 |
| Date Range | 1993-01-29 to 2025-12-12 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| Open | float64 | Opening price (USD) |
| High | float64 | Intraday high price (USD) |
| Low | float64 | Intraday low price (USD) |
| Close | float64 | Closing/adjusted price (USD) |
| Volume | int64 | Trading volume (shares) |

#### Statistics

| Statistic | Open | High | Low | Close | Volume |
|---|---|---|---|---|---|
| Count | 1,716 | 1,716 | 1,716 | 1,716 | 1,716 |
| Mean | 164.55 | 167.07 | 162.04 | 164.92 | 401,525,947 |
| Std | 148.29 | 150.44 | 146.32 | 148.76 | 420,246,773 |
| Min | 24.17 | 24.33 | 23.69 | 24.11 | 102,200 |
| 25% | 70.93 | 72.01 | 69.58 | 70.71 | 54,950,075 |
| 50% | 95.60 | 96.81 | 93.60 | 95.39 | 313,807,350 |
| 75% | 217.00 | 220.59 | 215.45 | 217.76 | 543,758,875 |
| Max | 686.59 | 689.70 | 682.19 | 685.69 | 3,281,575,900 |

---

### SPY_dataset_a20.parquet

| Property | Value |
|----------|-------|
| Path | data/processed/v1/SPY_dataset_a20.parquet |
| Rows | 8,073 |
| Columns | 26 |
| Date Range | 1993-11-11 to 2025-12-08 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| Open | float64 | Opening price (USD) |
| High | float64 | Intraday high price (USD) |
| Low | float64 | Intraday low price (USD) |
| Close | float64 | Closing/adjusted price (USD) |
| Volume | int64 | Trading volume (shares) |
| dema_9 | float64 | 9-period Double Exponential Moving Average |
| dema_10 | float64 | 10-period Double Exponential Moving Average |
| sma_12 | float64 | 12-period Simple Moving Average |
| dema_20 | float64 | 20-period Double Exponential Moving Average |
| dema_25 | float64 | 25-period Double Exponential Moving Average |
| sma_50 | float64 | 50-period Simple Moving Average |
| dema_90 | float64 | 90-period Double Exponential Moving Average |
| sma_100 | float64 | 100-period Simple Moving Average |
| sma_200 | float64 | 200-period Simple Moving Average |
| rsi_daily | float64 | 14-period Relative Strength Index (daily) |
| rsi_weekly | float64 | 14-period Relative Strength Index (weekly, forward-filled) |
| stochrsi_daily | float64 | Stochastic RSI %K (daily) |
| stochrsi_weekly | float64 | Stochastic RSI %K (weekly, forward-filled) |
| macd_line | float64 | MACD line (12/26/9 EMA) |
| obv | float64 | On-Balance Volume |
| adosc | float64 | Accumulation/Distribution Oscillator (3/10) |
| atr_14 | float64 | 14-period Average True Range |
| adx_14 | float64 | 14-period Average Directional Index |
| bb_percent_b | float64 | Bollinger Bands %B (20-period, 2 std dev) |
| vwap_20 | float64 | 20-period Volume-Weighted Average Price |

#### Statistics

| Statistic | Open | High | Low | Close | Volume | dema_9 | dema_10 | sma_12 | dema_20 | dema_25 | sma_50 | dema_90 | sma_100 | sma_200 | rsi_daily | rsi_weekly | stochrsi_daily | stochrsi_weekly | macd_line | obv | adosc | atr_14 | adx_14 | bb_percent_b | vwap_20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Count | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 |
| Mean | 167.86 | 168.83 | 166.80 | 167.88 | 85,343,281 | 167.88 | 167.88 | 167.43 | 167.87 | 167.87 | 165.91 | 167.73 | 163.95 | 160.28 | 55.11 | 58.15 | 54.82 | 57.79 | 0.56 | 7,320,684,792 | 19,480,231 | 2.22 | 23.32 | 0.59 | 166.75 |
| Std | 148.26 | 149.00 | 147.44 | 148.29 | 90,071,171 | 148.29 | 148.29 | 147.69 | 148.30 | 148.30 | 145.69 | 148.04 | 143.14 | 138.70 | 11.37 | 11.71 | 41.66 | 41.74 | 2.39 | 6,774,294,983 | 73,372,796 | 2.20 | 8.25 | 0.32 | 146.97 |
| Min | 24.73 | 25.12 | 24.73 | 25.05 | 5,200 | 25.26 | 25.25 | 25.45 | 25.26 | 25.32 | 25.74 | 25.81 | 25.67 | 25.24 | 16.80 | 16.66 | 0.00 | 0.00 | -21.68 | -897,338,500 | -730,535,034 | 0.14 | 7.81 | -0.47 | 25.42 |
| 25% | 73.40 | 73.88 | 72.89 | 73.34 | 17,199,300 | 73.36 | 73.34 | 73.43 | 73.45 | 73.47 | 73.18 | 72.76 | 72.73 | 72.09 | 46.96 | 50.69 | 1.95 | 7.69 | -0.20 | 53,441,000 | -2,907,069 | 0.96 | 17.14 | 0.34 | 73.30 |
| 50% | 97.20 | 97.89 | 96.44 | 97.18 | 64,463,800 | 97.14 | 97.15 | 96.99 | 97.07 | 96.98 | 96.68 | 96.39 | 96.30 | 95.52 | 56.09 | 59.72 | 61.73 | 70.08 | 0.43 | 5,420,739,500 | 5,454,454 | 1.41 | 21.90 | 0.67 | 96.70 |
| 75% | 226.49 | 227.56 | 225.47 | 226.85 | 113,408,900 | 226.30 | 226.34 | 226.84 | 226.18 | 226.07 | 224.88 | 228.32 | 220.31 | 214.14 | 63.62 | 66.55 | 100.00 | 100.00 | 1.22 | 14,543,402,700 | 51,801,540 | 2.40 | 28.10 | 0.85 | 226.39 |
| Max | 688.72 | 689.70 | 684.83 | 687.39 | 871,026,300 | 684.96 | 684.65 | 680.29 | 682.68 | 682.35 | 672.92 | 685.56 | 657.87 | 616.25 | 87.19 | 92.27 | 100.00 | 100.00 | 12.20 | 18,139,624,300 | 433,508,970 | 20.20 | 59.55 | 1.32 | 677.09 |

---

### SPY_dataset_c.parquet

| Property | Value |
|----------|-------|
| Path | data/processed/v1/SPY_dataset_c.parquet |
| Rows | 8,073 |
| Columns | 34 |
| Date Range | 1993-11-11 to 2025-12-08 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| Open | float64 | Opening price (USD) |
| High | float64 | Intraday high price (USD) |
| Low | float64 | Intraday low price (USD) |
| Close | float64 | Closing/adjusted price (USD) |
| Volume | int64 | Trading volume (shares) |
| dema_9 | float64 | 9-period Double Exponential Moving Average |
| dema_10 | float64 | 10-period Double Exponential Moving Average |
| sma_12 | float64 | 12-period Simple Moving Average |
| dema_20 | float64 | 20-period Double Exponential Moving Average |
| dema_25 | float64 | 25-period Double Exponential Moving Average |
| sma_50 | float64 | 50-period Simple Moving Average |
| dema_90 | float64 | 90-period Double Exponential Moving Average |
| sma_100 | float64 | 100-period Simple Moving Average |
| sma_200 | float64 | 200-period Simple Moving Average |
| rsi_daily | float64 | 14-period Relative Strength Index (daily) |
| rsi_weekly | float64 | 14-period Relative Strength Index (weekly, forward-filled) |
| stochrsi_daily | float64 | Stochastic RSI %K (daily) |
| stochrsi_weekly | float64 | Stochastic RSI %K (weekly, forward-filled) |
| macd_line | float64 | MACD line (12/26/9 EMA) |
| obv | float64 | On-Balance Volume |
| adosc | float64 | Accumulation/Distribution Oscillator (3/10) |
| atr_14 | float64 | 14-period Average True Range |
| adx_14 | float64 | 14-period Average Directional Index |
| bb_percent_b | float64 | Bollinger Bands %B (20-period, 2 std dev) |
| vwap_20 | float64 | 20-period Volume-Weighted Average Price |
| vix_close | float64 | VIX closing value |
| vix_sma_10 | float64 | 10-day VIX simple moving average |
| vix_sma_20 | float64 | 20-day VIX simple moving average |
| vix_percentile_60d | float64 | 60-day rolling percentile rank (0-100) |
| vix_zscore_20d | float64 | 20-day rolling z-score |
| vix_regime | object | Volatility regime: low (<15), normal (15-25), high (>=25) |
| vix_change_1d | float64 | 1-day percent change |
| vix_change_5d | float64 | 5-day percent change |

#### Statistics

| Statistic | Open | High | Low | Close | Volume | dema_9 | dema_10 | sma_12 | dema_20 | dema_25 | sma_50 | dema_90 | sma_100 | sma_200 | rsi_daily | rsi_weekly | stochrsi_daily | stochrsi_weekly | macd_line | obv | adosc | atr_14 | adx_14 | bb_percent_b | vwap_20 | vix_close | vix_sma_10 | vix_sma_20 | vix_percentile_60d | vix_zscore_20d | vix_change_1d | vix_change_5d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Count | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 |
| Mean | 167.86 | 168.83 | 166.80 | 167.88 | 85,343,281 | 167.88 | 167.88 | 167.43 | 167.87 | 167.87 | 165.91 | 167.73 | 163.95 | 160.28 | 55.11 | 58.15 | 54.82 | 57.79 | 0.56 | 7,320,684,792 | 19,480,231 | 2.22 | 23.32 | 0.59 | 166.75 | 19.69 | 19.69 | 19.69 | 46.45 | -0.05 | 0.25 | 0.97 |
| Std | 148.26 | 149.00 | 147.44 | 148.29 | 90,071,171 | 148.29 | 148.29 | 147.69 | 148.30 | 148.30 | 145.69 | 148.04 | 143.14 | 138.70 | 11.37 | 11.71 | 41.66 | 41.74 | 2.39 | 6,774,294,983 | 73,372,796 | 2.20 | 8.25 | 0.32 | 146.97 | 8.03 | 7.81 | 7.65 | 33.64 | 1.21 | 7.22 | 14.92 |
| Min | 24.73 | 25.12 | 24.73 | 25.05 | 5,200 | 25.26 | 25.25 | 25.45 | 25.26 | 25.32 | 25.74 | 25.81 | 25.67 | 25.24 | 16.80 | 16.66 | 0.00 | 0.00 | -21.68 | -897,338,500 | -730,535,034 | 0.14 | 7.81 | -0.47 | 25.42 | 9.14 | 9.63 | 9.78 | 0.00 | -2.88 | -35.75 | -46.67 |
| 25% | 73.40 | 73.88 | 72.89 | 73.34 | 17,199,300 | 73.36 | 73.34 | 73.43 | 73.45 | 73.47 | 73.18 | 72.76 | 72.73 | 72.09 | 46.96 | 50.69 | 1.95 | 7.69 | -0.20 | 53,441,000 | -2,907,069 | 0.96 | 17.14 | 0.34 | 73.30 | 13.95 | 14.09 | 14.14 | 13.56 | -1.01 | -3.80 | -7.58 |
| 50% | 97.20 | 97.89 | 96.44 | 97.18 | 64,463,800 | 97.14 | 97.15 | 96.99 | 97.07 | 96.98 | 96.68 | 96.39 | 96.30 | 95.52 | 56.09 | 59.72 | 61.73 | 70.08 | 0.43 | 5,420,739,500 | 5,454,454 | 1.41 | 21.90 | 0.67 | 96.70 | 17.88 | 17.88 | 17.88 | 44.07 | -0.26 | -0.45 | -0.82 |
| 75% | 226.49 | 227.56 | 225.47 | 226.85 | 113,408,900 | 226.30 | 226.34 | 226.84 | 226.18 | 226.07 | 224.88 | 228.32 | 220.31 | 214.14 | 63.62 | 66.55 | 100.00 | 100.00 | 1.22 | 14,543,402,700 | 51,801,540 | 2.40 | 28.10 | 0.85 | 226.39 | 23.11 | 23.07 | 23.19 | 77.97 | 0.81 | 3.36 | 7.10 |
| Max | 688.72 | 689.70 | 684.83 | 687.39 | 871,026,300 | 684.96 | 684.65 | 680.29 | 682.68 | 682.35 | 672.92 | 685.56 | 657.87 | 616.25 | 87.19 | 92.27 | 100.00 | 100.00 | 12.20 | 18,139,624,300 | 433,508,970 | 20.20 | 59.55 | 1.32 | 677.09 | 82.69 | 69.36 | 65.03 | 100.00 | 4.14 | 115.60 | 212.90 |

---

### SPY_features_a20.parquet

| Property | Value |
|----------|-------|
| Path | data/processed/v1/SPY_features_a20.parquet |
| Rows | 8,073 |
| Columns | 21 |
| Date Range | 1993-11-11 to 2025-12-08 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| dema_9 | float64 | 9-period Double Exponential Moving Average |
| dema_10 | float64 | 10-period Double Exponential Moving Average |
| sma_12 | float64 | 12-period Simple Moving Average |
| dema_20 | float64 | 20-period Double Exponential Moving Average |
| dema_25 | float64 | 25-period Double Exponential Moving Average |
| sma_50 | float64 | 50-period Simple Moving Average |
| dema_90 | float64 | 90-period Double Exponential Moving Average |
| sma_100 | float64 | 100-period Simple Moving Average |
| sma_200 | float64 | 200-period Simple Moving Average |
| rsi_daily | float64 | 14-period Relative Strength Index (daily) |
| rsi_weekly | float64 | 14-period Relative Strength Index (weekly, forward-filled) |
| stochrsi_daily | float64 | Stochastic RSI %K (daily) |
| stochrsi_weekly | float64 | Stochastic RSI %K (weekly, forward-filled) |
| macd_line | float64 | MACD line (12/26/9 EMA) |
| obv | float64 | On-Balance Volume |
| adosc | float64 | Accumulation/Distribution Oscillator (3/10) |
| atr_14 | float64 | 14-period Average True Range |
| adx_14 | float64 | 14-period Average Directional Index |
| bb_percent_b | float64 | Bollinger Bands %B (20-period, 2 std dev) |
| vwap_20 | float64 | 20-period Volume-Weighted Average Price |

#### Statistics

| Statistic | dema_9 | dema_10 | sma_12 | dema_20 | dema_25 | sma_50 | dema_90 | sma_100 | sma_200 | rsi_daily | rsi_weekly | stochrsi_daily | stochrsi_weekly | macd_line | obv | adosc | atr_14 | adx_14 | bb_percent_b | vwap_20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Count | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 | 8,073 |
| Mean | 167.88 | 167.88 | 167.43 | 167.87 | 167.87 | 165.91 | 167.73 | 163.95 | 160.28 | 55.11 | 58.15 | 54.82 | 57.79 | 0.56 | 7,320,684,792 | 19,480,231 | 2.22 | 23.32 | 0.59 | 166.75 |
| Std | 148.29 | 148.29 | 147.69 | 148.30 | 148.30 | 145.69 | 148.04 | 143.14 | 138.70 | 11.37 | 11.71 | 41.66 | 41.74 | 2.39 | 6,774,294,983 | 73,372,796 | 2.20 | 8.25 | 0.32 | 146.97 |
| Min | 25.26 | 25.25 | 25.45 | 25.26 | 25.32 | 25.74 | 25.81 | 25.67 | 25.24 | 16.80 | 16.66 | 0.00 | 0.00 | -21.68 | -897,338,500 | -730,535,034 | 0.14 | 7.81 | -0.47 | 25.42 |
| 25% | 73.36 | 73.34 | 73.43 | 73.45 | 73.47 | 73.18 | 72.76 | 72.73 | 72.09 | 46.96 | 50.69 | 1.95 | 7.69 | -0.20 | 53,441,000 | -2,907,069 | 0.96 | 17.14 | 0.34 | 73.30 |
| 50% | 97.14 | 97.15 | 96.99 | 97.07 | 96.98 | 96.68 | 96.39 | 96.30 | 95.52 | 56.09 | 59.72 | 61.73 | 70.08 | 0.43 | 5,420,739,500 | 5,454,454 | 1.41 | 21.90 | 0.67 | 96.70 |
| 75% | 226.30 | 226.34 | 226.84 | 226.18 | 226.07 | 224.88 | 228.32 | 220.31 | 214.14 | 63.62 | 66.55 | 100.00 | 100.00 | 1.22 | 14,543,402,700 | 51,801,540 | 2.40 | 28.10 | 0.85 | 226.39 |
| Max | 684.96 | 684.65 | 680.29 | 682.68 | 682.35 | 672.92 | 685.56 | 657.87 | 616.25 | 87.19 | 92.27 | 100.00 | 100.00 | 12.20 | 18,139,624,300 | 433,508,970 | 20.20 | 59.55 | 1.32 | 677.09 |

---

### VIX_features_c.parquet

| Property | Value |
|----------|-------|
| Path | data/processed/v1/VIX_features_c.parquet |
| Rows | 8,994 |
| Columns | 9 |
| Date Range | 1990-03-27 to 2025-12-10 |

#### Column Schema

| Column | Dtype | Description |
|--------|-------|-------------|
| Date | datetime64[ns] | Trading date (datetime index) |
| vix_close | float64 | VIX closing value |
| vix_sma_10 | float64 | 10-day VIX simple moving average |
| vix_sma_20 | float64 | 20-day VIX simple moving average |
| vix_percentile_60d | float64 | 60-day rolling percentile rank (0-100) |
| vix_zscore_20d | float64 | 20-day rolling z-score |
| vix_regime | object | Volatility regime: low (<15), normal (15-25), high (>=25) |
| vix_change_1d | float64 | 1-day percent change |
| vix_change_5d | float64 | 5-day percent change |

#### Statistics

| Statistic | vix_close | vix_sma_10 | vix_sma_20 | vix_percentile_60d | vix_zscore_20d | vix_change_1d | vix_change_5d |
|---|---|---|---|---|---|---|---|
| Count | 8,994 | 8,994 | 8,994 | 8,994 | 8,994 | 8,994 | 8,994 |
| Mean | 19.44 | 19.45 | 19.45 | 46.10 | -0.05 | 0.24 | 0.92 |
| Std | 7.81 | 7.59 | 7.43 | 33.47 | 1.21 | 7.13 | 14.60 |
| Min | 9.14 | 9.63 | 9.78 | 0.00 | -3.30 | -35.75 | -46.67 |
| 25% | 13.90 | 14.04 | 14.08 | 13.56 | -1.00 | -3.74 | -7.46 |
| 50% | 17.55 | 17.60 | 17.58 | 44.07 | -0.26 | -0.42 | -0.75 |
| 75% | 22.73 | 22.70 | 22.85 | 77.97 | 0.79 | 3.28 | 6.98 |
| Max | 82.69 | 69.36 | 65.03 | 100.00 | 4.14 | 115.60 | 212.90 |

---

## Indicator Reference

### Tier A20 Indicators

| Indicator | Description |
|-----------|-------------|
| dema_9 | 9-period Double Exponential Moving Average |
| dema_10 | 10-period Double Exponential Moving Average |
| sma_12 | 12-period Simple Moving Average |
| dema_20 | 20-period Double Exponential Moving Average |
| dema_25 | 25-period Double Exponential Moving Average |
| sma_50 | 50-period Simple Moving Average |
| dema_90 | 90-period Double Exponential Moving Average |
| sma_100 | 100-period Simple Moving Average |
| sma_200 | 200-period Simple Moving Average |
| rsi_daily | 14-period Relative Strength Index (daily) |
| rsi_weekly | 14-period Relative Strength Index (weekly, forward-filled) |
| stochrsi_daily | Stochastic RSI %K (daily) |
| stochrsi_weekly | Stochastic RSI %K (weekly, forward-filled) |
| macd_line | MACD line (12/26/9 EMA) |
| obv | On-Balance Volume |
| adosc | Accumulation/Distribution Oscillator (3/10) |
| atr_14 | 14-period Average True Range |
| adx_14 | 14-period Average Directional Index |
| bb_percent_b | Bollinger Bands %B (20-period, 2 std dev) |
| vwap_20 | 20-period Volume-Weighted Average Price |

### VIX Features

| Feature | Description |
|---------|-------------|
| vix_close | VIX closing value |
| vix_sma_10 | 10-day VIX simple moving average |
| vix_sma_20 | 20-day VIX simple moving average |
| vix_percentile_60d | 60-day rolling percentile rank (0-100) |
| vix_zscore_20d | 20-day rolling z-score |
| vix_regime | Volatility regime: low (<15), normal (15-25), high (>=25) |
| vix_change_1d | 1-day percent change |
| vix_change_5d | 5-day percent change |

### Timescales

| Name | Pandas Freq | Description |
|------|-------------|-------------|
| daily | D | Daily OHLCV (default) |
| 2d | 2D | 2D calendar days |
| 3d | 3D | 3D calendar days |
| 5d | 5D | 5D calendar days |
| weekly | W-FRI | Weekly (Friday close) |
