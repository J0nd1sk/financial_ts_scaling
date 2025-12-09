# Feature Tier Tracking

## Tier a20 (Phase 2 Baseline)

| # | Feature | Notes / Source |
|---|---------|----------------|
| 1 | DEMA(9) | TA-Lib DEMA, Close |
| 2 | DEMA(10) | TA-Lib DEMA |
| 3 | SMA(12) | TA-Lib SMA |
| 4 | DEMA(20) | TA-Lib DEMA |
| 5 | DEMA(25) | TA-Lib DEMA |
| 6 | SMA(50) | TA-Lib SMA |
| 7 | DEMA(90) | TA-Lib DEMA |
| 8 | SMA(100) | TA-Lib SMA |
| 9 | SMA(200) | TA-Lib SMA |
|10 | RSI (daily, 14) | TA-Lib RSI |
|11 | RSI (5-day resample) | Resample Monday-aligned, TA-Lib RSI, ffill to daily |
|12 | StochRSI (daily) | TA-Lib STOCHRSI (%K) |
|13 | StochRSI (5-day) | Weekly resample + STOCHRSI, ffill |
|14 | MACD line | TA-Lib MACD (fast12/slow26/signal9), use MACD line |
|15 | OBV | TA-Lib OBV |
|16 | ADOSC | TA-Lib ADOSC (3,10) |
|17 | ATR(14) | TA-Lib ATR |
|18 | ADX(14) | TA-Lib ADX |
|19 | Bollinger %B | From TA-Lib BBANDS (20,2); computed as (Close-Lower)/(Upper-Lower) |
|20 | VWAP (20-day rolling) | Custom rolling sum Close*Vol / sum Vol |

**Dataset**: `SPY.features.a20` (generated via `scripts/build_features_a20.py`, registered in data/processed manifest)

## Future Tiers

| Tier | Description | Status |
|------|-------------|--------|
| a50  | 50 indicators | TBD |
| a100 | 100 indicators | TBD |
| ...  | ... | ... |

Add rows as each tier is defined (indicators + references).

