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

## Tier Implementation Status

| Tier | Target | Actual | Status | Notes |
|------|--------|--------|--------|-------|
| a20 | 20 | 20 | ✅ Complete | Phase 2 baseline |
| a50 | 50 | 50 | ✅ Complete | Extended MAs, slopes, crosses |
| a100 | 100 | 100 | ✅ Complete | Volume, volatility, momentum extensions |
| a200 | 200 | **206** | ✅ Complete | Ichimoku, Donchian, entropy/regime indicators |
| a500 | 500 | - | Planned | Future |
| a1000 | 1000 | - | Planned | Future |
| a2000 | 2000 | - | Planned | Future |

**Note on actual counts:** Tier targets are approximate. Actual feature counts may differ based on indicator groupings (e.g., Ichimoku Cloud adds 6 related features together). Scientific validity depends on consistent tier definitions across experiments, not exact round numbers. The 206 features in tier_a200 provide richer signal than artificially constraining to exactly 200.

## Tier a50, a100, a200 Details

See `src/features/tier_a50.py`, `src/features/tier_a100.py`, and `src/features/tier_a200.py` for full feature lists and implementation details.

**tier_a200 categories (106 new features, 206 total):**
- Chunk 1 (101-120): TEMA, WMA, KAMA, HMA, VWMA and derived
- Chunk 2 (121-140): Duration counters, MA cross recency, proximity
- Chunk 3 (141-160): BB extension, RSI duration, mean reversion, patterns
- Chunk 4 (161-180): MACD extensions, volume dynamics, calendar, candle
- Chunk 5 (181-206): Ichimoku Cloud, Donchian Channel, divergence, entropy/regime

