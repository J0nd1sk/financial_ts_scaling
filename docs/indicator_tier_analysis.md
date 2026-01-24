# Indicator Tier Analysis - v1.0 Count-Based Tiers

**Date:** 2026-01-23
**Purpose:** Define cumulative indicator tiers where tier name = exact indicator count for proper scaling experiments.

---

## Executive Summary

**Problem Solved:** Previous tier naming (a100, a200) represented quality buckets, not feature counts. "a100" actually had ~180 features, "a200" had ~1,280 features. This conflated quality ranking with count, making feature scaling experiments imprecise.

**New Approach:** Cumulative count-based tiers where tier name = exact indicator count:
- a20 = exactly 20 indicators
- a50 = exactly 50 indicators (includes all of a20)
- a100 = exactly 100 indicators (includes all of a50)
- etc.

**Design Decision:** OHLCV (5 columns) are always present but NOT counted toward tier totals. Tiers measure **indicator scaling** - engineered features only.

---

## Tier Definitions

| Tier | Indicators | Added from Previous | Model Sees | Notes |
|------|------------|---------------------|------------|-------|
| a20 | 20 | - | 25 (20 + 5 OHLCV) | Existing implementation |
| a50 | 50 | +30 | 55 | High-signal additions |
| a100 | 100 | +50 | 105 | Core TA coverage |
| a200 | 200 | +100 | 205 | Standard TA toolkit |
| a500 | 500 | +300 | 505 | Extended coverage |
| a1000 | 1000 | +500 | 1005 | Comprehensive coverage |
| a2000 | 2000 | +1000 | 2005 | Near-complete catalog |
| aFULL | ~2209 | +~209 | ~2214 | Everything |

**Cumulative Property:** Each tier is a strict superset of all lower tiers.

---

## Ranking Methodology

### Hybrid: Signal Quality + Category Diversity

**For each tier increment:**
1. Identify underrepresented categories at current tier
2. Select highest-signal features from underrepresented categories first
3. Fill remaining slots with next-highest-signal features overall
4. Ensure no major category is completely absent at a100+

### Priority Factors (in order)

1. **Backtested Performance**: QQE, STC, VRP have documented high returns
2. **Fundamental Price Relationships**: Returns, momentum, gaps
3. **Standard Technical Analysis**: RSI, Stochastic, MACD, Bollinger Bands
4. **Volatility & Volume**: ATR, OBV, volume flow
5. **Risk Metrics**: Sharpe, Sortino, VaR
6. **Advanced/Experimental**: Fractal dimension, TDA, chaos theory

### Category Codes

| Code | Category | Priority |
|------|----------|----------|
| MOM | Momentum / Returns | HIGH |
| OSC | Oscillators (RSI, StochRSI, QQE, STC) | HIGH |
| VRP | Volatility Risk Premium | HIGH |
| RSK | Risk Metrics (Sharpe, Sortino) | HIGH |
| MA | Moving Averages & Derivatives | MEDIUM |
| VOL | Volatility (ATR, BB, KC) | MEDIUM |
| VLM | Volume (OBV, VWAP, KVO) | MEDIUM |
| TRD | Trend (MACD, ADX, SuperTrend) | MEDIUM |
| SR | Support/Resistance | MEDIUM |
| MTF | Multi-Timeframe | MEDIUM |
| CDL | Candlestick Patterns | MEDIUM |
| ENT | Entropy / Regime | LOW |
| ADV | Advanced Math (Fractal, TDA) | LOW |
| EXP | Experimental | LOW |

---

## Tier a20: Base (Rank 1-20) ✅ IMPLEMENTED

The existing `src/features/tier_a20.py` implementation serves as the base tier.

| Rank | Indicator | Category | Rationale |
|------|-----------|----------|-----------|
| 1 | dema_9 | MA | Fast trend (9-day DEMA) |
| 2 | dema_10 | MA | Short-term trend |
| 3 | sma_12 | MA | Short-term SMA |
| 4 | dema_20 | MA | Monthly trend |
| 5 | dema_25 | MA | Ichimoku-aligned |
| 6 | sma_50 | MA | Standard medium-term |
| 7 | dema_90 | MA | Quarterly trend |
| 8 | sma_100 | MA | Medium-long term |
| 9 | sma_200 | MA | Long-term standard |
| 10 | rsi_daily | OSC | Universal oscillator |
| 11 | rsi_weekly | OSC | Higher timeframe RSI |
| 12 | stochrsi_daily | OSC | Momentum of momentum |
| 13 | stochrsi_weekly | OSC | Weekly StochRSI |
| 14 | macd_line | TRD | Standard momentum |
| 15 | obv | VLM | Volume flow |
| 16 | adosc | VLM | Accumulation/Distribution |
| 17 | atr_14 | VOL | Volatility baseline |
| 18 | adx_14 | TRD | Trend strength |
| 19 | bb_percent_b | VOL | Bollinger position |
| 20 | vwap_20 | VLM | Institutional reference |

### Category Distribution (a20)
- MA: 9 (45%)
- OSC: 4 (20%)
- TRD: 2 (10%)
- VLM: 3 (15%)
- VOL: 2 (10%)

**Note:** a20 lacks high-signal indicators (QQE, STC, VRP, Sharpe/Sortino) - these are added in a50.

---

## Tier a50: +30 Indicators (Rank 21-50) - HIGH SIGNAL ADDITIONS

### Selection Rationale
Add the highest-signal indicators missing from a20, plus category diversity.

| Rank | Indicator | Category | Rationale |
|------|-----------|----------|-----------|
| 21 | return_1d | MOM | Fundamental: 1-day return |
| 22 | return_5d | MOM | Fundamental: 1-week return |
| 23 | return_21d | MOM | Fundamental: 1-month return |
| 24 | return_63d | MOM | Fundamental: 1-quarter return |
| 25 | return_252d | MOM | Fundamental: 1-year return |
| 26 | qqe_fast | OSC | High backtest: QQE fast line |
| 27 | qqe_slow | OSC | High backtest: QQE slow line |
| 28 | qqe_fast_slow_spread | OSC | QQE momentum spread |
| 29 | stc_value | OSC | High backtest: Schaff Trend Cycle |
| 30 | stc_from_50 | OSC | STC distance from neutral |
| 31 | vrp_10d | VRP | High alpha: Volatility Risk Premium |
| 32 | vrp_21d | VRP | VRP monthly horizon |
| 33 | implied_vs_realized_ratio | VRP | Fear gauge ratio |
| 34 | sharpe_20d | RSK | Risk-adjusted: 20-day Sharpe |
| 35 | sharpe_60d | RSK | Risk-adjusted: 60-day Sharpe |
| 36 | sortino_20d | RSK | Downside-focused: 20-day Sortino |
| 37 | sortino_60d | RSK | Downside-focused: 60-day Sortino |
| 38 | rsi_slope | OSC | RSI momentum direction |
| 39 | rsi_extreme_dist | OSC | RSI distance to overbought/oversold |
| 40 | price_pct_from_sma_50 | MA | Key level: % from 50 SMA |
| 41 | price_pct_from_sma_200 | MA | Key level: % from 200 SMA |
| 42 | sma_50_200_proximity | MA | Golden/death cross proximity |
| 43 | atr_pct | VOL | Normalized volatility |
| 44 | atr_pct_slope | VOL | Volatility expansion/contraction |
| 45 | bb_width | VOL | Bollinger Band width |
| 46 | overnight_gap | MOM | Overnight price change |
| 47 | open_to_close_pct | MOM | Intraday return |
| 48 | volume_ratio_20d | VLM | Relative volume |
| 49 | kvo_signal | VLM | Klinger Volume Oscillator |
| 50 | macd_histogram | TRD | MACD momentum |

### Category Distribution (a50)
- MA: 12 (24%) - added 3
- OSC: 10 (20%) - added 6
- MOM: 7 (14%) - added 7
- VOL: 5 (10%) - added 3
- VLM: 5 (10%) - added 2
- TRD: 3 (6%) - added 1
- VRP: 3 (6%) - added 3
- RSK: 4 (8%) - added 4

**a50 Achievement:** All major high-signal categories represented.

---

## Tier a100: +50 Indicators (Rank 51-100) - CORE TA COVERAGE

### Selection Rationale
Complete core technical analysis coverage with more oscillators, volatility, and trend indicators.

| Rank | Indicator | Category | Rationale |
|------|-----------|----------|-----------|
| 51 | return_1d_acceleration | MOM | Momentum shift detection |
| 52 | return_5d_acceleration | MOM | Weekly momentum shift |
| 53 | qqe_slope | OSC | QQE momentum |
| 54 | qqe_extreme_dist | OSC | QQE overbought/oversold distance |
| 55 | stc_slope | OSC | STC momentum |
| 56 | stc_extreme_dist | OSC | STC overbought/oversold distance |
| 57 | demarker_value | OSC | Exhaustion indicator |
| 58 | demarker_from_half | OSC | DeMarker bias |
| 59 | stoch_k_14 | OSC | Standard Stochastic %K |
| 60 | stoch_d_14 | OSC | Standard Stochastic %D |
| 61 | stoch_extreme_dist | OSC | Stochastic overbought/oversold |
| 62 | cci_14 | OSC | Commodity Channel Index |
| 63 | mfi_14 | OSC | Money Flow Index |
| 64 | williams_r_14 | OSC | Williams %R |
| 65 | vrp_5d | VRP | Short-term VRP |
| 66 | vrp_slope | VRP | VRP trend direction |
| 67 | sharpe_252d | RSK | Annual Sharpe |
| 68 | sortino_252d | RSK | Annual Sortino |
| 69 | sharpe_slope_20d | RSK | Sharpe trend |
| 70 | sortino_slope_20d | RSK | Sortino trend |
| 71 | var_95 | RSK | Value at Risk 95% |
| 72 | var_99 | RSK | Value at Risk 99% |
| 73 | cvar_95 | RSK | Conditional VaR 95% |
| 74 | sma_9_50_proximity | MA | Short vs medium cross |
| 75 | sma_50_slope | MA | 50 SMA direction |
| 76 | sma_200_slope | MA | 200 SMA direction |
| 77 | days_since_sma_50_cross | MA | Days since 50 SMA cross |
| 78 | days_since_sma_200_cross | MA | Days since 200 SMA cross |
| 79 | ema_12 | MA | EMA base for MACD |
| 80 | ema_26 | MA | EMA base for MACD |
| 81 | atr_pct_percentile_60d | VOL | ATR historical percentile |
| 82 | bb_width_percentile_60d | VOL | BB width historical percentile |
| 83 | parkinson_volatility | VOL | Efficient vol estimator |
| 84 | garman_klass_volatility | VOL | OHLC-based vol |
| 85 | vol_of_vol | VOL | Meta-volatility |
| 86 | adx_slope | TRD | Trend strength direction |
| 87 | plus_di_14 | TRD | Positive directional |
| 88 | minus_di_14 | TRD | Negative directional |
| 89 | di_spread | TRD | DI+/DI- difference |
| 90 | supertrend_direction | TRD | SuperTrend signal |
| 91 | obv_slope | VLM | OBV momentum |
| 92 | volume_price_trend | VLM | VPT indicator |
| 93 | kvo_histogram | VLM | KVO histogram |
| 94 | accumulation_dist | VLM | A/D line |
| 95 | expectancy_20d | MOM | Win rate × avg win |
| 96 | win_rate_20d | MOM | % positive days |
| 97 | buying_pressure_ratio | VLM | Candlestick-based pressure |
| 98 | fib_range_position | SR | Position in Fib range |
| 99 | prior_high_20d_dist | SR | Distance to 20d high |
| 100 | prior_low_20d_dist | SR | Distance to 20d low |

### Category Distribution (a100)
- MA: 20 (20%)
- OSC: 24 (24%)
- MOM: 11 (11%)
- VOL: 10 (10%)
- VLM: 10 (10%)
- TRD: 9 (9%)
- VRP: 5 (5%)
- RSK: 9 (9%)
- SR: 2 (2%)

**a100 Achievement:** Core technical analysis fully covered. All essential oscillators, volatility measures, and trend indicators present.

---

## Tier a200: +100 Indicators (Rank 101-200) - STANDARD TA TOOLKIT

### Selection Principles
- Complete MA type coverage (EMA, TEMA, WMA, KAMA, HMA, VWMA)
- Extended oscillator periods (5, 9, 21, 28)
- Divergence features
- Multi-timeframe features
- Candlestick pattern indicators

### Category Targets
- MA: +40 (complete all types × key periods)
- OSC: +20 (extended periods, divergences)
- VOL: +10 (channel features)
- VLM: +10 (volume flow extensions)
- TRD: +5 (Ichimoku, Donchian)
- MTF: +8 (weekly indicators)
- CDL: +5 (key candlestick patterns)
- ENT: +2 (basic entropy)

### Key Additions (101-200)

**Moving Averages (101-140):**
- tema_{periods} (9, 20, 50, 100)
- wma_{periods} (10, 20, 50, 200)
- kama_{periods} (10, 20, 50)
- hma_{periods} (9, 21, 50)
- vwma_{periods} (10, 20, 50)
- Additional slopes, accelerations, price distances

**Oscillators (141-160):**
- rsi_{periods} (5, 9, 21, 28) + derivatives
- stoch extended periods
- cmo_14, roc_14
- price_rsi_divergence
- price_stoch_divergence

**Volatility (161-170):**
- kc_width, kc_position
- gaussian_channel features
- historical_volatility_20d, 60d
- vol_regime_score

**Volume (171-180):**
- volume_delta_approx
- chaikin_money_flow
- force_index
- ease_of_movement

**Trend (181-185):**
- ichimoku_tenkan, kijun, cloud_thickness
- donchian_high_20, donchian_low_20

**Multi-Timeframe (186-193):**
- weekly_rsi, weekly_stoch, weekly_bb_percent_b
- weekly_sma_20, weekly_sma_50
- timeframe_alignment_score

**Candlestick (194-198):**
- doji_score, hammer_score, engulfing_score
- body_size_pct, upper_wick_pct

**Entropy (199-200):**
- permutation_entropy_20d
- hurst_exponent_100d

---

## Tier a500: +300 Indicators (Rank 201-500)

### Selection Principles
Extended coverage of all categories with deeper period combinations and advanced derivatives.

### Category Targets
- MA: +120 (all types × all periods × all derivatives)
- OSC: +60 (all periods × all derivatives)
- VOL: +30 (all estimators × periods)
- VLM: +25 (complete volume flow)
- TRD: +15 (all trend variants)
- SR: +15 (complete S/R)
- MTF: +10 (complete weekly)
- CDL: +25 (all candlestick patterns)
- ENT: +10 (complete entropy/regime)
- ADV: +10 (basic advanced math)

---

## Tier a1000: +500 Indicators (Rank 501-1000)

### Selection Principles
Comprehensive coverage including:
- All MA-to-MA relationships
- Complete divergence matrix
- SMC features (order blocks, FVG, liquidity)
- Complete candlestick pattern library
- Extended entropy/regime features
- Signal processing (VMD, wavelet, FFT)
- Topological Data Analysis basics

---

## Tier a2000: +1000 Indicators (Rank 1001-2000)

### Selection Principles
Near-complete catalog including:
- All derivative combinations (slope × acceleration × curvature)
- Calendar features
- Complete fractal dimension suite
- Chaos theory features
- Complete recurrence quantification
- Polynomial regression channels
- Ergodic economics features

---

## Tier aFULL: +~209 Indicators (Rank 2001-2209)

### Selection Principles
Everything remaining:
- Experimental/cutting-edge features
- Lyapunov exponent, correlation dimension
- MFDFA, DCCA, MF-DCCA
- Ruin probability, attractor features
- Any remaining feature combinations

---

## Implementation Priority

| Phase | Tier | Indicators | Status |
|-------|------|------------|--------|
| 1 | a20 | 20 | ✅ Implemented |
| 2 | a50 | 50 | ✅ Implemented |
| 3 | a100 | 100 | Pending |
| 4 | a200 | 200 | Pending |
| 5 | a500 | 500 | Pending |
| 6 | a1000 | 1000 | Pending |
| 7 | a2000 | 2000 | Pending |
| 8 | aFULL | ~2209 | Pending |

---

## Verification Checklist

- [x] Tier counts correct (a20=20, a50=50, a100=100, etc.)
- [x] Each indicator has exactly one priority rank
- [x] Category diversity achieved at a50+ tiers
- [x] Existing a20 implementation = indicators rank 1-20
- [x] a50 implementation matches rank 21-50
- [ ] a100 implementation matches rank 51-100

---

## Appendix A: Complete Priority Ranking (Indicators 1-100)

| Rank | Indicator | Category | Tier |
|------|-----------|----------|------|
| 1 | dema_9 | MA | a20 |
| 2 | dema_10 | MA | a20 |
| 3 | sma_12 | MA | a20 |
| 4 | dema_20 | MA | a20 |
| 5 | dema_25 | MA | a20 |
| 6 | sma_50 | MA | a20 |
| 7 | dema_90 | MA | a20 |
| 8 | sma_100 | MA | a20 |
| 9 | sma_200 | MA | a20 |
| 10 | rsi_daily | OSC | a20 |
| 11 | rsi_weekly | OSC | a20 |
| 12 | stochrsi_daily | OSC | a20 |
| 13 | stochrsi_weekly | OSC | a20 |
| 14 | macd_line | TRD | a20 |
| 15 | obv | VLM | a20 |
| 16 | adosc | VLM | a20 |
| 17 | atr_14 | VOL | a20 |
| 18 | adx_14 | TRD | a20 |
| 19 | bb_percent_b | VOL | a20 |
| 20 | vwap_20 | VLM | a20 |
| 21 | return_1d | MOM | a50 |
| 22 | return_5d | MOM | a50 |
| 23 | return_21d | MOM | a50 |
| 24 | return_63d | MOM | a50 |
| 25 | return_252d | MOM | a50 |
| 26 | qqe_fast | OSC | a50 |
| 27 | qqe_slow | OSC | a50 |
| 28 | qqe_fast_slow_spread | OSC | a50 |
| 29 | stc_value | OSC | a50 |
| 30 | stc_from_50 | OSC | a50 |
| 31 | vrp_10d | VRP | a50 |
| 32 | vrp_21d | VRP | a50 |
| 33 | implied_vs_realized_ratio | VRP | a50 |
| 34 | sharpe_20d | RSK | a50 |
| 35 | sharpe_60d | RSK | a50 |
| 36 | sortino_20d | RSK | a50 |
| 37 | sortino_60d | RSK | a50 |
| 38 | rsi_slope | OSC | a50 |
| 39 | rsi_extreme_dist | OSC | a50 |
| 40 | price_pct_from_sma_50 | MA | a50 |
| 41 | price_pct_from_sma_200 | MA | a50 |
| 42 | sma_50_200_proximity | MA | a50 |
| 43 | atr_pct | VOL | a50 |
| 44 | atr_pct_slope | VOL | a50 |
| 45 | bb_width | VOL | a50 |
| 46 | overnight_gap | MOM | a50 |
| 47 | open_to_close_pct | MOM | a50 |
| 48 | volume_ratio_20d | VLM | a50 |
| 49 | kvo_signal | VLM | a50 |
| 50 | macd_histogram | TRD | a50 |
| 51 | return_1d_acceleration | MOM | a100 |
| 52 | return_5d_acceleration | MOM | a100 |
| 53 | qqe_slope | OSC | a100 |
| 54 | qqe_extreme_dist | OSC | a100 |
| 55 | stc_slope | OSC | a100 |
| 56 | stc_extreme_dist | OSC | a100 |
| 57 | demarker_value | OSC | a100 |
| 58 | demarker_from_half | OSC | a100 |
| 59 | stoch_k_14 | OSC | a100 |
| 60 | stoch_d_14 | OSC | a100 |
| 61 | stoch_extreme_dist | OSC | a100 |
| 62 | cci_14 | OSC | a100 |
| 63 | mfi_14 | OSC | a100 |
| 64 | williams_r_14 | OSC | a100 |
| 65 | vrp_5d | VRP | a100 |
| 66 | vrp_slope | VRP | a100 |
| 67 | sharpe_252d | RSK | a100 |
| 68 | sortino_252d | RSK | a100 |
| 69 | sharpe_slope_20d | RSK | a100 |
| 70 | sortino_slope_20d | RSK | a100 |
| 71 | var_95 | RSK | a100 |
| 72 | var_99 | RSK | a100 |
| 73 | cvar_95 | RSK | a100 |
| 74 | sma_9_50_proximity | MA | a100 |
| 75 | sma_50_slope | MA | a100 |
| 76 | sma_200_slope | MA | a100 |
| 77 | days_since_sma_50_cross | MA | a100 |
| 78 | days_since_sma_200_cross | MA | a100 |
| 79 | ema_12 | MA | a100 |
| 80 | ema_26 | MA | a100 |
| 81 | atr_pct_percentile_60d | VOL | a100 |
| 82 | bb_width_percentile_60d | VOL | a100 |
| 83 | parkinson_volatility | VOL | a100 |
| 84 | garman_klass_volatility | VOL | a100 |
| 85 | vol_of_vol | VOL | a100 |
| 86 | adx_slope | TRD | a100 |
| 87 | plus_di_14 | TRD | a100 |
| 88 | minus_di_14 | TRD | a100 |
| 89 | di_spread | TRD | a100 |
| 90 | supertrend_direction | TRD | a100 |
| 91 | obv_slope | VLM | a100 |
| 92 | volume_price_trend | VLM | a100 |
| 93 | kvo_histogram | VLM | a100 |
| 94 | accumulation_dist | VLM | a100 |
| 95 | expectancy_20d | MOM | a100 |
| 96 | win_rate_20d | MOM | a100 |
| 97 | buying_pressure_ratio | VLM | a100 |
| 98 | fib_range_position | SR | a100 |
| 99 | prior_high_20d_dist | SR | a100 |
| 100 | prior_low_20d_dist | SR | a100 |

---

*Document Version: 1.0*
*Supersedes: v0.5 (quality-bucket approach)*
*Created: 2026-01-23*
*Related: docs/indicator_catalog.md, src/features/tier_a20.py*
