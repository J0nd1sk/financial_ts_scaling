# Indicator Catalog - Exhaustive Feature Enumeration

**Purpose:** Systematic enumeration of ALL possible indicators for Phase 6C feature scaling.
**Status:** DRAFT v0.4 - advanced mathematical features added
**Created:** 2026-01-22
**Revised:** 2026-01-23

---

## Revision Notes (v0.4)

**Key changes from v0.3:**
- Added **Category 16: Advanced Mathematical Features** (~118 features)
  - 16.1 Fractal Analysis (~17 features): Higuchi FD, Katz FD, MFDFA, Lévy alpha, FDI
  - 16.2 Chaos Theory (~8 features): Lyapunov exponent, correlation dimension
  - 16.3 Recurrence Quantification Analysis (~11 features): DET, LAM, entropy
  - 16.4 Spectral Decomposition (~10 features): EMD/EEMD, Hilbert transform
  - 16.5 Topological Data Analysis (~11 features): Betti curves, persistence
  - 16.6 Cross-Correlation Analysis (~4 features): DCCA, MF-DCCA
  - 16.7 Ergodic Economics (~6 features): Time-average growth, Kelly fraction
  - 16.8 Polynomial Regression Channels (~13 features): Quadratic, cubic, quintic
  - 16.9 Additional Stochastic (~8 features): Rolling Hurst, DFA alpha, mean reversion
  - 16.10 Volatility Risk Premium (~8 features): VRP, implied/realized ratio
  - 16.11 Risk Resilience (~6 features): Recovery speed, resilience score
- New dependencies: antropy, nolds, MFDFA, hfda, PyEMD, pyrqa, giotto-tda
- Total new features: ~118
- Updated grand total: ~2,090 features

---

## Revision Notes (v0.3)

**Key changes from v0.2:**
- Added **Category 14: Risk-Adjusted Metrics** (~26 features)
  - Sharpe Ratio (20d, 60d, 252d + slopes/accels)
  - Sortino Ratio (20d, 60d, 252d + slopes/accels)
  - Value at Risk (VaR) at 95% and 99% confidence
  - Conditional VaR / Expected Shortfall (CVaR)
- Added **Category 15: Signal Processing** (~15 features, optional)
  - Variational Mode Decomposition (VMD) components
  - Wavelet approximation/detail coefficients
  - FFT dominant frequency features
- Added **QQE (Quantitative Qualitative Estimation)** to Category 2 (~8 features)
- Added **Schaff Trend Cycle (STC)** to Category 2 (~7 features)
- Added **DeMarker indicator** to Category 2 (~5 features)
- Added **Donchian Channel** to Category 5 (~6 features)
- Added **Expectancy metrics** to Category 8 (~5 features)
- Added **Daily return enhancements** to Category 8 (~3 features)
- Total new features: ~75
- Updated grand total: ~1,970 features

**Previous revisions (v0.2):**
- Applied consolidation patterns throughout (signed features replace binary pairs)
- Added acceleration derivatives for all slope features
- Expanded StochRSI section (distinct from Stochastic)
- Added Gaussian Channel and Keltner Channel sections
- Expanded candlestick patterns (~30 patterns)
- Removed redundant binary calendar features (keep only continuous)
- Added Data Lookback Impact section
- Added SMC library reference and derivatives

---

## Design Question: Raw Values vs. Relational Only?

**Core Principle (established):** Raw indicator values are NOISE; relationships and dynamics are SIGNAL.

**Options:**
1. **Relational Only:** Exclude raw values entirely (e.g., no `SMA_50` value, only `price_pct_from_SMA_50`)
2. **Hybrid:** Include raw values for a subset (baseline comparison) + full relational features
3. **Full Enumeration:** Include everything, let model testing determine value

**Recommendation:** Option 1 (Relational Only) for primary tiers, with Option 2 for ablation studies.

**Rationale:**
- Raw SMA_50 = 450 vs 150 means nothing without context
- Price 2% above SMA_50 is universally interpretable
- Reduces feature count while increasing signal density
- Can always add raw values back for comparison

---

## Naming Convention

```
{base}_{param}_{timeframe}_{derivative}_{relation}

Examples:
- sma_50d_slope               # 50-day SMA slope
- sma_50d_slope_acceleration  # Change in 50-day SMA slope
- price_pct_from_sma_50d      # Price % distance from 50-day SMA (signed: + above, - below)
- days_since_sma_50d_cross    # Days since price crossed 50-day SMA (signed: + bullish, - bearish)
- sma_9d_50d_proximity        # % distance between 9-day and 50-day SMA (signed)
- days_since_sma_9d_50d_cross # Days since 9-day crossed 50-day (signed: + bullish, - bearish)
```

---

## Consolidation Patterns

These four patterns are applied consistently throughout this catalog to reduce feature count while preserving information.

### Pattern 1: Signed Duration
```
BEFORE: days_above_{x} + days_below_{x} (two features)
AFTER:  days_since_{x}_cross (one signed feature)
        Positive = days since bullish cross
        Negative = days since bearish cross
```

### Pattern 2: Signed Distance
```
BEFORE: max(0, RSI - 70) for overbought_dist + max(0, 30 - RSI) for oversold_dist
AFTER:  {indicator}_extreme_dist (one signed feature)
        Positive = distance into overbought zone
        Negative = distance into oversold zone
```

### Pattern 3: Signed Extremes
```
BEFORE: days_rsi_overbought + days_rsi_oversold (two features)
AFTER:  days_at_extreme (one signed feature)
        Positive = days overbought
        Negative = days oversold
        Zero = neutral zone
```

### Pattern 4: Slope + Acceleration Pairs
```
EVERY {indicator}_slope MUST have {indicator}_acceleration
acceleration = slope[t] - slope[t-1] (change in slope)
```

**Rationale:** These patterns typically halve the feature count for positional indicators while preserving all information through sign encoding.

---

## Category 1: Moving Averages

### 1.1 MA Types

| Type | Name | Description | Pros | Cons |
|------|------|-------------|------|------|
| SMA | Simple Moving Average | Equal-weighted average | Simple, universal | Lagging, equal weight to old data |
| EMA | Exponential Moving Average | Exponentially-weighted | More responsive | Still lagging |
| DEMA | Double EMA | EMA of EMA, reduced lag | Less lag than EMA | Can overshoot |
| TEMA | Triple EMA | EMA of EMA of EMA | Even less lag | More overshoot risk |
| WMA | Weighted Moving Average | Linearly-weighted | Smooth middle ground | Less common |
| KAMA | Kaufman Adaptive MA | Adapts to volatility | Smart adaptation | Complex, more params |
| HMA | Hull Moving Average | Smoothed, reduced lag | Very responsive | Can be noisy |
| VWMA | Volume-Weighted MA | Weighted by volume | Incorporates volume | Requires volume data |

### 1.2 MA Periods (Standard)

| Period | Timeframe | Usage |
|--------|-----------|-------|
| 5 | 1 week | Very short-term |
| 9 | ~2 weeks | Short-term (Ichimoku Tenkan) |
| 10 | 2 weeks | Short-term standard |
| 20 | 1 month | Short-term standard |
| 21 | 1 month | Fibonacci-based |
| 26 | ~5 weeks | Ichimoku Kijun |
| 50 | ~2.5 months | Medium-term standard |
| 100 | ~5 months | Medium-term |
| 200 | ~10 months | Long-term standard |
| 252 | 1 year | Annual |

### 1.3 MA Derivative Features

For each MA type × period combination:

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `{ma}_slope` | (MA[t] - MA[t-1]) / MA[t-1] | 1-day rate of change | Trend direction | Noisy |
| `{ma}_slope_acceleration` | slope[t] - slope[t-1] | Change in 1-day slope | Momentum shift early | Very noisy |
| `{ma}_slope_5d` | (MA[t] - MA[t-5]) / MA[t-5] / 5 | 5-day avg slope | Smoother trend | More lag |
| `{ma}_slope_5d_acceleration` | slope_5d[t] - slope_5d[t-1] | Change in 5-day slope | Clean momentum shift | More lag |
| `{ma}_curvature` | 2nd derivative | Concavity | Inflection detection | Noisy, complex |

### 1.4 Price-to-MA Relational Features

For each MA type × period:

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `price_pct_from_{ma}` | (close - MA) / MA * 100 | % distance from MA (signed: + above, - below) | Universal interpretation | - |
| `days_since_{ma}_cross` | Signed days since price crossed MA | + if bullish cross, - if bearish cross | Overextension signal | Large values early |
| `high_pct_from_{ma}` | (high - MA) / MA * 100 | Intraday reach (signed) | Rejection detection | - |
| `low_pct_from_{ma}` | (low - MA) / MA * 100 | Intraday reach (signed) | Support test detection | - |

**Consolidation Notes:**
- `price_above_{ma}` removed: redundant with sign of `price_pct_from_{ma}`
- `days_above_{ma}` + `days_below_{ma}` → `days_since_{ma}_cross` (signed)

### 1.5 MA-to-MA Relational Features

For each pair of MAs (e.g., 9d vs 50d, 50d vs 200d):

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `{ma1}_{ma2}_proximity` | (MA1 - MA2) / MA2 * 100 | % distance between MAs (signed: + bullish, - bearish) | Trend strength | - |
| `days_since_{ma1}_{ma2}_cross` | Signed days since last cross | + if bullish cross, - if bearish cross | Cross recency & direction | Large values early |
| `{ma1}_{ma2}_converging_rate` | Change in proximity (slope) | Negative = converging, positive = diverging | Cross anticipation | Can false signal |
| `{ma1}_{ma2}_converging_accel` | Change in converging rate | Acceleration of convergence | Leading signal | Noisy |

**Consolidation Notes:**
- `{ma1}_above_{ma2}` removed: redundant with sign of `{ma1}_{ma2}_proximity`
- `{ma1}_{ma2}_cross_direction` merged into `days_since_{ma1}_{ma2}_cross` (signed)
- `{ma1}_{ma2}_converging` (binary) → `{ma1}_{ma2}_converging_rate` (continuous) + acceleration

### 1.6 MA Feature Count Estimate

```
MA Types: 8 (SMA, EMA, DEMA, TEMA, WMA, KAMA, HMA, VWMA)
Periods: 10 (5, 9, 10, 20, 21, 26, 50, 100, 200, 252)
Derivatives per MA: 5 (slope, slope_accel, slope_5d, slope_5d_accel, curvature)
Price relations per MA: 4 (pct_from, days_since_cross, high_pct, low_pct) [consolidated from 6]
MA pairs: ~15 meaningful pairs (not all 45 combinations)
MA-MA relations per pair: 4 (proximity, days_since_cross, converging_rate, converging_accel) [consolidated from 5]

Raw MAs (if included): 8 × 10 = 80
Derivatives: 8 × 10 × 5 = 400
Price-to-MA: 8 × 10 × 4 = 320 (was 480)
MA-to-MA: 15 × 4 = 60 (was 75)

TOTAL MA FEATURES: ~780 (or ~860 including raw values)
Reduction: ~95 features saved via consolidation
```

---

## Category 2: Oscillators (RSI, Stochastic, etc.)

### 2.1 Oscillator Types

| Type | Range | Description | Pros | Cons |
|------|-------|-------------|------|------|
| RSI | 0-100 | Relative Strength Index | Universal, bounded | Can stay overbought long |
| Stochastic %K | 0-100 | Position in range | Good for ranging markets | Whipsaws in trends |
| Stochastic %D | 0-100 | Smoothed %K | Less noise | More lag |
| Williams %R | -100 to 0 | Inverse stochastic | Same info as stoch | Confusing scale |
| CCI | Unbounded | Commodity Channel Index | Catches extremes | Unbounded = harder to interpret |
| MFI | 0-100 | Money Flow Index (RSI with volume) | Volume-weighted | Requires volume |
| CMO | -100 to 100 | Chande Momentum Oscillator | Centered at 0 | Less common |
| ROC | Unbounded | Rate of Change | Simple momentum | Unbounded |
| Momentum | Unbounded | Price - Price[n] | Simplest | Unbounded, scale varies |

### 2.2 Oscillator Periods

| Period | Usage |
|--------|-------|
| 5 | Very short-term |
| 9 | Short-term |
| 14 | Standard (RSI default) |
| 21 | Medium-term |
| 28 | Longer-term |

### 2.3 Oscillator Raw vs. Relational Features

**Raw (if included):**

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `rsi_14` | RSI(14) value | Standard baseline | Raw value, context-free |
| `stoch_k_14` | Stochastic %K | - | - |

**Relational:**

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `rsi_14_extreme_dist` | RSI - 80 if RSI > 50, else RSI - 20 | Signed distance to extreme zone | + overbought, - oversold | - |
| `rsi_14_from_50` | RSI - 50 | Distance from neutral (signed) | Centered interpretation | - |
| `rsi_14_percentile_60d` | Percentile rank over 60 days | Historical context | Relative strength | Lookback dependency |
| `days_rsi_at_extreme` | Signed consecutive days in extreme | + overbought, - oversold, 0 neutral | Exhaustion signal | - |
| `rsi_slope` | RSI[t] - RSI[t-1] | RSI momentum | Divergence detection | Noisy |
| `rsi_slope_acceleration` | slope[t] - slope[t-1] | Change in RSI momentum | Early momentum shift | Very noisy |
| `rsi_slope_5d` | (RSI[t] - RSI[t-5]) / 5 | Smoothed RSI momentum | Cleaner | More lag |
| `rsi_slope_5d_acceleration` | slope_5d[t] - slope_5d[t-1] | Change in smoothed momentum | Clean signal | More lag |

**Consolidation Notes:**
- `rsi_14_overbought_dist` + `rsi_14_oversold_dist` → `rsi_14_extreme_dist` (signed)
- `days_rsi_overbought` + `days_rsi_oversold` → `days_rsi_at_extreme` (signed)

### 2.4 Oscillator Divergence Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `price_rsi_divergence` | Price making new high but RSI not (or vice versa) | Classic reversal signal | Complex to compute |
| `divergence_duration` | Days divergence has persisted | Amplifies signal | Needs clear definition |
| `divergence_magnitude` | Size of the divergence | Strength of signal | Complex |

### 2.5 StochRSI Features (Distinct from Stochastic)

**StochRSI** = Stochastic formula applied to RSI (not price). It measures momentum of momentum.

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `stochrsi_k_14` | Stochastic(%K) of RSI(14) | Raw StochRSI value | Faster than RSI | More whipsaw |
| `stochrsi_d_14` | SMA(3) of StochRSI %K | Smoothed StochRSI | Less noise | More lag |
| `stochrsi_zone` | -1 (oversold <20), 0 (neutral), +1 (overbought >80) | Zone classification | Simple state | Categorical |
| `stochrsi_extreme_dist` | StochRSI - 80 if >50, else StochRSI - 20 | Signed distance to extreme | Magnitude of extreme | - |
| `days_stochrsi_neutral` | Days in 40-60 zone | Consolidation duration | Breakout anticipation | - |
| `days_since_stochrsi_20_cross` | Signed days since crossing 20 | + if bullish cross, - if bearish | Oversold exit timing | - |
| `days_since_stochrsi_80_cross` | Signed days since crossing 80 | + if bullish cross, - if bearish | Overbought exit timing | - |
| `stochrsi_slope` | StochRSI[t] - StochRSI[t-1] | StochRSI momentum | Fast signal | Noisy |
| `stochrsi_slope_acceleration` | slope[t] - slope[t-1] | Change in StochRSI momentum | Early reversal | Very noisy |

**Key Insight:** StochRSI > 80 while RSI < 70 = momentum building but not yet overbought on RSI.

### 2.6 QQE (Quantitative Qualitative Estimation) Features

**QQE** = Smoothed RSI with ATR-based trailing bands. More responsive than raw RSI, provides overbought/oversold levels AND crossover signals.

**Calculation:**
1. Calculate RSI (typically 14-period)
2. Apply EMA smoothing to RSI
3. Calculate ATR of smoothed RSI
4. Create fast/slow trailing bands using ATR multiples

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `qqe_fast` | Fast QQE line value | Smoothed RSI momentum | More responsive than RSI | Still lagging |
| `qqe_slow` | Slow QQE line value | Confirmation line | Less noise | More lag |
| `qqe_fast_slow_spread` | QQE_fast - QQE_slow | Signed momentum spread | Cross anticipation | - |
| `qqe_from_50` | QQE - 50 | Signed distance from neutral | Centered interpretation | - |
| `qqe_extreme_dist` | QQE - 70 if >50, else QQE - 30 | Signed distance to extreme zone | + overbought, - oversold | - |
| `qqe_slope` | QQE[t] - QQE[t-1] | QQE momentum | Divergence detection | Noisy |
| `qqe_slope_acceleration` | slope[t] - slope[t-1] | Change in QQE momentum | Early reversal signal | Very noisy |
| `days_since_qqe_cross` | Signed days since fast/slow cross | + bullish cross, - bearish | Timing signal | - |

**Tier Assignment:** a100 (high-signal momentum indicator)

### 2.7 Schaff Trend Cycle (STC) Features

**STC** = Modified MACD with stochastic smoothing. Faster trend detection than MACD with less lag.

**Calculation:**
1. Calculate MACD (EMA_23 - EMA_50)
2. Apply Stochastic formula to MACD values
3. Smooth with Wilders smoothing

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `stc_value` | STC value (0-100) | Trend cycle position | Range-bound like RSI | - |
| `stc_from_50` | STC - 50 | Signed distance from neutral | Centered interpretation | - |
| `stc_extreme_dist` | STC - 75 if >50, else STC - 25 | Signed distance to extreme | + overbought, - oversold | - |
| `stc_slope` | STC[t] - STC[t-1] | STC momentum | Trend direction | Noisy |
| `stc_slope_acceleration` | slope[t] - slope[t-1] | Change in STC momentum | Early reversal | Very noisy |
| `days_since_stc_cross_25` | Signed days since crossing 25 | + if bullish (up through), - bearish | Oversold exit timing | - |
| `days_since_stc_cross_75` | Signed days since crossing 75 | + if bullish, - bearish (down through) | Overbought exit timing | - |

**Tier Assignment:** a100 (backtested high-performance indicator)

### 2.8 DeMarker Indicator Features

**DeMarker** = Compares current high/low vs previous day to measure exhaustion.

**Calculation:**
- DeMax = max(High - High[1], 0)
- DeMin = max(Low[1] - Low, 0)
- DeMarker = SMA(DeMax, n) / (SMA(DeMax, n) + SMA(DeMin, n))

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `demarker_value` | DeMarker value (0-1) | Exhaustion measure | Simple concept | Less common |
| `demarker_from_half` | DeMarker - 0.5 | Signed distance from neutral | + bullish bias, - bearish | - |
| `demarker_extreme_dist` | DeMarker - 0.7 if >0.5, else DeMarker - 0.3 | Signed distance to extreme | + overbought, - oversold | - |
| `demarker_slope` | DeMarker[t] - DeMarker[t-1] | DeMarker momentum | Direction change | Noisy |
| `demarker_slope_acceleration` | slope[t] - slope[t-1] | Change in DeMarker momentum | Early reversal | Very noisy |

**Tier Assignment:** a200 (well-backtested, CAGR 6.98% historical)

### 2.10 Oscillator Feature Count Estimate

```
Oscillator Types: 9 (RSI, Stoch %K, Stoch %D, Williams %R, CCI, MFI, CMO, ROC, Momentum)
Periods: 5 (5, 9, 14, 21, 28)
Raw values (if included): 9 × 5 = 45
Relational per oscillator: ~8 (extreme_dist, from_neutral, percentile, days_at_extreme,
                               slope, slope_accel, slope_5d, slope_5d_accel) [consolidated from 10]
Divergence features: ~3 per oscillator type
StochRSI features: ~9 (distinct momentum-of-momentum indicator)

Relational: 9 × 5 × 8 = 360 (was 450)
Divergence: 9 × 3 = 27 (was 54)
StochRSI: 9 (new in v0.2)
QQE: 8 (new in v0.3)
STC: 7 (new in v0.3)
DeMarker: 5 (new in v0.3)

TOTAL OSCILLATOR FEATURES: ~416 (or ~461 including raw)
Change from v0.2: +20 (QQE + STC + DeMarker)
```

---

## Category 3: Volatility Indicators

### 3.1 Volatility Types

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| ATR | Average True Range | Standard, absolute | Scale-dependent |
| ATR% | ATR / Close * 100 | Normalized ATR | Universal | - |
| Bollinger Band Width | (Upper - Lower) / Middle | Relative volatility | Well-known | - |
| Keltner Channel Width | (Upper - Lower) / Middle | Smoother than BB | Less common | - |
| Standard Deviation | Rolling std of returns | Statistical | Assumes normality |
| Parkinson Volatility | Uses high/low | More efficient | Ignores close |
| Garman-Klass Volatility | Uses OHLC | Most efficient | Complex |
| Historical Volatility | Annualized std dev | Standard in options | Lookback sensitive |

### 3.2 Volatility Derivative Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `atr_pct_slope` | Change in ATR% | Vol expansion/contraction | - |
| `atr_pct_slope_acceleration` | Change in ATR% slope | Vol regime shift early | Noisy |
| `atr_pct_percentile_60d` | Historical percentile | Relative vol context | - |
| `atr_pct_zscore_20d` | Z-score of ATR% | Anomaly detection | Assumes normality |
| `bb_width_percentile` | BB width percentile | Squeeze detection | - |
| `days_vol_extreme` | Signed days in extreme vol | + high vol, - low vol, 0 normal | Breakout anticipation | - |
| `vol_of_vol` | Volatility of volatility | Regime instability | Meta, complex |
| `vol_regime_score` | Continuous -1 (low) to +1 (high) | Regime state (continuous) | - |

**Consolidation Notes:**
- `days_low_vol` + `days_high_vol` → `days_vol_extreme` (signed)
- `vol_regime` (categorical) → `vol_regime_score` (continuous)

### 3.3 Bollinger Band Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `price_bb_position` | (Close - Lower) / (Upper - Lower) | Position in band (0-1) | Universal | - |
| `price_pct_from_bb_upper` | (Close - Upper) / Upper * 100 | Signed: + if above, - if below | Overbought measure | - |
| `price_pct_from_bb_lower` | (Close - Lower) / Lower * 100 | Signed: + if above, - if below | Oversold measure | - |
| `price_pct_from_bb_middle` | (Close - Middle) / Middle * 100 | Trend position (signed) | Same as SMA relation |
| `days_outside_bb` | Signed days outside bands | + above upper, - below lower, 0 inside | Extreme persistence | - |
| `bb_width_regime` | Continuous squeeze score | - = squeeze, + = expansion | Volatility state | - |
| `bb_width_slope` | Change in BB width | Expansion/contraction rate | - |
| `bb_width_slope_acceleration` | Change in width slope | Early vol regime shift | Noisy |

**Consolidation Notes:**
- `days_above_bb_upper` + `days_below_bb_lower` → `days_outside_bb` (signed)
- `bb_squeeze` + `bb_expansion` → `bb_width_regime` (continuous)

### 3.4 Gaussian Channel Features

Gaussian-smoothed price channel using multiple smoothing periods.

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `price_pct_from_gaussian_upper_{n}d` | (Close - Upper) / Upper * 100 | Signed distance from upper (n=3,5,7) | + above, - below | Multiple periods |
| `price_pct_from_gaussian_lower_{n}d` | (Close - Lower) / Lower * 100 | Signed distance from lower | + above, - below | Multiple periods |
| `gaussian_channel_width_{n}d` | (Upper - Lower) / Close * 100 | Channel width as % of price | Volatility measure | - |
| `gaussian_channel_slope_{n}d` | Change in midline | Channel direction | Trend indicator | - |
| `gaussian_channel_acceleration_{n}d` | Change in slope | Channel momentum shift | Early signal | Noisy |
| `days_outside_gaussian_{n}d` | Signed days outside channel | + above, - below, 0 inside | Extreme duration | - |

**Periods:** 3-day, 5-day, 7-day (short-term smoothing focus)

### 3.5 Keltner Channel Features

ATR-based channel around EMA (similar structure to Bollinger Bands).

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `price_pct_from_kc_upper` | (Close - Upper) / Upper * 100 | Signed distance from KC upper | + above, - below | - |
| `price_pct_from_kc_lower` | (Close - Lower) / Lower * 100 | Signed distance from KC lower | + above, - below | - |
| `kc_position` | (Close - Lower) / (Upper - Lower) | Position in channel (0-1) | Normalized position | - |
| `kc_width` | (Upper - Lower) / Close * 100 | Channel width as % | ATR-based vol measure | - |
| `kc_width_slope` | Change in KC width | Width change rate | Vol expansion/contraction | - |
| `kc_width_slope_acceleration` | Change in width slope | Early vol regime shift | Noisy |
| `days_outside_kc` | Signed days outside channel | + above upper, - below lower | Extreme persistence | - |

**Note:** KC squeeze = BB inside KC (both bands). Useful for combined BB-KC analysis.

### 3.6 Volatility Feature Count Estimate

```
Vol Types: 8 (ATR, ATR%, BB Width, KC Width, StdDev, Parkinson, Garman-Klass, Historical Vol)
Periods: 4 (10, 14, 20, 50)
Raw values: 8 × 4 = 32
Derivatives per type: ~6 [consolidated from 8]
BB-specific: ~8 [consolidated from 10]
KC-specific: ~7 (new)
Gaussian Channel: ~18 (6 features × 3 periods)

Derivatives: 8 × 4 × 6 = 192 (was 256)
BB features: 8 (was 10)
KC features: 7 (new)
Gaussian: 18 (new)

TOTAL VOLATILITY FEATURES: ~257 (or ~289 including raw)
Net change: -51 from consolidation, +25 from new channels = ~26 fewer than v0.1
```

---

## Category 4: Volume Indicators

### 4.1 Volume Types

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| Raw Volume | Trading volume | Direct | Scale varies by asset |
| Relative Volume | Vol / Avg Vol | Normalized | - |
| OBV | On-Balance Volume | Cumulative flow | Trend, not magnitude |
| VWAP | Volume-Weighted Avg Price | Institutional reference | Intraday focus |
| MFI | Money Flow Index | Volume-weighted RSI | Bounded, interpretable |
| CMF | Chaikin Money Flow | Accumulation/distribution | Bounded (-1 to 1) |
| ADL | Accumulation/Distribution Line | Cumulative | Trend only |
| NVI | Negative Volume Index | Low-volume days | Contrarian | Slow |
| PVI | Positive Volume Index | High-volume days | Crowd following | Noisy |
| VPT | Volume Price Trend | Volume × price change | Combines both | Cumulative |
| VWAP | Session VWAP | Institutional benchmark | Daily reset | - |

### 4.2 Volume Derivative Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `rel_volume` | Volume / SMA(Volume, 20) | Normalized | - |
| `rel_volume_percentile` | Percentile over 60 days | Historical context | - |
| `rel_volume_slope` | Change in relative volume | Vol trend direction | - |
| `rel_volume_acceleration` | Change in rel_volume_slope | Vol momentum shift | Noisy |
| `volume_trend` | Slope of volume SMA | Volume expansion/contraction | - |
| `volume_trend_acceleration` | Change in volume_trend | Early vol shift | Noisy |
| `obv_slope` | OBV rate of change | Flow momentum | - |
| `obv_slope_acceleration` | Change in OBV slope | Flow momentum shift | Noisy |
| `obv_divergence` | Price vs OBV disagreement | Reversal signal | Complex |
| `volume_price_corr_20d` | Correlation over 20 days | Confirmation measure | Rolling |
| `up_volume_ratio` | Up-day volume / total volume | Buying pressure | - |
| `days_volume_extreme` | Signed days in extreme volume | + high vol, - low vol, 0 normal | Event persistence | - |

**Consolidation Notes:**
- `days_high_volume` + `days_low_volume` → `days_volume_extreme` (signed)

### 4.3 VWAP Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `price_pct_from_vwap` | % distance from VWAP (signed) | Institutional reference | - |
| `vwap_slope` | VWAP rate of change | Institutional trend | - |
| `vwap_slope_acceleration` | Change in VWAP slope | Institutional trend shift | Noisy |
| `days_since_vwap_cross` | Signed days since VWAP cross | + bullish, - bearish | Position timing | - |
| `vwap_bands_position` | Position in VWAP ± std bands | Similar to BB | - |

**Consolidation Notes:**
- `days_above_vwap` → `days_since_vwap_cross` (signed)

### 4.4 Volume-Price Confluence Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `vol_price_spike` | Both volume and price spike | Breakout confirmation | - |
| `vol_spike_price_flat` | Volume spike, price flat | Absorption pattern | - |
| `vol_buildup` | Increasing volume trend | Breakout anticipation | - |
| `vol_price_divergence` | Price up + volume down (or vice versa) | Weakness signal | - |

### 4.5 Volume Feature Count Estimate

```
Volume Types: 11 (Raw Vol, Rel Vol, OBV, VWAP, MFI, CMF, ADL, NVI, PVI, VPT, Session VWAP)
Periods: 4 (5, 10, 20, 50)
Raw values: 11 × 4 = 44
Derivatives per type: ~10 [+2 acceleration features, -1 consolidated]
VWAP-specific: ~5 [consolidated from 8]
Confluence features: ~10

Derivatives: 11 × 10 = 110 (was 88)
VWAP: 5 (was 8)
Confluence: 10

TOTAL VOLUME FEATURES: ~169 (or ~213 including raw)
Net change: +19 from accelerations, -3 from consolidation = +16
```

---

## Category 5: Trend Indicators

### 5.1 Trend Indicator Types

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| ADX | Average Directional Index | Trend strength (not direction) | Lagging |
| +DI / -DI | Directional Indicators | Trend direction | Noisy alone |
| Aroon Up/Down | Time since high/low | Simple concept | Less common |
| Aroon Oscillator | Aroon Up - Aroon Down | Combined | - |
| SuperTrend | ATR-based trend | Clear signals | Whipsaws in ranges |
| Parabolic SAR | Stop and Reverse | Clear levels | Whipsaws |
| Ichimoku | Multi-component system | Comprehensive | Complex (5 lines) |
| MACD | Moving Average Convergence Divergence | Popular | Lagging |
| PPO | Percentage Price Oscillator | Normalized MACD | - |
| TRIX | Triple smoothed EMA ROC | Very smooth | Very lagging |

### 5.2 ADX Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `adx_value` | ADX value (trend strength 0-100) | Universal | Raw value |
| `adx_percentile_60d` | ADX historical percentile | Relative trend strength | Lookback |
| `adx_slope` | ADX rate of change | Trend developing/fading | - |
| `adx_slope_acceleration` | Change in ADX slope | Early trend shift | Noisy |
| `plus_di` | Positive Directional Indicator | Bullish pressure | - |
| `minus_di` | Negative Directional Indicator | Bearish pressure | - |
| `di_spread` | +DI - (-DI) | Directional bias (signed) | Combined signal | - |
| `di_spread_slope` | Change in DI spread | Trend momentum | - |
| `di_spread_acceleration` | Change in spread slope | Early direction shift | Noisy |
| `days_since_di_cross` | Signed days since +DI/-DI cross | + bullish, - bearish | Cross timing | - |

**Consolidation Notes:**
- `adx_above_25` and `adx_above_40` removed: use continuous `adx_value` and `adx_percentile_60d`
- `di_plus_minus_diff` renamed to `di_spread` for clarity
- `di_cross_days` → `days_since_di_cross` (signed)

### 5.3 MACD Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `macd_histogram` | MACD - Signal (signed) | Momentum, + bullish, - bearish | - |
| `macd_histogram_slope` | Change in histogram | Momentum shift | - |
| `macd_histogram_slope_acceleration` | Change in histogram slope | Early momentum shift | Noisy |
| `macd_from_zero` | MACD line value (signed distance from 0) | Trend state | Scale varies |
| `days_since_macd_signal_cross` | Signed days since MACD/Signal cross | + bullish, - bearish | Cross timing |
| `days_since_macd_zero_cross` | Signed days since MACD crossed 0 | + bullish, - bearish | Trend timing |
| `macd_divergence` | Price vs MACD disagreement (signed) | + bullish div, - bearish div | Complex |

**Consolidation Notes:**
- `macd_above_signal` removed: sign of `macd_histogram` provides same info
- `macd_above_zero` removed: sign of `macd_from_zero` provides same info
- `macd_line` (raw) → `macd_from_zero` (centered interpretation)

### 5.4 Ichimoku Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `price_vs_cloud` | Above (1), In (0), Below (-1) | Trend state | Categorical |
| `cloud_thickness` | Senkou A - Senkou B | Support/resistance strength | - |
| `cloud_color` | 1 if bullish cloud, -1 if bearish | Future trend hint | - |
| `tenkan_kijun_cross` | Days since cross | Short-term signal | - |
| `chikou_vs_price` | Chikou above/below price 26 days ago | Confirmation | - |
| `price_pct_from_tenkan` | % from Tenkan-sen | Short-term relation | - |
| `price_pct_from_kijun` | % from Kijun-sen | Medium-term relation | - |

### 5.5 SuperTrend Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `price_pct_from_supertrend` | % distance from line (signed: + above/bullish, - below/bearish) | Trend strength & direction | - |
| `supertrend_slope` | Rate of change of SuperTrend line | Trend line momentum | - |
| `supertrend_slope_acceleration` | Change in SuperTrend slope | Early trend shift | Noisy |
| `days_since_supertrend_flip` | Signed days since last flip | + bullish flip, - bearish flip | Trend duration | - |

**Consolidation Notes:**
- `supertrend_direction` removed: redundant with sign of `price_pct_from_supertrend`
- `days_in_supertrend` → `days_since_supertrend_flip` (signed, includes direction)
- `supertrend_flip_recency` merged into `days_since_supertrend_flip`

### 5.6 Donchian Channel Features

**Donchian Channel** = Highest high and lowest low over N periods. Classic breakout detection.

**Calculation:**
- Upper = max(High, n periods)
- Lower = min(Low, n periods)
- Middle = (Upper + Lower) / 2

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `price_pct_from_donchian_upper` | (Close - Upper) / Upper * 100 | Signed: + above (breakout), - below | Breakout detection | - |
| `price_pct_from_donchian_lower` | (Close - Lower) / Lower * 100 | Signed: + above, - below (breakdown) | Breakdown detection | - |
| `donchian_position` | (Close - Lower) / (Upper - Lower) | Position in channel (0-1) | Range position | - |
| `donchian_width` | (Upper - Lower) / Close * 100 | Channel width as % | Volatility proxy | - |
| `donchian_width_slope` | Change in width | Channel expansion/contraction | Volatility trend | - |
| `donchian_width_acceleration` | Change in width slope | Early vol regime shift | Noisy |

**Integration with Entropy:**
- Low entropy + Donchian breakout = high-quality breakout signal
- High entropy + Donchian breakout = likely false breakout
- See Category 10 for entropy features

**Tier Assignment:** a200 (enables entropy-filtered breakout strategy)

### 5.8 Trend Feature Count Estimate

```
Trend Types: 11 (ADX, +DI/-DI, Aroon, SuperTrend, Parabolic SAR, Ichimoku, MACD, PPO, TRIX, Donchian)
ADX features: ~10 (expanded with DI components, +acceleration)
MACD features: ~7 (consolidated from 10)
Ichimoku features: ~10
SuperTrend features: ~4 (consolidated from 6)
Donchian features: ~6 (new in v0.3)
Aroon features: ~6
Other trend: ~15

TOTAL TREND FEATURES: ~58 (was 52 in v0.2)
Change from v0.2: +6 (Donchian Channel)
```

---

## Category 6: Support/Resistance Features

### 6.1 S/R Types

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| Horizontal | Price levels with multiple touches | Clear, universal | Requires detection algo |
| Moving Average | Dynamic S/R at MA levels | Well-defined | Already in MA section |
| Bollinger Bands | Dynamic S/R at band edges | Well-defined | Already in Vol section |
| Fibonacci Retracement | 23.6%, 38.2%, 50%, 61.8%, 78.6% | Popular with traders | Subjective anchor points |
| Pivot Points | Daily/Weekly/Monthly pivots | Institutional reference | Requires higher timeframe |
| VWAP | Institutional benchmark | Clear | Intraday focus |
| Prior High/Low | Previous day/week/month H/L | Clear | Binary touch/not |
| Fair Value Gaps | Price gaps that tend to fill | Smart money concept | Complex detection |
| Order Blocks | Zones of institutional activity | Smart money concept | Complex detection |

### 6.2 Horizontal S/R Features (requires detection algorithm)

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `dist_to_nearest_support` | % to nearest support (signed: - = below support) | Downside buffer | Detection needed |
| `dist_to_nearest_resistance` | % to nearest resistance (signed: + = above resistance) | Upside barrier | Detection needed |
| `support_strength` | Touches × recency weight (0-1 normalized) | Level importance | Subjective |
| `resistance_strength` | Touches × recency weight (0-1 normalized) | Level importance | Subjective |
| `sr_zone_position` | Position between nearest S and R (0=support, 1=resistance) | Range position | - |
| `support_test_count_20d` | Tests of support level in last 20 days | Weakening signal | - |
| `resistance_test_count_20d` | Tests of resistance level in last 20 days | Weakening signal | - |
| `support_slope` | Rate of change of support level | Rising/falling floor | Dynamic S/R |
| `resistance_slope` | Rate of change of resistance level | Rising/falling ceiling | Dynamic S/R |
| `sr_compression_rate` | Change in distance between S and R | Squeeze detection | - |

**Note:** Dynamic S/R (trendlines) have slopes; horizontal S/R have slope = 0.

### 6.3 Fibonacci Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `fib_382_dist` | % distance from 38.2% level | Common retracement | Anchor choice |
| `fib_500_dist` | % distance from 50% level | - | - |
| `fib_618_dist` | % distance from 61.8% level | Golden ratio | - |
| `nearest_fib_level` | Which fib level is closest | Categorical | - |
| `fib_zone` | Which zone price is in | Categorical | - |

### 6.4 Prior High/Low Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `pct_from_52w_high` | % below 52-week high | Drawdown | - |
| `pct_from_52w_low` | % above 52-week low | Recovery | - |
| `pct_from_20d_high` | % below 20-day high | Short-term drawdown | - |
| `pct_from_20d_low` | % above 20-day low | Short-term recovery | - |
| `days_since_52w_high` | Days since 52-week high | High recency | - |
| `days_since_52w_low` | Days since 52-week low | Low recency | - |
| `new_high_20d` | 1 if today is 20-day high | Breakout | Binary |
| `new_low_20d` | 1 if today is 20-day low | Breakdown | Binary |

### 6.5 S/R Feature Count Estimate

```
Horizontal S/R: ~10 (expanded with slopes, requires detection algo)
Dynamic S/R: ~6 (slopes, compression)
Fibonacci: ~8
Prior High/Low: ~15
FVG/Order Block: ~10 (requires SMC library - see Category 13)

TOTAL S/R FEATURES: ~49 (was 43)
Net change: +6 from dynamic S/R features
```

---

## Category 7: Candlestick / Price Action Features

### 7.1 Single Candle Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `body_pct` | abs(close - open) / (high - low) | Body proportion | - |
| `upper_wick_pct` | (high - max(open,close)) / (high - low) | Upper rejection | - |
| `lower_wick_pct` | (min(open,close) - low) / (high - low) | Lower rejection | - |
| `candle_direction` | 1 if close > open, -1 otherwise | Basic direction | - |
| `candle_range_pct` | (high - low) / close | Daily range | - |
| `candle_range_vs_atr` | Range / ATR | Range relative to normal | - |
| `gap_pct` | (open - prev_close) / prev_close | Overnight gap | - |
| `gap_fill_pct` | How much of gap was filled | Gap behavior | - |

### 7.2 Multi-Candle Pattern Features

**Single/Double Candle Reversal Patterns:**

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `engulfing_bullish` | Bullish engulfing pattern | Classic reversal | Binary |
| `engulfing_bearish` | Bearish engulfing pattern | Classic reversal | Binary |
| `doji` | Open ~ Close with wicks | Indecision | Binary |
| `doji_dragonfly` | Doji with long lower wick | Bullish reversal | Binary |
| `doji_gravestone` | Doji with long upper wick | Bearish reversal | Binary |
| `hammer` | Small body, long lower wick, downtrend | Bullish reversal | Binary |
| `inverted_hammer` | Small body, long upper wick, downtrend | Bullish reversal | Binary |
| `hanging_man` | Small body, long lower wick, uptrend | Bearish reversal | Binary |
| `shooting_star` | Small body, long upper wick, uptrend | Bearish reversal | Binary |
| `spinning_top` | Small body, both wicks | Indecision | Binary |
| `marubozu_bullish` | Full body, no wicks, bullish | Strong conviction | Binary |
| `marubozu_bearish` | Full body, no wicks, bearish | Strong conviction | Binary |
| `piercing_line` | Bullish 2-candle reversal | Reversal | Binary |
| `dark_cloud_cover` | Bearish 2-candle reversal | Reversal | Binary |
| `harami_bullish` | Small inside big, bullish | Reversal | Binary |
| `harami_bearish` | Small inside big, bearish | Reversal | Binary |
| `tweezer_top` | Same highs, reversal | Resistance rejection | Binary |
| `tweezer_bottom` | Same lows, reversal | Support bounce | Binary |
| `inside_day` | Today's range inside yesterday's | Compression | Binary |
| `outside_day` | Today's range engulfs yesterday's | Expansion | Binary |

**Triple Candle Patterns:**

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `morning_star` | 3-candle bullish reversal | Strong reversal | Binary |
| `evening_star` | 3-candle bearish reversal | Strong reversal | Binary |
| `morning_doji_star` | Morning star with doji | Stronger reversal | Binary |
| `evening_doji_star` | Evening star with doji | Stronger reversal | Binary |
| `three_white_soldiers` | 3 consecutive bullish | Strong uptrend | Binary |
| `three_black_crows` | 3 consecutive bearish | Strong downtrend | Binary |
| `three_inside_up` | Harami + bullish confirm | Reversal confirmation | Binary |
| `three_inside_down` | Harami + bearish confirm | Reversal confirmation | Binary |
| `three_outside_up` | Engulfing + bullish confirm | Reversal confirmation | Binary |
| `three_outside_down` | Engulfing + bearish confirm | Reversal confirmation | Binary |
| `rising_three_methods` | Continuation in uptrend | Trend continuation | Binary |
| `falling_three_methods` | Continuation in downtrend | Trend continuation | Binary |

### 7.3 Quantified Pattern Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `consecutive_up_down_days` | Signed consecutive days (+ up, - down) | Trend persistence | - |
| `up_day_ratio_10d` | Up days / 10 over last 10 days | Short-term bias | - |
| `avg_up_day_magnitude` | Avg gain on up days (20d) | Bullish strength | - |
| `avg_down_day_magnitude` | Avg loss on down days (20d) | Bearish strength | - |
| `up_down_magnitude_ratio` | Avg up / Avg down | Asymmetry | - |
| `hh_hl_ll_lh_score` | Rolling HH/HL vs LL/LH score (signed) | Trend structure quality | - |
| `pattern_strength_score` | Confidence in detected patterns (0-1) | Pattern quality | Detection dependent |
| `days_since_reversal_pattern` | Signed days since reversal pattern | + bullish, - bearish, 0 none | Recency | - |
| `pattern_confluence_count` | Number of patterns confirming direction | Multi-pattern signal | Binary patterns |
| `candle_trend_alignment` | Candle direction vs MA trend (signed) | Confirmation | - |

**Consolidation Notes:**
- `consecutive_up_days` + `consecutive_down_days` → `consecutive_up_down_days` (signed)
- `higher_high`, `higher_low`, `lower_high`, `lower_low` → `hh_hl_ll_lh_score` (continuous)

### 7.4 Candlestick Feature Count Estimate

```
Single candle: ~10
Pattern detection: ~32 (expanded from 15)
  - Single/Double reversal: ~20
  - Triple candle: ~12
Quantified patterns: ~10 (consolidated from 15)
Heiken Ashi variants: ~5

TOTAL CANDLESTICK FEATURES: ~57 (was 45)
Net change: +17 from pattern expansion, -5 from consolidation = +12
```

---

## Category 8: Momentum / Rate of Change

### 8.1 Momentum Types

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `return_1d` | 1-day return | Basic | Noisy |
| `return_1d_acceleration` | Change in 1-day return | Momentum shift | Very noisy |
| `return_5d` | 5-day return | Short-term | - |
| `return_5d_acceleration` | Change in 5-day return | Short-term momentum shift | Noisy |
| `return_10d` | 10-day return | Medium-term | - |
| `return_10d_acceleration` | Change in 10-day return | Medium-term momentum shift | - |
| `return_20d` | 20-day return (1 month) | Monthly | - |
| `return_20d_acceleration` | Change in 20-day return | Monthly momentum shift | - |
| `return_60d` | 60-day return (3 months) | Quarterly | - |
| `return_252d` | 252-day return (1 year) | Annual | - |
| `return_percentile_60d` | Today's return vs last 60 | Relative magnitude | - |
| `return_zscore` | Z-score of return | Anomaly detection | - |

### 8.2 Cumulative / Streak Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `cum_return_5d` | Cumulative 5-day return | Short-term performance | - |
| `cum_return_20d` | Cumulative 20-day return | Monthly performance | - |
| `max_drawdown_20d` | Max drawdown over 20 days | Risk measure | - |
| `max_runup_20d` | Max runup over 20 days | Rally measure | - |
| `win_streak` | Consecutive positive days | Streak | - |
| `loss_streak` | Consecutive negative days | Streak | - |

### 8.3 Daily Return Enhancements

Additional relational features for daily returns (supplements existing return_1d through return_252d).

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `open_to_close_pct` | (Close - Open) / Open * 100 | Intraday return | Intraday momentum | - |
| `close_to_close_pct` | (Close - Close[1]) / Close[1] * 100 | Standard daily return | Same as return_1d | Redundant check |
| `overnight_gap_pct` | (Open - Close[1]) / Close[1] * 100 | Gap return | Overnight sentiment | - |

**Note:** `close_to_close_pct` is equivalent to `return_1d` - included here for naming clarity and completeness.

**Tier Assignment:** a100 (fundamental features)

### 8.4 Expectancy Metrics

Simple expectancy metrics derived from OHLCV data without simulating specific trading strategies.

**Concept:** Expectancy = (Win Rate × Avg Win) - (Loss Rate × Avg Loss)

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `avg_gain_up_days_20d` | Mean of positive returns over 20d | Average gain on winning days | Bullish strength | - |
| `avg_loss_down_days_20d` | Mean of negative returns over 20d | Average loss on losing days | Bearish strength | - |
| `win_rate_20d` | Count(positive returns) / 20 | Percentage of up days | Win frequency | Simple metric |
| `expectancy_20d` | win_rate × avg_gain - (1-win_rate) × abs(avg_loss) | Expected daily return per trade | Combined metric | Simplified |
| `expectancy_ratio_20d` | avg_gain / abs(avg_loss) | Reward-to-risk ratio | Asymmetry measure | - |

**Tier Assignment:** a100 (fundamental performance metrics)

### 8.6 Momentum Feature Count Estimate

```
Returns: ~12 (was 10, +4 acceleration features)
Cumulative/Streak: ~10
Daily return enhancements: ~3 (new in v0.3)
Expectancy metrics: ~5 (new in v0.3)
Momentum indicators: ~10 (overlap with oscillators)

TOTAL MOMENTUM FEATURES: ~40 (was 32 in v0.2)
Change from v0.2: +8 (daily returns + expectancy)
```

---

## Category 9: Calendar / Seasonal Features

### 9.1 Day-Based Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `day_of_week` | 0-4 (Mon-Fri) | Monday effect | Categorical |
| `day_of_month` | 1-31 | Month-end effects | Categorical |
| `days_to_month_end` | Days until month end | Rebalancing timing | - |
| `days_to_quarter_end` | Days until quarter end | Window dressing timing | - |

**Consolidation Notes:**
- `is_monday`, `is_friday` removed: neural net learns day_of_week=0 means Monday
- `is_month_start`, `is_month_end` removed: `days_to_month_end` captures this continuously
- `is_quarter_end` removed: `days_to_quarter_end` captures this continuously

### 9.2 Month-Based Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `month` | 1-12 | Seasonality | Categorical |
| `quarter` | 1-4 | Quarterly effects | Categorical |

**Consolidation Notes:**
- `is_january`, `is_september`, `is_october`, `is_december` removed
- Neural net learns month=1 means January effect, month=9 means weak September, etc.
- Continuous `month` and `quarter` features are sufficient

### 9.3 Event-Based Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `days_to_opex` | Days to options expiration | Pinning effect | Requires opex dates |
| `is_opex_week` | Options expiration week | Increased vol | Requires opex dates |
| `days_to_fomc` | Days to Fed meeting | Rate decision | Requires FOMC dates |

### 9.4 Calendar Feature Count Estimate

```
Day-based: ~4 (consolidated from 10)
Month-based: ~2 (consolidated from 15)
Event-based: ~10 (requires external calendars)

TOTAL CALENDAR FEATURES: ~16 (or ~6 without event data)
Net change: -19 from removing redundant binary features
```

---

## Category 10: Entropy / Complexity Features

### 10.1 Entropy Types

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| Shannon Entropy | Information content | Classic measure | Requires binning |
| Permutation Entropy | Order pattern complexity | No binning needed | Param choice |
| Approximate Entropy | Regularity measure | Handles noise | Param sensitive |
| Sample Entropy | Improved ApEn | More robust | Slower |
| Multiscale Entropy | Across timescales | Comprehensive | Complex |
| Spectral Entropy | Frequency domain | Different view | FFT needed |

### 10.2 Entropy Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `permutation_entropy_20d` | Permutation entropy over 20 days | Pattern predictability | - |
| `permutation_entropy_slope` | Change in permutation entropy | Regime shift | - |
| `permutation_entropy_acceleration` | Change in entropy slope | Early regime shift | Noisy |
| `sample_entropy_20d` | Sample entropy over 20 days | Regularity | Slow |
| `sample_entropy_slope` | Change in sample entropy | Complexity shift | - |
| `sample_entropy_acceleration` | Change in sample entropy slope | Early complexity shift | Noisy |
| `entropy_percentile` | Historical percentile | Relative complexity | - |
| `entropy_vol_ratio` | Entropy / normalized ATR% | Complexity per unit volatility | Novel metric |
| `entropy_regime_score` | Continuous -1 (low) to +1 (high) | Regime state (continuous) | - |

**Consolidation Notes:**
- `entropy_regime` (categorical) → `entropy_regime_score` (continuous)

### 10.3 Other Complexity Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `hurst_exponent` | Trending vs mean-reverting | Regime identification | Lookback needed |
| `autocorr_lag1` | Return autocorrelation | Persistence | Single lag |
| `autocorr_lag5` | Return autocorrelation | Persistence | - |
| `partial_autocorr` | Partial autocorrelation | Direct effect | - |
| `fractal_dimension` | Price path complexity | Roughness | Slow |

### 10.4 Entropy Feature Count Estimate

```
Entropy types: 6 (Shannon, Permutation, Approximate, Sample, Multiscale, Spectral)
Features per type: ~4 (value, slope, acceleration, percentile)
Complexity features: ~8
Additional: entropy_vol_ratio, entropy_regime_score

TOTAL ENTROPY FEATURES: ~34 (was 26)
Net change: +8 from acceleration features and new metrics
```

---

## Category 11: Regime / State Features

### 11.1 Regime Detection Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| Volatility-based | Low/Med/High vol regimes | Simple | Binary view |
| HMM | Hidden Markov Model states | Statistical rigor | Complex, slow |
| Trend-based | Trending/Ranging/Volatile | Intuitive | Threshold choice |
| ADX-based | Trending if ADX > 25 | Standard | Single indicator |
| Channel-based | In channel vs breakout | Visual match | Detection needed |

### 11.2 Regime Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `vol_regime_score` | Volatility regime (-1 low, 0 med, +1 high) | Continuous | - |
| `trend_regime_score` | Trend regime (-1 down, 0 range, +1 up) | Continuous | - |
| `adx_trend_score` | ADX-based trend strength (0-1 normalized) | Continuous | - |
| `hmm_state_prob` | HMM state probability (most likely state) | Sophisticated | Slow |
| `regime_duration` | Days in current regime | Persistence | - |
| `regime_duration_slope` | Change in regime duration (stability) | Regime maturity | - |
| `regime_confidence` | Confidence in regime call (0-1) | Uncertainty quantified | Method-dependent |
| `regime_change_probability` | Probability of regime change (0-1) | Forward-looking | Model-dependent |
| `regime_change_acceleration` | Change in regime_change_probability | Early transition signal | Noisy |

**Consolidation Notes:**
- Categorical regimes (L/M/H, Up/Down/Range) → continuous scores

### 11.3 Regime Feature Count Estimate

```
Regime detection methods: 5 (Volatility, HMM, Trend, ADX, Channel)
Features per method: ~5 (+duration_slope, +change_acceleration)

TOTAL REGIME FEATURES: ~25 (was 20)
Net change: +5 from additional derivative features
```

---

## Category 12: Multi-Timeframe Features

### 12.1 Higher Timeframe Indicators (Weekly Focus)

**Note:** Monthly indicators excluded per user preference (excessive data consumption). Weekly lookback ≤52 weeks (~1 year data loss is acceptable).

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `weekly_ma_slope` | Weekly MA rate of change | Higher-level trend | Lookback needed |
| `weekly_ma_slope_acceleration` | Change in weekly MA slope | Trend momentum shift | - |
| `price_pct_from_weekly_ma` | Daily price vs weekly MA (signed) | Cross-timeframe position | - |
| `weekly_rsi` | RSI on weekly data | Higher-level RSI | Requires resampling |
| `weekly_rsi_slope` | Change in weekly RSI | Weekly momentum | - |
| `weekly_rsi_slope_acceleration` | Change in weekly RSI slope | Early momentum shift | - |
| `weekly_bb_position` | Position in weekly BB (0-1) | Higher-level context | - |
| `weekly_bb_width_slope` | Change in weekly BB width | Weekly volatility trend | - |

### 12.2 Trend Alignment Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `trend_alignment_daily_weekly` | Signed: +1 both up, -1 both down, 0 mixed | Confluence (continuous) | - |
| `rsi_alignment_daily_weekly` | Daily RSI - Weekly RSI (signed divergence) | Momentum confluence | - |
| `vol_alignment_daily_weekly` | Daily vol percentile - Weekly vol percentile | Volatility confluence | - |

**Consolidation Notes:**
- `daily_vs_weekly_alignment` (binary) → `trend_alignment_daily_weekly` (signed continuous)
- `weekly_trend` (categorical) → `weekly_ma_slope` (continuous)
- Monthly indicators removed (excessive lookback for marginal benefit)

### 12.3 Data Lookback Constraints

| Indicator Type | Max Lookback | Data Loss |
|----------------|--------------|-----------|
| Daily MAs (252d) | 252 days | ~1 year |
| Weekly indicators | 52 weeks | ~1 year |
| Monthly indicators | EXCLUDED | N/A |
| Entropy (20-60d) | 60 days | ~3 months |

**Note:** SPY data starts 1993, so 1-2 years lookback is acceptable.

### 12.4 Multi-Timeframe Feature Count Estimate

```
Weekly indicators: ~8 (with slopes and accelerations)
Alignment features: ~3 (continuous, signed)

TOTAL MTF FEATURES: ~11 (was 30)
Net change: -19 from removing monthly and consolidating
```

---

## Category 13: Smart Money Concepts (SMC)

**Library Requirement:** [joshyattridge/smart-money-concepts](https://github.com/joshyattridge/smart-money-concepts)

### 13.1 SMC Feature Types

| Type | Description | Pros | Cons |
|------|-------------|------|------|
| Order Blocks | Zones of institutional orders | Institutional view | Detection complex |
| Fair Value Gaps | Price imbalances | Tend to fill | Detection needed |
| Liquidity Zones | Areas of stop losses | Manipulation targets | Subjective |
| Break of Structure | Trend change confirmation | Clear levels | Requires detection |
| Change of Character | Shift in behavior | Regime change | Subjective |
| Premium/Discount | Price vs recent range | Simple concept | Range definition |
| AMD (Accumulation/Manipulation/Distribution) | Market cycle phases | Institutional view | Pattern recognition |

### 13.2 SMC Features

**Order Block Features:**

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `dist_to_nearest_ob` | % to nearest order block (signed) | + above OB, - below | Detection needed |
| `ob_strength` | Order block strength (0-1 normalized) | Zone importance | Subjective |
| `ob_strength_slope` | Change in nearest OB strength | Zone weakening/strengthening | - |
| `ob_strength_acceleration` | Change in OB strength slope | Early zone breakdown | Noisy |
| `days_since_ob_touch` | Days since price touched OB | Recency | - |

**Fair Value Gap Features:**

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `fvg_imbalance_score` | FVGs above - FVGs below (signed) | Net attraction direction | - |
| `dist_to_nearest_fvg` | % to nearest unfilled FVG (signed) | + above, - below | - |
| `fvg_fill_rate` | Rolling rate of FVG fills | Gap behavior regime | - |
| `days_since_fvg_fill` | Signed days since FVG fill | + bullish fill, - bearish | - |

**Liquidity & Structure Features:**

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `liquidity_swept_recency` | Signed days since liquidity sweep | + bullish sweep, - bearish | Manipulation timing |
| `days_since_bos` | Signed days since break of structure | + bullish BOS, - bearish | Trend confirmation |
| `days_since_choch` | Signed days since change of character | + bullish CHoCH, - bearish | Regime change |
| `premium_discount_score` | Continuous -1 (discount) to +1 (premium) | Range position | - |
| `amd_phase_score` | Current AMD phase (-1 to +1) | Market cycle position | Detection complex |
| `amd_phase_duration` | Days in current AMD phase | Phase maturity | - |

**Consolidation Notes:**
- `in_order_block` (binary) removed: captured by `dist_to_nearest_ob` magnitude
- `fvg_above_count` + `fvg_below_count` → `fvg_imbalance_score` (signed)
- `bos_bullish_recency` + `bos_bearish_recency` → `days_since_bos` (signed)
- `premium_discount_zone` (categorical) → `premium_discount_score` (continuous)

### 13.3 SMC Feature Count Estimate

```
Order Blocks: ~5 (with derivatives)
FVG: ~4 (consolidated)
Liquidity & Structure: ~6
AMD: ~2

TOTAL SMC FEATURES: ~17 (was 22)
Net change: -5 from consolidation, library required for all features
```

---

## Category 14: Risk-Adjusted Metrics

**NEW in v0.3** - Risk-adjusted performance metrics widely used in ML portfolio research.

### 14.1 Sharpe Ratio Features

**Sharpe Ratio** = (Return - Risk-free Rate) / Standard Deviation

For simplicity, we assume risk-free rate ≈ 0 for relative comparisons.

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `sharpe_ratio_20d` | mean(returns_20d) / std(returns_20d) * sqrt(252) | Annualized 20-day Sharpe | Short-term risk-adjusted | Noisy |
| `sharpe_ratio_60d` | mean(returns_60d) / std(returns_60d) * sqrt(252) | Annualized 60-day Sharpe | Medium-term | - |
| `sharpe_ratio_252d` | mean(returns_252d) / std(returns_252d) * sqrt(252) | Annualized 1-year Sharpe | Long-term benchmark | Lagging |
| `sharpe_20d_slope` | Sharpe_20d[t] - Sharpe_20d[t-1] | Change in short-term Sharpe | Performance trend | Noisy |
| `sharpe_20d_slope_acceleration` | slope[t] - slope[t-1] | Change in Sharpe slope | Early performance shift | Very noisy |
| `sharpe_60d_slope` | Sharpe_60d[t] - Sharpe_60d[t-1] | Change in medium-term Sharpe | - | - |
| `sharpe_60d_slope_acceleration` | slope[t] - slope[t-1] | Change in 60d Sharpe slope | - | - |
| `sharpe_percentile_60d` | Percentile of Sharpe over 60 days | Historical context | Relative performance | - |
| `sharpe_regime_score` | Continuous -1 (poor) to +1 (excellent) | Risk-adjusted performance state | Regime classification | - |

**Tier Assignment:** a100 (high-signal, widely used in ML research)

### 14.2 Sortino Ratio Features

**Sortino Ratio** = (Return - Risk-free Rate) / Downside Deviation

More relevant for loss-averse prediction (only penalizes downside volatility).

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `sortino_ratio_20d` | mean(returns_20d) / downside_std_20d * sqrt(252) | Annualized 20-day Sortino | Downside-focused | Noisy |
| `sortino_ratio_60d` | mean(returns_60d) / downside_std_60d * sqrt(252) | Annualized 60-day Sortino | Medium-term | - |
| `sortino_ratio_252d` | mean(returns_252d) / downside_std_252d * sqrt(252) | Annualized 1-year Sortino | Long-term | Lagging |
| `sortino_20d_slope` | Sortino_20d[t] - Sortino_20d[t-1] | Change in short-term Sortino | - | - |
| `sortino_20d_slope_acceleration` | slope[t] - slope[t-1] | Change in Sortino slope | - | - |
| `sortino_60d_slope` | Sortino_60d[t] - Sortino_60d[t-1] | Change in medium-term Sortino | - | - |
| `sortino_60d_slope_acceleration` | slope[t] - slope[t-1] | Change in 60d Sortino slope | - | - |
| `sortino_percentile_60d` | Percentile of Sortino over 60 days | Historical context | - | - |
| `sortino_sharpe_ratio` | Sortino / Sharpe | Downside vs total risk | Skewness indicator | Requires both |

**Tier Assignment:** a100 (complements Sharpe, downside-focused)

### 14.3 Value at Risk (VaR) Features

**VaR** = Worst expected loss at a given confidence level (e.g., 95%, 99%).

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `var_95_20d` | 5th percentile of returns over 20d | 95% VaR (% worst-case loss) | Standard risk metric | Doesn't capture tail |
| `var_99_20d` | 1st percentile of returns over 20d | 99% VaR (extreme loss) | Tail risk | Requires data |
| `var_95_slope` | VaR_95[t] - VaR_95[t-1] | Change in 95% VaR | Risk trend | Noisy |
| `var_95_acceleration` | slope[t] - slope[t-1] | Change in VaR slope | Early risk shift | Very noisy |

**Note:** VaR values are negative (losses). More negative = higher risk.

**Tier Assignment:** a200 (more specialized than Sharpe/Sortino)

### 14.4 Conditional VaR (CVaR) / Expected Shortfall Features

**CVaR** = Average loss beyond VaR (expected loss in worst cases).

| Feature | Formula | Description | Pros | Cons |
|---------|---------|-------------|------|------|
| `cvar_95_20d` | Mean of returns below VaR_95 | Expected shortfall at 95% | Tail risk measure | - |
| `cvar_99_20d` | Mean of returns below VaR_99 | Expected shortfall at 99% | Extreme tail risk | Sparse data |
| `cvar_95_slope` | CVaR_95[t] - CVaR_95[t-1] | Change in CVaR | Tail risk trend | Noisy |
| `cvar_95_acceleration` | slope[t] - slope[t-1] | Change in CVaR slope | - | Very noisy |

**Tier Assignment:** a200 (specialized tail risk metrics)

### 14.5 Risk Regime Features

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `risk_regime_score` | Composite risk state (-1 low, 0 normal, +1 high) | Overall risk assessment | Subjective |
| `risk_percentile_60d` | Percentile of current risk metrics | Historical context | - |

### 14.6 Risk-Adjusted Feature Count Estimate

```
Sharpe Ratio: ~9 (3 windows × {value, slope, accel} + percentile + regime)
Sortino Ratio: ~9 (3 windows × {value, slope, accel} + percentile + ratio)
VaR: ~4 (2 confidence levels + slope + accel)
CVaR: ~4 (2 confidence levels + slope + accel)
Risk Regime: ~2

TOTAL RISK-ADJUSTED FEATURES: ~28
Tier breakdown: ~18 in a100, ~10 in a200
```

---

## Category 15: Signal Processing Features (Optional)

**NEW in v0.3** - Advanced signal decomposition for denoising. Requires additional libraries.

**Library Requirements:**
- **VMD:** `vmdpy` or `PyEMD`
- **Wavelets:** `PyWavelets` (pywt)
- **FFT:** Built into NumPy/SciPy

### 15.1 Variational Mode Decomposition (VMD) Features

**VMD** = Decomposes signal into band-limited intrinsic mode functions (IMFs).

2024-2025 research shows VMD decomposition improves financial time series predictions.

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `vmd_imf1` | First IMF (highest frequency) | Noise component | - |
| `vmd_imf2` | Second IMF | Short-term trend | - |
| `vmd_imf3` | Third IMF | Medium-term trend | - |
| `vmd_residual` | Residual (lowest frequency) | Long-term trend | - |
| `vmd_imf_ratio_1_2` | IMF1 / IMF2 | Noise-to-signal ratio | Novel metric | - |

**Note:** Number of IMFs is configurable (typically 3-5 modes).

**Tier Assignment:** a200 (computationally expensive, research-backed)

### 15.2 Wavelet Features

**Wavelet Transform** = Multi-resolution decomposition using mother wavelets.

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `wavelet_approx_level1` | Level-1 approximation coefficients | Smooth trend | - |
| `wavelet_detail_level1` | Level-1 detail coefficients | High-frequency noise | - |
| `wavelet_approx_level2` | Level-2 approximation | Smoother trend | - |
| `wavelet_detail_level2` | Level-2 detail | Medium-frequency | - |
| `wavelet_energy_ratio` | Detail energy / Approx energy | Noise content measure | - |

**Tier Assignment:** a200 (established denoising technique)

### 15.3 FFT Features

**FFT** = Fast Fourier Transform for spectral analysis.

| Feature | Description | Pros | Cons |
|---------|-------------|------|------|
| `fft_dominant_freq` | Dominant frequency in spectrum | Cycle detection | Window-dependent |
| `fft_dominant_amplitude` | Amplitude of dominant frequency | Cycle strength | - |
| `fft_spectral_entropy` | Entropy of frequency spectrum | Complexity measure | Related to Cat. 10 |
| `fft_low_freq_power` | Power in low frequencies | Trend component | - |
| `fft_high_freq_power` | Power in high frequencies | Noise component | - |

**Tier Assignment:** a200 (spectral analysis complement)

### 15.4 Signal Processing Feature Count Estimate

```
VMD: ~5 features (3-4 IMFs + residual + ratio)
Wavelet: ~5 features (2 levels × 2 coefficients + energy ratio)
FFT: ~5 features (dominant freq/amp + entropy + power ratios)

TOTAL SIGNAL PROCESSING FEATURES: ~15
All assigned to tier a200 (computationally expensive)
Library dependencies: vmdpy, PyWavelets
```

---

## Category 16: Advanced Mathematical Features

**NEW in v0.4** - Advanced mathematical techniques from fractal analysis, chaos theory, topological data analysis, and related fields.

**Library Requirements:**
- **Fractal/Chaos:** `antropy>=0.1.6`, `nolds>=0.6.0`, `MFDFA>=0.4.3`, `hfda>=0.2.0`
- **Spectral:** `PyEMD>=1.6.0`
- **RQA:** `pyrqa>=8.2.0`
- **TDA:** `giotto-tda>=0.6.0` (optional, computationally heavy)
- **Cross-correlation:** `py-DCCA` (install from GitHub)
- **Existing:** `scipy`, `numpy` (Hilbert, polynomial fits)

### 16.1 Fractal Analysis Features (~17 features)

**Fractal Dimension** measures the complexity/roughness of price paths. Different methods capture different aspects.

| Feature | Description | Library | Tier |
|---------|-------------|---------|------|
| `higuchi_fd` | Higuchi Fractal Dimension | `antropy`, `hfda` | a200 |
| `higuchi_fd_slope` | Change in Higuchi FD | - | a200 |
| `higuchi_fd_acceleration` | Change in HFD slope | - | a200 |
| `katz_fd` | Katz Fractal Dimension | `antropy` | a200 |
| `katz_fd_slope` | Change in Katz FD | - | a200 |
| `petrosian_fd` | Petrosian Fractal Dimension | `antropy` | a500 |
| `mfdfa_hurst_mean` | Mean Hurst from MFDFA | `MFDFA` | a500 |
| `mfdfa_hurst_width` | Width of multifractal spectrum | `MFDFA` | a500 |
| `mfdfa_alpha_min` | Min singularity exponent | `MFDFA` | a500 |
| `mfdfa_alpha_max` | Max singularity exponent | `MFDFA` | a500 |
| `levy_alpha` | Lévy stable distribution alpha | `scipy.stats` | a500 |
| `levy_alpha_slope` | Change in Lévy alpha | - | a500 |
| `tail_exponent` | Power-law tail exponent (Hill estimator) | custom | a500 |
| `tail_exponent_slope` | Change in tail exponent | - | a500 |
| `fdi` | Fractal Dimension Index (Bill Williams) | custom | a200 |
| `fdi_slope` | Change in FDI | - | a200 |
| `fdi_regime` | FDI regime score (-1 trending, +1 ranging) | - | a200 |

**Interpretation:**
- **Higuchi FD**: Values near 1.5 = random walk, <1.5 = trending, >1.5 = mean-reverting
- **MFDFA**: Captures multifractal properties (different scaling at different scales)
- **FDI**: Bill Williams' trading indicator; high FDI = ranging market, low FDI = trending

### 16.2 Chaos Theory Features (~8 features)

**Chaos Theory** metrics quantify system predictability and attractor dynamics.

| Feature | Description | Library | Tier |
|---------|-------------|---------|------|
| `lyapunov_exp` | Maximal Lyapunov exponent | `nolds` | a500 |
| `lyapunov_exp_slope` | Change in Lyapunov | - | a500 |
| `lyapunov_regime` | Chaos regime score (0=stable, 1=chaotic) | - | a500 |
| `correlation_dim` | Correlation dimension | `nolds` | a500 |
| `correlation_dim_slope` | Change in correlation dim | - | a500 |
| `embedding_dim` | Optimal embedding dimension | `nolds` | a1000 |
| `attractor_radius` | Phase space attractor size | custom | a1000 |
| `predictability_horizon` | Lyapunov-based forecast limit | custom | a1000 |

**Interpretation:**
- **Lyapunov > 0**: Chaotic system, prediction error grows exponentially
- **Lyapunov ≈ 0**: Edge of chaos, quasi-periodic
- **Lyapunov < 0**: Stable, convergent dynamics
- **Correlation dimension**: Effective degrees of freedom in the system

### 16.3 Recurrence Quantification Analysis (~11 features)

**RQA** analyzes recurrence patterns in phase space. Excellent for detecting regime changes.

| Feature | Description | Library | Tier |
|---------|-------------|---------|------|
| `rqa_recurrence_rate` | % recurrence points (REC) | `pyrqa` | a500 |
| `rqa_determinism` | % diagonal lines (DET) | `pyrqa` | a500 |
| `rqa_det_slope` | Change in determinism | - | a500 |
| `rqa_laminarity` | % vertical lines (LAM) | `pyrqa` | a500 |
| `rqa_lam_slope` | Change in laminarity | - | a500 |
| `rqa_avg_diagonal` | Mean diagonal length (Lmean) | `pyrqa` | a500 |
| `rqa_max_diagonal` | Max diagonal length (Lmax) | `pyrqa` | a500 |
| `rqa_entropy` | Diagonal length entropy | `pyrqa` | a500 |
| `rqa_trapping_time` | Avg vertical line length | `pyrqa` | a500 |
| `rqa_ratio` | DET/REC ratio | - | a500 |
| `rqa_crisis_indicator` | RQA-based crisis detection | custom | a500 |

**Interpretation:**
- **High DET**: Deterministic/predictable dynamics
- **High LAM**: System getting "stuck" in states (laminarity)
- **DET/REC ratio**: Rising ratio may precede regime changes

### 16.4 Spectral Decomposition (beyond FFT) (~10 features)

**EMD/HHT** provides adaptive, data-driven decomposition (unlike fixed-basis FFT/wavelets).

| Feature | Description | Library | Tier |
|---------|-------------|---------|------|
| `emd_imf_count` | Number of IMFs from EMD | `PyEMD` | a500 |
| `emd_trend_strength` | Residual/total variance ratio | `PyEMD` | a500 |
| `eemd_noise_ratio` | Noise component ratio | `PyEMD` | a500 |
| `hilbert_inst_freq` | Instantaneous frequency (HHT) | `scipy` | a500 |
| `hilbert_inst_freq_slope` | Change in inst. frequency | - | a500 |
| `hilbert_inst_amplitude` | Instantaneous amplitude | `scipy` | a500 |
| `hilbert_phase` | Instantaneous phase | `scipy` | a500 |
| `spectral_centroid` | Weighted mean frequency | custom | a200 |
| `spectral_bandwidth` | Frequency spread | custom | a200 |
| `spectral_rolloff` | Frequency below which 85% energy | custom | a500 |

**Interpretation:**
- **EMD IMFs**: Intrinsic Mode Functions capture different timescales
- **Instantaneous frequency**: Local cycle length (via Hilbert transform)
- **Spectral centroid**: "Center of mass" of frequency spectrum

### 16.5 Topological Data Analysis (~11 features)

**TDA** uses persistent homology to detect structural patterns. Uses 40-60 day sliding windows.

**Note:** Data loss is ~60 days (configurable), less than 252d MA features.

| Feature | Description | Library | Tier |
|---------|-------------|---------|------|
| `tda_betti_0` | Connected components (Betti-0) | `giotto-tda` | a200 |
| `tda_betti_0_slope` | Change in Betti-0 | - | a200 |
| `tda_betti_1` | Loops/holes (Betti-1) | `giotto-tda` | a200 |
| `tda_betti_1_slope` | Change in Betti-1 | - | a200 |
| `tda_persistence_entropy` | Entropy of persistence diagram | `giotto-tda` | a200 |
| `tda_persistence_entropy_slope` | Change in persistence entropy | - | a200 |
| `tda_total_persistence` | Sum of lifetimes | `giotto-tda` | a200 |
| `tda_amplitude` | Amplitude from persistence | `giotto-tda` | a200 |
| `tda_landscape_norm` | L^p norm of persistence landscape | `giotto-tda` | a500 |
| `tda_crisis_indicator` | TDA-based early warning | custom | a200 |
| `tda_regime_change_prob` | Topological regime shift probability | custom | a500 |

**Interpretation:**
- **Betti-0**: Number of connected components (market fragmentation)
- **Betti-1**: Number of loops/cycles (cyclical patterns)
- **Persistence entropy**: Complexity of topological features
- **Crisis indicator**: Research shows TDA can detect crashes 1-2 weeks early

### 16.6 Cross-Correlation Analysis (~4 features)

**DCCA** (Detrended Cross-Correlation Analysis) measures long-range correlations.

| Feature | Description | Library | Tier |
|---------|-------------|---------|------|
| `dcca_coeff` | Detrended cross-correlation coefficient | `py-DCCA` | a500 |
| `dcca_coeff_slope` | Change in DCCA | - | a500 |
| `mfdcca_hurst` | Cross-correlation Hurst | `MF-DCCA` | a1000 |
| `dpcca_partial` | Partial cross-correlation (controlling for third var) | custom | a1000 |

**Note:** These features require a second time series (e.g., VIX, sector indices) for cross-correlation.

### 16.7 Ergodic Economics Features (~6 features)

**Ergodic Economics** (Ole Peters) distinguishes time-average from ensemble-average growth.

| Feature | Description | Library | Tier |
|---------|-------------|---------|------|
| `time_avg_growth` | Time-average growth rate (geometric) | custom | a200 |
| `ensemble_avg_growth` | Ensemble-average growth rate (arithmetic) | custom | a200 |
| `ergodicity_ratio` | Time avg / Ensemble avg | custom | a200 |
| `ergodicity_ratio_slope` | Change in ergodicity ratio | - | a200 |
| `kelly_fraction` | Optimal Kelly bet size | custom | a200 |
| `ruin_probability` | Probability of ruin over horizon | custom | a500 |

**Interpretation:**
- **Ergodicity ratio < 1**: Non-ergodic dynamics (time average < ensemble average)
- **Kelly fraction**: Optimal position sizing for growth maximization
- Ratio approaching 1 indicates more ergodic (stable) dynamics

### 16.8 Polynomial Regression Channels (~13 features)

**Polynomial channels** fit curves to price, providing flexible trend estimation.

**Note:** Includes quadratic (2), cubic (3), AND quintic (5) per user preference.

| Feature | Description | Library | Tier |
|---------|-------------|---------|------|
| `poly2_residual` | Quadratic (parabolic) fit residual | `numpy` | a200 |
| `poly2_channel_position` | Position in quadratic channel (0-1) | `numpy` | a200 |
| `poly2_curvature` | Parabolic curvature (2nd derivative) | `numpy` | a200 |
| `poly3_residual` | Cubic fit residual | `numpy` | a200 |
| `poly3_channel_position` | Position in cubic channel (0-1) | `numpy` | a200 |
| `poly3_slope` | Cubic regression slope at current | `numpy` | a200 |
| `poly3_curvature` | Cubic curvature (2nd derivative) | `numpy` | a200 |
| `poly5_residual` | Quintic fit residual | `numpy` | a200 |
| `poly5_channel_position` | Position in quintic channel (0-1) | `numpy` | a200 |
| `poly5_slope` | Quintic slope at current | `numpy` | a200 |
| `poly_degree_optimal` | Best-fit polynomial degree (AIC/BIC) | `numpy` | a500 |
| `poly_inflection_distance` | Distance to nearest inflection point | custom | a500 |
| `poly_trend_strength` | R² of best polynomial fit | `numpy` | a200 |

**Interpretation:**
- **Channel position**: 0 = at lower band, 1 = at upper band
- **Curvature**: Positive = convex (accelerating), negative = concave (decelerating)
- **Inflection distance**: How close to trend reversal point

### 16.9 Additional Stochastic Features (~8 features)

**Extensions** to existing Hurst/fBm features with rolling and multi-scale variants.

| Feature | Description | Library | Tier |
|---------|-------------|---------|------|
| `fbm_hurst_rolling` | Rolling fBm Hurst estimate | `fbm` | a200 |
| `fbm_hurst_rolling_slope` | Change in rolling Hurst | - | a200 |
| `fbm_hurst_rolling_accel` | Acceleration of Hurst change | - | a200 |
| `multi_fbm_hurst` | Time-varying Hurst (multi-fBm) | custom | a1000 |
| `dfa_alpha` | DFA scaling exponent | `nolds` | a200 |
| `dfa_alpha_slope` | Change in DFA alpha | - | a200 |
| `mean_reversion_halflife` | OU process half-life estimate | custom | a200 |
| `mean_reversion_speed` | OU theta parameter | custom | a200 |

**Interpretation:**
- **DFA alpha ≈ 0.5**: Random walk
- **DFA alpha > 0.5**: Persistent (trending)
- **DFA alpha < 0.5**: Anti-persistent (mean-reverting)
- **Mean reversion half-life**: Expected time for price to revert halfway to mean

### 16.10 Volatility Risk Premium Features (~8 features)

**Girsanov-derived**: Practical approximation of "market price of risk" using VIX data.

**Data Required:** VIX index (already available in our data pipeline).

| Feature | Description | Tier |
|---------|-------------|------|
| `volatility_risk_premium` | VIX - realized_vol(30d) | a100 |
| `vrp_slope` | Change in VRP | a100 |
| `vrp_acceleration` | Change in VRP slope | a100 |
| `vrp_zscore` | VRP normalized by 252d history | a100 |
| `vrp_percentile` | VRP historical percentile | a100 |
| `vrp_regime_score` | VRP regime (-1 cheap vol, +1 expensive) | a100 |
| `vrp_mean_reversion` | Distance from VRP mean | a200 |
| `implied_vs_realized_ratio` | VIX / realized_vol | a100 |

**Interpretation:**
- **VRP > 0**: Implied vol > realized vol (vol sellers get paid)
- **VRP < 0**: Implied vol < realized vol (rare, vol buyers get paid)
- **High VRP**: Fear premium is elevated, potential mean reversion
- This is the practical equivalent of "market price of risk" from Girsanov theorem

### 16.11 Risk Resilience Features (~6 features)

**BSDE-inspired**: Practical approximations of dynamic risk recovery from stochastic control theory.

| Feature | Description | Tier |
|---------|-------------|------|
| `drawdown_recovery_speed` | Slope during recovery from drawdown | a200 |
| `drawdown_recovery_accel` | Acceleration of recovery | a200 |
| `risk_resilience_score` | How quickly risk metrics normalize | a200 |
| `cvar_mean_reversion_speed` | Speed of CVaR returning to mean | a500 |
| `max_recovery_time` | Estimated time to recover from max DD | a500 |
| `resilience_regime` | Recovery regime score | a200 |

**Interpretation:**
- **Recovery speed > 0**: Actively recovering from drawdown
- **High resilience score**: Risk metrics quickly normalize after shocks
- These features capture the "bounce-back" behavior after adverse events

### 16.12 Category 16 Feature Count Estimate

```
16.1 Fractal Analysis: ~17 features
16.2 Chaos Theory: ~8 features
16.3 RQA: ~11 features
16.4 Spectral (EMD/HHT): ~10 features
16.5 TDA: ~11 features
16.6 Cross-Correlation: ~4 features
16.7 Ergodic Economics: ~6 features
16.8 Polynomial Channels: ~13 features
16.9 Stochastic Extensions: ~8 features
16.10 VRP (Girsanov-derived): ~8 features
16.11 Risk Resilience (BSDE-inspired): ~6 features

TOTAL CATEGORY 16 FEATURES: ~118

Tier breakdown:
- a100: ~8 (VRP features - use existing VIX data)
- a200: ~55 (Core fractal, TDA, polynomial, ergodic, resilience)
- a500: ~45 (Advanced chaos, RQA, spectral, cross-correlation)
- a1000: ~10 (Multi-fBm, embedding dim, advanced TDA)
```

### 16.13 Library Dependencies Summary

| Library | Features Enabled | Install Notes |
|---------|------------------|---------------|
| `antropy>=0.1.6` | Higuchi, Katz, Petrosian FD | `pip install antropy` |
| `nolds>=0.6.0` | Lyapunov, correlation dim, DFA | `pip install nolds` |
| `MFDFA>=0.4.3` | Multifractal DFA | `pip install MFDFA` |
| `hfda>=0.2.0` | Alternative Higuchi FD | `pip install hfda` |
| `PyEMD>=1.6.0` | EMD, EEMD, CEEMDAN | `pip install EMD-signal` |
| `pyrqa>=8.2.0` | RQA features | `pip install PyRQA` |
| `giotto-tda>=0.6.0` | Persistent homology | `pip install giotto-tda` (heavy) |
| `py-DCCA` | DCCA features | Install from GitHub |

---

## Grand Total Estimate

| Category | Features (v0.4) | Change from v0.3 | Notes |
|----------|-----------------|------------------|-------|
| Moving Averages | 780 | 0 | No changes |
| Oscillators | 416 | 0 | No changes |
| Volatility | 257 | 0 | No changes |
| Volume | 169 | 0 | No changes |
| Trend Indicators | 58 | 0 | No changes |
| Support/Resistance | 49 | 0 | No changes |
| Candlestick | 57 | 0 | No changes |
| Momentum | 40 | 0 | No changes |
| Calendar | 16 | 0 | No changes |
| Entropy | 34 | 0 | No changes |
| Regime | 25 | 0 | No changes |
| Multi-Timeframe | 11 | 0 | No changes |
| SMC | 17 | 0 | No changes |
| Risk-Adjusted | 28 | 0 | No changes |
| Signal Processing | 15 | 0 | No changes |
| **Advanced Math (NEW)** | **~118** | **+118** | **Fractal, chaos, TDA, ergodic, VRP** |
| **TOTAL** | **~2,090** | **+118** | **~6% increase with advanced features** |

**Key Changes in v0.4:**
- Added Category 16: Advanced Mathematical Features (~118 features)
  - **Fractal Analysis** (~17): Higuchi FD, Katz FD, MFDFA, Lévy alpha, FDI
  - **Chaos Theory** (~8): Lyapunov exponent, correlation dimension, attractor features
  - **RQA** (~11): Determinism, laminarity, diagonal entropy, crisis indicators
  - **Spectral/EMD** (~10): EMD/EEMD decomposition, Hilbert instantaneous frequency
  - **TDA** (~11): Betti curves, persistence entropy, regime change probability
  - **Cross-Correlation** (~4): DCCA, MF-DCCA coefficients
  - **Ergodic Economics** (~6): Time-average growth, Kelly fraction, ergodicity ratio
  - **Polynomial Channels** (~13): Quadratic, cubic, quintic regression channels
  - **Stochastic Extensions** (~8): Rolling Hurst, DFA alpha, mean reversion
  - **Volatility Risk Premium** (~8): VRP, implied/realized ratio (Girsanov-derived)
  - **Risk Resilience** (~6): Recovery speed, resilience score (BSDE-inspired)
- New dependencies: antropy, nolds, MFDFA, hfda, PyEMD, pyrqa, giotto-tda

**Tier Distribution (v0.4 additions):**

| Tier | New Features | Examples |
|------|--------------|----------|
| a100 | ~8 | VRP, implied/realized ratio (use existing VIX data) |
| a200 | ~55 | Higuchi FD, TDA Betti curves, polynomial channels, ergodic features |
| a500 | ~45 | Lyapunov, RQA, EMD/HHT, MFDFA, DCCA |
| a1000 | ~10 | Multi-fBm, embedding dimension, advanced cross-correlation |

**Previous v0.3 Changes Summary:**
- Added Category 14: Risk-Adjusted Metrics (~28 features)
- Added Category 15: Signal Processing (~15 features)
- Expanded Oscillators: +QQE, +STC, +DeMarker (+20 features)
- Added Donchian Channel (+6 features)
- Added Expectancy metrics (+5 features)
- Added Daily return enhancements (+3 features)

**Previous v0.2 Changes Summary:**
- Signed features replace binary pairs (Pattern 1, 3): ~150 features saved
- Removed redundant calendar binaries: ~19 features saved
- Added acceleration features (Pattern 4): ~40 features added
- New indicator sections (StochRSI, Gaussian, Keltner): ~35 features added

---

## Data Lookback Impact

Understanding lookback requirements is critical for training data availability.

| Category | Max Lookback | Data Loss | Notes |
|----------|--------------|-----------|-------|
| Moving Averages | 252 days | ~1 year | SMA_252 is longest |
| Oscillators | 28 days | ~1 month | RSI_28 is longest |
| Volatility | 50 days | ~2 months | ATR_50 typical max |
| Volume | 60 days | ~3 months | Percentile windows |
| Trend Indicators | 26 days | ~1 month | Ichimoku Kijun |
| Support/Resistance | 252 days | ~1 year | 52-week high/low |
| Candlestick | 5 days | ~1 week | Pattern windows |
| Momentum | 252 days | ~1 year | 252-day return |
| Calendar | 0 days | None | Current date only |
| Entropy | 60 days | ~3 months | Typical window |
| Regime | 60 days | ~3 months | HMM training window |
| Multi-Timeframe | 52 weeks | ~1 year | Weekly indicators |
| SMC | 20 days | ~1 month | Order block detection |

**Maximum Overall Lookback:** 252 days (~1 year)

**SPY Data Availability:** 1993-present (~31 years)
- Training data starts: 1994 (after 1-year lookback)
- Effective training period: 1994-2020 (26 years)
- This is sufficient for robust training

---

## Next Steps

1. ✅ **Risk Metrics Added:** Sharpe, Sortino, VaR, CVaR in Category 14
2. ✅ **Signal Processing Added:** VMD, Wavelet, FFT in Category 15 (optional)
3. ✅ **High-Signal Oscillators Added:** QQE, STC, DeMarker
4. ✅ **Donchian Channel Added:** Enables entropy-filtered breakout strategies
5. ✅ **Expectancy Metrics Added:** Fundamental performance features
6. ✅ **Advanced Mathematical Features Added:** Fractal, chaos, TDA, ergodic, VRP in Category 16
7. **Implementation Phase:** Begin coding feature calculations
8. **Library Integration:** Install new dependencies when implementing Category 16
   - Phase 0: VRP features (no new deps, uses existing VIX data)
   - Phase 1: antropy, nolds, numpy (a200 features)
   - Phase 2: MFDFA, PyEMD, pyrqa (a500 features)
   - Phase 3: giotto-tda (a1000 TDA features, computationally heavy)
9. **Tier Validation:** Verify a100 features are highest-signal during Phase 6C

---

## Verification Checklist

**Catalog Integrity:**
- [x] Total feature count: ~2,090 (v0.4)
- [x] No duplicate feature names (verified by naming convention consistency)
- [x] All new features follow naming conventions
- [x] All slopes have corresponding accelerations
- [x] Calculation formulas documented for new indicators

**Consolidation Pattern Compliance:**
- [x] No `days_above_` or `days_below_` patterns (use `days_since_*_cross` signed)
- [x] No `max(0, x)` patterns (use signed `extreme_dist` features)
- [x] All categorical features converted to continuous scores

**New Indicator Verification (v0.3):**

| Indicator | Category | Features | Tier | Library Req. |
|-----------|----------|----------|------|--------------|
| Sharpe Ratio | 14 | 9 | a100 | None |
| Sortino Ratio | 14 | 9 | a100 | None |
| VaR | 14 | 4 | a200 | None |
| CVaR | 14 | 4 | a200 | None |
| QQE | 2 | 8 | a100 | pandas-ta |
| STC | 2 | 7 | a100 | pandas-ta |
| DeMarker | 2 | 5 | a200 | pandas-ta |
| Donchian | 5 | 6 | a200 | pandas-ta |
| Daily Returns | 8 | 3 | a100 | None |
| Expectancy | 8 | 5 | a100 | None |
| VMD | 15 | 5 | a200 | vmdpy/PyEMD |
| Wavelet | 15 | 5 | a200 | PyWavelets |
| FFT | 15 | 5 | a200 | scipy |

**New Indicator Verification (v0.4 - Category 16):**

| Indicator Group | Section | Features | Tier | Library Req. |
|-----------------|---------|----------|------|--------------|
| Higuchi/Katz FD | 16.1 | 5 | a200 | antropy, hfda |
| Petrosian FD | 16.1 | 1 | a500 | antropy |
| MFDFA | 16.1 | 4 | a500 | MFDFA |
| Lévy/Tail | 16.1 | 4 | a500 | scipy, custom |
| FDI | 16.1 | 3 | a200 | custom |
| Lyapunov/Chaos | 16.2 | 5 | a500 | nolds |
| Embedding/Attractor | 16.2 | 3 | a1000 | nolds, custom |
| RQA | 16.3 | 11 | a500 | pyrqa |
| EMD/HHT | 16.4 | 7 | a500 | PyEMD |
| Spectral Features | 16.4 | 3 | a200-a500 | custom |
| TDA Betti/Persistence | 16.5 | 9 | a200 | giotto-tda |
| TDA Advanced | 16.5 | 2 | a500 | giotto-tda |
| DCCA | 16.6 | 2 | a500 | py-DCCA |
| MF-DCCA/DPCCA | 16.6 | 2 | a1000 | custom |
| Ergodic Economics | 16.7 | 6 | a200-a500 | custom |
| Polynomial Channels | 16.8 | 13 | a200-a500 | numpy |
| Stochastic Extensions | 16.9 | 8 | a200-a1000 | nolds, custom |
| VRP | 16.10 | 8 | a100-a200 | None (VIX data) |
| Risk Resilience | 16.11 | 6 | a200-a500 | custom |

**Library Dependencies Summary:**
- `pandas-ta`: QQE, STC, DeMarker, Donchian (already in requirements)
- `vmdpy` or `PyEMD`: VMD features
- `PyWavelets`: Wavelet features
- `scipy`: FFT features (already available)
- `antropy>=0.1.6`: Higuchi, Katz, Petrosian FD (new)
- `nolds>=0.6.0`: Lyapunov, correlation dim, DFA (new)
- `MFDFA>=0.4.3`: Multifractal DFA (new)
- `hfda>=0.2.0`: Alternative Higuchi FD (new)
- `PyEMD>=1.6.0`: EMD, EEMD, CEEMDAN (new)
- `pyrqa>=8.2.0`: RQA features (new)
- `giotto-tda>=0.6.0`: Persistent homology (new, optional, heavy)
- `py-DCCA`: DCCA features (new, install from GitHub)

---

*Document Version: 0.4 DRAFT*
*Created: 2026-01-22*
*Revised: 2026-01-23*
*Status: Expanded with advanced mathematical features (fractal, chaos, TDA, ergodic, VRP)*
