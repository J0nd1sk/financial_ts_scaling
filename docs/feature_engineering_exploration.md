# Feature Engineering Exploration

## Document Purpose

This document tracks **brainstorming, discovery, and exploration** of features for Phase 6C (Feature Scaling). We are building a comprehensive feature vocabulary - casting a wide net before filtering.

**Stage**: Discovery & Exploration (not implementation planning yet)

**Target Feature Counts**:
- Tier a50: ~50 features (curated, high-signal)
- Tier a100: ~100 features
- Tier a200: ~200 features
- Tier a500: ~500 features
- Tier a1000: ~1000 features (comprehensive)
- Tier a2000: ~2000 features (exhaustive - includes experimental/novel features)

**Last Updated**: 2026-01-22

---

## Core Principle: Signal Over Noise

**THE NUMBER ONE GOAL**: Give the neural network the ability to discern signal from noise.

### The Problem with Raw Values

Raw indicator values (SMA, DEMA, RSI, etc.) are largely **noise** for a neural network because:

1. **Absolute values lack context** - An SMA of 450 vs 150 means nothing without knowing where price is
2. **Scale varies over time** - SPY at $100 in 2010 vs $500 in 2024 makes raw values incomparable
3. **The model must learn relationships from scratch** - We're forcing the NN to discover what we already know

### What Actually Matters: Relationships and Dynamics

The **signal** lies in:

| Relationship Type | Examples | Why It Matters |
|-------------------|----------|----------------|
| **Position relative to level** | Price above/below SMA | Defines bullish/bearish context |
| **Distance from level** | % difference from MA | Measures extension/reversion potential |
| **Duration of relationship** | Days above/below MA | Overextension signal (e.g., >9 days above 9MA often precedes pullback) |
| **Slope of indicator** | MA slope direction/magnitude | Trend strength; being above a flat/declining MA is less bullish |
| **Acceleration** | Change in slope | Momentum shifts before price does |
| **Proximity to regime change** | How close are two MAs to crossing | Anticipates crossover signals |
| **Recency of events** | Days since last cross | Crossovers have decaying influence |

---

## Feature Categories to Develop

### 1. Price-to-MA Relationships

**Current problem**: We have SMA_12, SMA_100, SMA_200, DEMA_9, etc. as raw values.

**Better approach**:
- `price_pct_from_sma_X` - % distance of close from SMA (negative = below)
- `days_above_sma_X` / `days_below_sma_X` - Duration counters
- `sma_X_slope` - Rate of change of the MA itself
- `sma_X_slope_change` - Acceleration/deceleration of MA slope

**Key insight from user**: Prices tend to pull back when closed above 9-day MA for >9 days (and vice versa). This is a *duration-based* signal that raw MA values cannot capture.

### 2. MA Cross Relationships

**Current problem**: No cross features at all.

**Better approach**:
- `days_since_X_cross_Y` - How recently did SMA_X cross SMA_Y (signed: + for golden, - for death)
- `pct_to_X_Y_cross` - % difference between two MAs (proximity to cross)
- `X_above_Y` - Binary: is shorter MA above longer MA

**Important crosses to track**:
- 9/50 (short-term momentum)
- 50/200 (golden/death cross - institutional signal)
- 9/200 (extreme momentum shifts)

### 3. Bollinger Band Relationships

**Current problem**: We have `bband_width` (volatility measure).

**Better approach**:
- `price_pct_from_upper_band` - % distance to upper band (negative = inside)
- `price_pct_from_lower_band` - % distance to lower band (positive = inside)
- `price_position_in_bands` - Normalized position (0 = lower band, 1 = upper band)
- `days_outside_bands` - Duration of extreme extension

### 4. Momentum Indicator Relationships

**Current features**: RSI, StochRSI values

**Better approach**:
- `rsi_distance_from_50` - Centered measure of momentum bias
- `rsi_slope` - Is RSI rising or falling?
- `days_in_overbought` / `days_in_oversold` - Duration in extreme zones
- `rsi_divergence` - Price making new high while RSI isn't (requires careful implementation)

### 5. MACD Components

**Current features**: Just `macd_line`

**Better approach** (MACD has 3 components):
- `macd_line` - MACD line value (keep, but normalize)
- `macd_signal` - Signal line
- `macd_histogram` - Difference (momentum of momentum)
- `macd_histogram_slope` - Is histogram expanding or contracting?
- `days_since_macd_cross` - Recency of signal line cross

### 6. ADX Components

**Current features**: Just `adx_14`

**Better approach** (ADX has 3 components):
- `adx_14` - Trend strength (keep)
- `plus_di` - Positive directional indicator
- `minus_di` - Negative directional indicator
- `di_spread` - plus_di - minus_di (directional bias)
- `adx_slope` - Is trend strengthening or weakening?

### 7. Volume Relationships

**Current features**: OBV, A/D, VWAP_20

**Better approach**:
- `price_pct_from_vwap` - Extension from volume-weighted average
- `volume_vs_avg` - Today's volume as multiple of 20-day average
- `obv_slope` - Accumulation/distribution trend
- `price_volume_divergence` - Price up but volume declining (weakness signal)

---

## Research Findings (2026-01-22)

### Sources Consulted

- [Feature Engineering Methods on Multivariate Time-Series Data](https://arxiv.org/abs/2303.16117) - ArXiv paper on financial data science competitions
- [Alpha Factor Research - ML for Trading](https://stefan-jansen.github.io/machine-learning-for-trading/04_alpha_factor_research/) - Stefan Jansen's comprehensive guide
- [Feature Engineering in Trading - LuxAlgo](https://www.luxalgo.com/blog/feature-engineering-in-trading-turning-data-into-insights/)
- [The Alpha Scientist - Feature Engineering](https://alphascientist.com/feature_engineering.html) - Practical recipes
- [TradingView - Consecutive Closes Above/Below SMA](https://www.tradingview.com/script/8JujhZmh-Consecutive-Closes-Above-or-Below-a-SMA/) - Duration-based indicator
- [TrendSpider - Moving Average Crossover Strategies](https://trendspider.com/learning-center/moving-average-crossover-strategies/)

### Key Findings

#### 1. Derived Features Over Raw Values (Validated)
The Alpha Scientist explicitly states: *"It's often more important to know how a value is **changing** than to know the value itself."* This directly supports our core principle.

#### 2. Normalization is Critical
Research consistently shows ML algorithms perform better with normalized data. Key techniques:
- **Percentile transformation**: Rank values 0.0-1.0 relative to trailing period (e.g., 200 days)
- **Log transformation**: For values like volume, market cap that span orders of magnitude
- **Binning**: Converting continuous to discrete (loses info but can remove noise > signal)

#### 3. Duration Features Exist in Practice
TradingView has indicators tracking "consecutive closes above/below SMA" - confirming this is a recognized signal. The indicator description: *"provides a quick visual cue that there is a strong trend in play."* However, this is **under-studied in ML literature**.

#### 4. Cross Proximity is Recognized
The "Touch & Go" pattern (MAs approach but don't cross) is documented as indicating *"potential trend acceleration"*. This validates our idea of measuring % distance between MAs as a cross-proximity feature.

#### 5. Time-Lag Limitation of MAs
Research notes: *"All MA-based technical indicators have a 'time-lag' limitation because buy and sell signals are mostly generated after price trends have already been developed."* Our relational features (slope, acceleration) may help predict the signals earlier.

#### 6. Feature Selection is Critical
One study examined 124 technical indicators but found *"feature selection methods are applied to shrink the feature set aiming to eliminate redundant information from similar indicators."* We should expect many of our raw indicators to be redundant.

#### 7. Interaction Terms Matter
Research highlights: *"Interaction terms emphasize the relationship between features... Interactions between changes in volume and price, or between short-run returns and long-run returns are often analyzed."*

### Techniques to Implement

| Technique | Description | Our Application |
|-----------|-------------|-----------------|
| Percentile rank | Value as 0-1 percentile over trailing window | RSI percentile over 60 days |
| Rate of change | diff() of any indicator | MA slope, RSI momentum |
| Cross-sectional rank | Rank vs other assets | Future: multi-asset phase |
| Interaction terms | Multiply/combine features | Volume × price change |
| Lag features | Past values as features | Price 5 days ago vs today |
| Binning | Discretize continuous | RSI zones (oversold/neutral/overbought) |

### Gap Identified

**Duration-based features** (days above/below MA) are used by traders but **not well-studied in academic ML literature**. This could be a novel contribution of our research.

---

## Research Questions (Updated)

1. ~~What other relational features are used in quantitative finance?~~ ✅ Answered above
2. ~~Are there established libraries/papers on "derived features" for financial ML?~~ ✅ Yes - see sources
3. How do we handle lookback periods for duration counters without look-ahead bias? → **Use causal computation only!!**
4. ~~Should we normalize all features to similar scales?~~ ✅ Yes - research confirms this

---

---

## Expanded Feature Categories (2026-01-22)

### 8. Socio-Psychological Indicators (Derived from OHLCV)

**Core insight**: Markets are human behavior, not physics. Price action reflects collective psychology.

| Feature | Calculation | Psychological Signal |
|---------|-------------|---------------------|
| `candle_body_pct` | \|Close - Open\| / Open | Conviction vs indecision |
| `candle_range_pct` | (High - Low) / Low | Mania (FOMO or FUD) |
| `body_to_range_ratio` | \|Close - Open\| / (High - Low) | Decisiveness of move |
| `upper_wick_pct` | (High - max(O,C)) / (High - Low) | Rejection of highs (selling pressure) |
| `lower_wick_pct` | (min(O,C) - Low) / (High - Low) | Rejection of lows (buying pressure) |
| `range_vs_avg_range` | Today's range / 20-day avg range | Unusual volatility = emotion |
| `gap_pct` | (Open - prev Close) / prev Close | Overnight sentiment shift |

**Derived patterns**:
- Large body + small wicks = strong conviction
- Small body + large wicks = indecision (doji-like)
- Large range + small body = battle between bulls/bears
- Consecutive small bodies = consolidation/uncertainty

### 9. Seasonal & Calendar Features

**Core insight**: Human behavior has rhythms - weekly, monthly, yearly, event-driven.

| Feature | Calculation | Why It Matters |
|---------|-------------|----------------|
| `trading_day_of_week` | 0-4 (Mon-Fri) | Monday effect, Friday positioning |
| `is_monday` / `is_friday` | Binary flags | Start/end of week dynamics |
| `day_of_month_bucket` | early(1-10)/mid(11-20)/late(21-31) | Month-end rebalancing, options expiry |
| `month_of_year` | 1-12 | September-October effect, January effect |
| `quarter` | 1-4 | Earnings seasons, rebalancing |
| `day_of_year_normalized` | 0-1 | Continuous annual cycle |
| `is_holiday_adjacent` | Days since/until market holiday | Thin volume periods |
| `days_to_month_end` | Countdown | Options expiry, rebalancing |

**Detected events** (from data patterns):
- `is_post_holiday` - Gap in trading day sequence detected
- `trading_days_this_month` - Running count

**Future (if data available)**:
- FOMC meeting proximity
- Earnings season indicator
- Quad witching dates

### 10. Signal Processing / Mathematical Transforms

**Core insight**: Price series contain hidden periodicities and patterns detectable via transforms.

| Technique | Application | Features Generated |
|-----------|-------------|-------------------|
| **Fourier Transform** | Frequency decomposition | Dominant cycle periods, spectral power at key frequencies |
| **Wavelet Transform** | Time-frequency localization | Multi-scale trend/noise separation |
| **Fibonacci Ratios** | Retracement levels from swings | Price position relative to 23.6%, 38.2%, 50%, 61.8%, 78.6% |
| **Williams %R** | Momentum oscillator | Overbought/oversold with specific ranges |
| **Hilbert Transform** | Instantaneous phase/amplitude | Trend mode vs cycle mode detection |

**Fourier-derived features**:
- `dominant_cycle_period` - Strongest frequency component
- `spectral_entropy` - How "noisy" vs "periodic" the signal is
- `power_ratio_short_long` - Energy in short vs long cycles

**Fibonacci-derived features**:
- `pct_retracement_from_swing_high` - Where in the retracement are we?
- `nearest_fib_level` - Which Fib level is closest?
- `distance_to_fib_level` - % distance to nearest Fib

### 11. Support/Resistance Levels

**Core insight**: Historical price levels have memory - more touches = stronger level.

**Challenge**: S/R detection is algorithmic, not formulaic. Requires peak/trough detection.

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `pct_to_nearest_resistance` | % distance to nearest resistance above | Headroom |
| `pct_to_nearest_support` | % distance to nearest support below | Cushion |
| `support_strength` | # of touches at nearest support | Level reliability |
| `resistance_strength` | # of touches at nearest resistance | Level reliability |
| `days_since_support_test` | Recency of support test | Fresh vs stale level |
| `in_consolidation_zone` | Close to both S and R | Breakout potential |

**Multi-timeframe S/R**:
- Weekly S/R levels (strongest)
- Daily S/R levels
- Position relative to higher-timeframe levels matters most

**Implementation approach**:
1. Detect local minima/maxima over rolling windows
2. Cluster nearby levels (within X%)
3. Count touches per cluster
4. Track most recent/strongest levels

### 12. Volume-Price Relationships

**Core insight**: Volume confirms or contradicts price moves.

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `volume_price_trend` | Cumulative (volume × price change) | Accumulation vs distribution |
| `volume_on_up_days_ratio` | Avg vol on up days / avg vol on down days | Bullish/bearish volume bias |
| `price_volume_divergence` | Price making new high but volume declining | Weakness signal |
| `volume_breakout` | Volume > 2× 20-day average | Institutional activity |
| `relative_volume` | Today's volume / 20-day avg | Activity level |

### 13. Regime Detection Features

**Core insight**: Markets operate in different regimes - trending vs ranging, high vs low volatility.

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `adx_regime` | ADX > 25 = trending, < 20 = ranging | Trend strength |
| `volatility_regime` | ATR percentile over 60 days | High/low vol environment |
| `trend_regime` | Price vs 50 MA + slope sign | Bull/bear/neutral |
| `vix_regime` | VIX percentile | Fear level |

---

## Feature Tier Architecture (Proposed)

### Tier a50 (~50 features)
- Current 25 raw features (OHLCV + indicators)
- +15 relational features (% from MAs, slopes)
- +10 socio-psychological (candle analysis)

### Tier a100 (~100 features)
- Tier a50
- +20 duration counters (days above/below)
- +15 cross features (proximity, recency)
- +15 seasonal/calendar features

### Tier a200 (~200 features)
- Tier a100
- +30 multi-timeframe features
- +25 signal processing (Fourier, wavelets)
- +25 S/R level features
- +20 regime features

### Tier a500+ (future)
- Tier a200
- +Fibonacci levels
- +Advanced pattern detection
- +Cross-asset features (Phase 6D)

---

## Implementation Notes

- All duration counters must be computed causally (no future information)
- % differences are naturally normalized and comparable across time
- Consider interaction features (e.g., RSI oversold + price at lower BB = stronger signal)
- Test incrementally: add feature groups and measure impact on AUC
- S/R detection requires algorithmic approach - plan separately

---

## Session Log

### 2026-01-22: Initial Exploration

**Key insight from user**: Raw indicator values are noise. The neural net needs relationships:
- Position relative to levels (above/below, % distance)
- Duration of relationships (days above/below)
- Dynamics (slope, acceleration)
- Regime change proximity (distance to crosses, recency of crosses)

**Example given**: Being above the 50 MA is bullish, but less so if the 50 MA's slope is near 0 or negative. This nuance is lost with raw MA values.

**Action items**:
1. Research web for feature engineering approaches in financial ML
2. Design feature tiers (a50, a100, a200) with relational features
3. Implement and test incrementally

---
