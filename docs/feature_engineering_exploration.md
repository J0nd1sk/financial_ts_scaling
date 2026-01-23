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

## Additional Ideas & Riffs (Agent Contributions)

### 14. Momentum Divergences

**Core insight**: When price and momentum disagree, momentum usually wins.

| Feature | Description | Signal |
|---------|-------------|--------|
| `price_rsi_divergence` | Price making new high but RSI isn't | Hidden weakness |
| `price_obv_divergence` | Price up but OBV declining | Volume not confirming |
| `price_macd_divergence` | Price/MACD histogram disagreement | Momentum shift coming |
| `divergence_duration` | How many days has divergence persisted | Stronger signal with duration |

**Implementation**: Compare price percentile rank vs indicator percentile rank over N days.

### 15. Mean Reversion Indicators

**Core insight**: Extended moves tend to revert. How "stretched" is price?

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `zscore_from_20d_mean` | (Price - 20d MA) / 20d StdDev | Statistical extension |
| `zscore_from_50d_mean` | Same, 50-day | Medium-term extension |
| `percentile_in_52wk_range` | Where in 52-week range (0-1) | Near highs or lows |
| `days_since_52wk_high` | Recency of high | Momentum vs exhaustion |
| `days_since_52wk_low` | Recency of low | Recovery age |
| `distance_from_52wk_high_pct` | % below 52-week high | Drawdown depth |

### 16. Volatility Clustering & Regime

**Core insight**: Volatility clusters - high vol follows high vol. Regime matters for strategy.

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `atr_expansion` | ATR today / ATR 5 days ago | Volatility accelerating |
| `atr_percentile_60d` | ATR rank over 60 days | High/low vol regime |
| `consecutive_high_vol_days` | Days with range > 1.5× average | Volatility persistence |
| `vol_of_vol` | StdDev of daily ATR over 20 days | Volatility stability |
| `realized_vs_implied` | ATR vs VIX-implied vol | Vol premium/discount |

### 17. Trend Exhaustion Indicators

**Core insight**: Strong trends eventually exhaust. How "tired" is the current move?

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `consecutive_up_days` | Count of consecutive higher closes | Trend persistence |
| `consecutive_down_days` | Count of consecutive lower closes | Trend persistence |
| `up_days_in_last_20` | Ratio of up days | Trend strength |
| `avg_up_day_magnitude` | Mean % gain on up days | Move quality |
| `avg_down_day_magnitude` | Mean % loss on down days | Move quality |
| `up_down_magnitude_ratio` | Avg up / avg down | Bulls vs bears strength |

### 18. Liquidity & Market Microstructure

**Core insight**: Thin markets behave differently. Volume patterns reveal institutional activity.

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `volume_autocorrelation` | Correlation of volume with lag-1 | Clustering of activity |
| `volume_trend` | Slope of 20-day volume MA | Increasing/decreasing interest |
| `high_volume_up_ratio` | % of high-vol days that were up | Institutional bias |
| `volume_price_correlation` | Correlation of vol and abs(return) | Normal = positive |
| `overnight_vs_intraday` | Gap % vs intraday range % | Who's moving price |

### 19. Pattern-Based Features (Quantified)

**Core insight**: Classic patterns can be quantified rather than visually matched.

| Pattern | Quantification | Feature |
|---------|----------------|---------|
| **Higher highs/lows** | Compare recent peaks/troughs | `trend_structure_score` |
| **Inside days** | Today's range inside yesterday's | `is_inside_day`, `consecutive_inside_days` |
| **Outside days** | Today's range engulfs yesterday's | `is_outside_day` |
| **Narrow range** | Range < 0.5× 10-day avg | `is_narrow_range_day` |
| **Wide range** | Range > 2× 10-day avg | `is_wide_range_day` |
| **Gap fill** | Did price fill previous gap? | `gap_filled_pct` |

### 20. Autocorrelation & Memory

**Core insight**: Markets have varying degrees of memory/momentum at different lags.

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `return_autocorr_1d` | Correlation of returns with lag-1 | Short-term momentum/reversal |
| `return_autocorr_5d` | Correlation with lag-5 | Weekly patterns |
| `return_autocorr_20d` | Correlation with lag-20 | Monthly patterns |
| `partial_autocorr_1d` | Partial autocorrelation | Direct lag effect |
| `hurst_exponent` | Long-range dependence measure | Trending (>0.5) vs mean-reverting (<0.5) |

### 21. Cross-Timeframe Features

**Core insight**: Higher timeframe context informs daily decisions.

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `weekly_trend_direction` | Weekly close vs weekly 10-MA | Higher TF bias |
| `monthly_trend_direction` | Monthly close vs monthly 10-MA | Macro bias |
| `daily_vs_weekly_alignment` | Do daily and weekly trends agree? | Confluence |
| `price_vs_weekly_vwap` | Daily close vs weekly VWAP | Institutional reference |
| `atr_daily_vs_weekly_ratio` | Daily ATR / Weekly ATR | Relative volatility |

### 22. Event-Driven Proxies (Without External Data)

**Core insight**: Major events leave footprints in price/volume even without knowing the event.

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `anomalous_volume` | Volume > 3× 60-day avg | Something happened |
| `anomalous_range` | Range > 3× 60-day avg | Major news/event |
| `gap_magnitude_percentile` | Gap size vs historical gaps | Unusual overnight news |
| `post_anomaly_days` | Days since last anomalous event | Decay of event impact |

---

## Open Questions for Further Exploration

1. **Interactions**: Which feature combinations are synergistic? (e.g., RSI oversold + price at support + volume spike)
2. **Optimal lookback windows**: Should we include same feature at multiple windows? (e.g., RSI_14, RSI_7, RSI_21)
3. **Feature stability**: Which features are robust vs which are overfit-prone?
4. **Correlation structure**: How do we handle highly correlated feature groups?
5. **Non-linear features**: Should we include polynomial/interaction terms explicitly, or let the NN learn them?
6. **Lagged features**: Include t-1, t-2, t-5 versions of key indicators?

---

## Expanded Ideas (Session 2026-01-22 Continued)

### 23. Granular Anomaly Detection

**User insight**: Not just one anomaly type - categorize and track each separately.

| Anomaly Type | Detection | Feature |
|--------------|-----------|---------|
| **Price spike** | Close > X σ above MA | `days_since_price_spike_Nd` |
| **Price dump** | Close > X σ below MA | `days_since_price_dump_Nd` |
| **Volatility spike** | Range > X σ above avg | `days_since_vol_spike_Nd` |
| **Volume spike** | Volume > X σ above avg | `days_since_volume_spike_Nd` |
| **Dead volume** | Volume < X σ below avg | `days_since_dead_volume_Nd` |
| **Combined** | Multiple anomalies same day | `days_since_multi_anomaly_Nd` |

**Window variants**: Calculate over 1-day, 2-day, 3-day, 5-day windows (N = window size).

### 24. Trend Channels (Linear & Non-Linear)

**User insight**: Trend channels define bounds; % distance to bounds matters.

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `trend_channel_upper` | Upper bound of fitted channel | Resistance |
| `trend_channel_lower` | Lower bound of fitted channel | Support |
| `pct_to_channel_upper` | % distance to upper (negative = below) | Headroom |
| `pct_to_channel_lower` | % distance to lower (positive = above) | Cushion |
| `channel_width_pct` | (Upper - Lower) / Price | Volatility/consolidation |
| `channel_slope` | Slope of channel midline | Trend direction/strength |

**Channel types**:
- Linear regression channel (both bounds linear)
- Non-linear: Linear lower + curved upper (or vice versa)
- Bollinger-like: Expanding/contracting bounds

**Range-bound detection**: Narrow channel + low slope = range-bound regime.

### 25. Multi-Timeframe Indicators

**User insight**: Daily data should include weekly/monthly indicator context.

| Feature | Calculation | Notes |
|---------|-------------|-------|
| `pct_from_weekly_sma_50` | Price vs 50-week MA | Major trend context |
| `pct_from_monthly_sma_10` | Price vs 10-month MA | Macro context |
| `weekly_rsi_14` | RSI on weekly data | Higher TF momentum |
| `weekly_macd_histogram` | MACD on weekly | Higher TF momentum |

**Constraint**: Be conservative with lookback to preserve early training data.

### 26. Interaction Features (Trader-Used Combinations)

**User insight**: Some combinations are already used by traders → self-fulfilling prophecy → include explicitly.

**Research needed**: What combinations do trading communities use?

| Combination | Rationale | Feature |
|-------------|-----------|---------|
| RSI × ATR | Momentum adjusted for volatility | `rsi_atr_product` |
| RSI + StochRSI alignment | Momentum confirmation | `rsi_stochrsi_alignment` |
| MA cross proximity | Distance between MAs (e.g., 20 vs 200) | `pct_diff_ma_20_200` |
| Price at BB + RSI oversold | Confluence | `bb_rsi_confluence_score` |
| Volume spike + breakout | Confirmed breakout | `volume_confirmed_breakout` |

**Principle**: Explicitly encode what traders watch → captures market psychology.

### 27. Support/Resistance Expanded

**User insight**: Multiple S/R types, timeframes, and strength measures.

**S/R Types**:
1. **Horizontal levels** - Historical price pivots
2. **Moving averages** - Dynamic S/R (50 MA, 200 MA)
3. **Bollinger bands** - Volatility-based S/R
4. **Trend lines** - Diagonal S/R
5. **Fibonacci levels** - Retracement-based S/R
6. **VWAP** - Volume-weighted S/R
7. **Fair value gaps** - Imbalance zones

**Strength measurement**:
- `sr_level_touches` - Number of times level tested
- `sr_level_recency` - Days since last touch
- `sr_level_timeframe` - Which timeframe (daily < weekly < monthly strength)

**Features per S/R type**:
- `pct_to_nearest_[type]_support`
- `pct_to_nearest_[type]_resistance`
- `[type]_support_strength`
- `[type]_resistance_strength`

### 28. Ichimoku Cloud Components

**Japanese technical analysis** - Good for trending assets.

| Component | Description | Feature |
|-----------|-------------|---------|
| Tenkan-sen | 9-period midpoint | Conversion line |
| Kijun-sen | 26-period midpoint | Base line |
| Senkou Span A | (Tenkan + Kijun) / 2, plotted 26 ahead | Cloud boundary 1 |
| Senkou Span B | 52-period midpoint, plotted 26 ahead | Cloud boundary 2 |
| Chikou Span | Close plotted 26 periods back | Lagging span |

**Derived features**:
- `price_vs_cloud` - Above, below, or inside cloud
- `cloud_thickness` - Senkou A - Senkou B (trend strength)
- `tenkan_kijun_cross` - Bullish/bearish cross
- `chikou_vs_price` - Momentum confirmation

### 29. Candlestick Pattern Quantification

**User insight**: Patterns represent psychology; quantify rather than pattern-match.

| Pattern | Psychology | Quantification |
|---------|------------|----------------|
| **Wedge** | Condensing bias | `range_compression_rate` - slope of range over N days |
| **Breakout** | Momentum shift | `breakout_magnitude` - % move beyond channel |
| **Head & shoulders** | Weakening conviction | `hs_pattern_score` - algorithmic detection |
| **Engulfing** | Mania | `engulfing_score` - body ratio × direction |
| **Doji** | Indecision | `doji_score` - body/range ratio < threshold |
| **Hammer/shooting star** | Rejection | `wick_rejection_score` |

**Gaps**:
- `gap_size_pct` - Magnitude of gap
- `gap_direction` - Up or down
- `gap_filled_pct` - How much has been filled
- `days_since_unfilled_gap` - Gap persistence

### 30. Fair Value Gaps (FVG)

**User insight**: FVGs across timeframes; general rule is they get filled.

| Feature | Calculation | Signal |
|---------|-------------|--------|
| `nearest_fvg_above_pct` | % distance to nearest FVG above | Upside target |
| `nearest_fvg_below_pct` | % distance to nearest FVG below | Downside target |
| `fvg_above_size` | Size of nearest FVG above | Imbalance magnitude |
| `fvg_below_size` | Size of nearest FVG below | Imbalance magnitude |
| `unfilled_fvg_count` | Number of unfilled FVGs | Imbalance backlog |
| `days_since_fvg_created` | Age of nearest FVG | Fill probability |

### 31. Volume Deep Dive

**User insight**: Volume is HUGE. Volume vectors, trends, and derived indicators.

**Volume Vectors**:
| Feature | Calculation | Signal |
|---------|-------------|--------|
| `volume_trend_3d` | Slope of volume over 3 days | Short-term interest |
| `volume_trend_5d` | Slope over 5 days | Medium-term interest |
| `consecutive_increasing_vol` | Count of days vol > prior | Building interest |
| `consecutive_decreasing_vol` | Count of days vol < prior | Waning interest |

**VWAP Derivatives**:
| Feature | Calculation | Signal |
|---------|-------------|--------|
| `pct_from_vwap` | (Price - VWAP) / VWAP | Extension from fair value |
| `vwap_trend_5d` | Slope of VWAP | Institutional bias |
| `vwap_pct_change_1d` | Daily VWAP change | Shift in fair value |
| `vwap_ma_10` | 10-day MA of VWAP | Smoothed institutional level |
| `vwap_ma_slope` | Slope of VWAP MA | Institutional trend |

**Additional Volume Indicators**:
| Indicator | Description |
|-----------|-------------|
| VPT (Volume Price Trend) | Cumulative volume × price change % |
| MFI (Money Flow Index) | Volume-weighted RSI (0-100) |
| CMF (Chaikin Money Flow) | Accumulation/distribution over period |
| Ease of Movement | Price change / volume ratio |
| NVI (Negative Volume Index) | Tracks price on low-volume days |
| A/D Line | Accumulation/Distribution |

### 32. Advanced Indicators (User's Chart Favorites)

| Indicator | Description | Category |
|-----------|-------------|----------|
| **Heiken Ashi** | Smoothed candles (avg OHLC) | Trend clarity |
| **Aroon** | Time since high/low (0-100) | Trend strength/direction |
| **Kalman Trend** | Kalman filter smoothing | Noise reduction |
| **Gaussian Channel** | Gaussian-smoothed channel (3, 5, 7 day) | Trend + volatility |
| **ASI Oscillator** | Absolute Strength Index | Momentum |
| **Squeeze Momentum** | BB squeeze + momentum | Breakout anticipation |
| **SuperTrend** | ATR-based trend indicator | Trend direction |
| **Polynomial Regression Channel** | Non-linear trend fit | Curved trends |

**Smart Money Concepts**:
| Indicator | Description |
|-----------|-------------|
| Liquidity Void Detector | Identifies liquidity gaps |
| Smart Money Breakout Channels | Institutional breakout zones |
| Smart Money Concepts (LuxAlgo) | Order blocks, FVGs, liquidity |
| Volume Cluster Profile | Volume at price clusters |
| Volume Channel Flow | Volume distribution in channels |

### 33. Fibonacci Extensions

**User note**: Fibonacci fans in addition to retracements.

| Feature | Description |
|---------|-------------|
| **Fib Retracements** | 23.6%, 38.2%, 50%, 61.8%, 78.6% of swing |
| **Fib Extensions** | 127.2%, 161.8%, 261.8% beyond swing |
| **Fib Fans** | Diagonal lines from pivot at Fib angles |
| **Fib Time Zones** | Vertical lines at Fib intervals |

**Features**:
- `nearest_fib_retracement_pct` - Distance to nearest retracement level
- `nearest_fib_extension_pct` - Distance to nearest extension level
- `fib_fan_position` - Above/below/at fan line

---

## Novel & Research-Based Approaches (2026-01-22 Research)

### 34. Entropy-Based Complexity Measures

**Research sources**: [Entropy in ML Review (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11675792/), [Permutation Transition Entropy](https://www.sciencedirect.com/science/article/abs/pii/S0960077920303611), [Complexity-Entropy Causality Plane](https://www.sciencedirect.com/science/article/abs/pii/S0378437110000397)

**Core insight**: Entropy measures complexity/predictability. Low entropy = predictable (trust your signals). High entropy = chaotic (be skeptical).

| Entropy Measure | What It Captures | Feature |
|-----------------|------------------|---------|
| **Permutation Entropy** | Pattern complexity in ordinal patterns | `permutation_entropy_20d` |
| **Approximate Entropy (ApEn)** | Regularity/predictability | `approx_entropy_20d` |
| **Sample Entropy** | Self-similarity (less biased than ApEn) | `sample_entropy_20d` |
| **Permutation Transition Entropy** | Dynamic complexity, momentum effects | `pte_20d` |
| **Spectral Entropy** | Frequency distribution uniformity | `spectral_entropy_20d` |

**Key finding**: Post-2008 crisis, entropy measures gained attention because "indicators did not signal any danger incoming." Entropy can detect regime instability before price shows it.

**Derived features**:
- `entropy_regime` - High/medium/low entropy state
- `entropy_trend` - Is complexity increasing or decreasing?
- `entropy_vs_volatility` - Entropy normalized by ATR (complexity per unit volatility)

### 35. Hidden Markov Model Regime Features

**Research sources**: [QuantStart HMM Tutorial](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/), [LSEG Regime Detection](https://developers.lseg.com/en/article-catalog/article/market-regime-detection), [HMM GitHub](https://github.com/theo-dim/regime_detection_ml)

**Core insight**: Markets operate in distinct regimes (bull/bear/sideways, high/low vol). HMM can identify current regime probabilistically.

| Feature | Description | Signal |
|---------|-------------|--------|
| `hmm_regime_state` | Current regime (0, 1, 2...) | Which state are we in? |
| `hmm_regime_prob` | Probability of current state | Confidence in regime |
| `hmm_transition_prob` | Probability of regime change | Instability signal |
| `days_in_current_regime` | Duration in current state | Regime persistence |

**Practical application**: "Risk manager checks whether current state is low or high volatility regime. If low, trades pass; if high, trades are filtered."

### 36. Smart Money Concepts (SMC)

**Research sources**: [SMC Python Library](https://github.com/joshyattridge/smart-money-concepts), [LuxAlgo SMC](https://www.luxalgo.com/library/indicator/smart-money-concepts-smc/), [ICT Concepts](https://atas.net/technical-analysis/what-is-the-smart-money-concept-and-how-does-the-ict-trading-strategy-work/)

**Core insight**: Institutional traders leave footprints. Order blocks, liquidity sweeps, and FVGs reveal their activity.

**Order Blocks**:
| Feature | Description |
|---------|-------------|
| `nearest_ob_bullish_pct` | % distance to nearest bullish order block |
| `nearest_ob_bearish_pct` | % distance to nearest bearish order block |
| `ob_strength` | Volume intensity of order block |
| `ob_age_days` | How old is the nearest order block |

**Liquidity Zones**:
| Feature | Description |
|---------|-------------|
| `liquidity_above_pct` | % to nearest liquidity pool above (stops) |
| `liquidity_below_pct` | % to nearest liquidity pool below (stops) |
| `liquidity_swept` | Was liquidity recently swept? |

**AMD Framework (Accumulation-Manipulation-Distribution)**:
| Feature | Description |
|---------|-------------|
| `amd_phase` | Current phase estimate (A/M/D) |
| `consolidation_duration` | Days in accumulation range |
| `manipulation_detected` | Recent false breakout? |

### 37. Indicator Deep Dives (Researched)

**Aroon Indicator** ([Fidelity](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/aroon-indicator), [StockCharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/aroon-oscillator)):
- `aroon_up` = ((N - Days Since N-day High) / N) × 100
- `aroon_down` = ((N - Days Since N-day Low) / N) × 100
- `aroon_oscillator` = aroon_up - aroon_down (-100 to +100)

**Derived features**:
- `aroon_trend_strength` - abs(aroon_oscillator)
- `aroon_crossover_recency` - Days since last aroon cross
- `aroon_consolidation` - Both aroon up and down between 30-70

**TTM Squeeze** ([StockCharts](https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ttm-squeeze), [TrendSpider](https://trendspider.com/learning-center/introduction-to-ttm-squeeze/)):
- Squeeze ON when Bollinger Bands inside Keltner Channels
- Squeeze OFF (fires) when BB expands outside KC
- Momentum histogram shows breakout direction

**Derived features**:
- `squeeze_on` - Binary: is squeeze active?
- `squeeze_duration` - Days in current squeeze
- `squeeze_momentum` - Direction and magnitude of momentum oscillator
- `squeeze_fire_recency` - Days since last squeeze fired

**SuperTrend** ([LuxAlgo](https://www.luxalgo.com/blog/supertrend-indicator-trailing-stop-strategy/), [TrendSpider](https://trendspider.com/learning-center/supertrend-indicator-a-comprehensive-guide/)):
- Upper/Lower Band = (H+L)/2 ± (Multiplier × ATR)
- Trailing stop that adapts to volatility

**Derived features**:
- `supertrend_direction` - Above or below price (+1/-1)
- `pct_from_supertrend` - % distance from SuperTrend line
- `supertrend_flip_recency` - Days since last direction change
- `supertrend_slope` - Is the stop level rising or falling?

### 38. Behavioral/Psychological State Proxies

**Research sources**: [MarketPsych](https://www.marketpsych.com/home), [Behavioral Finance Review](https://acr-journal.com/article/behavioral-finance-and-investor-psychology-understanding-market-volatility-in-crisis-scenarios-1763/)

**Without external sentiment data, we can infer psychology from price/volume:**

| Proxy | Calculation | Psychological State |
|-------|-------------|---------------------|
| `fear_greed_proxy` | Composite of RSI, BB position, volume | Market emotion |
| `herd_behavior_score` | Trend persistence × volume | Following the crowd |
| `overreaction_score` | Magnitude of move vs historical | Emotional excess |
| `capitulation_signal` | High volume + large down move + RSI<20 | Panic selling |
| `euphoria_signal` | High volume + large up move + RSI>80 | FOMO buying |
| `indecision_score` | Consecutive small-body candles | Uncertainty |

**Put/Call ratio proxy** (if options data unavailable):
- `implied_sentiment` = VIX relative level + price momentum divergence

### 39. Multi-Scale Analysis

**Core insight**: Same data at different scales reveals different patterns.

| Scale | Features | Insight |
|-------|----------|---------|
| 1-day | All standard indicators | Noise + signal |
| 3-day | Smoothed versions | Less noise |
| 5-day | Weekly proxy | Short-term trend |
| 20-day | Monthly proxy | Medium-term trend |

**Cross-scale features**:
- `trend_alignment_1_5_20` - Do all scales agree on direction?
- `momentum_divergence_scales` - RSI disagreement across scales
- `volatility_ratio_1_5` - Is short-term vol higher than medium-term?

### 40. Volume-Price Confluence Features

**User insight**: Volume spike + price spike matters SIGNIFICANTLY. Sequential patterns matter too.

**Confluence patterns**:
| Pattern | Calculation | Signal |
|---------|-------------|--------|
| `volume_price_spike_confluence` | Both volume AND price > 2σ same day | Confirmed institutional move |
| `volume_spike_price_flat` | Volume > 2σ but price < 0.5σ | Absorption (quiet accumulation/distribution) |
| `sequential_volume_buildup` | 3+ days of increasing volume | Building pressure |
| `volume_buildup_with_price_accel` | Sequential vol increase + increasing price moves | Momentum confirmation |
| `volume_buildup_then_spike` | Sequential increase → sudden volume explosion | Breakout confirmation |

**Implementation approach**:
```
# Confluence score
volume_zscore = (volume - vol_ma) / vol_std
price_move_zscore = abs(close - open) / atr
confluence_score = volume_zscore * price_move_zscore  # High when both spike

# Sequential buildup
vol_increasing_days = count consecutive days where vol[t] > vol[t-1]
price_accel = (return[t] - return[t-1]) / return[t-1]  # Acceleration
buildup_score = vol_increasing_days * sign(price_accel)
```

### 41. Nested/Multi-Channel Analysis

**User insight**: Price can be in multiple channels simultaneously (e.g., parabolic channel inside larger range-bound channel).

**Approach**: Track inner and outer channels at different timeframes.

| Feature | Description |
|---------|-------------|
| `outer_channel_type` | Longer-term channel (linear/curved/range) |
| `outer_channel_slope` | Slope of outer channel |
| `outer_channel_width_pct` | Width as % of price |
| `inner_channel_type` | Shorter-term channel within outer |
| `inner_channel_slope` | Slope of inner channel |
| `inner_channel_width_pct` | Width as % of price |
| `channel_nesting_ratio` | Inner width / outer width |
| `price_position_in_outer` | 0-1 position in outer channel |
| `price_position_in_inner` | 0-1 position in inner channel |

**Edge cases**:
- If only single channel detected: `inner_channel_type` = `outer_channel_type`, nesting_ratio = 1.0
- If no clear channel: `channel_type` = "none", position features = 0.5

**Channel types beyond price**:
- **Volume channels** - Volume trending in a channel (expanding/contracting interest)
- **RSI channels** - Multiple lower highs in RSI = bearish divergence channel
- **MACD channels** - Histogram bounded in a channel

### 42. Entropy vs Volatility (Key Distinction)

**User question**: How is entropy different than volatility?

| Measure | What It Captures | Example |
|---------|------------------|---------|
| **Volatility (ATR)** | Magnitude of price swings | "How big are the moves?" |
| **Entropy** | Predictability/complexity of patterns | "How orderly vs chaotic is the sequence?" |

**Key difference**: You can have HIGH volatility with LOW entropy (orderly large moves, like a clean trend), or LOW volatility with HIGH entropy (small but chaotic, unpredictable moves).

**Example**:
- Clean uptrend: Large moves (high vol) but predictable pattern (low entropy)
- Choppy sideways: Small moves (low vol) but random direction (high entropy)

**Entropy measures available**:
| Entropy Type | What It Measures | Computation |
|--------------|------------------|-------------|
| **Shannon Entropy** | Information content | Probability distribution of states |
| **Permutation Entropy** | Ordinal pattern complexity | Frequency of up/down/flat permutations |
| **Approximate Entropy (ApEn)** | Regularity/self-similarity | Pattern recurrence at scale r |
| **Sample Entropy (SampEn)** | ApEn without self-matching bias | More robust version of ApEn |
| **Multiscale Entropy** | Complexity at different scales | SampEn at multiple coarse-graining levels |
| **Spectral Entropy** | Frequency distribution uniformity | Entropy of power spectrum |
| **Permutation Transition Entropy** | Dynamic complexity | Transitions between ordinal patterns |

**Proposed features**:
- `entropy_vol_ratio` = entropy / normalized_volatility → High when chaotic relative to move size
- `entropy_regime` = categorical (orderly/moderate/chaotic)
- `entropy_trend_5d` = Is chaos increasing or decreasing?

### 43. Hurst Exponent Refinement

**User insight**: H = 0.4-0.6 might not be "random walk" but rather **range-bound** or **condensing channel**.

**Refined interpretation**:
| Hurst Value | Interpretation | Feature Value |
|-------------|----------------|---------------|
| H > 0.6 | Strong trending (momentum works) | `hurst_regime` = "trending" |
| 0.5 < H < 0.6 | Weak trending | `hurst_regime` = "weak_trend" |
| 0.4 < H < 0.5 | Range-bound / condensing | `hurst_regime` = "range_bound" |
| H < 0.4 | Mean reverting (contrarian works) | `hurst_regime` = "reverting" |

**Additional features**:
- `hurst_value` - Raw H value
- `hurst_trend_5d` - Is H increasing or decreasing?
- `hurst_acceleration` - Rate of change of hurst_trend
- `hurst_vs_channel_width` - Correlation between H and channel compression

### 44. Elliott Wave Theory Features

**User insight**: Elliott Waves map the accumulation → major move → distribution pattern that institutions follow.

**Elliott Wave structure**:
```
Impulse Wave (5 waves in trend direction):
  Wave 1: Initial move (accumulation)
  Wave 2: Pullback (typically 50-61.8% retracement)
  Wave 3: Strongest move (institutional participation)
  Wave 4: Consolidation (typically doesn't overlap Wave 1)
  Wave 5: Final push (distribution begins)

Corrective Wave (3 waves against trend):
  Wave A: Initial correction
  Wave B: Counter-rally (often traps late bulls)
  Wave C: Final correction (capitulation)
```

**Proposed features**:
| Feature | Description |
|---------|-------------|
| `elliott_wave_position` | Estimated current wave (1-5 or A-C) |
| `elliott_wave_confidence` | Confidence in wave count |
| `wave_progress_pct` | How far through current wave (0-100%) |
| `fib_retracement_from_wave_start` | Current retracement level |
| `wave_3_probability` | Probability we're in Wave 3 (strongest) |
| `distribution_signal` | Indicators of Wave 5 exhaustion |

**Detection heuristics**:
- Wave 2: Retraces 50-61.8% of Wave 1
- Wave 3: Often 1.618× length of Wave 1
- Wave 4: Typically retraces 38.2% of Wave 3, doesn't overlap Wave 1
- Wave 5: Often equals Wave 1 in length

**Implementation challenge**: Elliott Wave counting is subjective. We could:
1. Algorithmic detection using swing high/low + Fibonacci ratios
2. Multiple interpretations with confidence scores
3. Focus on simpler pattern: accumulation → impulse → distribution phases

### 45. Accumulation-Distribution Cycle Features

**Simplified Elliott Wave / Wyckoff hybrid**:

| Phase | Characteristics | Features |
|-------|-----------------|----------|
| **Accumulation** | Range-bound, declining volume, institutions buying | `accumulation_score` |
| **Markup** | Breaking out, increasing volume, trend begins | `markup_score` |
| **Distribution** | Range-bound at top, high volume, institutions selling | `distribution_score` |
| **Markdown** | Breaking down, increasing volume, trend down | `markdown_score` |

**Detection signals**:
- Accumulation: Narrow range + volume declining + price near support
- Markup: Breakout + volume spike + price acceleration
- Distribution: Narrow range at highs + high volume + RSI divergence
- Markdown: Breakdown + volume spike + acceleration down

---

## User Feedback on Testing Approach

**On FVG hypothesis testing**: User notes that testing whether "gaps get filled" is interesting but may not be critical - the neural net can learn this, and we'll find best features through actual model testing.

**Principle**: Build comprehensive feature set → test with model → identify highest-signal features empirically.

---

## Open Research Items

1. **Elliott Wave algorithmic detection** - How to implement robustly?
2. **Entropy calculation libraries** - What Python packages support these?
3. **Nested channel detection algorithm** - How to identify inner vs outer channels?
4. **Volume-price confluence scoring** - Optimal thresholds for "significant"?

---

## Session Log

---

## Session Log

### 2026-01-22: Initial Exploration

**Key insight from user**: Raw indicator values are noise. The neural net needs relationships:
- Position relative to levels (above/below, % distance)
- Duration of relationships (days above/below)
- Dynamics (slope, acceleration)
- Regime change proximity (distance to crosses, recency of crosses)

**Example given**: Being above the 50 MA is bullish, but less so if the 50 MA's slope is near 0 or negative. This nuance is lost with raw MA values.

### 2026-01-22: Expanded Exploration (Continued)

**User leads explored**:
1. **Socio-psychological indicators** - Markets are human, not mechanical
2. **Granular anomaly types** - `days_since_X_anomaly` for different anomaly types
3. **Trend channels** - Linear and non-linear bounds, range-bound detection
4. **Multi-timeframe** - Weekly/monthly context for daily data
5. **Interaction features** - Trader-used combinations (self-fulfilling prophecy)
6. **S/R types** - Horizontal, MA, BB, trend lines, Fib, VWAP, FVG
7. **Ichimoku clouds** - Japanese TA for trending assets
8. **Candlestick pattern quantification** - Wedges, engulfing, gaps
9. **Fair Value Gaps** - Multi-timeframe imbalance zones
10. **Volume deep dive** - VWAP derivatives, VPT, MFI, CMF, etc.
11. **Advanced indicators** - Heiken Ashi, Aroon, Kalman, Gaussian Channel, Squeeze, SuperTrend
12. **Smart Money Concepts** - Order blocks, liquidity zones, AMD framework

**Research conducted**:
- Entropy measures (permutation, approximate, sample entropy)
- Hidden Markov Models for regime detection
- Smart Money Concepts Python library discovered
- Aroon, TTM Squeeze, SuperTrend calculations researched

**Key research findings**:
- Entropy can detect regime instability before price shows it
- Post-2008 crisis: "indicators did not signal danger" → entropy fills this gap
- SMC Python library exists: [joshyattridge/smart-money-concepts](https://github.com/joshyattridge/smart-money-concepts)
- HMM-based strategies outperformed buy-and-hold 2006-2023

### 2026-01-22: User Feedback Round 2

**Volume-price confluence**:
- Volume spike + price spike is SIGNIFICANT
- Sequential volume buildup matters (3+ days increasing)
- Volume buildup → sudden spike = breakout confirmation
- Volume spike + flat price = absorption (quiet accumulation)

**Nested channels**:
- Price can be in multiple channels (parabolic inside range-bound)
- Need inner/outer channel tracking
- Single channel case: inner = outer, nesting_ratio = 1.0
- Also: volume channels, RSI channels (multiple lower highs)

**Entropy vs volatility**:
- Volatility = magnitude of moves
- Entropy = predictability/orderliness of pattern
- Can have high vol + low entropy (clean trend) or low vol + high entropy (choppy)
- Multiple entropy types: Shannon, permutation, approximate, sample, multiscale, spectral

**Hurst exponent refinement**:
- User insight: H = 0.4-0.6 is likely range-bound, not random walk
- Condensing channel would show moderate H with decreasing channel width

**Elliott Wave Theory**:
- Maps accumulation → major move → distribution pattern
- 5-wave impulse + 3-wave correction structure
- Wave 3 is strongest (institutional participation)
- Wave 5 is distribution phase
- Implementation: algorithmic detection using swing high/low + Fibonacci ratios

**Testing philosophy**:
- Build comprehensive feature set
- Let model testing identify highest-signal features empirically
- NN can learn patterns like "FVGs get filled" - don't need to pre-validate everything

---
