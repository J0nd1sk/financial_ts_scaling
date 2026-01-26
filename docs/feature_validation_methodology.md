# Feature Tier Validation Methodology

This document describes the three-layer validation approach used for financial indicator tiers. Use this methodology when creating validation scripts for new tiers (a500, a1000, a2000).

## Overview

Feature validation ensures that computed indicators are:
1. **Correct** - formulas match specifications
2. **Safe** - no lookahead bias (future data leakage)
3. **Sensible** - values make sense in real market contexts

## Layer 1: Deterministic Validation

Automated, reproducible checks with binary pass/fail outcomes.

### 1.1 Data Quality Checks

**Required for all tiers:**
- All expected columns present
- No NaN values after warmup dropout
- No Inf values in any column

```python
# Pattern
missing_cols = set(FEATURE_LIST) - set(df.columns)
nan_cols = [c for c in FEATURE_LIST if df[c].isnull().any()]
inf_cols = [c for c in FEATURE_LIST if np.isinf(df[c]).any()]
```

### 1.2 Formula Verification

Hand-calculate indicators at random indices and compare to stored values.

**Methods by indicator type:**

| Type | Verification Method |
|------|---------------------|
| TA-Lib indicators | Compare against direct `talib.FUNC()` calls |
| Custom formulas | Implement from spec, compare at 5+ indices |
| Derived features | Verify transformation: slope = X - X.shift(5) |
| Ratios | Verify numerator/denominator relationship |

**Tolerance:** `max_diff < 1e-6` for most, `< 1e-5` for complex multi-step

### 1.3 Range/Boundary Checks

Bounded features must stay within expected ranges.

```python
BOUNDED_FEATURES = {
    "ratio_feature": (0, 1),      # Ratios
    "percentile_feature": (0, 1),  # Percentiles
    "binary_feature": (0, 1),      # Binary flags
    "day_of_week": (0, 4),         # Categorical
    "regime_state": (-1, 1),       # Categorical
}

# Allow floating-point tolerance
FP_TOLERANCE = 1e-9
in_range = vals.min() >= min_val - FP_TOLERANCE and vals.max() <= max_val + FP_TOLERANCE
```

### 1.4 Lookahead Detection (Truncation Test)

**Critical for any feature used in ML models.**

Test that features computed on truncated data match features computed on full data:

```python
def test_no_lookahead(feature, full_df, raw_df, vix_df, test_idx):
    # Get value from full computation
    full_value = full_df.loc[test_idx, feature]

    # Truncate raw data at test_idx
    truncated_raw = raw_df[raw_df["Date"] <= test_date]
    truncated_vix = vix_df[vix_df["Date"] <= test_date]

    # Recompute features on truncated data
    truncated_features = build_features(truncated_raw, truncated_vix)
    truncated_value = truncated_features.loc[test_idx, feature]

    # Values MUST match - any difference indicates lookahead
    assert abs(full_value - truncated_value) < 1e-6
```

**Test at 5+ random indices after warmup period (260+ days).**

### 1.5 Logical Constraints

Domain-specific rules that must always hold:

| Constraint Type | Example |
|-----------------|---------|
| Mutual exclusivity | `days_above_X > 0` implies `days_below_X == 0` |
| Sign constraints | `distance_from_52wk_high` always <= 0 |
| Increment logic | Counter increases by 1 when condition persists |
| Reset logic | Counter resets to 0 when condition changes |

## Layer 2: Semantic Validation

Human-guided checks that verify indicators behave sensibly in real market contexts.

### 2.1 Sample Date Audit

Select 5-10 significant market dates and manually verify feature values make sense.

**Recommended dates for SPY:**

| Date | Context | Expected Behavior |
|------|---------|-------------------|
| 2020-03-16 | COVID crash worst day | High vol, negative z-scores, oversold |
| 2020-03-23 | COVID bottom | Extreme oversold, high vol, reversal signals |
| 2021-11-19 | ATH before bear | Overbought, near 52wk high, low vol |
| 2022-06-16 | 2022 bear low | Oversold, elevated vol, below cloud |
| 2023-10-27 | Oct 2023 correction | Oversold, volatility spike |
| 2024-07-16 | Mid-2024 high | Overbought, bullish indicators |

**For each date, print key indicators and assess:**
- Do volatility measures match market conditions?
- Do momentum/trend indicators show expected direction?
- Do mean reversion z-scores reflect price extremes?

### 2.2 Cross-Indicator Consistency

Related indicators must agree logically. Validate these rules programmatically:

```python
# RSI-duration sync
if rsi > 70: assert days_rsi_overbought > 0
if rsi < 30: assert days_rsi_oversold > 0

# Duration exclusivity
if days_above_X > 0: assert days_below_X == 0

# Calendar consistency
if is_monday == 1: assert trading_day_of_week == 0
if is_quarter_end == 1: assert month_of_year in {3, 6, 9, 12}

# Regime state thresholds
if vol_regime_state == 1: assert atr_regime_pct > 0.7
if vol_regime_state == -1: assert atr_regime_pct < 0.3

# Derived formulas
divergence_magnitude == max(|price_rsi_div|, |price_obv_div|)
```

**Allow 1-2% tolerance for edge cases at state transitions.**

### 2.3 Manual Reasoning Review

**This step cannot be automated.** After running automated checks, a human must:

1. **Inspect raw indicator values** at significant dates and reason through whether they make sense
2. **Verify causal logic** - do indicator relationships follow expected market behavior?
3. **Check edge cases** - extreme values, boundary conditions, state transitions

**Process:**
1. Run the automated validation script
2. Export key indicators for 5-10 sample dates
3. For EACH new indicator, answer these questions:
   - Does the value make sense given the market context?
   - Is the sign/direction correct?
   - Are related indicators consistent?
   - Does the indicator behave correctly at boundaries?

**Example Manual Reasoning (tier_a200 COVID crash 2020-03-16):**

| Indicator | Value | Reasoning |
|-----------|-------|-----------|
| `vol_regime_state` | 1 (high) | ✓ Correct - extreme volatility during crash |
| `atr_regime_pct_60d` | 1.00 | ✓ Correct - ATR at 100th percentile |
| `rsi_distance_from_50` | -19.9 | ✓ RSI=30.1, just above oversold threshold |
| `days_rsi_oversold` | 0 | ✓ Correct - RSI=30.1 > 30, not oversold |
| `donchian_position` | 0.024 | ✓ Correct - near bottom of 20-day range |
| `permutation_entropy` | 0.606 | ✓ Lower than avg (0.716) = trending market |

**Key verification patterns:**
- Duration counters: Verify increment (+1) and reset (→0) logic
- Threshold indicators: Verify boundary logic (RSI>70 vs RSI>=70)
- Signed features: Verify sign convention (+/- for bullish/bearish)
- Entropy features: Verify lower values = more predictable/trending

**Document your reasoning.** Create a file `outputs/validation/tier_aNNN_manual_review.md` with:
- Dates inspected
- Indicators reviewed
- Reasoning for each key finding
- Any anomalies or concerns

### 2.4 Known Event Verification

Verify indicators show expected behavior during well-known market events:

**Event: COVID Crash (2020-03-09 to 2020-03-23)**
- [ ] vol_regime_state == 1 for majority of days
- [ ] atr_regime_pct_60d > 0.8 mean
- [ ] zscore_from_20d_mean < -1 mean
- [ ] distance_from_52wk_high_pct < -20% at worst

**Event: 2022 Bear Market (Jan-Oct 2022)**
- [ ] Significant time below Ichimoku cloud (>30%)
- [ ] Consecutive down day streaks >= 3
- [ ] Elevated volatility periods

**Event: 2023-2024 Bull Run**
- [ ] Significant time above Ichimoku cloud (>50%)
- [ ] High Donchian position mean (>0.6)
- [ ] Near 52wk highs during breakouts

## Layer 3: Documentation & Reporting

### 3.1 Output Files

Each validation run produces:

| File | Content |
|------|---------|
| `tier_aNNN_validation.json` | Full results, machine-readable |
| `tier_aNNN_validation.md` | Human-readable report |
| `tier_aNNN_sample_audit.md` | Sample date audit details |

### 3.2 Report Structure

```markdown
# Tier ANNN Validation Report

## Summary
- Total Checks: X
- Passed: Y
- Failed: Z
- Pass Rate: P%

## Results by Chunk
| Chunk | Passed | Failed | Total |

## Detailed Results
### [PASS/FAIL] `feature_name`
- Check 1: ...
- Check 2: ...

## Failed Checks Summary (if any)
```

## Checklist for New Tiers

When validating a new tier (a500, a1000, a2000):

1. [ ] Create `scripts/validate_tier_aNNN.py` following this pattern
2. [ ] Define CHUNK_MAP grouping features logically
3. [ ] Define BOUNDED_FEATURES for all bounded indicators
4. [ ] Implement formula verification for each chunk
5. [ ] Define cross-indicator consistency rules for new features
6. [ ] Select 5-10 sample dates relevant to the data period
7. [ ] Define expected behavior for known events
8. [ ] Run validation, achieve 100% Layer 1 pass rate
9. [ ] **CRITICAL: Manual reasoning review** - inspect actual values and verify logical soundness
10. [ ] Document manual review findings in `outputs/validation/tier_aNNN_manual_review.md`
11. [ ] Archive reports in `outputs/validation/`

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Floating point mismatch | Precision differences | Use tolerance 1e-6 |
| NaN in features | Insufficient warmup | Check warmup periods |
| Lookahead detected | Rolling window includes future | Fix window calculation |
| Range violation | Edge case not handled | Add bounds clamping |
| Cross-indicator mismatch | Logic error | Fix one of the indicators |

## Validation Metrics

**Success criteria:**
- Layer 1: 100% pass rate required (automated)
- Layer 2: All sample audits "SENSIBLE" or documented exceptions (automated)
- Layer 2.3: Manual reasoning review completed with documented findings (human)
- Layer 3: Reports archived for reproducibility

**Runtime targets:**
- Deterministic checks: < 2 minutes
- Lookahead test (5 indices): 3-5 minutes
- Full validation: < 10 minutes
