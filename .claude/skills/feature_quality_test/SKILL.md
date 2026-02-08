---
name: feature_quality_test
description: Test quality of individual features before adding to curated tiers. Use when importing features from btc_ts_experimentation, implementing new indicators, or expanding feature tiers. Provides structured pass/fail verdicts based on quality thresholds.
---

# Feature Quality Test Skill

Test individual features for quality before adding to curated tiers.

## When to Use

- Adding new features from btc_ts_experimentation
- Implementing new indicators
- Before any feature tier expansion
- Validating feature quality after refactoring
- User says "test feature", "feature quality", "validate feature"

## Execution Flow

### Phase 1: DATA QUALITY (Must All Pass)

These are hard requirements - any failure blocks the feature:

| Check | Requirement | Action on Fail |
|-------|-------------|----------------|
| NaN ratio | < 5% | BLOCK |
| Inf values | 0% | BLOCK |
| Variance | > 0 | BLOCK |
| Unique values | > 10 | BLOCK |

### Phase 2: SIGNAL QUALITY (Weighted Scoring)

| Metric | PASS | MARGINAL | FAIL |
|--------|------|----------|------|
| Target correlation (abs) | >= 0.02 | 0.01-0.02 | < 0.01 |
| Mutual information | >= 0.001 | 0.0005-0.001 | < 0.0005 |
| Temporal stability CV | < 0.5 | 0.5-0.8 | >= 0.8 |
| Permutation precision drop | >= 0.002 | 0.001-0.002 | < 0.001 |

### Phase 3: REDUNDANCY Check

| Metric | PASS | MARGINAL | FAIL |
|--------|------|----------|------|
| Max redundancy (max corr with existing) | < 0.90 | 0.90-0.95 | >= 0.95 |

### Phase 4: VERDICT

Overall verdict logic:
- **FAIL**: Any metric is FAIL → Do not add feature
- **MARGINAL**: No FAIL but any MARGINAL → Consider carefully, may add with caveats
- **PASS**: All metrics PASS → Safe to add

## Running the Test

```bash
# Test a single feature with synthetic data
./venv/bin/python scripts/test_feature_quality.py --feature rsi_14

# Test with actual processed data
./venv/bin/python scripts/test_feature_quality.py \
    --feature rsi_14 \
    --data data/processed/a200/SPY.parquet

# Test multiple features
./venv/bin/python scripts/test_feature_quality.py \
    --features rsi_14,macd_line,bb_width
```

## Output Format

```
================================================================================
Feature Quality Test: rsi_14
================================================================================

DATA QUALITY
  NaN ratio:     0.5%  ✅ PASS (< 5%)
  Inf values:    0.0%  ✅ PASS (= 0%)
  Variance:      142.3 ✅ PASS (> 0)
  Unique values: 847   ✅ PASS (> 10)

SIGNAL QUALITY
  Target correlation: 0.034   ✅ PASS (>= 0.02)
  Mutual information: 0.0015  ✅ PASS (>= 0.001)
  Temporal CV:        0.42    ✅ PASS (< 0.5)
  Precision drop:     0.003   ✅ PASS (>= 0.002)

REDUNDANCY
  Max correlation:    0.78    ✅ PASS (< 0.90)
  Most similar to:    rsi_21

================================================================================
VERDICT: PASS
================================================================================
Feature rsi_14 is safe to add to curated tier.
```

## Threshold Reference

From `src/features/curation/quality.py`:

```python
# Target Correlation (absolute value) - higher is better
TARGET_CORRELATION_PASS = 0.02
TARGET_CORRELATION_MARGINAL = 0.01

# Mutual Information - higher is better
MUTUAL_INFORMATION_PASS = 0.001
MUTUAL_INFORMATION_MARGINAL = 0.0005

# Max Redundancy - lower is better
MAX_REDUNDANCY_PASS = 0.90
MAX_REDUNDANCY_MARGINAL = 0.95

# Temporal Stability CV - lower is better
TEMPORAL_STABILITY_PASS = 0.5
TEMPORAL_STABILITY_MARGINAL = 0.8

# Permutation Precision Drop - higher is better
PRECISION_DROP_PASS = 0.002
PRECISION_DROP_MARGINAL = 0.001
```

## Integration with Tier Development

When developing new tiers (e.g., expanding from a500 to a1000):

1. **Before implementing**: Run quality test on proposed features
2. **Filter candidates**: Only implement PASS features
3. **Handle MARGINAL**: Document reasoning if including
4. **Never include FAIL**: Unless ablation study justifies

## Related Scripts

| Script | Purpose |
|--------|---------|
| `scripts/test_feature_quality.py` | Interactive quality testing |
| `scripts/curate_features.py` | Batch quality analysis |
| `experiments/feature_curation/reduce_a200.py` | a200 reduction experiment |
| `experiments/feature_curation/reduce_a500.py` | a500 reduction experiment |

## Critical Notes

- Data quality checks are NON-NEGOTIABLE blockers
- Signal quality uses the overall verdict logic (FAIL if any FAIL)
- Redundancy check uses existing tier features as reference
- Always run `make test` after adding new features
- Document MARGINAL features in decision_log.md
