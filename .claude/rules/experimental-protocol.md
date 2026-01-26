# Experimental Protocol Rules

These constraints are non-negotiable. They ensure scientific rigor and reproducibility.

---

## Parameter Budgets

**2M, 20M, 200M ONLY**

- No intermediate values (e.g., no 5M, 10M, 50M)
- Each budget tests a specific scaling hypothesis
- Clean isolation required for valid scaling law analysis

---

## Batch Size Re-Tuning

| Trigger | Requirement |
|---------|-------------|
| Parameter budget changes | **REQUIRED** - Must re-tune |
| Dataset changes | **RECOMMENDED** - Should re-tune |
| Feature count changes | **OPTIONAL** - May re-tune |

Before training at a new parameter budget, run batch size discovery:
```bash
python scripts/find_batch_size.py --param-budget 20M
```

---

## Architecture Constraints

### PatchTST Only
- No alternative architectures during scaling experiments
- Clean isolation for parameter scaling analysis
- Modifications only to scale parameters, not architecture

### One Model Per Task
- 6 tasks × 8 timescales = 48 models per dataset configuration
- No multi-task models
- No shared encoders between tasks

---

## Data Splits (Fixed)

| Split | Date Range |
|-------|------------|
| Training | Through 2022-12-31 |
| Validation | 2023-01-01 to 2024-12-31 |
| Testing | 2025-01-01 onwards |

These splits are fixed. No leakage permitted.

---

## Dataset Matrix

### Rows (Asset Scaling)
- A: SPY only
- B: SPY + DIA + QQQ
- C: + major stocks (AAPL, MSFT, etc.)
- D: + sentiment data
- E: + economic indicators

### Columns (Quality Scaling)
- a: OHLCV + technical indicators
- b: + sentiment
- c: + VIX/volatility
- d: + search trends

---

## Timescales

1. daily
2. 2-day
3. 3-day
4. 5-day
5. weekly
6. 2-week
7. monthly
8. daily + multi-resolution (weekly/monthly indicators as features)

---

## Tasks

### Binary Classification (5)
1. Direction (up/down)
2. Threshold >1%
3. Threshold >2%
4. Threshold >3%
5. Threshold >5%

### Regression (1)
6. Price prediction

---

## Feature Tiers

Approximate targets: 20 → 50 → 100 → 200 → 500 → 1000 → 2000 indicators

Each tier is a superset of the previous. Actual feature counts may differ slightly from targets based on indicator groupings and implementation details. This is acceptable and desirable—scientific validity depends on consistent tier definitions, not exact round numbers.

**Implemented tiers:**
- a20: 20 features (baseline)
- a50: 50 features
- a100: 100 features
- a200: 206 features (6 extra from Chunk 5's Ichimoku/Donchian/Entropy groupings)

---

## Reproducibility Requirements

- Fixed random seeds for all experiments
- Deterministic operations where possible
- Version-pinned dependencies
- Environment snapshots saved with results
- Checksums for all data files

---

## Experiment Naming Convention

```
[param_budget]-[timescale]-[task]

Examples:
2M-daily-direction
20M-weekly-threshold-3pct
200M-monthly-regression
```
