# HPO Supplemental Test Plan

**Created**: 2025-12-13
**Purpose**: Fill gaps in completed HPO experiments without full re-runs

## Summary of Gaps

| Experiment | Issue | Supplemental Tests | Est. Trials |
|------------|-------|-------------------|-------------|
| 2M_h1/h3/h5 | n_heads not logged in old format | Recovery via arch_idx lookup | 0 (data fix) |
| 2M_h1/h3/h5 | L=64 not in original grid | Need to test L=64 with various d_model | 3-5 each |
| 20M_h1 | Stopped at 31/50 (timeout) | Full re-run with fixed scripts | 50 |
| 20M_h3 | Stopped at 32/50 (timeout) | Full re-run with fixed scripts | 50 |
| 20M_h5 | L=160, 180 not tested | Test new L values | 5-8 |

**Total supplemental trials**: ~20-25 for gap-filling, +100 for 20M re-runs

---

## 1. n_heads Recovery for 2M Experiments

### Issue
The original log format did not include `n_heads` in the trial start message:
```
Trial N: arch_idx=X, d_model=X, n_layers=X, params=X, ...
```

### Solution
Since `arch_idx` is logged and the architecture grid is deterministic, we can recover `n_heads` by looking up the architecture:

```python
from src.models.arch_grid import get_architectures_for_budget

archs = get_architectures_for_budget('2M', num_features=25)

# For each trial with arch_idx, get n_heads:
n_heads = archs[trial['arch_idx']]['n_heads']
```

### Action
Update `scripts/recover_trial_data.py` to:
1. Load the architecture grid for the budget
2. For trials with `n_heads=None`, lookup from `arch_idx`
3. Re-generate the JSON files with recovered `n_heads`

**Status**: Can be done without re-running experiments.

---

## 2. L=64 Testing for 2M Experiments

### Issue
The original grid only went down to L=48 for small d_model values. L=64 is now valid for 2M budget with d=64.

### Architecture Gap Analysis
```
2M Budget (1.5M - 2.5M params):
- L=64 with d=64, h=8: ~2.1M params ✓ valid
- L=64 with d=64, h=4: ~2.1M params ✓ valid
- L=64 with d=64, h=2: ~2.1M params ✓ valid
```

### Supplemental Test Plan

For each horizon (h1, h3, h5):

| Trial | d_model | n_layers | n_heads | Expected Params |
|-------|---------|----------|---------|-----------------|
| S1 | 64 | 64 | 8 | ~2.1M |
| S2 | 64 | 64 | 4 | ~2.1M |
| S3 | 64 | 64 | 2 | ~2.1M |

**Total**: 9 supplemental trials (3 per horizon)

### Execution
Create a supplemental script that forces these specific architectures:

```bash
# Create experiments/phase6a/supplemental_2M_L64.py
# Force arch_idx to L=64 configurations
```

---

## 3. 20M Re-runs

### Issue
20M_h1 and 20M_h3 stopped early due to the old 4-hour timeout (now removed).

### Solution
Full re-run from scratch with new scripts that:
1. Have no timeout (`TIMEOUT_HOURS = None`)
2. Include expanded n_layers grid (L=160, 180)
3. Include n_heads in log messages
4. Force extreme testing in first 6 trials

### Timing
Run AFTER all other HPO experiments complete (200M, 2B, 20M_h5).

---

## 4. L=160, 180 Testing for 20M_h5

### Issue
Current 20M_h5 run (49/50 trials) used old grid without L=160, 180.
These values are valid for 20M budget and should be tested.

### Architecture Gap Analysis
```
20M Budget with d=128:
- L=160, d=128, h=8, d_ff=256: ~23.5M params ✓ valid
- L=160, d=128, h=4, d_ff=256: ~23.5M params ✓ valid
- L=180, d=128, h=8, d_ff=256: ~24.9M params ✓ valid (near max)
- L=180, d=128, h=4, d_ff=256: ~24.9M params ✓ valid
```

### Supplemental Test Plan

| Trial | d_model | n_layers | n_heads | d_ff | Expected Params |
|-------|---------|----------|---------|------|-----------------|
| S1 | 128 | 160 | 8 | 256 | ~23.5M |
| S2 | 128 | 160 | 4 | 256 | ~23.5M |
| S3 | 128 | 160 | 8 | 512 | ~25.0M |
| S4 | 128 | 180 | 8 | 256 | ~24.9M |
| S5 | 128 | 180 | 4 | 256 | ~24.9M |

**Total**: 5 supplemental trials

### Execution Option A: Quick Supplemental
Create script that only tests these 5 specific architectures with best learning rate from main run.

### Execution Option B: Partial Re-run
Since 20M_h5 when re-run will include L=160, 180 via forced extremes, we can skip supplemental and rely on the re-run.

**Recommendation**: Use Option B - when we re-run 20M_h5 with the new scripts (after 200M/2B complete), the forced extreme testing will cover these cases.

---

## Execution Priority

1. **Immediate** (no experiment time needed):
   - Update recovery script to lookup n_heads from arch_idx
   - Regenerate 2M JSON files with recovered n_heads

2. **After 20M_h5 completes** (~10 min):
   - Start 200M_h1 with new scripts

3. **After 200M/2B HPO complete** (days/weeks):
   - Re-run 20M_h1, 20M_h3 with new scripts
   - Optionally re-run 20M_h5 to get L=160, 180 coverage

4. **Optional - if needed for publication**:
   - Run L=64 supplemental tests for 2M experiments
   - Create dedicated supplemental script

---

## Verification Checklist

After fixes and supplemental tests:

- [ ] All 2M trials have n_heads recovered/logged
- [ ] 2M experiments tested L=64 configurations
- [ ] 20M_h1/h3 re-run with 50 complete trials each
- [ ] 20M experiments tested L=160, 180 configurations
- [ ] All experiments have extreme testing in first 6 trials
- [ ] All log messages include n_heads for recovery script compatibility

---

## Notes

- Supplemental tests are NOT full 50-trial re-runs
- Focus on filling specific architecture gaps
- Main HPO provided good coverage; supplements target edge cases
- For scaling law analysis, having coverage across the full architecture space is more important than maximizing trial count
