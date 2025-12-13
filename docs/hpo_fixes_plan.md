# HPO Fixes Plan

**Created**: 2025-12-13
**Status**: Ready for execution
**Priority**: High - blocks remaining HPO experiments

## Problem Summary

Audit of past HPO runs revealed several issues:

| Issue | Impact | Affected Runs |
|-------|--------|---------------|
| n_heads not in log message | Can't recover n_heads from logs | All 2M, 20M_h1 |
| n_layers grid gaps | L=160, 180, 188 never tested for 20M | All 20M |
| No forced extreme testing | Random sampling misses extremes | All runs |
| 2M_h5 script error | TrialLogger bug at end | 2M_h5 |
| Incomplete trials | Timeout stopped early | 20M_h1 (31), 20M_h3 (32) |

## Approved Changes

### Task 1: Expand n_layers Search Space

**File**: `src/models/arch_grid.py`

**Change**:
```python
# Line 29, change from:
"n_layers": [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],

# To:
"n_layers": [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 160, 180, 192, 256],
```

**Rationale**:
- 20M budget with d=128 supports up to L=188
- Current grid jumps from 128 to 192, missing valid architectures
- Adding 160 and 180 covers the gap without bloating the grid

**Tests**: Run `make test` - arch_grid tests should still pass

---

### Task 2: Add n_heads to Trial Log Message

**File**: `src/experiments/templates.py`

**Location**: In `generate_hpo_script()`, the trial logging section

**Change**: Update the log message format to include n_heads:
```python
# From:
print(f"Trial {trial.number}: arch_idx={arch_idx}, d_model={arch['d_model']}, n_layers={arch['n_layers']}, params={arch['param_count']:,}, lr={lr:.6f}, epochs={epochs}, batch_size={batch_size}")

# To:
print(f"Trial {trial.number}: arch_idx={arch_idx}, d_model={arch['d_model']}, n_layers={arch['n_layers']}, n_heads={arch['n_heads']}, params={arch['param_count']:,}, lr={lr:.6f}, epochs={epochs}, batch_size={batch_size}")
```

**Tests**: Run `make test` - template tests should pass

---

### Task 3: Add Forced Extreme Testing

**File**: `src/experiments/templates.py`

**Location**: In `generate_hpo_script()`, add new function and modify objective

**Design**: First 6 trials force-test extremes before random sampling:

```python
def get_extreme_architecture_indices(architectures: list[dict]) -> list[int]:
    """Get indices of extreme architectures to force-test first.

    Returns 6 indices:
    - min/max d_model (with middle n_heads=8, middle n_layers)
    - min/max n_layers (with middle n_heads=8, middle d_model)
    - min/max n_heads (with middle d_model, middle n_layers)
    """
    # Find middle values
    d_models = sorted(set(a['d_model'] for a in architectures))
    n_layers_vals = sorted(set(a['n_layers'] for a in architectures))
    middle_d = d_models[len(d_models)//2]
    middle_L = n_layers_vals[len(n_layers_vals)//2]
    middle_h = 8  # Always use h=8 as middle

    extreme_indices = []

    # Min d_model (middle h, middle L)
    for i, a in enumerate(architectures):
        if a['d_model'] == min(d_models) and a['n_heads'] == middle_h:
            extreme_indices.append(i)
            break

    # Max d_model (middle h, middle L)
    for i, a in enumerate(architectures):
        if a['d_model'] == max(d_models) and a['n_heads'] == middle_h:
            extreme_indices.append(i)
            break

    # Min n_layers (middle h, middle d)
    for i, a in enumerate(architectures):
        if a['n_layers'] == min(n_layers_vals) and a['n_heads'] == middle_h and a['d_model'] == middle_d:
            extreme_indices.append(i)
            break

    # Max n_layers (middle h, middle d)
    for i, a in enumerate(architectures):
        if a['n_layers'] == max(n_layers_vals) and a['n_heads'] == middle_h:
            extreme_indices.append(i)
            break

    # Min n_heads (h=2, middle d, middle L)
    for i, a in enumerate(architectures):
        if a['n_heads'] == 2 and a['d_model'] == middle_d:
            extreme_indices.append(i)
            break

    # Max n_heads (h=32, middle d, middle L)
    for i, a in enumerate(architectures):
        if a['n_heads'] == 32 and a['d_model'] == middle_d:
            extreme_indices.append(i)
            break

    return extreme_indices

# In objective function:
EXTREME_INDICES = get_extreme_architecture_indices(ARCHITECTURES)

def objective(trial):
    # Force extremes for first 6 trials
    if trial.number < len(EXTREME_INDICES):
        arch_idx = EXTREME_INDICES[trial.number]
    else:
        arch_idx = trial.suggest_int("arch_idx", 0, len(ARCHITECTURES) - 1)
    ...
```

**Tests**: Add test for `get_extreme_architecture_indices()` function

---

### Task 4: Update Log Recovery Script

**File**: `scripts/recover_trial_data.py`

**Change**: Update regex to capture n_heads:
```python
# Line 18-22, change from:
TRIAL_START_PATTERN = re.compile(
    r'\[I (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\] '
    r'Trial (\d+): arch_idx=(\d+), d_model=(\d+), n_layers=(\d+), params=([\d,]+), '
    r'lr=([\d.e-]+), epochs=(\d+), batch_size=(\d+)'
)

# To:
TRIAL_START_PATTERN = re.compile(
    r'\[I (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\] '
    r'Trial (\d+): arch_idx=(\d+), d_model=(\d+), n_layers=(\d+), n_heads=(\d+), params=([\d,]+), '
    r'lr=([\d.e-]+), epochs=(\d+), batch_size=(\d+)'
)

# Also update the parse logic to extract n_heads (around line 90)
```

---

### Task 5: Regenerate HPO Scripts

**Scripts to regenerate** (9 total):
- `experiments/phase6a/hpo_200M_h1_threshold_1pct.py`
- `experiments/phase6a/hpo_200M_h3_threshold_1pct.py`
- `experiments/phase6a/hpo_200M_h5_threshold_1pct.py`
- `experiments/phase6a/hpo_2B_h1_threshold_1pct.py`
- `experiments/phase6a/hpo_2B_h3_threshold_1pct.py`
- `experiments/phase6a/hpo_2B_h5_threshold_1pct.py`
- `experiments/phase6a/hpo_20M_h1_threshold_1pct.py` (for re-run)
- `experiments/phase6a/hpo_20M_h3_threshold_1pct.py` (for re-run)
- `experiments/phase6a/hpo_20M_h5_threshold_1pct.py` (for re-run with expanded grid)

**Command**:
```bash
./venv/bin/python3 -c "
from src.experiments.templates import generate_hpo_script
from pathlib import Path

configs = [
    ('200M', 1), ('200M', 3), ('200M', 5),
    ('2B', 1), ('2B', 3), ('2B', 5),
    ('20M', 1), ('20M', 3), ('20M', 5),
]

for budget, horizon in configs:
    generate_hpo_script(
        budget=budget,
        horizon=horizon,
        task='threshold_1pct',
        output_dir=Path('experiments/phase6a'),
    )
    print(f'Generated: hpo_{budget}_h{horizon}_threshold_1pct.py')
"
```

---

### Task 6: Document Supplemental Test Plan

**File**: `docs/hpo_supplemental_tests.md`

Create documentation for targeted tests to fill gaps in completed experiments:

| Experiment | Gap | Supplemental Tests | Est. Trials |
|------------|-----|-------------------|-------------|
| 2M_h1 | L=64 not tested | Test L=64 with various d_model | 3-5 |
| 2M_h3 | L=64 not tested | Test L=64 with various d_model | 3-5 |
| 2M_h5 | L=64 not tested | Test L=64 with various d_model | 3-5 |
| 20M_h5 | L=160, 180 not in grid | Test new L values after grid expansion | 5-8 |
| All 2M | n_heads not recorded | Re-run recovery with arch lookup | 0 (data fix) |

**Total supplemental trials**: ~20-25 (not full 50-trial re-runs)

---

## Execution Order

1. **Task 1**: Expand n_layers (arch_grid.py) - 5 min
2. **Task 2**: Add n_heads to log (templates.py) - 5 min
3. **Task 3**: Add forced extreme testing (templates.py) - 20 min
4. **Task 4**: Update recovery script - 10 min
5. `make test` - verify all changes
6. **Task 5**: Regenerate 9 HPO scripts - 5 min
7. **Task 6**: Document supplemental plan - 10 min
8. `git add -A && git commit` - commit all changes

**Total estimated time**: ~1 hour

---

## Verification Checklist

- [ ] `make test` passes after Task 1-4
- [ ] New n_layers values appear in architecture grids
- [ ] Generated scripts include n_heads in log message
- [ ] Generated scripts have EXTREME_INDICES and forced testing
- [ ] Recovery script regex updated
- [ ] All 9 scripts regenerated
- [ ] Supplemental test plan documented

---

## Files Modified

| File | Change Type |
|------|-------------|
| `src/models/arch_grid.py` | Edit (1 line) |
| `src/experiments/templates.py` | Edit (~50 lines) |
| `scripts/recover_trial_data.py` | Edit (~10 lines) |
| `experiments/phase6a/hpo_*.py` | Regenerate (9 files) |
| `docs/hpo_supplemental_tests.md` | Create |
| `docs/hpo_fixes_plan.md` | Create (this file) |
