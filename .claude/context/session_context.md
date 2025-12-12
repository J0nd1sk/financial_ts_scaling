# Session Handoff - 2025-12-12 (Integration Test Bug Fixing)

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `0044e22` docs: update runbook for architectural HPO (Task 7)
- **Uncommitted**: 14 files (12 scripts + templates.py + hpo.py)
- **Status**: Bug fixes in progress, NOT YET TESTED

### Project Phase
- **Phase 6A**: IN PROGRESS - Task 8 (integration test) revealed bugs

### Task Status
- **Working on**: Task 8 integration smoke test
- **Status**: BLOCKED - fixing bugs discovered during test

---

## Critical Bugs Found & Fixed (UNCOMMITTED)

Integration test (Task 8) discovered 3 bugs. All fixes applied but NOT yet tested/verified:

### Bug 1: Wrong Import Path ✅ FIXED
**File**: `src/experiments/templates.py` line 75
```python
# Was:
from src.data.splitter import ChunkSplitter
# Fixed to:
from src.data.dataset import ChunkSplitter
```

### Bug 2: Wrong ChunkSplitter API ✅ FIXED
**File**: `src/experiments/templates.py` lines 131-140
```python
# Was:
splitter = ChunkSplitter(
    val_size=252,
    test_size=252,
    train_window=None,
)
return splitter.split(df)

# Fixed to:
splitter = ChunkSplitter(
    total_days=len(df),
    context_length=60,
    horizon=HORIZON,
    val_ratio=0.15,
    test_ratio=0.15,
)
return splitter.split()
```

### Bug 3: Missing num_features Parameter ✅ FIXED
**File**: `src/training/hpo.py` - `create_architectural_objective()`
```python
# Was:
num_features=experiment_config.num_features  # ExperimentConfig doesn't have this

# Fixed to:
# Added num_features parameter to function signature
# Pass num_features=len(FEATURE_COLUMNS) from template
```

---

## Files Modified (Uncommitted)

| File | Change |
|------|--------|
| `src/experiments/templates.py` | Fixed import + ChunkSplitter API + added num_features param |
| `src/training/hpo.py` | Added num_features parameter to create_architectural_objective |
| `experiments/phase6a/*.py` (12 files) | Regenerated with fixes |

---

## Next Session MUST Do

1. **Run tests**: `make test` to verify fixes don't break anything
2. **Regenerate scripts**: Scripts were regenerated but need fresh regeneration after all fixes
3. **Run smoke test**: 3-trial HPO to validate end-to-end
4. **Commit all fixes**: Single commit with all bug fixes
5. **Push**: Get fixes to origin

---

## Commands for Next Session

```bash
# 1. Verify environment
source venv/bin/activate
make test

# 2. Regenerate all 12 scripts (REQUIRED after template fixes)
PYTHONPATH=. ./venv/bin/python3 << 'SCRIPT'
from pathlib import Path
from src.experiments.templates import generate_hpo_script

BUDGETS = ['2M', '20M', '200M', '2B']
HORIZONS = [1, 3, 5]
EXPERIMENTS_DIR = Path('experiments/phase6a')
DATA_PATH = 'data/processed/v1/SPY_dataset_a25.parquet'
FEATURE_COLUMNS = ['dema_9', 'dema_10', 'sma_12', 'dema_20', 'dema_25', 'sma_50', 'dema_90', 'sma_100', 'sma_200', 'rsi_daily', 'rsi_weekly', 'stochrsi_daily', 'stochrsi_weekly', 'macd_line', 'obv', 'adosc', 'atr_14', 'adx_14', 'bb_percent_b', 'vwap_20']

for budget in BUDGETS:
    for horizon in HORIZONS:
        experiment = f"phase6a_{budget}_h{horizon}_threshold_1pct"
        script = generate_hpo_script(
            experiment=experiment, phase="phase6a", budget=budget, task="threshold_1pct",
            horizon=horizon, timescale="daily", data_path=DATA_PATH, feature_columns=FEATURE_COLUMNS,
        )
        filepath = EXPERIMENTS_DIR / f"hpo_{budget}_h{horizon}_threshold_1pct.py"
        filepath.write_text(script)
print("✓ Regenerated 12 scripts")
SCRIPT

# 3. Run 3-trial smoke test
PYTHONPATH=. ./venv/bin/python3 experiments/phase6a/hpo_2M_h1_threshold_1pct.py
# (modify N_TRIALS to 3 first, or run inline test)

# 4. Commit all
git add -A
git commit -m "fix: correct ChunkSplitter API and add num_features to arch HPO

- Fix import: src.data.splitter -> src.data.dataset
- Fix ChunkSplitter API: use total_days, context_length, horizon, val_ratio
- Add num_features param to create_architectural_objective
- Regenerate all 12 HPO scripts with fixes

Bugs discovered during Task 8 integration test."
```

---

## Test Status
- **Last `make test`**: 317 passed (before bug fixes)
- **Current**: UNKNOWN - tests not run after fixes

---

## Key Context

- GPT-5 identified two gaps: runbook outdated + no integration test
- Task 7 (runbook) was completed and committed
- Task 8 (integration test) revealed 3 bugs in template/hpo code
- Bugs were from Task 5/6 implementation - missed during testing because unit tests don't run actual HPO
- Integration test is essential - catches real runtime issues

---

## Memory Entities

- `Task7_RunbookUpdate_Plan`: Planning and completion of Task 7

---

*Session: 2025-12-12 (context window limit reached during bug fixing)*
