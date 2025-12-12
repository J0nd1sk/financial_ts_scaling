# Architectural HPO Implementation Plan

**Created:** 2025-12-11
**Design Document:** `docs/architectural_hpo_design.md`
**Estimated Total:** 6-10 hours

---

## Overview

This plan implements the architectural HPO design, which adds model architecture search (d_model, n_layers, n_heads, d_ff) to the existing training parameter HPO.

---

## Task Breakdown

### Task 1: Architecture Grid Generator (NEW)
**File:** `src/models/arch_grid.py`
**Est:** 1-2 hours

**Deliverables:**
1. `estimate_param_count()` - Calculate params for any architecture config
2. `generate_architecture_grid()` - Generate all valid (d_model, n_layers, n_heads, d_ff) combos
3. `filter_by_budget()` - Filter grid to architectures within ±25% of target
4. `get_architectures_for_budget()` - Main entry point, returns list of valid arch dicts
5. `ARCH_SEARCH_SPACE` - Module constant with definitive search values

**Tests:** `tests/test_arch_grid.py`
- Test param count estimation matches actual model.parameters()
- Test constraint filtering (n_heads divides d_model)
- Test budget filtering (±25% tolerance)
- Test extremes are included
- Test each budget returns 25-35 architectures

---

### Task 2: Update HPO Search Space Config
**File:** `configs/hpo/architectural_search.yaml` (NEW)
**Est:** 30 min

**Changes:**
- Create new config for architectural HPO
- Define narrow training param ranges (from design doc)
- Keep `default_search.yaml` for backwards compatibility

**New config structure:**
```yaml
n_trials: 50
timeout_hours: null
direction: minimize

# Training params only - architecture comes from arch_grid
training_search_space:
  learning_rate:
    type: log_uniform
    low: 1.0e-4
    high: 1.0e-3
  epochs:
    type: categorical
    choices: [50, 75, 100]
  batch_size:
    type: categorical
    choices: [32, 64, 128, 256]
  weight_decay:
    type: log_uniform
    low: 1.0e-5
    high: 1.0e-3
  warmup_steps:
    type: categorical
    choices: [100, 200, 300, 500]
```

---

### Task 3: Update HPO Module
**File:** `src/training/hpo.py`
**Est:** 2-3 hours

**Changes:**
1. Add `create_architectural_objective()` - New objective that samples architecture + training params
2. Modify objective to:
   - Accept pre-computed architecture list
   - Sample architecture index from list
   - Sample training params from narrow ranges
   - Build PatchTSTConfig dynamically from sampled architecture
   - Return val_loss
3. Update `save_best_params()` to include architecture in output
4. Add logging for architecture params per trial

**Key implementation detail:**
```python
def create_architectural_objective(
    config_path: str,
    budget: str,
    architectures: list[dict],  # Pre-computed valid architectures
    training_search_space: dict,
    split_indices: SplitIndices | None = None,
) -> Callable[[optuna.Trial], float]:
    """Create objective that searches both architecture and training params."""

    def objective(trial: optuna.Trial) -> float:
        # Sample architecture from pre-computed list
        arch_idx = trial.suggest_categorical("arch_idx", list(range(len(architectures))))
        arch = architectures[arch_idx]

        # Sample training params from narrow ranges
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
        epochs = trial.suggest_categorical("epochs", [50, 75, 100])
        # ... etc

        # Build config dynamically
        config = PatchTSTConfig(
            d_model=arch["d_model"],
            n_layers=arch["n_layers"],
            # ... etc
        )

        # Train and return val_loss
        ...
```

**Tests:** Update `tests/test_hpo.py`
- Test architectural objective samples from provided list
- Test training params sampled from narrow ranges
- Test output includes architecture info

---

### Task 4: Update Experiment Runner
**File:** `src/experiments/runner.py`
**Est:** 1 hour

**Changes:**
1. Add architecture columns to CSV logging
2. Update `update_experiment_log()` to handle new columns
3. Ensure backwards compatibility with existing logs

**New CSV columns:**
- `d_model`, `n_layers`, `n_heads`, `d_ff`, `param_count`

---

### Task 5: Update Templates
**File:** `src/experiments/templates.py`
**Est:** 1 hour

**Changes:**
1. Update HPO script template to use architectural objective
2. Add architecture grid generation to script
3. Update logging to show architecture per trial

---

### Task 6: Regenerate HPO Scripts
**Files:** `experiments/phase6a/hpo_*.py` (12 files)
**Est:** 30 min

**Process:**
1. Delete existing 12 scripts
2. Generate new scripts using updated templates
3. Verify scripts use new architectural HPO approach

---

### Task 7: Update Runbook
**File:** `docs/phase6a_hpo_runbook.md`
**Est:** 30 min

**Changes:**
1. Document new architectural HPO approach
2. Update monitoring instructions for architecture logging
3. Update output format documentation
4. Add section on interpreting architectural results

---

### Task 8: Integration Testing
**Est:** 1 hour

**Tests:**
1. Run 3-trial validation on 2M/h1 with new system
2. Verify architecture varies between trials
3. Verify output format includes architecture
4. Verify param counts are within budget

---

## Files Summary

### New Files
| File | Purpose |
|------|---------|
| `src/models/arch_grid.py` | Architecture grid generation |
| `tests/test_arch_grid.py` | Tests for arch grid |
| `configs/hpo/architectural_search.yaml` | New search space config |

### Modified Files
| File | Changes |
|------|---------|
| `src/training/hpo.py` | Add architectural objective |
| `src/experiments/runner.py` | Add architecture columns to logging |
| `src/experiments/templates.py` | Update HPO script template |
| `tests/test_hpo.py` | Add architectural HPO tests |
| `docs/phase6a_hpo_runbook.md` | Update documentation |

### Regenerated Files
| File | Notes |
|------|-------|
| `experiments/phase6a/hpo_*.py` | All 12 HPO scripts regenerated |

---

## Execution Order

```
Task 1 (arch_grid.py + tests)
    ↓
Task 2 (new search config)
    ↓
Task 3 (hpo.py updates + tests)
    ↓
Task 4 (runner.py updates)
    ↓
Task 5 (templates.py updates)
    ↓
Task 6 (regenerate scripts)
    ↓
Task 7 (runbook updates)
    ↓
Task 8 (integration test)
```

Tasks 1-3 are the core implementation. Tasks 4-8 are integration and cleanup.

---

## Rollback Plan

If issues arise:
1. Keep `configs/hpo/default_search.yaml` unchanged (backwards compatible)
2. Old scripts in git history can be restored
3. New `create_architectural_objective()` is additive, doesn't break existing `create_objective()`

---

## Success Criteria

1. `make test` passes with all new tests
2. 3-trial validation run completes successfully
3. Output shows different architectures per trial
4. `best_params.json` includes architecture info
5. All architectures within ±25% of budget
