# Workstream 2 Context: foundation
# Last Updated: 2026-02-01 10:00

## Identity
- **ID**: ws2
- **Name**: foundation
- **Focus**: Foundation model & alternative architecture investigation
- **Status**: active - two-phase budget-aware HPO implemented

---

## Current Task
- **Working on**: Two-phase budget-aware HPO strategy implementation
- **Status**: Core implementation COMPLETE, ready for experiments

---

## Progress Summary

### Completed
- [2026-01-28] Methodology correction: v1/v2 invalid (MAE vs Bernoulli mismatch)
- [2026-01-29] v3 HPO: 50 trials iTransformer, 50 trials Informer (Focal Loss)
- [2026-01-30 AM] direction_accuracy bug fix in common.py
- [2026-01-30 AM] Comprehensive HPO audit completed
- [2026-01-30 12:15] Context ablation implementation COMPLETE
- [2026-01-30 15:15] Runner script created: `scripts/run_context_ablation.sh`
- [2026-01-31 08:15] **Context ablation experiments COMPLETE** (10 runs)
- [2026-01-31 22:00] **--data-tier support added** (Task 1 of 6-point validation)
- [2026-02-01] **Two-phase budget-aware HPO strategy IMPLEMENTED**:
  - `src/training/hpo_budget_extremes.py` - 150 lines, BUDGET_CONFIGS, generate_forced_configs, check_early_stopping_convergence
  - `tests/test_hpo_budget_extremes.py` - 30 tests, all passing
  - CLI integration in `hpo_neuralforecast.py` - 6 new flags
  - `docs/hpo_strategy_phase6.md` - comprehensive documentation

### Pending
1. **Run forced extremes HPO** - Execute 18 forced configs + TPE trials
2. **Analyze Phase 1 results** - Identify top 2 budgets for Phase 2
3. **Phase 2 supplementary trials** - Deep dive on top budgets

---

## Last Session Work (2026-02-01 ~10:00)

### Two-Phase Budget-Aware HPO Strategy Implemented

**New Files Created:**
1. `src/training/hpo_budget_extremes.py` (~150 lines)
   - `BUDGET_CONFIGS`: Architecture configs for 750k/2M/20M/200M
   - `DEFAULT_REGULARIZATION`: dropout=0.5, lr=1e-4, wd=1e-3
   - `generate_forced_configs()`: Generates 18 forced extreme configs
   - `check_early_stopping_convergence()`: Early stopping logic
   - `compute_budget_aware_extremes()`: Get config for budget/style
   - `estimate_params()`: Parameter count estimation

2. `tests/test_hpo_budget_extremes.py` (~200 lines)
   - 30 tests covering all functions
   - All tests passing

3. `docs/hpo_strategy_phase6.md` (~100 lines)
   - Strategy overview
   - Phase 1 (18 forced configs) tables
   - Phase 2 supplementary approach
   - Usage examples

**Modified Files:**
4. `experiments/architectures/hpo_neuralforecast.py`
   - Added import for hpo_budget_extremes
   - Added 6 new CLI flags: `--forced-extremes`, `--budgets`, `--early-stop-patience`, `--early-stop-threshold`, `--supplementary`, `--param-budget`
   - Updated `run_hpo()` signature with new parameters

5. `tests/test_hpo_neuralforecast.py`
   - Added 10 new tests for CLI flags and imports
   - Total: 46 tests (all passing)

**Test Results:** 76 HPO-related tests passing

---

### Previous Session (2026-01-31 ~22:10)

### --data-tier Support Implemented (Task 1 of Plan)

**Completed:**
1. Added `DATA_PATH_A200` to `common.py`
2. Added `DATA_PATHS` dict and `get_data_path()` helper
3. Added `--data-tier` CLI argument to `hpo_neuralforecast.py`
4. Added `data_tier` param to `run_hpo()`
5. Updated `prepare_hpo_data()` to accept `data_path` param
6. Added 7 new tests - all pass (36/36 total)

**Data Verification:**
- a200 file: 212 columns, 7977 rows, 0 NaN
- Path: `data/processed/v1/SPY_dataset_a200_combined.parquet`

### Validation Runs (Exposed Methodology Gap)

| Model | Tier | Context | AUC | Issue |
|-------|------|---------|-----|-------|
| iTransformer | a200 | 80d | 0.47 | Random params |
| Informer | a200 | 180d | 0.56 | Random params |

**Root Cause**: `--trials 1` samples random params from search space, not optimal params from prior HPO.

**Prior Best (a20 HPO, itransformer_v2):**
- AUC: 0.62
- Params: dropout=0.6, lr=5e-5, hidden=512, layers=5, heads=8, steps=2000

**Gap**: Random params vs optimal params explains the AUC difference.

---

## Files Owned/Modified
- `experiments/architectures/common.py` - MODIFIED (direction_accuracy fix + data tier support)
- `experiments/architectures/hpo_neuralforecast.py` - MODIFIED (--input-size, --data-tier, **--forced-extremes, --budgets, --early-stop-*, --supplementary, --param-budget**)
- `experiments/architectures/context_ablation_nf.py` - NEW (~350 lines)
- `scripts/run_context_ablation.sh` - NEW (runner script)
- `scripts/audit_hpo_results.py` - NEW
- `src/training/hpo_budget_extremes.py` - **NEW** (~150 lines, budget-aware HPO)
- `tests/test_evaluation.py` - NEW
- `tests/test_context_ablation_nf.py` - NEW (~150 lines)
- `tests/test_hpo_neuralforecast.py` - MODIFIED (+17 tests, **46 total**)
- `tests/test_hpo_budget_extremes.py` - **NEW** (~200 lines, **30 tests**)
- `docs/hpo_strategy_phase6.md` - **NEW** (~100 lines, strategy docs)
- `outputs/hpo/architectures/itransformer_ctx80_a200/` - NEW (validation result)
- `outputs/hpo/architectures/informer_ctx180_a200/` - NEW (validation result)

---

## Key Decisions (Workstream-Specific)

### Validation Methodology Gap (2026-01-31)
- **Issue**: Single-trial validation samples random params, not optimal
- **Options**:
  - A: Run full HPO (50 trials) on a200 - finds a200-specific optima
  - B: Add `--use-best-from <path>` to transfer known-good params - faster validation
- **Recommendation**: Option B for quick validation, then Option A for optimization
- **Status**: User decision pending

---

## Best Configurations (from 80d HPO on a20)

### iTransformer (AUC: 0.620 from itransformer_v2)
| Parameter | Value |
|-----------|-------|
| hidden_size | 512 |
| learning_rate | 5e-5 |
| max_steps | 2000 |
| dropout | 0.6 |
| n_layers | 5 |
| n_heads | 8 |
| batch_size | 64 |
| weight_decay | 0.0 |

### Informer (AUC: 0.574)
| Parameter | Value |
|-----------|-------|
| hidden_size | 256 |
| learning_rate | 1e-4 |
| max_steps | 1000 |
| dropout | 0.4 |
| n_layers | 2 |
| n_heads | 2 |
| focal_gamma | 0.5 |
| focal_alpha | 0.9 |

---

## Context Length Results (COMPLETE)

| Architecture | 60d | 80d | 120d | 180d | 220d | Best |
|--------------|-----|-----|------|------|------|------|
| PatchTST | 0.703 | **0.718** | 0.695 | 0.683 | - | 80d |
| iTransformer | 0.552 | **0.590** | 0.503 | 0.548 | 0.583 | 80d |
| Informer | 0.539 | 0.554 | 0.512 | **0.585** | 0.557 | 180d |

---

## Session History

### 2026-01-31 22:10 (This Session)
- Implemented --data-tier support (TDD, 7 tests)
- Ran a200 validation: iTransformer 0.47, Informer 0.56
- Identified gap: validation used random params, not optimal
- Proposed fix: --use-best-from or full HPO

### 2026-01-31 09:00
- Ran context ablation experiments (10 runs)
- Results: iTransformer best @ 80d (0.590), Informer best @ 180d (0.585)
- Updated all documentation (5 files)
- Created 6-point validation plan for a200 training

### 2026-01-30 15:15
- Created `scripts/run_context_ablation.sh` runner script
- Context lengths: 60, 80, 120, 180, 220 (max 220d due to data constraint)

---

## Next Session Should

### Run Phase 1 Forced Extremes HPO
```bash
# Verify configs (dry run)
./venv/bin/python experiments/architectures/hpo_neuralforecast.py \
    --model itransformer \
    --data-tier a200 \
    --trials 70 \
    --forced-extremes \
    --dry-run

# Full run with early stopping
./venv/bin/python experiments/architectures/hpo_neuralforecast.py \
    --model itransformer \
    --data-tier a200 \
    --trials 70 \
    --forced-extremes \
    --early-stop-patience 20 \
    --early-stop-threshold 0.02
```

### After Phase 1 Completes
1. Analyze results to identify top 2 budgets
2. Run Phase 2 supplementary trials on those budgets
3. Document findings in research paper notes

### Commit Changes
- All new files created this session (uncommitted)
- Update global_context.md summary

---

## Memory Entities (Workstream-Specific)
- `Alternative_Architecture_Methodology_Lesson_20260128` - v1/v2 flaw
- `iTransformer_FocalLoss_Finding_20260129` - smaller model discovery
