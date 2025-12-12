# Phase Tracker

## Phase 0: Development Discipline ✅ COMPLETE (2025-11-26)
- SpecKit + Superpowers installation verified
- Core skills implemented: planning_session, test_first, approval_gate, task_breakdown, session_handoff, session_restore, thermal_management
- Claude/Cursor rules synced (global, experimental protocol, testing, development, context handoff): 2025-12-07

## Phase 1: Environment Setup ✅ COMPLETE (2025-12-08)
- Directory scaffold + Makefile created: ✅ 2025-12-07
- CLAUDE.md + project rules framework: ✅ 2025-12-07
- `.gitignore`: ✅ Present
- Python 3.12 venv + all requirements: ✅ 2025-12-07 (30+ packages installed)
- Test infrastructure functional: ✅ 2025-12-07 (pytest operational)
- Context files populated & maintained: ✅ 2025-12-08
- Verification tooling (`scripts/verify_environment.py` + `make verify`): ✅ 2025-12-08
- Agentic tools documentation: ✅ Present in docs/rules_and_skills_background.md

## Phase 2: Data Pipeline ✅ COMPLETE (2025-12-08)
- Planning session completed: ✅ 2025-12-07
- Plan documented in docs/project_phase_plans.md: ✅ 2025-12-07
- Test plan defined (8 test cases): ✅ 2025-12-08
- TDD cycle (RED→GREEN): ✅ 2025-12-08
- `scripts/download_ohlcv.py` implemented: ✅ 2025-12-08
- Data directories created (raw, processed, samples): ✅ 2025-12-08
- SPY.OHLCV.daily downloaded (8,272 rows, 1993-2025): ✅ 2025-12-08
- Manifest registered with MD5: ✅ 2025-12-08

## Phase 3: Pipeline Design ✅ COMPLETE (2025-12-08)
- Feature engineering implemented: ✅ 2025-12-08 (tier_a20.py with 20 indicators)
- Build script created: ✅ 2025-12-08 (build_features_a20.py)
- Manifest registration integrated: ✅ 2025-12-08
- Training infrastructure decisions: ✅ 2025-12-08 (documented in project_phase_plans.md)
- Config schema defined: ✅ 2025-12-08 (YAML format, target construction rules)
- All tests passing: ✅ 2025-12-08 (17/17 tests)

## Phase 4: Boilerplate ✅ COMPLETE (2025-12-09)
- All 7 sub-tasks completed with TDD
- All tests passing: ✅ 2025-12-09 (94/94 tests)
- 2B parameter config added: ✅ 2025-12-09

## Phase 5: Data Acquisition ✅ COMPLETE (2025-12-10)
- Plan approved: ✅ 2025-12-09 (docs/phase5_data_acquisition_plan.md v1.3)
- Task 1: ✅ Generalize download script (download_ticker + retry logic)
- Task 2: ✅ Download ETFs + Indices (2025-12-10)
  - DIA: 7,018 rows (1998+), QQQ: 6,731 rows (1999+)
  - ^DJI: 8,546 rows (1992+), ^IXIC: 13,829 rows (1971+)
  - Decision: Use indices for extended training history (1992+ vs 1999+)
- Task 3: ✅ Download VIX (2025-12-10)
  - ^VIX: 9,053 rows (1990+)
  - Relaxed Volume validation to allow NaN/0 (VIX has no volume)
  - 103 tests passing
- Task 4: ✅ Generalize feature pipeline (2025-12-10)
  - Added --ticker CLI arg to build_features_a20.py
  - Dynamic path/dataset name construction
  - 110 tests passing
- Task 5: ✅ Build DIA/QQQ features (2025-12-10)
  - DIA: 6,819 rows, QQQ: 6,532 rows
  - Fixed date normalization bug in load_raw_data()
  - 116 tests passing
- Task 6: ✅ VIX feature engineering (2025-12-10)
  - 8 VIX features: close, sma_10, sma_20, percentile_60d, zscore_20d, regime, change_1d, change_5d
  - VIX_features_c.parquet: 8,994 rows
  - 131 tests passing
- Task 7: ✅ Combined dataset builder (2025-12-10)
  - VIX integration: --include-vix flag, --vix-path parameter
  - SPY_dataset_c.parquet: 8,073 rows, 34 columns (Date + 5 OHLCV + 20 ind + 8 VIX)
  - Date overlap validation with clear error messages
  - 136 tests passing
- Task 8: ⏸️ Multi-asset builder (optional stretch goal)

## Phase 5.5: Experiment Setup ✅ COMPLETE (2025-12-11)
**Plan Document:** `docs/phase5_5_experiment_setup_plan.md`
**Total Estimate:** 10-14 hours
**Execution:** One task per session, sequential with approval gates

| Task | Name | Est. | Status | Deliverables |
|------|------|------|--------|--------------|
| 5.5.1 | Config Templates | 30 min | ✅ COMPLETE | configs/experiments/threshold_{2,3,5}pct.yaml |
| 5.5.2 | Timescale Resampling | 2-3 hrs | ✅ COMPLETE | src/features/resample.py, CLI, 10 tests |
| 5.5.3 | Data Dictionary | 1-2 hrs | ✅ COMPLETE | docs/data_dictionary.md, generator script, 9 tests |
| 5.5.4 | Optuna HPO | 3-4 hrs | ✅ COMPLETE | src/training/hpo.py, run_hpo.py, 18 tests |
| 5.5.5 | Scaling Analysis | 2 hrs | ✅ COMPLETE | src/analysis/scaling_curves.py, 26 tests |
| 5.5.6 | Result Aggregation | 1-2 hrs | ✅ COMPLETE | src/analysis/aggregate_results.py, 8 tests |

**Dependencies:** 5.5.1 → 5.5.2 → 5.5.3 → 5.5.4 → 5.5.5 → 5.5.6

**Memory Entities:** Phase5_5_Plan, Phase5_5_Task{1-6}_* (8 entities total)

## Phase 6A Prep: Experiment Skills ✅ COMPLETE (2025-12-11)
**Plan Document:** `docs/experiment_skills_design.md`
**Completed:** 2025-12-11
**Tests:** 239 passing

| Task | Name | Status |
|------|------|--------|
| 1 | Create src/experiments/ module | ✅ COMPLETE |
| 2 | Write tests for runner (TDD RED) | ✅ COMPLETE |
| 3 | Implement runner.py (TDD GREEN) | ✅ COMPLETE |
| 4 | Implement templates.py (TDD) | ✅ COMPLETE |
| 5 | Create experiment-generation skill | ✅ COMPLETE |
| 6 | Create experiment-execution skill | ✅ COMPLETE |
| 7 | Manual test end-to-end | ✅ COMPLETE |

**Key Decisions:**
- Thin wrapper scripts (~50-80 lines) with all params visible
- Dynamic data assembly (no pre-built datasets)
- Per-budget HPO: 12 runs (skip 2% task, borrow params)
- Hybrid logging: append-only CSV + regenerated markdown

**Implementation Complete:**
- `src/experiments/runner.py`: 4 functions (~310 lines)
- `src/experiments/templates.py`: 2 functions (~240 lines)
- `.claude/skills/experiment_generation/SKILL.md`: 228 lines (Task 5)
- `.claude/skills/experiment_execution/SKILL.md`: 329 lines (Task 6)

## Phase 6A: Parameter Scaling 🔄 IN PROGRESS
- 12 HPO runs: 4 scales × 3 horizons (testing horizon variance)
- Hold: 25 features, SPY
- Vary: 2M → 20M → 200M → 2B parameters × 1d/3d/5d horizons
- Research: Does error ∝ N^(-α)? Do optimal params vary by horizon?

**Status (2025-12-11):**
- ✅ Fixed feature pipeline integration issues (vix_regime encoding, non-numeric column filtering)
- ✅ Config file created: `configs/experiments/threshold_1pct.yaml`
- ✅ **CRITICAL FIX: Implemented train/val/test data splits (commit 0e9ec1b)**
  - ChunkSplitter class for hybrid splits (val/test=chunks, train=sliding window)
  - HPO now optimizes val_loss instead of train_loss
  - 264 tests passing
- ✅ HPO validation test passed (3 trials, 2M/1-day, ~50s/trial)
- ✅ **12 HPO scripts generated (commit 93e43b8)**
  - `experiments/phase6a/hpo_{2M,20M,200M,2B}_h{1,3,5}_threshold_1pct.py`
  - Consistent naming convention: `hpo_{budget}_h{horizon}_{task}.py`
- ✅ **Runbook created: `docs/phase6a_hpo_runbook.md`**
- ⚠️ **CRITICAL DISCOVERY (2025-12-11): HPO only searched training params, not architecture!**
  - Original HPO varied: lr, epochs, weight_decay, warmup_steps, dropout
  - Missing: d_model, n_layers, n_heads, d_ff (architecture search)
  - This is essential for scaling law research - must find best arch per budget
- ✅ **Architectural HPO redesign complete**
  - Design doc: `docs/architectural_hpo_design.md`
  - Implementation plan: `docs/architectural_hpo_implementation_plan.md`
  - 8 tasks, 6-10 hours estimated
  - Pre-compute valid arch grid per budget, search arch + training params
- ✅ **Task 1 COMPLETE: Architecture Grid Generator (2025-12-11)**
  - `src/models/arch_grid.py`: estimate_param_count(), generate_architecture_grid(), filter_by_budget(), get_architectures_for_budget()
  - `tests/test_arch_grid.py`: 28 tests, all passing
  - Param estimation matches actual model within 0.1%
  - 292 total tests passing
- ✅ **Task 2 COMPLETE: Architectural Search Config (2025-12-11)**
  - `configs/hpo/architectural_search.yaml`: narrow training param ranges
  - 11 new tests in test_hpo.py validating config structure/values
  - 303 total tests passing
- ✅ **Task 3 COMPLETE: Architectural Objective Function (2025-12-11)**
  - `create_architectural_objective()` in hpo.py (~80 lines)
  - Samples arch from pre-computed list + training params from narrow ranges
  - `save_best_params()` now includes architecture info when available
  - Fixed `_sample_hyperparameter()` categorical handling
  - 6 new tests, 309 total tests passing
- ✅ **Task 4 COMPLETE: Runner CSV Architecture Columns (2025-12-12)**
  - Added 5 columns to EXPERIMENT_LOG_COLUMNS: d_model, n_layers, n_heads, d_ff, param_count
  - Backwards compatible: missing arch fields auto-set to None
  - 3 new tests, 312 total tests passing
- ✅ **Task 5 COMPLETE: Template Update for Architectural HPO (2025-12-12)**
  - Rewrote `generate_hpo_script()` in templates.py (~127 lines changed)
  - Generated scripts now: import arch_grid, compute architectures, use create_architectural_objective()
  - Self-contained scripts with visible architecture search
  - 5 new tests, 317 total tests passing
- ✅ **Task 6 COMPLETE: Regenerate 12 HPO Scripts (2025-12-12)**
  - Deleted old scripts (training-only HPO)
  - Regenerated 12 scripts with architectural HPO template
  - All scripts use get_architectures_for_budget(), create_architectural_objective()
  - All scripts reference configs/hpo/architectural_search.yaml
  - 317 tests still passing
- ✅ **Task 7 COMPLETE: Update Runbook (2025-12-12)**
  - Updated docs/phase6a_hpo_runbook.md for architectural HPO
  - Sections updated: Overview, CLI Output, Outputs, Analyzing Results, Next Steps
  - Added new section: "Interpreting Architectural Results"
  - Document: 429 lines (+173/-44 from original)
  - 317 tests still passing
- 🔜 **Next: Task 8 (integration test), then re-run experiments**

## Phase 6B: Horizon Scaling ⏸️ NOT STARTED
- 64 runs (reuse 6A HPO)
- Hold: 20 features, SPY
- Add: 2-day, 3-day, 5-day, weekly horizons
- Research: Does scaling vary with horizon?

## Phase 6C: Feature × Horizon Scaling ⏸️ NOT STARTED
- 288 runs: 48 HPO + 240 final
- Hold: SPY
- Add: 100, 250, 500 feature tiers
- Full matrix: 3 tiers × 5 horizons × 4 params × 4 tasks

## Phase 6D: Data Scaling ⏸️ GATED
- Proceed after 6A-6C results
- Add: DIA, QQQ, stocks, FRED
- Research: Does data diversity improve scaling?

## Experiment Totals
- Phase 6A-6C: 384 runs
- Estimated runtime: ~1,728 hours (~2-3 months)
