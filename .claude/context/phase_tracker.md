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

## Phase 5.5: Experiment Setup 🔄 NEXT
**Plan Document:** `docs/phase5_5_experiment_setup_plan.md`
**Total Estimate:** 10-14 hours
**Execution:** One task per session, sequential with approval gates

| Task | Name | Est. | Status | Deliverables |
|------|------|------|--------|--------------|
| 5.5.1 | Config Templates | 30 min | ⏸️ PENDING | configs/experiments/threshold_{2,3,5}pct.yaml |
| 5.5.2 | Timescale Resampling | 2-3 hrs | ⏸️ PENDING | src/features/resample.py, CLI |
| 5.5.3 | Data Dictionary | 1-2 hrs | ⏸️ PENDING | docs/data_dictionary.md, generator script |
| 5.5.4 | Optuna HPO | 3-4 hrs | ⏸️ PENDING | src/training/hpo.py, run_hpo.py |
| 5.5.5 | Scaling Analysis | 2 hrs | ⏸️ PENDING | src/analysis/scaling_curves.py |
| 5.5.6 | Result Aggregation | 1-2 hrs | ⏸️ PENDING | src/analysis/aggregate_results.py |

**Dependencies:** 5.5.1 → 5.5.2 → 5.5.3 → 5.5.4 → 5.5.5 → 5.5.6

**Memory Entities:** Phase5_5_Plan, Phase5_5_Task{1-6}_* (7 entities total)

## Phase 6A: Parameter Scaling ⏸️ NOT STARTED
- 32 runs: 16 HPO + 16 final evaluation
- Hold: 20 features, 1-day horizon, SPY
- Vary: 2M → 20M → 200M → 2B parameters
- Research: Does error ∝ N^(-α)?

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
