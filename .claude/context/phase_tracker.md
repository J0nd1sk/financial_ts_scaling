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

**Status (2026-01-01):**
- ✅ **2M HPO Complete** (all 3 horizons, 50 trials each)
  - h1: best=0.3199 (d=64, L=48, h=2)
  - h3: best=0.2630 (d=64, L=32, h=2) ← BEST OVERALL
  - h5: best=0.3371 (d=64, L=64, h=16)
  - Finding: All prefer narrow-deep (d=64), h3 easiest to predict
- ✅ **20M HPO Complete** (all 3 horizons, 50 trials each)
  - h1: best=0.3483 (d=128, L=180, h=16) — worse than 2M
  - h3: best=0.3191 (d=256, L=32, h=2) — worse than 2M
  - h5: best=0.3458 (d=384, L=12, h=4) — worse than 2M
  - Finding: More params did NOT help at this data scale
- ✅ **200M HPO Complete** (all 3 horizons, 50 trials each)
  - h1: best=0.3564 (d=384, L=96, h=4)
  - h3: complete (see outputs)
  - h5: best=0.3612 (d=384, L=180, h=4)
- ✅ **2B HPO Complete** (all 3 horizons, 50 trials each) - 2026-01-10
  - h1: best=0.3609 (d=1024, L=128, h=2)
  - h3: best=0.3948 (d=768, L=256, h=32)
  - h5: best=0.3592 (d=1024, L=180, h=4)
  - 14 diverged trials (val_loss=100.0), all L=256 architectures
  - Finding: 2B scale did NOT improve over smaller models - data-limited regime
- ✅ **HPO Diversity Enhancement** (2026-01-03)
  - Added n_startup_trials=20 to TPESampler
  - Added forced variation logic for same-arch trials
  - Fixed arch_idx=0 falsy bug (0 or x returns x)
  - 4 new tests, 365 total passing
- ✅ **Supplementary 2M Experiments Complete** (10 runs)
  - Tested h3-optimal config on h1/h5
  - Finding: Architecture does NOT transfer across horizons
  - h3-optimal on h5: +132% worse (catastrophic)
  - n_heads tuning has zero effect when depth is wrong

**Historical Status (2025-12-11):**
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
- ✅ **Task 8 COMPLETE: Integration smoke test (2025-12-12)**
  - Fixed 4 bugs: import path, ChunkSplitter API, num_features param, SplitIndices attrs
  - Smoke test passed: 3 trials, ~100s/trial, val_loss=0.385
  - Runner script created: scripts/run_phase6a_hpo.sh
  - Commits: 23b0356 (bug fixes), 6e7363c (runner script)
- ✅ **Hardware Monitoring Enhancement COMPLETE (2025-12-12)**
  - ✅ Task A: Hardware monitoring provider (commit c39753b)
    - psutil + powermetrics for CPU/mem/temp
    - 10 new tests, 327 total tests passing
  - ✅ Task B: HPO template thermal callback (commit b1ed2bc)
    - ThermalCallback in generated scripts
    - Pauses on warning, stops on critical
    - 5 new tests, 332 total tests passing
  - ✅ Task C: Runner pre-flight + monitoring (commit 81205e5)
    - Pre-flight checks (MPS, memory, data file)
    - Background hardware monitor (5-min CSV logging)
    - Trap handler for cleanup
- ✅ **200M HPO Complete (2025-12-21)**
  - 200M_h1: 46/50 trials, best=0.3633 (d=1024, L=12, h=16, 151M)
  - **200M_h3: 50/50** trials, best=0.3081 (d=768, L=24, h=16, 170M) ← BEST OVERALL
  - **200M_h5: 50/50** trials, best=0.3507 (d=256, L=256, h=16, 202M) ← NARROW-DEEP!
  - Key finding: h1/h3 prefer wide-medium, h5 prefers narrow-deep
  - Architecture analysis: h=16 optimal, h=8 underperforms
- 🔄 **2B HPO Starting (2025-12-21)**
  - 2B_h1: Trial 0 in progress (d=768, L=256, 1.8B params - very slow)
  - 2B_h3/h5: Not started
  - Warning: 2B trials taking hours per trial due to model size
- ✅ **Supplementary Trials Generated (2025-12-21)**
  - 9 training scripts in experiments/phase6a_supplementary/
  - Cross-horizon validation: d=768 L=24, d=1024 L=12 on missing horizons
  - New architectures: d=1024 L=16 (201.8M), d=768 L=28 (198.7M)
- ⚠️ **2B HPO Memory Issues (2025-12-26)**
  - 2B_h1 Trial 4 (d=1024, L=256, batch=128) consumed 115GB, swap thrashed for days
  - Root cause: Deep+wide architectures exceed 128GB unified memory
  - Solution: Stage detour for optimization work
- 🔄 **Stage: HPO Time Optimization** (temporary detour, Task 6 of 6)
  - Plan: `docs/hpo_time_optimization_plan.md` (revised 2025-12-27)
  - ✅ Task 1: `get_memory_safe_batch_config()` in arch_grid.py (6 tests)
  - ✅ Task 2: Gradient accumulation in trainer.py (3 tests)
  - ✅ Task 3: Early stopping in trainer.py (5 tests)
  - ✅ Task 4: Wire HPO to use new training features (6 tests, 2025-12-28)
    - Config: removed batch_size, added dropout (0.1-0.3), added early_stopping
    - hpo.py: import arch_grid, sample dropout, use dynamic batch, pass new Trainer params
  - ✅ Task 5: Regenerate 12 HPO scripts + runner 'q' quit (2025-12-28)
    - Regenerated all 12 scripts (auto-pick up new features from hpo.py)
    - Added graceful stop: `touch outputs/logs/STOP_HPO` to stop between experiments
    - Documented in: Memory MCP, runner comments, runbook
  - ✅ **Task 6: Integration smoke test (2B, 3 trials)** — COMPLETE (2025-12-29)
    - 3/3 trials completed successfully before system crash
    - Best: Trial 1 (d=2048, L=32, 1.6B params) val_loss=0.3863
    - All features verified: dynamic batch, gradient accum, early stopping, dropout
  - 361 tests passing
  - **Stage COMPLETE** — all 6 tasks done, code ready for production HPO
- ✅ **Documentation Consolidation COMPLETE** (2025-12-30, commit 245024a)
  - ✅ Task 1: Created `phase6a_implementation_history.md` (277 lines, 20 tasks)
  - ✅ Task 2: Updated `config_architecture.md` to v2.0 (364→561 lines, +54%)
  - ✅ Task 3: Updated `phase6a_execution_plan.md` to v2.0 (155→290 lines, +87%)
  - ✅ Task 4: Deleted 5 stale files (content in history doc)
  - ✅ Task 5: N_TRIALS reverted
  - ✅ Task 6: Committed all changes
- 🔄 **HPO Re-Run Infrastructure** (Tasks 1-5 complete, Task 6 pending)
  - All 12 HPO runs need re-running with new optimized scripts
  - Old runs (2M/20M/200M) lacked: dropout search, dynamic batch, early stopping, gradient accum
  - ✅ Task 1-4: Archived old outputs, reset CSVs (2025-12-30)
  - ✅ Task 5: Updated `docs/phase6a_execution_plan.md` to v2.1 (2025-12-30)
  - ⏳ Task 6: Update runner script comments (optional)
  - Ready to commit and start production HPO
  - 📝 HPO Logging Infrastructure: deferred (plan in `docs/hpo_logging_plan.md`)
- 🔄 **Supplementary Scripts Rewrite** (2025-12-31)
  - All 10 scripts REWRITTEN with correct API (ExperimentConfig, PatchTSTConfig, ChunkSplitter, Trainer)
  - Plan archived: `docs/archive/supplementary_scripts_rewrite_plan.md`
  - ✅ Task A: Fixed runner script pipefail
  - ✅ Task B: Wrote ONE template script (`train_h1_d64_L32_h2_drop010.py`)
  - ✅ Task C: Generated remaining 9 scripts from template
  - ✅ All 10 scripts pass syntax check (py_compile) and parameter verification
  - ⏳ **PENDING**: End-to-end validation not yet run (scripts written but untested)
  - Memory: `Supplementary_Scripts_Rewrite_Plan`, `Lesson_VerifyAPIBeforeGenerating`
- 🔄 **HPO Analysis Stage** (2026-01-10)
  - Plan: `docs/hpo_analysis_data_plan.md`
  - ⏳ Task 1: Implement extraction script `scripts/extract_hpo_analysis.py`
  - ⏳ Task 2: Generate analysis data files
  - ⏳ Task 3: Deep analysis (architecture patterns, training dynamics, horizon effects)
  - ⏳ Task 4: Diverged trials analysis (14 trials, all 2B scale)
- ✅ **Final Training Stage COMPLETE** (2026-01-19)
  - ✅ Task 0: Refresh SPY data (download + features + combined dataset)
  - ✅ Task 1: Add contiguous split mode to ChunkSplitter
  - ✅ Task 2: Interpolate H2 architectures from H1/H3
  - ✅ Task 3: Implement best checkpoint saving in Trainer
  - ✅ Task 4: Create final training script template
  - ✅ Task 5: Generate 16 final training scripts
  - ✅ Task 6: Create runner script with thermal monitoring
  - ✅ Task 7: Bug fix - batch_config key mismatch (commit e7ac3ed)
  - ✅ **ALL 16 MODELS TRAINED** - checkpoints in `outputs/final_training/`
  - Val loss results (NOT test accuracy):
    - h1: 0.232-0.244 (best: 2B)
    - h2-h5: 0.515-0.647 (harder horizons)
- ✅ **Backtest Evaluation Stage COMPLETE** (2026-01-19)
  - ✅ Task 1: Create `scripts/evaluate_final_models.py` (315 lines, 6 tests)
  - ✅ Task 2: Run backtest on 2025 data (256-260 samples per model)
  - ✅ Task 3: Analyze confidence-accuracy relationship
  - **CRITICAL FINDING: Probability Collapse**
    - All models output near-constant probabilities (<1% spread)
    - 2M/h1: [0.518-0.524], 2B/h1: [0.519-0.520]
    - Cannot filter by confidence - no variation exists
    - AUC-ROC 0.53-0.65 (signal exists but compressed)
  - Results: `outputs/results/phase6a_backtest_2025.csv`
  - Analysis: `docs/phase6a_backtest_analysis.md`
- ✅ **Loss Function Investigation Stage** (2026-01-20) - BLOCKED BY VALIDATION SIZE
  - ✅ Task 1: Implement SoftAUCLoss (TDD, 11 tests)
  - ✅ Task 2: Add criterion parameter to Trainer
  - ✅ Task 3: Initial validation - 7.8x spread improvement
  - ✅ Task 4: Test 1 - BCE vs SoftAUC → SoftAUC -5.8% worse (INVALID - 19 sample val)
  - ✅ Task 5: Test 2 - AUC early stopping → Stopped epoch 1 (INVALID - 19 sample val)
  - ⏳ Task 6: Look-ahead bias audit (still valid, doesn't depend on val size)
  - **ROOT CAUSE FOUND**: ChunkSplitter contiguous mode = only 19 val samples
  - **All loss function tests need re-run after validation fix**
- 🔄 **Validation Exploration Stage** (2026-01-20)
  - Plan: `docs/phase6a_validation_exploration_plan.md`
  - Tracker: `.claude/context/phase6a_exploration_tracker.md`
  - ⏳ Phase 1: Validation strategy sweep (4 options: 19/38/500/125 samples)
  - ⏳ Phase 2: Loss function sweep (5 options: BCE/pos_weight/Focal/SoftAUC/combo)
  - ⏳ Phase 3: Early stopping sweep (val_loss vs val_auc)
  - ⏳ Phase 4: Scale validation (2M/20M)
  - ⏳ Full factorial: 40 experiments after sequential (~3-4 hours)
  - **Code needed**: Time-based splitter, rolling splitter, FocalLoss, combined loss
  - **Success criteria**: AUC >0.60, spread >10%, val/test correlation >0.7
- ✅ **Research Paper Analysis Stage** (2026-01-19)
  - ✅ Comprehensive Phase 6A analysis document
  - ✅ Statistical analysis appendix (ANOVA, effect sizes)
  - ✅ Figure data files (5 CSVs)
  - ✅ Table files (3 CSVs + LaTeX)
  - ✅ Discussion draft and conclusions documents
  - Core finding: **Inverse scaling** - 2M outperforms 2B by 21%
- 📝 **Future Research Backlog**
  - Variable-width transformer architectures (user suggestion)
  - Funnel/hourglass/bottleneck designs
  - Deferred to Phase 6B or 6C

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
