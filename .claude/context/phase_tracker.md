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
- Planning session: ✅ 2025-12-08 (Option A: Sequential TDD approved)
- Plan documented: ✅ 2025-12-08 (docs/phase4_boilerplate_plan.md)
- Execution strategy: 7 sub-tasks with individual approval gates
- All tests passing: ✅ 2025-12-09 (94/94 tests)
- 2B parameter config added: ✅ 2025-12-09

### Sub-Tasks
1. ✅ Config System (src/config/experiment.py) - 2025-12-08
2. ✅ Dataset Class (src/data/dataset.py) - 2025-12-08
3. ✅ PatchTST Model & Configs (REVISED 2025-12-08: implement from scratch, not HuggingFace)
   - 3a. ✅ PatchTST Backbone (src/models/patchtst.py) - 2025-12-09
   - 3b. ✅ Parameter Budget Configs (configs/model/patchtst_{2m,20m,200m,2b}.yaml) - 2025-12-09
   - 3c. ✅ Integration Tests (tests/test_patchtst_integration.py) - 2025-12-09
4. ✅ Thermal Callback (src/training/thermal.py) - 2025-12-09
5. ✅ Tracking Integration (src/training/tracking.py) - 2025-12-09
6. ✅ Training Script (scripts/train.py) - 2025-12-09
7. ✅ Batch Size Discovery (scripts/find_batch_size.py) - 2025-12-09

## Phase 5: Data Acquisition 🔄 IN PROGRESS (Started 2025-12-09)
- Plan approved: ✅ 2025-12-09 (docs/phase5_data_acquisition_plan.md v1.2)
- Task 1: ✅ Generalize download script (download_ticker + retry logic)
- Task 2: ⏸️ Download DIA + QQQ
- Task 3: ⏸️ Download VIX
- Task 4: ⏸️ Generalize feature pipeline
- Task 5: ⏸️ Build DIA/QQQ features
- Task 6: ⏸️ VIX feature engineering
- Task 7: ⏸️ Combined dataset builder
- Task 8: ⏸️ Multi-asset builder (optional)

## Phase 5.5: Experiment Setup ⏸️ PROPOSED
- Config templates for 4 threshold tasks
- HPO infrastructure (Optuna + W&B/MLflow)
- Timescale resampling
- Scaling curve analysis tools
- Result aggregation

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
