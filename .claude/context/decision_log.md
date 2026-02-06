# Decision Log

## 2025-11-26 PatchTST-Only Scaling Architecture

**Context**: Need a single architecture to isolate scaling-law effects across parameter budgets.

**Decision**: Use Hugging Face PatchTST as the exclusive model for all experiments; scale only via parameter count and feature/data dimensions.

**Rationale**: PatchTST has proven TS performance, clean parameterization, and avoids architectural confounds.

**Alternatives Considered**:
- Lag-Llama or other sequence models — rejected for higher complexity and lack of mature tooling.
- Multi-architecture comparison — rejected until baseline scaling behavior is established.

**Implications**: All training/evaluation code must target PatchTST; any architectural change requires a new approval gate.

## 2025-11-26 Fixed Dataset Matrix & Splits

**Context**: Scaling experiments require consistent data ranges and dataset definitions.

**Decision**: Lock train/val/test splits to ≤2020 / 2021-2022 / 2023+, and adopt the 5×4 asset-quality dataset matrix (Aa..Ed) with defined feature tiers.

**Rationale**: Prevents leakage, keeps comparisons fair, and enables reproducible scaling curves.

**Alternatives Considered**:
- Rolling or adaptive splits — rejected for added variance.
- Ad-hoc dataset definitions — rejected due to poor reproducibility.

**Implications**: All data work must respect the matrix and date ranges; deviations require documented approval.

## 2025-12-07 Context File Enforcement

**Context**: Rules review showed `.claude/context` artifacts existed but lacked explicit usage requirements.

**Decision**: Added mandatory read/write conditions for `session_context.md`, `phase_tracker.md`, and `decision_log.md` across both Claude and Cursor rule sets.

**Rationale**: Ensure every session start, end, phase change, and architectural decision is captured, keeping restore/handoff workflows reliable.

**Alternatives Considered**:
- Rely on skills alone — rejected because Cursor lacks Superpowers automation.
- Update docs only — rejected since enforcement belongs in active rules.

**Implications**: Agents must now update context files whenever the defined triggers occur; omissions are treated as process violations.

## 2025-12-08 Data Version Manifest System

**Context**: Need reproducible tracking of raw/processed datasets before Phase 2 pipeline work.

**Decision**: Introduced JSON manifests (`data/raw/manifest.json`, `data/processed/manifest.json`) plus `scripts/manage_data_versions.py` and `make verify` integration to register artifacts with MD5 checksums.

**Rationale**: Guarantees provenance for every dataset, enables automated verification, and ensures session handoffs capture latest data state.

**Alternatives Considered**:
- Git LFS or DVC — rejected for added tooling overhead at this stage.
- Manual spreadsheets — rejected due to high risk of drift and poor automation.

**Implications**: All future data downloads/processing steps must register manifest entries; handoff/restore summaries must note latest entries; verification failures block merges.

## 2025-12-08 Dataset Naming Convention

**Context**: Registering first SPY download in manifest required choosing a dataset identifier format.

**Decision**: Adopt hierarchical naming: `{TICKER}.{DATA_TYPE}.{FREQUENCY}` (e.g., `SPY.OHLCV.daily`).

**Rationale**: Enables future expansion to multiple data types (fundamentals, options, sentiment) and frequencies (1min, hourly, weekly) while maintaining clear provenance.

**Alternatives Considered**:
- Simple ticker name (`SPY`) — rejected as ambiguous when multiple data types exist.
- Filename-based (`SPY.parquet`) — rejected as it conflates storage with logical identity.

**Implications**: All manifest registrations must follow this convention; scripts should validate format on registration.

## 2025-12-08 Phase 3 Pipeline Design Complete

**Context**: Review of phase tracker vs actual implementation revealed Phase 3 is complete but not marked.

**Evidence**:
- Feature engineering implemented: tier_a20.py with 20 TA-Lib indicators (DEMA, SMA, RSI, MACD, Bollinger Bands, etc.)
- Build script: build_features_a20.py with automatic manifest registration
- Tests: 17/17 passing, including feature calculation tests
- Training infrastructure: Config schema, target construction rules, and tracking decisions documented in project_phase_plans.md

**Deliverables**:
- `src/features/tier_a20.py` - 20 feature calculations
- `scripts/build_features_a20.py` - Build script with manifest integration
- `tests/features/test_indicators.py` - Feature tests
- Training infrastructure decisions in project_phase_plans.md

**Next Phase**: Phase 4 (Boilerplate) - Implement training infrastructure (PatchTST configs, train.py, config loader, dataset, batch size discovery, W&B/MLflow integration)

## 2025-12-08 Testing & Manifest Automation Lessons

**Context**: Phase 2 implementation had three critical issues identified by code review:
1. Tests called real yfinance API (slow, flaky, non-reproducible)
2. Each test downloaded full SPY history independently (8× redundant downloads)
3. Manifest registration was manual, not automatic in download script

**Decision**: Establish mandatory patterns:
- **Data download tests must use mocks/fixtures**: Never call real APIs in tests; mock external dependencies for speed and reproducibility
- **Shared test fixtures**: Use pytest fixtures to download/mock data once, reuse across tests
- **Automatic manifest registration**: Download scripts must register artifacts programmatically, not rely on manual post-download steps

**Rationale**: Tests must be fast, deterministic, and offline-capable per PRD reproducibility requirements. Manual registration steps violate automation principles and invite manifest drift.

**Alternatives Considered**:
- Live API tests in separate suite — adds complexity, still flaky
- Optional manifest registration flag — easy to forget, defeats automation purpose

**Implications**:
- All future data-fetching code must include mocking strategy in test plan
- Download scripts must integrate manifest registration before merge
- CI/CD must run offline without external API dependencies

## 2025-12-08 Phase 4 Boilerplate Planning Approved

**Context**: Phase 3 complete, ready to implement training infrastructure. Needed structured plan for complex multi-component implementation.

**Decision**: Adopt Option A (Sequential TDD) with 7 independent sub-tasks, each with its own approval gate.

**Sub-Tasks Defined**:
1. Config System - YAML loader with dataclass validation
2. Dataset Class - PyTorch Dataset with binary threshold target generation
3. Model Configs - PatchTST JSON configs for 2M/20M/200M params
4. Thermal Callback - Temperature monitoring with powermetrics
5. Tracking Integration - W&B + MLflow dual logging
6. Training Script - Main training loop integrating all components
7. Batch Size Discovery - Automated batch size finding per budget

**Rationale**:
- ~1,400 lines across 20 files requires decomposition
- Sequential TDD ensures each component is solid before integration
- Individual approval gates prevent scope creep
- Parallel execution possible for Tasks 1, 3, 4, 5 (no dependencies)

**Alternatives Considered**:
- Option B (Grouped Implementation) - Faster but higher risk, 3 approval gates
- Option C (Single Task with Checkpoints) - Most efficient but less granular control

**Key Design Decisions**:
- Target construction: `future_max = max(close[t+1:t+horizon])`, label = 1 if exceeds threshold
- Config format: Plain YAML with dataclass validation (no OmegaConf)
- Thermal thresholds: warn ≥85°C, stop ≥95°C
- Tracking: Both W&B (dashboards) and MLflow (local artifacts)

**Implications**:
- Each sub-task follows full TDD cycle (RED → GREEN → REFACTOR)
- Plan documented in docs/phase4_boilerplate_plan.md
- Estimated 17-25 hours total implementation time
- Must build processed features before training (run build_features_a20.py)

## 2025-12-08 Phase 4 Plan Review Improvements (v1.1)

**Context**: GPT5-Codex reviewed `docs/phase4_boilerplate_plan.md` and identified 7 gaps related to reproducibility, testing, and experimental protocol compliance.

**Decision**: Accept all 7 review recommendations and update plan document.

**Changes Made**:
1. **Config format standardization**: Changed model configs from JSON to YAML to avoid dual schema/loaders
2. **Reproducibility section added**: Seed handling (config field, torch seeds, dataloader generator), determinism notes for MPS
3. **Dataset edge case tests added**: NaN handling, short sequence rejection, warmup exclusion tests
4. **Parameter counting helper**: Added `src/models/utils.py` with `count_parameters()` for budget verification
5. **Manifest verification**: Training script pre-flight check validates manifest entries and MD5 before training
6. **Testing environment assumptions**: Clarified W&B disabled mode, MLflow file:// backend, powermetrics mocking
7. **Data version logging**: Training logs MD5 hash to trackers for reproducibility audit trail

**Rationale**: Strengthens experimental protocol compliance, ensures reproducibility for scaling law research, and prevents CI failures from credential/platform dependencies.

**Implications**:
- Model config files now `.yaml` instead of `.json`
- All Task 2-6 test counts increased
- Training script has additional pre-flight verification step
- Tests run fully offline without external service dependencies

## 2025-12-08 PatchTST From-Scratch Implementation

**Context**: Task 3 (Model Configs) originally assumed HuggingFace transformers library for PatchTST. However, `requirements.txt` had a comment indicating intent to implement from scratch, and user confirmed this preference.

**Decision**: Implement PatchTST architecture from scratch using pure PyTorch instead of HuggingFace transformers.

**Rationale**:
- Full control over architecture and modifications
- Minimal dependencies (only torch, no transformers library)
- Educational value - understand the architecture deeply
- Avoids potential MPS compatibility issues with HuggingFace
- Cleaner integration with our training pipeline

**Trade-offs**:
- More code to write and test (~300 lines model, ~150 lines tests)
- Risk of implementation bugs vs well-tested HuggingFace
- No pretrained weights available (not needed for scaling experiments)

**Scope Change**:
Task 3 expanded into 3 sub-tasks:
- 3a: PatchTST Backbone Implementation (`src/models/patchtst.py`)
- 3b: Parameter Budget Configs (`configs/model/patchtst_*.yaml`)
- 3c: Integration Tests

**Architecture Components**:
1. PatchEmbedding: Split time series into patches, project to d_model
2. PositionalEncoding: Learnable position embeddings
3. TransformerEncoder: Self-attention + FFN layers
4. PredictionHead: Linear projection for binary classification

**Implications**:
- No HuggingFace transformers dependency needed
- Must verify MPS compatibility for all PyTorch ops (attention, etc.)
- Parameter counting must be implemented in `src/models/utils.py`
- Plan document updated: `docs/phase4_boilerplate_plan.md` (assumption #3 changed)

## 2025-12-10 Index Tickers for Extended Training History

**Context**: Phase 5 Task 2 downloaded DIA (1998) and QQQ (1999) ETFs, but this limited training data to 1999+ when used as features with SPY (1993). User wanted to train as far back as possible using DJIA and NASDAQ price data as features.

**Decision**: Use index tickers (^DJI, ^IXIC) instead of ETF tickers (DIA, QQQ) for feature data to extend training history to 1992.

**Data Comparison**:
| Ticker | Type | Start Date | Rows |
|--------|------|------------|------|
| DIA | ETF | 1998-01-20 | 7,018 |
| QQQ | ETF | 1999-03-10 | 6,731 |
| ^DJI | Index | 1992-01-02 | 8,546 |
| ^IXIC | Index | 1971-02-05 | 13,829 |

**Training Window Impact**:
- With ETFs only: 1999+ (~26 years)
- With indices: 1992+ (~33 years)
- **+7 years of training data recovered**

**Rationale**:
- Index data provides longer history than ETFs (ETFs are relatively recent financial instruments)
- All indices have OHLCV data suitable for feature engineering
- ^DJI (1992) is the limiting factor; ^IXIC goes back to 1971
- SPY remains the prediction target (1993+)

**Implementation Details**:
- Added `_sanitize_ticker_for_filename()` to handle ^ in index tickers
- ^DJI → DJI.parquet, ^IXIC → IXIC.parquet
- Manifest uses sanitized names: DJI.OHLCV.daily, IXIC.OHLCV.daily
- yfinance API called with original ticker (^DJI) for correct data retrieval

**Alternatives Considered**:
- FRED data sources (SP500, NASDAQCOM) — deferred; yfinance sufficient for now
- ^GSPC (S&P 500 index back to 1927) — available if deeper history needed later
- MeasuringWorth for pre-1990 Dow — rejected as unnecessary complexity

**Implications**:
- Feature engineering must handle both ETF and index data
- Training date range now limited by ^DJI start (1992-01-02) when using all indices
- DIA and QQQ ETF data retained for potential ETF-specific experiments
- Future work may add ^GSPC for even longer S&P 500 history

## 2025-12-10 Phase 5.5 Selected as Next Phase

**Context**: Phase 5 Task 7 (VIX integration) completed. Decision needed on next phase: Task 8 (multi-asset, optional), Phase 5.5 (experiment setup), or Phase 6A (experiments).

**Decision**: Proceed to Phase 5.5 (Experiment Setup) before Phase 6A experiments.

**Rationale**:
- Phase 6A requires 32 runs (16 HPO + 16 final evaluation)
- HPO infrastructure (Optuna) not yet integrated with Trainer
- Config templates needed for all 4 threshold tasks
- Timescale resampling not yet implemented
- Scaling curve analysis tools needed for publication-quality results

**Scope for Phase 5.5**:
1. Config templates for 4 threshold tasks
2. Optuna HPO integration with Trainer
3. Timescale resampling (daily → 2d, 3d, 5d, weekly, 2wk, monthly)
4. Scaling curve analysis tools (power law fitting)
5. Result aggregation

**Implications**:
- Task 8 (multi-asset builder) deferred as optional stretch goal
- Phase 5.5 infrastructure required before any Phase 6 experiments
- Estimated 5 sub-tasks following same TDD pattern as Phase 4/5

## 2025-12-10 Memory MCP API Correction

**Context**: Session restore consistently returned empty Memory results despite handoff storing data.

**Investigation**: Both session_handoff and session_restore skills referenced non-existent Memory MCP functions (`mcp__memory__store_memory`, `mcp__memory__search_memory`).

**Decision**: Update both skills to use correct Memory MCP API and add explicit entity tracking.

**Changes**:
- session_handoff: Use `create_entities`/`add_observations` instead of `store_memory`
- session_handoff: Add "Memory Entities Updated" section to session_context.md template
- session_restore: Use `open_nodes` with entity names from context file instead of generic search
- Established entity naming convention: `Phase[N]_[Topic]_[Type]`

**Rationale**: Context file becomes reliable index of Memory entities, enabling deterministic retrieval via `open_nodes` rather than unreliable keyword search.

**Implications**:
- Future handoffs must list Memory entity names in session_context.md
- Future restores use `open_nodes` with those exact names
- More reliable cross-session knowledge transfer

## 2025-12-11 Hybrid Chunk-Based Data Splits

**Context**: HPO was running on ALL data with no train/val/test splits. This violated experimental protocol (lines 46-55 in `.claude/rules/experimental-protocol.md`) and would produce scientifically invalid results.

**Problem Identified**:
- Trainer created ONE dataloader from entire dataset
- HPO optimized train_loss on full data (no val_loss)
- No held-out test set for final evaluation
- User asked: "are you testing/doing HPO on the full data available or only a subset?"

**Decision**: Implement hybrid chunk-based splits instead of pure chronological splits.

**Design**:
- Val/Test: Non-overlapping chunks of 61 days (context_length + horizon), randomly assigned
- Train: Sliding window on remaining data (maximizes samples)
- Constraint: No training sample's `[context, target]` can overlap any val/test chunk
- Split ratio: 70% train, 15% val, 15% test
- HPO uses 30% of train samples for faster iteration

**Sample Counts**:
| Approach | Train | Val | Test |
|----------|-------|-----|------|
| Pure non-overlapping | 92 | 20 | 20 |
| **Hybrid (approved)** | ~3,157 | 20 | 20 |

**Rationale**:
- User rejected pure chronological splits (train pre-2020, val 2021-22, test 2023+) as regime-dependent
- Random chunk assignment gives exposure to different market conditions across all splits
- Sliding window for train maximizes training data (~34x more samples)
- Non-overlapping val/test ensures strict isolation and valid held-out evaluation

**Alternatives Considered**:
- Pure chronological splits — rejected by user as regime-dependent
- All non-overlapping chunks — produces too few training samples (92)
- Sliding window for all splits — rejected due to data leakage risk

**Implications**:
- Must implement ChunkSplitter class in dataset.py
- Trainer needs train_loader and val_loader, must compute val_loss
- HPO objective must return val_loss instead of train_loss
- Skills and templates must include split validation
- Experimental protocol documentation needs update

## 2025-12-10 Phase 5.5 Plan Approved

**Context**: Phase 5 complete, ready for experiment infrastructure. Planning session held to define Phase 5.5 scope.

**Decision**: Adopt 6-task sequential plan for Phase 5.5 (Experiment Setup).

**Tasks Defined**:
1. **5.5.1 Config Templates** (30 min): Create threshold_2pct, threshold_3pct, threshold_5pct YAML configs
2. **5.5.2 Timescale Resampling** (2-3 hrs): OHLCV resampling to 2d/3d/5d/weekly
3. **5.5.3 Data Dictionary** (1-2 hrs): Comprehensive docs/data_dictionary.md with schema + stats
4. **5.5.4 Optuna HPO** (3-4 hrs): HPO integration with W&B/MLflow tracking
5. **5.5.5 Scaling Analysis** (2 hrs): Power law fitting, scaling curve plots
6. **5.5.6 Result Aggregation** (1-2 hrs): Collect and export experiment results

**Total Estimated Effort**: 10-14 hours

**Key Design Decisions**:
- Data Dictionary: Auto-generated via script, includes dtype, statistics (mean/std/min/max/percentiles)
- HPO: Optuna with pruning, thermal checks between trials
- Scaling Analysis: Fit `error = a * params^(-alpha)`, compute R² for fit quality
- Execution: One task per session, sequential with approval gates

**Documentation**:
- Plan: `docs/phase5_5_experiment_setup_plan.md`
- Memory: 7 entities created (Phase5_5_Plan + 6 task specs)

**Implications**:
- Any coding agent can pick up a task by reading the plan document
- Sequential dependencies: 5.5.1 → 5.5.2 → 5.5.3 → 5.5.4 → 5.5.5 → 5.5.6
- After Phase 5.5, ready for Phase 6A (Parameter Scaling Baseline)

## 2025-12-11 OHLCV as Core Training Data

**Context**: Documentation incorrectly stated OHLCV should be excluded from features. User clarified core data architecture.

**Decision**: OHLCV (Open, High, Low, Close, Volume) is CORE training data that must ALWAYS be included. Indicators/features are ADDITIONAL and will expand during feature scaling tests.

**Rationale**: If there was nothing else to train on, OHLCV alone would be the training data. Indicators enhance but do not replace core price data.

**Implementation**:
- Code was already correct: `NON_FEATURE_COLUMNS = {"Date"}` - only Date excluded
- Fixed incorrect documentation in `docs/feature_pipeline_integration_issues.md`
- Fixed misleading test comment in `tests/test_training.py`

**Feature Counts by Phase**:
- **Phase 6A-6C**: 25 features = 5 OHLCV + 20 indicators (use `SPY_dataset_a25.parquet`)
- **Phase 6D (data scaling)**: 33 features = 5 OHLCV + 20 indicators + 8 VIX (use `SPY_dataset_c.parquet`)

**Implications**:
- All experiments train on OHLCV + indicators
- VIX features added only in data scaling phase (6D)
- Feature scaling will EXPAND indicator count, never remove OHLCV
- Model `num_features` must match dataset being used

## 2025-12-11 Experiment Execution Workflow Clarification

**Context**: Clarifying how HPO and training experiments should be monitored and executed.

**Decision**: Agent monitors first run of each model budget only. After validation, user runs remaining scripts manually.

**Workflow**:
1. **First HPO run per budget**: Agent runs in background, checks every 30 minutes
2. **Subsequent runs**: User runs manually, agent provides script list and instructions
3. **Results**: Output to CLI, logs, AND `docs/experiment_results.csv`
4. **No timeout limits**: Different budgets take different times (2B could take days)

**Runtime Estimates** (for planning monitoring cadence):
- 2M: ~30 min per full training, HPO with 30% subset ~15-20 min/trial
- 20M: ~1.5 hr per full training
- 200M: ~4 hr per full training
- 2B: ~12 hr per full training

**Implications**:
- Remove hardcoded TIMEOUT_HOURS from HPO scripts
- Results CSV stored in `docs/` for versioning
- Agent provides complete script execution list for each phase

## 2025-12-11 Horizon Testing Strategy for Phase 6A

**Context**: Planning session for first HPO run. User confirmed testing 1-day, 3-day, and 5-day horizons to determine if optimal hyperparameters vary significantly by prediction horizon.

**Decision**: Implement staged execution with horizon variance testing before full HPO matrix.

**Stage Breakdown**:
1. **Stage 1 (Validation)**: Run 2-3 trial HPO to validate pipeline works end-to-end
2. **Stage 2 (Horizon Variance)**: Run full HPO for 2M budget across all 3 horizons (1d, 3d, 5d)
3. **Stage 3 (Full Matrix)**: Based on Stage 2 results:
   - If params vary >20% by horizon: Run separate HPO per horizon (36 runs)
   - If params similar: Borrow params across horizons (12 runs)
4. **Stage 4 (Training)**: Final training with best params

**Rationale**:
- Testing horizon variance early informs whether we need 3x the HPO runs for future phases
- 2M budget is cheapest (~12-15 hrs per HPO), best for exploratory testing
- Could save 200+ hours of HPO if params don't vary significantly

**Key Decision Point**: After Stage 2, compare `best_params` across horizons:
- If learning_rate, batch_size, or d_model differ >20% → separate HPO per horizon
- If similar → borrow params (significant time savings for Phase 6B/6C)

**Implications**:
- Execution plan updated in `docs/phase6a_execution_plan.md`
- Config files default to horizon=1 (varies by script)
- Need to generate 3-day and 5-day variant scripts after Stage 1 validates

## 2025-12-13 No Timeout Policy

**Context**: HPO experiments were stopping early due to `timeout_hours=4.0` default. 20M_h1 stopped at 31/50 trials.

**Decision**: Remove ALL timeout defaults - experiments run to completion, even if they take days/weeks.

**Changes Made**:
- `run_hpo()`: `timeout_hours: float | None = None`
- `run_hpo_experiment()`: `timeout_hours: float | None = None`
- All 12 generated HPO scripts: `TIMEOUT_HOURS = None`
- Fixed arithmetic: `timeout_hours * 3600 if timeout_hours else None`

**Rationale**: User explicitly stated experiments should run to completion. Scaling law research needs complete datasets, not truncated ones.

**Implications**: HPO runs may take extended periods. Monitor via `ps aux | grep hpo_`.

## 2025-12-13 Re-run 20M_h1/h3 from Scratch (Option C)

**Context**: 20M_h1 (31/50) and 20M_h3 (32/50) stopped early due to old 4-hour timeout. Options considered:
- Option A: Resume existing Optuna study
- Option B: Run supplemental trials only (19 + 18)
- Option C: Re-run from scratch with new scripts

**Decision**: Re-run both from scratch AFTER all other HPO runs complete (20M_h5, 200M, 2B).

**Rationale**:
1. Old runs used architecture grid with max L=48
2. New scripts have L=128 max for 20M budget (added L=64, 96, 128)
3. For scaling law research, need to test if deeper architectures help at 20M scale
4. Would need to re-run anyway to test expanded architecture space

**Timing**: After 20M_h5, 200M_h1/h3/h5, and 2B_h1/h3/h5 complete.

**Implications**:
- Discards 63 trials of partial work
- Gains access to deeper architectures (L=64, 96, 128) for 20M budget
- Total HPO queue: 20M_h5 → 200M (3) → 2B (3) → 20M_h1 → 20M_h3

## 2025-12-13 HPO Architecture Search Fixes

**Context**: Audit of past HPO runs revealed several issues:
1. n_heads not in log message (can't recover from logs)
2. n_layers grid gap: L=128 → L=192 misses valid 20M architectures (max L=188)
3. No forced extreme testing (random sampling misses extremes)
4. 2M_h5 had script error (TrialLogger bug)

**Decision**: Implement 6-task fix plan before running 200M/2B experiments.

**Changes Planned**:
1. Add n_layers [160, 180] to arch_grid.py (covers 20M gap)
2. Add n_heads to trial log message in templates.py
3. Add forced extreme testing (first 6 trials test min/max of d_model, n_layers, n_heads)
4. Update recover_trial_data.py regex for n_heads
5. Regenerate 9 HPO scripts (200M×3, 2B×3, 20M×3)
6. Document supplemental test plan for gaps

**Key Design - Extreme Testing**:
- When testing d_model extremes: use middle n_heads (h=8), middle n_layers
- When testing n_layers extremes: use middle n_heads (h=8), middle d_model
- When testing n_heads extremes: use middle d_model, middle n_layers

**h=64 Analysis**: NOT reasonable - d_head would be 1-4 for most d_model values (BERT/GPT use d_head=64)

**Supplemental Tests**: ~20-25 targeted trials to fill gaps in completed experiments, not full 50-trial re-runs

**Plan Document**: `docs/hpo_fixes_plan.md`

**Implications**:
- Future HPO scripts will force-test extremes first
- 20M grid will include L=160, 180 (in addition to 128)
- Supplemental tests needed for 2M (L=64 gap) and 20M_h5 (new L values)

## 2025-12-13 HPO DATA_PATH Fix

**Context**: HPO batch run failed - 7 of 12 experiments failed due to bugs introduced when scripts were regenerated mid-run.

**Root Causes Identified**:
1. **200M/2B instant failures**: `DATA_PATH` missing `v1/` subdirectory (FileNotFoundError)
2. **2M_h5 runtime failure**: Old TrialLogger bug (since fixed in regenerated scripts)

**Fix Applied**: Added `v1/` to DATA_PATH in 9 scripts (20M×3, 200M×3, 2B×3).

**Commit**: `70cce96` - fix: correct DATA_PATH in HPO scripts

**Lesson Learned**: When regenerating scripts, always verify `data_path` parameter matches actual file location (`data/processed/v1/...`).

## 2025-12-13 Full Re-run Strategy (Option A)

**Context**: Completed 2M experiments (h1, h3, h5) have gaps:
- L=64 architectures not tested (original grid max was L=48 for small d_model)
- No forced extreme testing (first 6 trials testing min/max of d_model, n_layers, n_heads)
- n_heads not logged in original format (recoverable via arch_idx lookup)

**Options Considered**:
- **Option A**: Full re-runs of all 2M experiments with new scripts
- **Option B**: Targeted supplemental tests for missing configurations only

**Decision**: Option A - Full re-runs for all experiments lacking forced extremes and complete grid coverage.

**Rationale**:
1. 2M experiments are fast (~2-4 hours each)
2. Clean, consistent methodology across all experiments
3. Simpler data aggregation for publication
4. New scripts have all improvements: forced extremes, expanded grid, n_heads logging, no timeout

**Experiments Requiring Re-run**:

| Experiment | Reason | Priority |
|------------|--------|----------|
| 2M_h1 | No forced extremes, L=64 gap | After 200M/2B |
| 2M_h3 | No forced extremes, L=64 gap | After 200M/2B |
| 2M_h5 | No forced extremes, L=64 gap, TrialLogger bug | After 200M/2B |
| 20M_h1 | Stopped at 31/50, no forced extremes | After 200M/2B |
| 20M_h3 | Stopped at 32/50, no forced extremes | After 200M/2B |

**Total HPO Queue** (in order):
1. 200M_h1, 200M_h3, 200M_h5 (new runs)
2. 2B_h1, 2B_h3, 2B_h5 (new runs)
3. 2M_h1, 2M_h3, 2M_h5 (re-runs)
4. 20M_h1, 20M_h3 (re-runs)

**Implications**:
- 20M_h5 does NOT need re-run (completed 50/50, will get L=160/180 coverage via forced extremes if re-run desired later)
- Discard partial 2M results; fresh runs will have complete coverage
- Supplemental scripts (supplemental_2M_L64.py) NOT needed

## 2025-12-13 20M_h5 Must Also Re-run (Verification Finding)

**Context**: Pre-run verification discovered that 20M_h5 (marked as "complete" with 50 trials) actually ran BEFORE forced extremes was implemented.

**Evidence**:
- 20M_h5 trial 0 started: `2025-12-13 09:44:32`
- Forced extremes commit (`7163c7d`): `2025-12-13 14:45:53`
- Trial 0 used `arch_idx=56` (random) instead of expected extreme `idx=12`
- Expected extreme for trial 0: d=128, L=128, h=8 (deep narrow architecture)

**Decision**: Add 20M_h5 to re-run list. All 12 HPO experiments will now run.

**Rationale**: Without forced extreme testing, we cannot guarantee the scaling law analysis tested boundary conditions. Consistent methodology across all experiments is essential for publication.

**Updated Queue** (12 total):
1. 200M_h1, 200M_h3, 200M_h5 (new)
2. 2B_h1, 2B_h3, 2B_h5 (new)
3. 2M_h1, 2M_h3, 2M_h5 (re-run)
4. 20M_h1, 20M_h3, 20M_h5 (re-run)

**Lesson Learned**: Always verify experiment outputs used current code features before marking complete. Check trial 0 architecture against expected forced extreme.

## 2025-12-13 Batch Size Config Increase

**Context**: HPO queue running on M4 MacBook Pro (128GB unified memory). Hardware severely underutilized - 90% memory free, fans not spinning.

**Decision**: Update batch_size choices from `[32, 64, 128, 256]` to `[64, 128, 256, 512]`.

**Rationale**:
- Hardware underutilized (90% free memory) suggested larger batches could improve throughput
- Dropped 32 (too small for modern hardware), added 512 (pushes utilization higher)
- Kept 64 as minimum for generalization safety (user preference)

**Alternatives Considered**:
- `[128, 256, 512, 768]` — rejected; user preferred safer option keeping 64 for generalization
- `[128, 256, 512, 1024]` — rejected; too aggressive, risk of generalization degradation

**Trade-offs**:
- Larger batches: faster epochs, better GPU utilization
- Smaller batches: better generalization (gradient noise helps escape sharp minima)
- HPO samples all options and selects by val_loss, so impact is self-correcting

**Impact**: Experiment 1 (200M_h1) uses old config; experiments 2-12 use new config.

## 2025-12-14 CRITICAL: Architecture Parameter Logging Gap

**Context**: During HPO progress review, discovered that `_best.json` and `all_trials.json` files were missing architecture parameters (d_model, n_layers, n_heads, d_ff, param_count) for most trials, despite these being the PRIMARY parameters of interest for scaling law research.

**Problem Identified**:
1. **`_best.json`** contained ONLY training params (learning_rate, epochs, batch_size, weight_decay, warmup_steps)
2. **Architecture params completely absent** from best params output
3. **Individual trial files** (`trial_XXXX.json`) DID have architecture in `user_attrs.architecture`
4. **Aggregation logic failed** to extract from `user_attrs`

**Root Cause Analysis**:
```python
# In save_best_params() at hpo.py:647-649
if architectures is not None and "arch_idx" in study.best_params:
    arch_idx = study.best_params["arch_idx"]
    result["architecture"] = architectures[arch_idx]
```

**The Bug**: For forced extreme trials (0-9), `arch_idx` is stored via `set_user_attr()`, NOT `trial.suggest_*()`. This means:
- `arch_idx` is in `trial.user_attrs`, NOT in `trial.params`
- The condition `"arch_idx" in study.best_params` is FALSE for forced trials
- Architecture info never gets included in output files

**Why This Matters**:
- **Scaling law research depends on architecture params** - they determine parameter count
- Training params (lr, epochs) are secondary - they optimize within an architecture
- A "best params" file without architecture is scientifically useless
- We know Trial 0 won (val_loss=0.3756) but `_best.json` doesn't tell us WHY (d=256, L=192)

**Evidence from HPO Output**:
```json
// _best.json (WRONG - missing architecture)
{
  "best_params": {
    "learning_rate": 0.000158,
    "epochs": 75,
    "batch_size": 64,
    "weight_decay": 5.23e-05,
    "warmup_steps": 300
  }
}

// trial_0000.json (CORRECT - has architecture in user_attrs)
{
  "user_attrs": {
    "architecture": {
      "d_model": 256,
      "n_layers": 192,
      "n_heads": 8,
      "d_ff": 1024,
      "param_count": 151706881
    }
  }
}
```

**Required Fix**:
```python
# Check user_attrs first (where we store it), then fall back to architectures list
best_trial = study.best_trial
if "architecture" in best_trial.user_attrs:
    result["architecture"] = best_trial.user_attrs["architecture"]
elif architectures is not None and "arch_idx" in study.best_params:
    arch_idx = study.best_params["arch_idx"]
    result["architecture"] = architectures[arch_idx]
```

**Files Requiring Updates**:
1. `src/training/hpo.py` - `save_best_params()` function
2. `src/training/hpo.py` - all_trials export logic (same issue)
3. `experiments/phase6a/*.py` - regenerate after fix

**Preventive Measures**:
1. **Test requirement**: Add test verifying `_best.json` contains architecture for forced extreme trials
2. **Code review checklist item**: "Do output files contain ALL scientifically relevant parameters?"
3. **Template review**: When generating scripts, verify logging captures primary research variables
4. **Hierarchy rule**: Architecture params > Training params in importance for scaling research

**Lessons Learned**:
1. **Verify outputs, not just execution**: Code ran successfully but outputs were incomplete
2. **Test the full pipeline**: Unit tests passed but integration output was wrong
3. **Primary vs secondary params**: Always log PRIMARY research variables first, verify they're captured
4. **user_attrs vs params**: Optuna stores `set_user_attr` differently from `suggest_*` - must handle both

**Impact Assessment**:
- Individual trial files ARE complete (architecture in user_attrs)
- Best/summary files are INCOMPLETE (missing architecture)
- Data is recoverable from trial files
- Fix required before any publication or formal analysis

**Status**: Fix identified, awaiting implementation in next session.

## 2025-12-14 HPO Output Postprocessor Script

**Context**: Running HPO process has old code in memory before architecture logging bug fix. Output files (_best.json, all_trials.json) missing architecture info.

**Decision**: Create postprocessor script rather than restart HPO or modify running process.

**Rationale**: 
- Individual trial files DO have architecture in user_attrs
- Restarting HPO would lose 12+ trials of compute
- Postprocessor is safe, idempotent, and reusable

**Implementation**: `scripts/postprocess_hpo_output.py` (~180 lines)
- Reads architecture from trial JSON files
- Regenerates _best.json and all_trials.json with architecture
- Creates backups before overwriting

**Tests Added**: 4 tests in `TestPostprocessHpoOutput` class

## 2025-12-14 n_heads Parameter Finding

**Context**: Question about whether n_heads affects parameter count or training time.

**Finding**: n_heads does NOT affect parameter count.

**Evidence**: Architecture grid shows identical param counts for same d_model/n_layers:
```
d=384, L=192, h=2,  params=227,412,865
d=384, L=192, h=8,  params=227,412,865
d=384, L=192, h=32, params=227,412,865
```

**Explanation**: In multi-head attention, each head has dimension d_k = d_model/n_heads, but total Q,K,V projections remain (d_model × d_model) matrices. Heads partition dimensions, don't add new parameters.

**Training impact**: Minimal - similar FLOPs, slightly different parallelization patterns.

**Implication**: n_heads is purely an architectural design choice for attention pattern learning, not a scaling parameter.

## 2025-12-14 HPO Trial 13 New Best Result

**Context**: During HPO monitoring, Trial 13 surpassed Trial 0 as best architecture.

**Finding**: Mid-depth architecture (d=768, L=32, h=2) achieving val_loss=0.3673, beating deep narrow (d=256, L=192, h=8) at 0.3756.

**Implication**: Initial hypothesis that "deeper is better" may be wrong. Need more trials to confirm pattern.

**Action**: Continue monitoring; record as potential pattern shift in scaling behavior.

## 2025-12-14 Trial 22 New Best - Deep Narrow Validated

**Context**: Trial 22 surpassed Trial 13 as best architecture, reversing the earlier pattern shift.

**Finding**: Deep narrow (d=384, L=192, h=32, batch=128) at val_loss=0.3670 beats mid-depth (d=768, L=32, h=2) at 0.3673.

**Key Insights**:
1. **Batch size is dominant factor**: batch=128 avg=0.3808 vs batch=32 avg=0.4089
2. **Deep architectures win** when paired with proper batch size
3. **High heads (h=32) work well** with batch=128 (Trial 17 failure was batch=32, not h=32)

**Implication**: Deep narrow + high heads + batch=128 is currently winning formula for 200M budget.

## 2025-12-14 200M Architecture Optimization Study Plan

**Context**: User wants systematic exploration of head count across two architecture families.

**Decision**: Test h ∈ {8, 12, 16, 24, 32} for both architecture families with batch=128.

**Architecture Families**:
1. **Mid-depth wide**: d=768, L=32 (~227M params)
2. **Deep narrow**: d=384, L=192 (~227M params)

**HPO Grid Coverage**:
- In grid (may get tested): h=8, h=16, h=32
- Manual tests required: h=12, h=24 (not powers of 2)

**Manual Tests Queued** (4 total):
1. d=768, L=32, h=12, batch=128
2. d=768, L=32, h=24, batch=128
3. d=384, L=192, h=12, batch=128
4. d=384, L=192, h=24, batch=128

**Research Question**: Does optimal head count differ between deep-narrow vs mid-wide architectures?

**Memory Entities**: Phase6A_User_Hypothesis_d768_L32_h12, Phase6A_User_Hypothesis_d384_L192_heads, Phase6A_200M_Optimization_Study

## 2025-12-16 d_ff Ratio Discovery - Two Architecture Variants

**Context**: User proposed d=768, L=36, h=32 expecting 151M params. Investigation revealed mismatch in understanding.

**Finding**: The HPO grid includes TWO d_ff ratios for d=768:
- d_ff = 3072 (4x d_model): d=768, L=32 → 227M params
- d_ff = 1536 (2x d_model): d=768, L=32 → **151M params**

**Implication for User's Hypothesis**: With d_ff=1536 (2x ratio), the 200M budget allows:
- L=40 → 189M ✅
- **L=44 → 208M ✅ (optimal)**
- L=48 → 227M ❌

**Optimal config for user's intent**: `d=768, L=44, h=32, d_ff=1536, batch=512` (~208M params)

**Note**: User also mentioned batch=512 which is outside current HPO range [64, 128, 256, 512]. This would require manual testing.

## 2025-12-16 Parallel HPO Strategy - Separate 2M Runner

**Context**: Hardware severely underutilized (83% CPU idle, ~8GB RAM free). Current runner script queues all 12 experiments sequentially including 2M which could run in parallel with larger models.

**Decision**: Create separate `run_phase6a_2M.sh` runner for 2M experiments to enable parallel execution.

**Rationale**:
1. 2M models use ~50x less GPU memory than 200M
2. Can run alongside 200M/2B HPO without significant contention
3. Better hardware utilization
4. Reduces total experiment time

**Implementation**:
- Created `scripts/run_phase6a_2M.sh` - runs only 2M_h{1,3,5}
- User will cancel main runner, regenerate without 2M experiments
- 2M runner can execute in separate tmux window

**Trade-offs**:
- Some GPU memory contention possible (monitor required)
- Slightly more complex orchestration
- But: significant time savings from parallelization

**Memory Entity**: Phase6A_Parallel_HPO_Strategy

## 2025-12-17 NEW BEST: Wide-Shallow Architecture (d=1024, L=12)

**Context**: HPO h=3 experiment Trial 23 achieved val_loss=0.3120, significantly beating all previous results.

**Winning Architecture**:
| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| n_layers | **12** (shallow!) |
| n_heads | 16 |
| d_ff | 4096 (4x ratio) |
| param_count | 151,446,529 |
| **batch_size** | **512** |
| val_loss | **0.3120** |

**Paradigm Shift**: Previous hypothesis favored deep-narrow architectures (d=384, L=192 or d=256, L=256). This result shows **wide-shallow (d=1024, L=12) with batch_size=512** dramatically outperforms deep architectures.

**Comparison**:
| Architecture Style | d_model | L | Best val_loss |
|-------------------|---------|---|---------------|
| **Wide-Shallow (NEW)** | **1024** | **12** | **0.3120** |
| Deep-Narrow | 256 | 256 | 0.3644 |
| Mid-Wide | 768 | 32 | 0.3673 |

**Implications**:
1. batch_size=512 validated as effective (user hypothesis correct)
2. Shallow models (L=12) may be optimal for financial time series
3. Wide d_model (1024) with standard 4x d_ff ratio preferred
4. Previous deep-narrow investigation may have been wrong direction
5. Should test d=1024, L=12, h=16, batch=512 on h=1 and h=5 tasks

**Memory Entity**: Phase6A_Wide_Shallow_Discovery

## 2025-12-17 User Hypothesis: d=1024, L=16 Optimal

**Context**: Based on Trial 23's success with d=1024, L=12, user hypothesizes that slightly more layers (L=16-18) and different head count (h≈18) might be optimal.

**Analysis**:

| Config | Params | Budget Status |
|--------|--------|---------------|
| d=1024, L=12 | 151M | ⚠️ Under |
| d=1024, L=14 | 176M | ⚠️ Under |
| **d=1024, L=16** | **201M** | **✅ Perfect fit** |
| d=1024, L=18 | 227M | ❌ Over |

**Head Count Constraint**: h=18 is invalid for d=1024 (must divide evenly). Valid options: h=8, 16, 32, 64.

**Recommended Test Config**: `d=1024, L=16, h=16, d_ff=4096, batch=512` (~201M params)

**Rationale**:
1. L=16 maximizes layers within 200M budget for d=1024
2. h=16 matches current best (Trial 23)
3. batch=512 validated by Trial 23's success
4. Tests whether additional 4 layers (L=12→L=16) improve performance

**Alternative**: Test h=32 variant to explore higher head counts

**Status**: Queued for manual testing after HPO completes

**Memory Entity**: Phase6A_User_Hypothesis_d1024_L16

## 2025-12-22 Batch Size Minimum for GPU Utilization

**Context**: During 2B HPO, discovered GPU was only 14% utilized despite running large models. Benchmarking revealed MPS only outperforms CPU for matrix operations >= 4000x4000.

**Decision**: Set minimum batch_size to 256 (128 acceptable) for all HPO and training runs.

**Rationale**: With batch_size=64 and seq_len=60, effective matrix size is ~3840x768, below the MPS crossover point. GPU was doing almost no work (130mW power vs expected 20-40W).

**Alternatives Considered**:
- Keep small batches for regularization — rejected; overfitting is dominated by params>>data ratio, not batch size
- Mixed batch strategy — rejected; simplicity of fixed minimum is preferred

**Implications**: HPO search space must be updated; may need to increase dropout/weight_decay to compensate for reduced gradient noise.

**Memory Entity**: MPS_GPU_Utilization_Finding

## 2025-12-22 Research Paper Documentation Structure

**Context**: Need to compile evidence and notes for eventual research paper as experiments progress.

**Decision**: Create `docs/research_paper/` subdirectory with appendices/, notes/, figures/ structure. Use Markdown with LaTeX math, convert via pandoc for final submission.

**Rationale**: Captures learnings in real-time rather than reconstructing later. Markdown is easy to write and version control.

**Alternatives Considered**:
- Direct LaTeX — rejected; too heavyweight for iterative note-taking
- Flat docs/ structure — rejected; research paper materials are distinct from project documentation

**Implications**: CLAUDE.md updated with exception to no-subfolders rule. Feature dictionaries (CSV) and notes accumulated as we go.

**Memory Entity**: Research_Paper_Documentation_Plan

## 2025-12-22 Feature Scaling Hypothesis

**Context**: Phase 6A shows no scaling benefit from 200M→2B with only 20 features. User hypothesizes that feature richness (20→2000 indicators) will unlock scaling laws.

**Decision**: Prioritize Phase 6C (feature scaling) as potentially more important than Phase 6D (sample scaling) for demonstrating scaling laws.

**Rationale**: With 2000 features, ~2M pairwise relationships exist (vs ~190 with 20 features). Transformers excel at learning relationships, so more features = more for larger models to learn. Effective_Data ≈ Samples × Features × Relationship_Complexity.

**Alternatives Considered**:
- Focus on sample scaling (6D) first — still valid, but feature scaling may be more impactful
- Skip to combined scaling — rejected; need isolated evidence for each dimension

**Implications**: Research narrative centers on "feature richness as prerequisite for scaling laws" — a publishable contribution challenging naive "just scale up" approaches.

**Memory Entity**: Research_Narrative_Core_Thesis

## 2025-12-26 HPO Time Optimization Plan

**Context**: 2B HPO trials failing due to memory exhaustion. Trial 4 (d=1024, L=256, batch=128) consumed 115GB memory, triggered swap thrashing, and stalled for 2+ days with 0% CPU.

**Decision**: Implement 4-pronged optimization: (1) dynamic batch sizing based on architecture, (2) gradient accumulation to maintain effective batch size, (3) early stopping with patience=10, (4) higher regularization (dropout, weight_decay).

**Rationale**:
- Deep+wide architectures (d≥1024, L≥192) with batch≥64 exceed 128GB unified memory
- Gradient accumulation allows smaller physical batches while maintaining training dynamics
- Early stopping saves ~33% time by avoiding overtraining
- Higher regularization compensates for large-batch training

**Alternatives Considered**:
- Skip 2B entirely — rejected; user wants complete scaling curve evidence
- Gradient checkpointing — deferred; more complex, dynamic batching should suffice
- Filter architecture grid — partial; helps but doesn't solve core issue

**Implementation**: 8-task plan documented in `docs/hpo_time_optimization_plan.md`

**Memory Entity**: Phase6A_HPO_Time_Optimization_Plan

## 2025-12-27 Memory-Safe Batch Config Implementation

**Context**: Task 1 of HPO time optimization plan - needed function to determine safe batch sizes based on architecture.

**Decision**: Implement `get_memory_safe_batch_config()` in `src/models/arch_grid.py` using memory score heuristic.

**Implementation**:
```python
memory_score = (d_model ** 2) * n_layers / 1e9
# Thresholds: ≤0.1→256, ≤0.5→128, ≤1.5→64, ≤3.0→32, >3.0→16
```

**Rationale**: Memory pressure scales approximately with d_model² × n_layers (attention + FFN activations). Normalized to ~1.0 for d=1024, L=256 (known problematic config).

**Tests**: 6 new tests in `TestGetMemorySafeBatchConfig` class (348 → 354 total, then reduced to 351 after consolidation).

## 2025-12-27 Gradient Accumulation Implementation

**Context**: Task 2 of HPO time optimization plan - needed to maintain effective batch size while using smaller physical batches.

**Decision**: Add `accumulation_steps` parameter to Trainer, modify `_train_epoch` to accumulate gradients.

**Implementation**:
- Zero gradients once at epoch start
- Scale loss by accumulation_steps for proper averaging
- Call optimizer.step() every N batches
- Handle leftover batches at end of epoch

**Rationale**: Gradient accumulation is a well-established technique that provides equivalent training dynamics to large batches while using less memory per forward pass.

**Tests**: 3 new tests in `TestGradientAccumulation` class (351 total tests passing).

## 2025-12-27 Project Terminology Convention

**Context**: Confusion between phases, stages, and tasks in documentation and session context. HPO time optimization work was being conflated with a separate phase rather than a stage within Phase 6A.

**Decision**: Establish clear hierarchy: Phase > Stage > Task > Subtask.

**Definitions**:
- **Phase**: Major project milestone (defined in `phase_tracker.md`)
- **Stage**: Focused work block within a phase (may have temporary plan doc)
- **Task**: Discrete deliverable with tests
- **Subtask**: Atomic step within a task

**Rationale**: Prevents scope creep and confusion. HPO Time Optimization is a stage (temporary detour) within Phase 6A, not a new phase. Once the 8 tasks complete, Phase 6A resumes.

**Implications**:
- CLAUDE.md updated with terminology section
- Stage plan docs in `docs/` are temporary - deleted when stage completes
- Don't create new phases for detour work

## 2025-12-29 Documentation Consolidation Approach

**Context**: Audit revealed 8 completed implementation plans sitting as standalone files in docs/. Initial proposal was to delete them as obsolete.

**Decision**: Consolidate completed plans into a single historical record rather than delete. Create `docs/phase6a_implementation_history.md` that preserves the "what we did and why" context.

**Files to consolidate**:
- `hpo_fixes_plan.md` (6 tasks complete)
- `hpo_time_optimization_plan.md` (6 tasks complete)
- `architectural_hpo_implementation_plan.md` (8 tasks complete)
- `hpo_supplemental_tests.md` (strategy changed)
- `config_architecture.md` (merge into design doc)
- `timeseries_transformer_experimentation_project.md` (early draft)
- `feature_pipeline_integration_issues.md` (fixed issues)
- `rules_and_skills_background.md` (background context)

**Rationale**: User preference - preserving historical context allows future sessions to understand not just the current state but the journey. "What we did and why" is valuable for:
1. Debugging if issues resurface
2. Understanding design decisions
3. Onboarding future agents or collaborators
4. Research paper methodology section

**Alternatives Considered**:
- Pure deletion - rejected; loses valuable context
- Keep all files as-is - rejected; fragmented and confusing
- Archive to `.archive/` folder - rejected; still fragmented

**Implications**:
- Create comprehensive history document before deleting originals
- Include timeline, rationale, and outcomes for each implementation
- Reference Memory MCP entities for additional context

## 2025-12-29 User Preferences Durability Protocol

**Context**: Session handoff inadvertently reduced fidelity of user preferences by summarizing them. User explicitly requested restoration of full precision.

**Decision**: Implement 6-location durability chain for user preferences with explicit verification during handoff.

**Locations**:
1. `session_context.md` — authoritative full list (User Preferences section)
2. `CLAUDE.md` — key preferences enforced every session
3. `.claude/rules/context-handoff.md` — Step 2.5 verification requirement
4. `.cursor/rules/context-handoff.mdc` — Cursor sync of verification
5. Memory MCP — `User_Preferences_Authoritative` entity as backup
6. `session_handoff` skill — Step 7 verification before handoff completes

**Rationale**: User preferences are stable configuration that must never be inadvertently summarized or reduced. Multiple locations ensure durability across crashes and session boundaries.

**Key Preferences Preserved**:
- TDD approach, planning sessions before implementation
- Uses tmux for long-running experiments
- Insists on durability for pending actions (document in multiple places)
- Prefers consolidation of docs/ over deletion
- Flat docs/ structure (no subdirectories except research_paper/)
- Precision in language — never reduce fidelity

**Implications**:
- Handoff skill now verifies preferences section exists and is complete
- Future agents can reconstruct from Memory MCP if files corrupted
- NEVER summarize or reduce precision of user preferences

## 2025-12-29 Execution Plan vs Implementation Plan Distinction

**Context**: Planning documentation consolidation revealed confusion about which docs to archive vs. update.

**Decision**: Distinguish between two types of plan documents:
- **Execution plans** (what experiments we run) — KEEP and UPDATE
- **Implementation plans** (how we built tooling) — CONSOLIDATE to history when complete

**Files by Category**:
| Type | File | Action |
|------|------|--------|
| Execution | `phase6a_execution_plan.md` | UPDATE to v2.0 |
| Methodology | `config_architecture.md` | UPDATE to v2.0 |
| Implementation | `hpo_fixes_plan.md` | CONSOLIDATE |
| Implementation | `hpo_time_optimization_plan.md` | CONSOLIDATE |
| Implementation | `architectural_hpo_implementation_plan.md` | CONSOLIDATE |
| Implementation | `hpo_supplemental_tests.md` | CONSOLIDATE |
| Implementation | `feature_pipeline_integration_issues.md` | CONSOLIDATE |

**Rationale**:
- Execution plan defines WHAT experiments we run — stays active as the "current state" document
- `config_architecture.md` defines methodology for research paper — needs update not deletion
- Implementation plans define HOW we built infrastructure — historical value only once tasks complete

**User Input**: `config_architecture.md` will be referenced in research paper methodology section. Must reflect current reality (2B budget, dynamic batch sizing, architectural HPO).

**Implications**:
- Create `phase6a_implementation_history.md` for 20 completed tasks
- Update `config_architecture.md` to v2.0 (add 2B, dynamic batch, architectural HPO)
- Update `phase6a_execution_plan.md` to v2.0 (mark stages 1-3 complete)
- Full plan in `docs/documentation_consolidation_plan.md`

## 2025-12-29 Session Handoff Skill Gap Identified

**Context**: Audit of plan files revealed they have stale status fields (e.g., "In Progress Task 4") while `phase_tracker.md` shows all tasks complete.

**Problem**: Session handoff skill only updates `phase_tracker.md`, not individual plan files.

**Decision**: Note for future skill revision — session handoff should update ALL applicable plan documentation when tasks complete.

**Evidence**:
- `hpo_time_optimization_plan.md` says "Status: In Progress — Task 4 of 6"
- Reality: All 6 tasks complete (verified in code and phase_tracker)

**Scope of Required Fix**:
- Any plan doc in `docs/` with task checkboxes or status fields
- Corresponding Memory MCP entities

**Memory Entity**: `Session_Handoff_Skill_Gap` — captures this process improvement item

**Implications**:
- Next session restores may see misleading status in plan files
- Use `phase_tracker.md` as source of truth until skill is revised
- Documentation consolidation will naturally fix this for Phase 6A plans

## 2026-01-03 HPO Diversity Enhancement

**Context**: 2B HPO was reusing the same architecture (arch_idx) with very similar hyperparameters, wasting trials on near-duplicate configurations.

**Problem Identified**:
1. TPESampler with default n_startup_trials=10 converged too quickly to similar configs
2. No mechanism to force variation when same architecture sampled with similar params
3. Bug: arch_idx=0 was being skipped due to Python's falsy zero behavior

**Decision**: Implement three-part diversity enhancement:
1. Increase n_startup_trials to 20 for broader random exploration
2. Add forced variation logic: when same arch_idx with similar dropout (<0.08) AND epochs (<20), force dropout to opposite extreme
3. Fix falsy bug: use explicit `if prev_arch_idx is None` instead of `prev_arch_idx or fallback`

**Implementation Details**:
- `create_study()`: Added `n_startup_trials=20` parameter to TPESampler
- `create_architectural_objective()`: Added forced variation logic at lines 276-303
- Fixed comparison at lines 282-284: `prev_arch_idx = t.params.get("arch_idx")` then `if prev_arch_idx is None: prev_arch_idx = t.user_attrs.get("arch_idx")`

**Rationale**:
- 20 startup trials ensures more architecture diversity before TPE optimization
- Forcing dropout variation ensures each trial with same architecture learns something new
- Falsy fix prevents arch_idx=0 from being incorrectly skipped

**Tests Added**: 4 new tests in `TestTPESamplerConfig` and `TestArchObjectiveForcesVariation` classes (365 total passing)

**Memory Entity**: `HPO_Diversity_Enhancement`, `Lesson_FalsyZeroBug`

## 2026-01-03 2B HPO Resume Script

**Context**: 2B HPO h1 had 11 trials complete (0-10) but needed to skip problematic arch_idx=52 (d=1024, L=256) which caused memory exhaustion.

**Decision**: Create dedicated resume script rather than modifying main HPO script.

**Implementation**: `experiments/phase6a/hpo_2B_h1_resume.py`
- Loads existing trials 0-10 from JSON files
- Injects them into new Optuna study via `study.add_trial()`
- Filters architecture list to exclude arch_idx=52
- Runs remaining 39 trials

**Bug Fix During Testing**: Historical trials had `warmup_steps=200` which wasn't in current CategoricalDistribution([100, 300, 500]). Changed to `IntDistribution(100, 500)` for injection to accept any historical value.

**Rationale**: Resume approach preserves 11 trials of compute (~hours of work) while skipping known-problematic configurations.

**2B HPO Status**:
- 11/50 trials complete
- Best: Trial 4, val_loss=0.3778, d=1024, L=180, h=16
- Skip: arch_idx=52 (d=1024, L=256) - memory exhaustion
- h3/h5 pending h1 completion

**Memory Entity**: `HPO_2B_Resume_Script`, `Phase6A_2B_HPO_Status`

## 2026-01-03 Lesson Learned: Python Falsy Zero Bug

**Context**: Diversity forcing logic wasn't triggering for arch_idx=0 trials.

**Root Cause**: Expression `t.params.get("arch_idx") or t.user_attrs.get("arch_idx")` returns the fallback when value is 0 because 0 is falsy in Python.

**Pattern to Avoid**:
```python
# WRONG: fails when value is 0
x = value or default

# CORRECT: explicit None check
x = value if value is not None else default
# OR
x = value
if x is None:
    x = default
```

**Applicability**: Any code dealing with indices, counts, or numeric IDs that can legitimately be zero.

**Memory Entity**: `Lesson_FalsyZeroBug`

## 2026-01-23 Expanded Architecture Scope for Foundation Investigation

**Context**: Original investigation plan included only 4 architecture comparisons (PatchTST, iTransformer, TimeMixer, Informer). Expanding scope to include more encoder-decoder architectures for comprehensive comparison.

**Decision**: Add 5 additional architectures to Tier 2 (trained from scratch), bringing total to 9 architectures.

**New Architectures**:
| Model | Type | Key Innovation |
|-------|------|----------------|
| Autoformer | Enc-Dec | Autocorrelation replaces attention |
| FEDformer | Enc-Dec | Frequency domain decomposition |
| ETSformer | Enc-Dec | Exponential smoothing + neural |
| Crossformer | Enc-Dec | Cross-variate attention |
| ARMA-Attention | Hybrid | Statistical + neural (conditional) |

**Rationale**:
- These are architectures, NOT foundation models - they contribute to scaling law research
- Encoder-decoder designs may have different inductive biases than encoder-only PatchTST
- Provides comprehensive coverage of transformer variants for financial time series
- All have published implementations available for porting

**Impact**:
- Added experiments FD-09 through FD-18 (10 new experiments)
- Added Tasks 7-12 to implementation plan
- Timeline updated: 20-36 hours → 40-70 hours

**Alternatives Considered**:
- Only foundation models → rejected; architecture comparison equally valuable
- All architectures at once → rejected; staged implementation more manageable
- Skip encoder-decoder → rejected; may reveal different attention patterns

**Implications**:
- More comprehensive architecture comparison for research paper
- Can identify if attention pattern (causal vs bidirectional) matters
- Extended timeline but more publishable findings

## 2026-01-23 TimesFM Execution Path: Docker + Colab

**Context**: TimesFM requires JAX, which is incompatible with ARM64/Apple Silicon. Native installation fails on M4 MacBook Pro.

**Decision**: Use Docker x86 emulation (primary) and Google Colab notebooks (secondary) for TimesFM experiments.

**Approaches**:
| Approach | Role | When to Use |
|----------|------|-------------|
| Docker x86 | Primary | Local development, faster iteration |
| Colab | Secondary | If Docker too slow or for cloud GPU access |

**Rationale**:
- Docker with `--platform linux/amd64` uses Rosetta 2 to run x86 containers on ARM
- Colab notebooks provide self-contained experiments runnable in user's browser
- Both approaches avoid modifying the project's Python environment

**Alternatives Considered**:
- Native JAX on ARM → rejected; unstable, requires manual metal plugin installation
- Skip TimesFM → rejected; it's a key foundation model in the investigation
- Separate venv → rejected; JAX ARM issues persist regardless of venv

**Implications**:
- TimesFM experiments will be slower than native models
- Colab notebooks must be self-contained (no local imports)
- Results exported as JSON for integration with analysis pipeline
- Investigation plan updated with execution strategy section

## 2026-01-21 CRITICAL: Target Calculation Correction

**Context**: Discovered that target calculation was using the wrong price series. All prior documentation and code used `max(close[t+1:t+1+horizon])` but this is incorrect for trading purposes.

**Problem**:
- WRONG: Predicting if future CLOSE will exceed threshold → unrealistic trading assumption
- RIGHT: Predicting if future HIGH will exceed threshold → matches actual trade execution

**Decision**: Establish canonical target calculation rules (see Memory entity `Target_Calculation_Definitive_Rule`):

| Target Type | Question | Formula |
|-------------|----------|---------|
| **UPSIDE threshold** | Will HIGH in next N days be ≥X% above today's CLOSE? | `max(high[t+1:t+1+horizon]) >= close[t] * (1+X%)` |
| **DOWNSIDE threshold** (future) | Will LOW in next N days be ≥X% below today's CLOSE? | `min(low[t+1:t+1+horizon]) <= close[t] * (1-X%)` |

**Rationale**: A trade entered at today's close achieves profit when the HIGH reaches the target price, not when the CLOSE does. This is fundamental to how trading actually works.

**Impact on Class Balance**:
- 0.5% threshold with CLOSE-based (wrong): 29.1% positive
- 0.5% threshold with HIGH-based (correct): ~50% positive (balanced!)

**Deprecated Terminology**: The terms "Close-to-Close" and "Close-to-High" caused confusion and are now deprecated. Use "upside threshold (HIGH-based)" and "downside threshold (LOW-based)" instead.

**Historical Note**: Line 140 of this file (Phase 4 decision) documents the original incorrect implementation. That decision captured what was implemented at the time, but the approach was conceptually wrong.

**Files Affected**:
- `src/data/dataset.py` - CODE (needs implementation fix)
- `docs/project_history.md` - Historical docs (note added)
- This file (clarification added)
- `.claude/context/session_context.md` - Updated with correct terminology

**Memory Entity**: `Target_Calculation_Definitive_Rule` - canonical definition

**Implications**:
- All prior experiments used incorrect targets - results need revalidation
- TDD implementation required to add HIGH-based target calculation
- User prefers 0.5% threshold (achieves ~50/50 class balance with HIGH-based target)

## 2026-01-28 Alternative Architecture Methodology Correction

**Context**: After running 50-trial HPO for iTransformer and Informer (v2), discovered that both achieved 0% recall despite moderate AUC (0.621 and 0.669 respectively).

**Problem Identified**:
- v1/v2 trained as **regressors** (MAE loss on returns)
- Evaluated as **classifiers** (binary AUC, precision, recall)
- Model outputs clustered in [0.004, 0.006] range
- When thresholded at 0.5, no predictions were positive

**Decision**: Redesign experiments with proper classification training (v3 design).

**Rationale**:
1. Task mismatch caused 0% recall - invalid results
2. AUC was misleading (measures ranking, not calibration)
3. PatchTST worked because it used BCE loss on binary targets from the start
4. NeuralForecast supports `DistributionLoss('Bernoulli')` for classification

**Correct Approach (v3)**:
- Loss: `DistributionLoss(distribution='Bernoulli')`
- Target: Binary (0/1) threshold target
- Output: Probabilities in [0, 1]
- Evaluation: Standard classification metrics

**Alternatives Considered**:
- Rerun foundation models (Lag-Llama, TimesFM) with classification loss: Rejected - pre-trained models can't easily be retrained with BCE. Domain mismatch is the primary issue.
- Accept v2 results as valid: Rejected - 0% recall means model is not learning class separation.

**Implications**:
- v1/v2 results discarded as scientifically invalid
- v3 design documented in `docs/architecture_hpo_v3_design.md`
- Lessons documented in `docs/methodology_lessons_v1_v2.md`
- Foundation model findings stand (domain mismatch is the issue)
- Focus v3 on iTransformer/Informer only

**Workstream**: ws2 (foundation)

**Memory Entity**: `Alternative_Architecture_Methodology_Lesson_20260128`

## 2026-01-23 Lag-Llama Integration Complete

**Context**: Task 2 of Foundation Model Investigation - implementing LagLlamaWrapper for binary classification.

**Decision**: Implemented `LagLlamaWrapper` class that adapts the pre-trained Lag-Llama foundation model for threshold-based binary classification.

**Key Technical Decisions**:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CDF Implementation | Normal approximation via `ndtr` | PyTorch lacks `betainc`; normal approx is differentiable and accurate for df > 5 |
| Context Length | 1150 (min 1124) | Lag-Llama requires max_lag (1092) + 32 |
| Fine-tuning Mode | "full" (all params trainable) | Maximize adaptation to financial domain |
| Input Handling | Two approaches: Close-only OR feature projection | Compare univariate vs multivariate adaptation |

**Files Created**:
- `src/models/foundation/lag_llama.py` - LagLlamaWrapper class
- `tests/test_lag_llama.py` - 16 unit tests (all passing)
- `experiments/foundation/train_lagllama_h1_close.py` - FD-01a
- `experiments/foundation/train_lagllama_h1_proj.py` - FD-01b
- `experiments/foundation/train_lagllama_h3_close.py` - FD-02a
- `experiments/foundation/train_lagllama_h3_proj.py` - FD-02b

**Success Criteria**:
| Horizon | Baseline (PatchTST 200M) | Target (5% improvement) |
|---------|--------------------------|-------------------------|
| H1 | 0.718 | ≥0.74 |
| H3 | 0.622 | ≥0.65 |

**Dependencies Installed**:
- `lag-llama` (from GitHub)
- `gluonts` 0.15.1
- numpy 1.26.4, pandas 2.1.4 (pinned for gluonts compatibility)

**Alternatives Considered**:
- scipy-based CDF → rejected; not differentiable, breaks gradient flow
- PyTorch betainc → not available in PyTorch 2.9.1
- Sigmoid approximation → less accurate than normal approximation

**Implications**:
- Experiments FD-01/FD-02 ready to run
- CDF approximation accuracy validated in tests (relative ordering preserved)
- Full gradient flow enables end-to-end fine-tuning

## 2026-01-31 Context Ablation Results - Architecture-Specific Optima

**Context**: Ran context ablation experiments for iTransformer and Informer across 60d, 80d, 120d, 180d, 220d to determine optimal context length per architecture.

**Findings**:

| Model | 60d | 80d | 120d | 180d | 220d | **Best** |
|-------|-----|-----|------|------|------|----------|
| iTransformer | 0.552 | **0.590** | 0.503 | 0.548 | 0.583 | **80d** |
| Informer | 0.539 | 0.554 | 0.512 | **0.585** | 0.557 | **180d** |

**Decision**: Use architecture-specific optimal context lengths for a200 training:
- iTransformer: 80d (matches PatchTST optimal)
- Informer: 180d (benefits from longer context due to ProbSparse attention)

**Key Observations**:
1. Both architectures show non-monotonic relationship with context length
2. Both dip at 120d (possibly noise or regime-specific artifact)
3. Neither approaches PatchTST performance (0.718) - gap of ~12-15% persists
4. Informer's ProbSparse attention may be better suited to longer sequences

**Rationale**: Context length significantly affects performance, but the optimal differs by architecture. Using architecture-specific optima ensures fair comparison and best possible performance before feature scaling tests.

**Implications**:
- a200 training must use optimal context per architecture
- Performance gap vs PatchTST is architectural, not configuration-related
- Future research should investigate why patching mechanism outperforms attention variants

**Workstream**: ws2 (foundation)

---

## 2026-01-30 Feature Scaling Requires Higher Regularization

**Context**: Phase 6C loss sweep experiments revealed that feature tier (a50 vs a100) requires different architecture configurations, with more features needing more regularization.

**Evidence**:

| Tier | Features | Best Architecture | Best Precision | Best AUC |
|------|----------|-------------------|----------------|----------|
| a50 | 55 | d_model=128, layers=6, **dropout=0.3** | **100%** | **0.738** |
| a100 | 105 | d_model=64, layers=4, **dropout=0.7** | 58.3% | 0.714 |

**Key Observations**:
1. a100 requires **2.3x higher dropout** (0.7 vs 0.3) compared to a50
2. a100 requires **4x smaller model** (d_model 64 vs 128)
3. Despite optimal architecture per tier, a100 still underperforms a50
4. 20 more configs achieve ≥50% precision on a50 vs a100 (20 vs 10)

**Decision**: When scaling to more features, proactively increase regularization strength. Use a100's architecture (high dropout, smaller model) as baseline for a200+.

**Rationale**: Additional features introduce more noise that the model can overfit to. Higher regularization (dropout, smaller model) is necessary but not sufficient - even with optimal regularization, more features may hurt precision for this task.

**Interpretation**: This is a **scaling law violation** - more features should help, but for financial time series with limited signal, additional technical indicators may be redundant or contradictory, adding noise faster than signal.

**Alternatives Considered**:
- Use a50's architecture for a200: Rejected - would likely overfit severely with 2x more features
- Skip a200 entirely: Rejected - need data point to confirm pattern continues

**Implications**:
- a200 sweep uses a100 architecture as starting point (high regularization)
- Feature quality > feature quantity hypothesis strengthened
- Research paper should highlight this inverse scaling finding
- Future feature engineering should prioritize signal-to-noise ratio over feature count

**Workstream**: ws3 (loss_function_optimization)

**Memory Entities**: `Feature_Scaling_Violation`, `Feature_Regularization_Relationship`

## 2026-02-01 Two-Phase Budget-Aware HPO Strategy

**Context**: Previous HPO focused on single parameter budget (2M). Need systematic exploration across all scales (750k → 2M → 20M → 200M) to understand scaling laws and identify optimal architectures.

**Decision**: Implement two-phase HPO strategy with budget-aware forced extremes.

**Phase 1 Design (18 forced configs + TPE)**:
- Group 1: Budget × Architecture (8 configs) - all 4 budgets × 2 styles (shallow-wide vs deep-narrow)
- Group 2: Dropout extremes (4 configs) - 0.1, 0.3, 0.7 on 2M and 200M
- Group 3: Learning rate extremes (3 configs) - 1e-5, 1e-3 on 2M and 20M
- Group 4: Weight decay extremes (3 configs) - 0.0, 1e-2 on 2M and 200M
- Early stopping when top-5 trials converge within 0.02 AUC

**Phase 2 Design (supplementary)**:
- Focus on top 2 budgets from Phase 1
- Explore n_heads, extreme LRs, high weight decay, batch sizes
- ~20-30 trials per budget

**Architecture Sizing Formula**: params ≈ 12 × n_layers × d_model²

| Budget | Shallow (d, L, h) | Deep (d, L, h) |
|--------|-------------------|----------------|
| 750k | 192, 2, 4 | 128, 4, 4 |
| 2M | 320, 2, 4 | 192, 5, 4 |
| 20M | 768, 3, 8 | 384, 12, 8 |
| 200M | 1536, 8, 16 | 768, 28, 16 |

**Rationale**:
1. Forced extremes ensure boundary conditions are tested (not left to random sampling)
2. Budget-aware sizing ensures valid architectures per parameter scale
3. Early stopping avoids wasting compute on converged trials
4. Two-phase approach focuses resources on promising budgets

**Alternatives Considered**:
- Single-phase TPE only: Rejected - may miss extremes due to random sampling
- Full grid search: Rejected - too expensive (would require 1000+ trials)
- Focus on single budget: Rejected - need multi-scale data for scaling laws

**Implementation**:
- `src/training/hpo_budget_extremes.py` - Core logic (~150 lines)
- `tests/test_hpo_budget_extremes.py` - 30 tests
- CLI flags: `--forced-extremes`, `--budgets`, `--early-stop-patience`, `--early-stop-threshold`, `--supplementary`, `--param-budget`
- Documentation: `docs/hpo_strategy_phase6.md`

**Workstream**: ws2 (foundation)
