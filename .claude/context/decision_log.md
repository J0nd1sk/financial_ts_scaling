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
