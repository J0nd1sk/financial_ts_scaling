# Financial TS Transformer Scaling: Project History

**Purpose:** Historical record of decisions, rationale, and evolution of the project. Captures "what we planned and why" at various points in time. For current approaches, see `project_prd.md` and `project_phase_plans.md`.

---

## 1. Original Research Vision (November 2025)

### 1.1 Primary Research Questions

The project set out to answer five core questions:

1. **Parameter Scaling**: Does increasing model parameters (2M → 20M → 200M) improve prediction accuracy following a power law?
2. **Feature Scaling**: Does increasing feature count (20 → 100 → 500 → 1000 → 2000) improve accuracy?
3. **Data Scaling**: Does increasing training data diversity (single asset → multiple assets → cross-asset) improve accuracy?
4. **Data Quality Scaling**: Does adding richer data types (price-only → +sentiment → +volatility → +search trends) improve accuracy?
5. **Interaction Effects**: How do these scaling dimensions interact?

### 1.2 Success Criteria

**Minimum Viable:**
- Complete Phase 1 (parameter scaling) experiments
- Generate scaling curves
- Determine if power law holds
- Publish findings (Medium minimum)

**Full Success:**
- Complete all phases
- Clear scaling law characterization
- Publish on arXiv or SSRN
- Reproducible code released
- Community engagement

### 1.3 Publication Targets

- **Primary**: Medium (Towards Data Science)
- **Secondary**: arXiv (Quantitative Finance / Machine Learning)
- **Tertiary**: SSRN
- **Code**: GitHub with full reproducibility

### 1.4 Baseline Reference

Existing ML model achieved >70% accuracy on next-day SPY direction prediction. The goal was to determine if transformer scaling improves upon this baseline and follows predictable laws.

---

## 2. Technology Stack Decisions (Original)

### 2.1 Hardware

- **Machine**: MacBook Pro M4 Max, 128GB Unified Memory
- **Storage**: Internal SSD + External SSD for backups
- **Cooling**: Basement environment (50-60°F ambient), elevated hard surface

The M4 Max with 128GB unified memory was chosen for its ability to handle larger models without discrete GPU memory constraints.

### 2.2 Software Stack Rationale

| Category | Choice | Rationale |
|----------|--------|-----------|
| Python | 3.12.x | Stable middle ground between features and compatibility |
| ML Framework | PyTorch 2.6+ | MPS backend for M4, mature ecosystem |
| Transformers | Hugging Face | PatchTST implementation available |
| HPO | Optuna | Bayesian optimization, good integration |
| Tracking | W&B + MLflow | W&B for visualization, MLflow for local model registry |
| Data Storage | Parquet | 10-100x faster than CSV, type-safe, columnar |
| Indicators | pandas-ta + TA-Lib | TA-Lib for speed, pandas-ta for convenience |

### 2.3 Why PatchTST

PatchTST (Patch Time Series Transformer) was selected for:
- Clean parameter scaling without architectural confounds
- Proven time-series performance
- Modular design enabling systematic testing
- Encoder-only architecture (simpler than encoder-decoder)
- Channel-independent design (each feature series shares backbone)

### 2.4 Why Parquet Over CSV

- Binary columnar format, 2-10x smaller than CSV
- Schema-aware (preserves dtypes without inference)
- Supports partial column reads
- Industry standard for data engineering
- Note: LLMs cannot read Parquet directly, so CSV samples created in `data/samples/` for code review

---

## 3. Experimental Design Philosophy (Original)

### 3.1 Parameter Budget Design

The original design specified three parameter budgets with 10x scaling intervals:
- **2M**: Small baseline
- **20M**: Medium scale
- **200M**: Large scale

The 2B budget was added later during Phase 4 implementation.

Batch size re-tuning was specified as:
- **Required** when parameter budget changes
- **Recommended** when dataset changes
- **Optional** when feature count changes

### 3.2 Dataset Matrix Concept

A two-dimensional naming scheme was designed:

**Asset Dimension (rows):**
- A = SPY only
- B = SPY + DIA + QQQ
- C = + major stocks (AAPL, MSFT, GOOGL, AMZN, NVDA, etc.)
- D = + sentiment data
- E = + economic indicators (FRED)

**Feature Quality Dimension (columns):**
- a = OHLCV + calculated indicators only
- b = a + SF Fed Sentiment
- c = b + VIX
- d = c + Google Trends

**History Constraints by Column:**
- Column a: Longest history (1950s+ for some assets)
- Column b: 1980+ (SF Fed Sentiment start)
- Column c: 1990+ (VIX start)
- Column d: 2004+ (Google Trends start)

### 3.3 Feature Tier System

Seven tiers were planned for feature scaling experiments:

| Tier | Feature Count | Composition |
|------|---------------|-------------|
| Minimal | ~20 | Basic OHLCV + returns + 3 MAs + RSI + MACD |
| Small | ~50 | + more MAs + momentum + basic volatility |
| Medium | ~100 | + volume indicators + trend + cross-signals |
| Large | ~200 | + multiple periods per indicator + derived |
| XL | ~500 | + all indicator variants + lagged features |
| XXL | ~1000 | + interaction features + polynomials |
| Max | ~2000 | + all possible combinations |

### 3.4 Timescales (8 Total)

| # | Timescale | Bar Period | Notes |
|---|-----------|------------|-------|
| 1 | Daily | 1 day | Base timescale |
| 2 | 2-Day | 2 days | |
| 3 | 3-Day | 3 days | |
| 4 | 5-Day | 5 days | Trading week |
| 5 | Weekly | 7 days | |
| 6 | 2-Week | 14 days | |
| 7 | Monthly | ~21 trading days | |
| 8 | Daily + Multi-Resolution | 1 day | Daily with weekly/monthly indicators as features |

### 3.5 Tasks (6 Per Timescale)

| Task | Type | Output | Head | Loss |
|------|------|--------|------|------|
| Direction | Binary Classification | P(up) | Sigmoid | BCE |
| Threshold >1% | Binary Classification | P(>1%) | Sigmoid | BCE |
| Threshold >2% | Binary Classification | P(>2%) | Sigmoid | BCE |
| Threshold >3% | Binary Classification | P(>3%) | Sigmoid | BCE |
| Threshold >5% | Binary Classification | P(>5%) | Sigmoid | BCE |
| Price Regression | Regression | Predicted price/return | Linear | MSE |

Each task was to be a separate model: 8 timescales × 6 tasks = 48 models per dataset configuration.

### 3.6 Original Data Splits

The original experimental design specified fixed temporal splits:
- **Training**: Through 2020-12-31
- **Validation**: 2021-01-01 to 2022-12-31
- **Testing**: 2023-01-01 onwards

Critical constraint: No future leakage. All features computed with only past data.

### 3.7 Experiment Phases (Original Plan)

1. **Phase 1 (Parameter Scaling)**: Fixed features/data, vary parameters
2. **Phase 2 (Feature Scaling)**: Fixed parameters/data, vary features
3. **Phase 3 (Data Scaling)**: Fixed parameters/features, vary dataset size
4. **Phase 4 (Quality Scaling)**: Fixed parameters/features, vary data quality
5. **Phase 5 (Interaction Effects)**: Test coupled scaling dimensions
6. **Phase 6 (Full Expansion)**: Complete parameter space exploration

---

## 4. Thermal Management Design

### 4.1 Temperature Thresholds

Designed specifically for M4 MacBook Pro:

| Temperature | Status | Action |
|-------------|--------|--------|
| < 70°C | Normal | Full operation |
| 70-85°C | Acceptable | Monitor closely |
| 85-95°C | Warning | Consider pause, reduce batch size |
| > 95°C | Critical | STOP IMMEDIATELY |

### 4.2 Mitigation Strategies

**If sustained > 70°C:**
- Verify laptop elevated on hard surface
- Check vents not blocked
- Point fan at intake if needed
- Verify basement ambient 50-60°F

**If sustained > 85°C:**
- Reduce batch size by 50%
- Add delay between batches
- Consider overnight training (cooler ambient)

**If hits > 95°C:**
- Auto-pause via thermal callback
- Save checkpoint immediately
- Cool for 30+ minutes before resuming

### 4.3 Thermal Callback Design

The training loop was designed to include a powermetrics-based callback:
- Check temperature every epoch
- Log to thermal log file
- Warn at ≥85°C
- Abort at ≥95°C

---

## 5. Development Methodology Decisions

### 5.1 RACI Matrix (2025-11-26)

| Activity | Human (Alex) | LLM/Agent |
|----------|--------------|-----------|
| Experimental design | Lead, Accountable | Consult |
| Architecture decisions | Lead, Accountable | Propose |
| Phase planning | Lead, Approve | Execute |
| Writing tests | Review, Approve | Execute |
| Writing implementation | Review, Approve | Execute |
| Code review | Execute, Accountable | Inform |
| Git operations | Approve | Execute |
| Data downloads | Monitor | Execute |
| Model training | Monitor thermal | Execute |

### 5.2 Test-Driven Development Protocol

Mandatory TDD workflow established:
1. Write failing test FIRST
2. Run test, confirm failure
3. Write MINIMAL code to pass
4. Run test, confirm pass
5. Refactor if needed
6. Commit with test + implementation

### 5.3 Approval Gates

Automatic triggers for human approval:
- Any refactoring >50 lines
- Architecture changes
- Dependency additions/upgrades
- Branch merges to staging or main
- Parameter budget changes
- Phase transitions
- Model training with new configuration
- Data deletion or overwriting

### 5.4 SpecKit and Superpowers Integration

The project integrated two agentic development frameworks:
- **SpecKit**: For structured specifications and planning (`/specify`, `/plan`, `/tasks`)
- **Superpowers**: For skill-based workflows (session_handoff, test_first, approval_gate, thermal_management, planning_session, task_breakdown)

Skills implemented in `.claude/skills/` directory.

---

## 6. Completed Phase History

### 6.1 Phase 0: Development Discipline (2025-11-26)

Established:
- SpecKit + Superpowers installation
- Core skills implemented
- Claude/Cursor rules synced

### 6.2 Phase 1: Environment Setup (2025-12-07 to 2025-12-08)

Completed:
- Directory scaffold + Makefile
- CLAUDE.md + project rules framework
- Python 3.12 venv with 30+ packages
- Test infrastructure (pytest)
- Verification tooling (`scripts/verify_environment.py`, `make verify`)

### 6.3 Phase 2: Data Pipeline Foundation (2025-12-07 to 2025-12-08)

**Planning Session Output (2025-12-07):**

Objective: Create minimal, testable data pipeline for SPY OHLCV data.

In Scope:
- Data directory structure (raw/, processed/, samples/)
- SPY OHLCV download script using yfinance
- Basic data validation
- Parquet storage format

Out of Scope (deferred):
- Multi-asset data (DIA, QQQ, stocks, econ)
- Indicator calculations
- Data versioning system
- Advanced validation

Assumptions validated:
1. yfinance 0.2.66 works for SPY historical data
2. Yahoo Finance has SPY data from ~1993 to present
3. Parquet format appropriate for this data

Deliverables:
- `tests/test_data_download.py` with 5 test cases
- `scripts/download_ohlcv.py` implementing download
- SPY data: 8,272 rows (1993-2025)

### 6.4 Phase 3: Pipeline Design (2025-12-08)

Completed:
- Feature engineering: `src/features/tier_a20.py` with 20 indicators
- Build script: `scripts/build_features_a20.py`
- Manifest integration for processed data versioning

### 6.5 Phase 4: Training Boilerplate (2025-12-09)

Deliverables:

| Component | Files | Tests |
|-----------|-------|-------|
| Config System | `src/config/experiment.py` | 5 |
| Dataset Class | `src/data/dataset.py` | 8 |
| PatchTST Model | `src/models/patchtst.py` | 6 |
| Parameter Configs | `configs/model/patchtst_{2m,20m,200m,2b}.yaml` | 4 |
| Thermal Callback | `src/training/thermal.py` | 10 |
| Tracking Integration | `src/training/tracking.py` | 9 |
| Training Script | `scripts/train.py`, `src/training/trainer.py` | 7 |
| Batch Size Discovery | `scripts/find_batch_size.py` | 10 |

Parameter budget configurations finalized:

| Budget | d_model | n_heads | n_layers | d_ff | Actual Params |
|--------|---------|---------|----------|------|---------------|
| 2M | 192 | 6 | 4 | 768 | ~1.82M |
| 20M | 512 | 8 | 6 | 2048 | ~19M |
| 200M | 1024 | 16 | 16 | 4096 | ~202M |
| 2B | 2304 | 24 | 32 | 9216 | ~2.04B |

### 6.6 Training Infrastructure Decisions (2025-12-08)

Key decisions documented:

- **Configuration format**: Plain YAML with lightweight dataclass loaders. Deliberately avoided OmegaConf to keep configs simple and reviewable.
- **Execution model**: No generic launcher scripts. Every experiment run explicitly via CLI to preserve one-task-per-model discipline.
- **Tracking**: All runs log to both W&B and local MLflow.
- **Batch-size discovery**: Mandatory when changing parameter budgets.

Original config schema example:
```yaml
dataset:
  features_path: data/processed/v1/SPY_features_a20.parquet
  feature_tier: a20
target:
  type: threshold
  horizon_days: 1
  pct_threshold: 0.01
training:
  param_budget: 2M
  epochs: 100
  batch_size: 32
```

Target construction rule: For each example at index t, compute `future_max = max(close[t+1 : t+horizon])`. Label = 1 if `future_max >= close[t] * (1 + pct_threshold)` else 0.

> **⚠️ CORRECTION (2026-01-21)**: This documents the original implementation which was **conceptually incorrect**. The correct approach uses HIGH prices, not CLOSE prices: `future_max = max(high[t+1 : t+horizon])`. A trade achieves profit when the HIGH reaches the target, not when the CLOSE does. See Memory entity `Target_Calculation_Definitive_Rule` and decision_log.md entry 2026-01-21 for canonical rules.

### 6.7 Phase 5: Data Acquisition (2025-12-10)

Extended data pipeline beyond SPY:

| Task | Deliverable |
|------|-------------|
| Generalize download | download_ticker + retry logic |
| Download ETFs/Indices | DIA (7,018 rows), QQQ (6,731 rows), ^DJI (8,546 rows), ^IXIC (13,829 rows) |
| Download VIX | ^VIX (9,053 rows), relaxed volume validation |
| Generalize feature pipeline | --ticker CLI arg, dynamic paths |
| Build DIA/QQQ features | 6,819 and 6,532 rows respectively |
| VIX feature engineering | 8 features: close, sma_10/20, percentile_60d, zscore_20d, regime, change_1d/5d |
| Combined dataset builder | SPY_dataset_c.parquet: 8,073 rows, 34 columns |

**Key Decision:** Use indices (^DJI, ^IXIC) for extended training history (1992+ vs 1999+ for ETFs).

### 6.8 Phase 5.5: Experiment Setup (2025-12-11)

Infrastructure for running scaling experiments:

| Task | Deliverable |
|------|-------------|
| Config Templates | configs/experiments/threshold_{2,3,5}pct.yaml |
| Timescale Resampling | src/features/resample.py with CLI |
| Data Dictionary | docs/data_dictionary.md with generator script |
| Optuna HPO | src/training/hpo.py, run_hpo.py |
| Scaling Analysis | src/analysis/scaling_curves.py |
| Result Aggregation | src/analysis/aggregate_results.py |

### 6.9 Phase 6A Prep: Experiment Skills (2025-12-11)

Created skill-based experiment workflow:

| Component | Implementation |
|-----------|----------------|
| src/experiments/runner.py | 4 functions (~310 lines) |
| src/experiments/templates.py | 2 functions (~240 lines) |
| experiment_generation skill | 228 lines |
| experiment_execution skill | 329 lines |

**Key Decisions:**
- Thin wrapper scripts (~50-80 lines) with all params visible
- Dynamic data assembly (no pre-built datasets)
- Per-budget HPO: 12 runs (skip 2% task, borrow params)
- Hybrid logging: append-only CSV + regenerated markdown

### 6.10 Phase 6A HPO: Architectural Search (2025-12-12 to 2026-01-17)

Ran 12 HPO studies (4 budgets × 3 horizons) to find optimal architectures.

**Key Finding:** Larger models did NOT improve over smaller models. Best val_loss achieved by 2M model (0.2538 on h3). This indicates a **data-limited regime** where model capacity exceeds information content.

| Budget | Horizon | Best Architecture | val_loss |
|--------|---------|-------------------|----------|
| 2M | h1 | d=64, L=48, h=2 | 0.3136 |
| 2M | h3 | d=64, L=32, h=2 | 0.2538 |
| 2M | h5 | d=64, L=64, h=16 | 0.3368 |
| 20M | h1 | d=128, L=180, h=16 | 0.3461 |
| 20M | h3 | d=256, L=32, h=2 | 0.3035 |
| 20M | h5 | d=384, L=12, h=4 | 0.3457 |
| 200M | h1 | d=384, L=96, h=4 | 0.3488 |
| 200M | h3 | d=768, L=24, h=16 | 0.3281 |
| 200M | h5 | d=256, L=256, h=16 | 0.3521 |
| 2B | h1 | d=1024, L=128, h=2 | 0.3599 |
| 2B | h3 | d=768, L=256, h=32 | 0.3716 |
| 2B | h5 | d=1024, L=180, h=4 | 0.3575 |

**HPO Infrastructure Additions:**
- Contiguous split mode for ChunkSplitter (train through Sept 2024, val Oct-Dec 2024)
- Gradient accumulation for memory-constrained large models
- Early stopping with patience and min_delta parameters
- Dropout as HPO parameter

### 6.11 Phase 6A: Infrastructure Fixes (2026-01-19 to 2026-01-20)

Critical issues discovered during backtest evaluation:

**Problems Identified:**
1. **ChunkSplitter Bug**: Validation used only 19 samples (1 per chunk), not ~500
2. **Probability Collapse**: Models output near-constant predictions (0.52-0.57 range)
3. **Normalization Interaction**: Global z-score + RevIN caused distribution issues
4. **Context Length**: 60 days suboptimal for prediction task

**Solutions Implemented:**
1. **SimpleSplitter**: Date-based contiguous splits with sliding window for ALL regions
   - Train: through 2022-12-31 (~7,255 samples)
   - Validation: 2023-01-01 to 2024-12-31 (~420 samples)
   - Test: 2025-01-01 onwards (~180 samples)
2. **RevIN Only**: Removed global z-score, use per-instance normalization only
3. **Context Length**: Changed from 60 to 80 days (ablation-validated)

**Validation of Fixes:**
- Prediction spread: 0.52-0.57 → 0.01-0.94 (no collapse)
- AUC improved: 0.53-0.65 → 0.60-0.72
- Validation samples: 19 → 420+

### 6.12 Phase 6A: Ablation Studies (2026-01-20 to 2026-01-21)

Systematic validation of hyperparameters through controlled experiments:

**Context Length Ablation (2026-01-20):**
| Context | Val AUC | Δ vs Baseline |
|---------|---------|---------------|
| 60 days | 0.601 | baseline |
| **80 days** | **0.695** | **+15.5%** |
| 120 days | 0.688 | +14.4% |
| 180 days | 0.549 | -8.7% |
| 252 days | 0.477 | -20.7% |

**Finding**: 80-day context optimal, longer contexts hurt due to noise accumulation.

**Head Dropout Ablation (2026-01-21):**
| Scale | head_dropout | Test AUC | Δ vs Baseline |
|-------|--------------|----------|---------------|
| 2M | 0.00 (baseline) | 0.713 | — |
| 2M | 0.05-0.30 | 0.711-0.713 | ~0% |
| 20M | 0.00 (baseline) | 0.712 | — |
| 20M | 0.05-0.15 | 0.612-0.614 | **-14%** |
| 20M | 0.30 | 0.708 | -0.6% |

**Finding**: Head dropout provides no benefit; encoder dropout (0.5) sufficient.

**Head Count Comparison (2026-01-21):**
- 2M scale: h=8 best (AUC 0.713) > h=4 (0.709) > h=2 (0.707)
- 20M scale: h=4 best (AUC 0.712) > h=8 (0.697) > h=2 (0.629)

**Finding**: Optimal head count varies by scale—not transferable.

### 6.13 Phase 6A: Final Results (2026-01-21)

12 experiments with corrected infrastructure (3 budgets × 4 horizons):

**AUC Results:**
| Budget | H1 | H2 | H3 | H5 | Mean |
|--------|-----|-----|-----|-----|------|
| 2M | 0.706 | 0.639 | 0.618 | 0.605 | 0.642 |
| 20M | 0.715 | 0.635 | 0.615 | 0.596 | 0.640 |
| 200M | 0.718 | 0.635 | 0.622 | 0.599 | 0.644 |

**Key Conclusions:**
1. **Minimal scaling benefit**: 200M only +1.7% over 2M at H1
2. **Horizon dominates scale**: H1→H5 = -16% vs scale effect +1.7%
3. **Feature bottleneck confirmed**: 25 features insufficient for larger models to benefit
4. **Recall problem at H1**: Models miss 87-96% of positive opportunities

**Implication**: Feature expansion (Phase 6C) is the logical next step to test whether scaling laws emerge with richer inputs.

See `docs/phase6a_final_results.md` for complete analysis.

### 6.14 Phase 6A: Threshold Sweep Analysis (2026-01-22)

Post-training analysis examining precision/recall tradeoffs across probability thresholds:

**Script:** `scripts/threshold_sweep.py` (250 lines, TDD with 5 tests in `tests/test_threshold_sweep.py`)

**Methodology:**
- Swept probability thresholds 0.1-0.8 across all 12 models
- Computed precision, recall, F1, and AUC at each threshold
- Generated visualization (`outputs/phase6a_final/threshold_sweep_plots.png`)

**Key Findings:**

| Finding | Evidence |
|---------|----------|
| H1 models poorly calibrated | Predictions rarely exceed 0.5, limiting threshold selection |
| H5 models best calibrated | Prediction range 0.26-0.90, enabling meaningful threshold tuning |
| Best F1 at low thresholds | 0.1-0.2 optimal (trading precision for recall) |
| AUC confirms horizon effect | H1≈0.72 >> H2≈0.64 > H3≈0.62 > H5≈0.60 |

**Scaling Effect Quantified:**
- Parameter scaling (2M→200M): +1.7% AUC improvement
- Horizon selection (H1→H5): -16% AUC degradation
- **Horizon choice has 10x more impact than parameter count**

**Outputs:**
- `outputs/phase6a_final/threshold_sweep.csv` (96 rows: 12 models × 8 thresholds)
- `outputs/phase6a_final/threshold_sweep_plots.png` (4-panel visualization)

**Conclusion:** Confirms data-limited regime finding. With only 25 features, parameter scaling provides minimal benefit. Feature expansion (Phase 6C) is the logical next step.

### 6.15 Foundation Model & Decoder Architecture Investigation (2026-01-22)

**Motivation**: Phase 6A concluded with a data-limited regime finding (+1.7% scaling benefit). Before proceeding to feature scaling (Phase 6C), we investigate whether the architectural choice of encoder-only PatchTST is itself a limitation.

**Key Insight**: PatchTST uses bidirectional (encoder) attention. Decoder architectures use causal attention and may:
- Extract different temporal patterns
- Benefit from pre-training on diverse time series (foundation models)
- Be more sample-efficient through transfer learning

**Models Under Investigation**:

| Model | Architecture | Pre-trained | Source |
|-------|-------------|-------------|--------|
| Lag-Llama | Decoder-only | ✅ General TS | Salesforce |
| TimesFM | Decoder-only | ✅ General TS | Google |
| iTransformer | Inverted attention | ❌ | Tsinghua |
| TimeMixer | MLP (no attention) | ❌ | ICLR 2024 |

**Research Questions**:
1. Can foundation models improve over from-scratch PatchTST via transfer learning?
2. Does decoder attention outperform encoder attention for classification?
3. Is self-attention even necessary (TimeMixer is pure MLP)?

**Experimental Design**:
- Same data/splits/metrics as Phase 6A for fair comparison
- 8 experiments: 4 models × 2 horizons (H1, H3)
- Success criterion: ≥5% AUC improvement over PatchTST baseline

**Decision Criteria**:
- If foundation models help → incorporate into main project
- If architecture doesn't matter → return to feature scaling (Phase 6C)

**Branch**: `experiment/foundation-decoder-investigation`
**Plan**: `docs/foundation_decoder_investigation_plan.md`

---

## 7. Deferred and Discarded Ideas

### 7.1 Meta-Module (Discarded)

Originally proposed as a tentative addition:
- **Purpose**: Combine outputs from multiple Base Modules into final trading signal
- **Architecture**: CatBoost classifier/regressor on concatenated predictions
- **Status**: Marked "tentative" from inception, never implemented
- **Reason for discard**: Focus remained on understanding base transformer scaling behavior

### 7.2 Walk-Forward Validation (Deferred)

Originally proposed for robustness:
- Train on years 1-10, validate on year 11
- Train on years 2-11, validate on year 12
- etc.

Deferred in favor of simpler fixed temporal splits.

### 7.3 Conformal Prediction (Deferred)

Originally planned for regression tasks:
- Use MAPIE library for uncertainty quantification
- Provide prediction intervals with coverage guarantees

Deferred to focus on classification tasks first.

### 7.4 Google Trends Integration (Deferred)

Originally part of dataset quality tier 'd':
- Weekly data from 2004+
- Search momentum indicators

Not implemented due to shorter history and complexity.

---

## 8. Original Risk Assessment

| Risk | Original Likelihood | Original Impact | Planned Mitigation |
|------|---------------------|-----------------|-------------------|
| Thermal throttling | Medium | Medium | Batch reduction, overnight runs, basement cooling |
| Training instability | Medium | Medium | Gradient clipping, LR reduction, checkpoint resume |
| Data issues | Low | High | Validation scripts, backup raw data |
| Overfitting | Medium | Medium | Early stopping, validation monitoring |
| No scaling effect found | Medium | Medium | Still publishable as null result |
| Disk space exhaustion | Low | Medium | Monitor usage, prune old checkpoints |
| Hardware failure | Low | High | Cloud backups, checkpoint redundancy |

### 8.1 Contingency Plans

**If no scaling law observed:**
- Document as null result (still valuable)
- Analyze why (data quality? task difficulty? architecture limit?)
- Compare to published benchmarks

**If training keeps failing:**
- Reduce model size temporarily
- Simplify to binary classification only
- Debug on smallest dataset first

---

## 9. Original Experiment Runtime Estimates

| Model | Avg Time | Count | Hours |
|-------|----------|-------|-------|
| 2M | ~30 min | 96 | 48 |
| 20M | ~1.5 hr | 96 | 144 |
| 200M | ~4 hr | 96 | 384 |
| 2B | ~12 hr | 96 | 1,152 |
| **Total** | | **384** | **~1,728 hrs** |

Original realistic timeline estimate: 2-3 months with thermal constraints.

---

## 10. Data Sources (Original Plan)

| Data Type | Source | Frequency | History Start |
|-----------|--------|-----------|---------------|
| OHLCV (SPY, DOW, QQQ) | Yahoo Finance | Daily | 1990s |
| OHLCV (Major Stocks) | Yahoo Finance | Daily | Varies |
| Treasury Yields | FRED | Daily | 1960s |
| Fed Funds Rate | FRED | Daily | 1954 |
| VIX | Yahoo/CBOE | Daily | 1990 |
| SF Fed News Sentiment | SF Fed | Daily | 1980 |
| Google Trends | pytrends | Weekly | 2004 |

**Asset Universe Tiers:**

- **Tier 1 (Core)**: SPY, DIA, QQQ
- **Tier 2 (Major Stocks)**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Tier 3 (Economic)**: ^TNX, ^VIX, LIBOR rates, Fed Funds Rate

---

*Document created: 2026-01-17*
*Updated: 2026-01-18 (added Phases 5, 5.5, 6A Prep, 6A HPO)*
*Updated: 2026-01-21 (added sections 6.11-6.13: Infrastructure Fixes, Ablation Studies, Final Results)*
*Updated: 2026-01-22 (added section 6.14: Threshold Sweep Analysis, archived 12 stale docs)*
*Consolidates: timeseries_transformer_experimentation_project.md, completed phases from project_phase_plans.md*
