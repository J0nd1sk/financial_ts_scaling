# Workstream 2 Context: foundation
# Last Updated: 2026-01-26 12:00

## Identity
- **ID**: ws2
- **Name**: foundation
- **Focus**: Foundation model & alternative architecture investigation
- **Status**: HPO SCRIPT READY - Awaiting full HPO runs

---

## Current Task
- **Working on**: Alternative Architecture HPO Investigation
- **Status**: HPO script created, smoke test passed, ready for full runs

---

## Investigation Summary

### Research Questions
1. Can pre-trained foundation models (Lag-Llama, TimesFM) beat task-specific PatchTST?
2. Can alternative transformer architectures (iTransformer, Informer) beat PatchTST?
3. **NEW**: Can proper HPO (dropout tuning, LR exploration) fix alternative architectures?

### Original Results - Foundation Models

| Experiment | Val AUC | vs PatchTST | Status |
|------------|---------|-------------|--------|
| **PatchTST 200M** | **0.718** | Baseline | BEST |
| TimesFM TFM-01 | 0.364 | -49% | Anti-correlated |
| TimesFM TFM-07 (50 features) | 0.364 | -49% | **IDENTICAL to TFM-01** |
| TimesFM (inverted) | 0.636 | -11% | Still below baseline |
| Lag-Llama (all modes) | 0.499-0.576 | -20% to -30% | FAILED |

### Original Results - Alternative Architectures (UNFAIR - 1 config each)

| Experiment | Val AUC | vs PatchTST | Status |
|------------|---------|-------------|--------|
| **PatchTST 200M** | **0.718** | Baseline | BEST |
| iTransformer (ARCH-01) | 0.517 | **-28%** | 1 config only |
| Informer (ARCH-02) | 0.587 | **-18%** | 1 config only |

### NEW: HPO Investigation Plan (2026-01-26)

Key insight: PatchTST went through 50+ trials of HPO. Alternative architectures only ran with ONE configuration each - no dropout tuning, no LR exploration, only 500 steps.

**Hypothesis**: dropout=0.5 (critical for PatchTST) might fix probability collapse in alternative architectures.

**HPO Search Space**:
- dropout: [0.3, 0.4, 0.5]
- learning_rate: [5e-5, 1e-4, 2e-4]
- hidden_size: [64, 128, 256]
- n_layers: [2, 3, 4]
- n_heads: [2, 4, 8]
- max_steps: [1000, 2000]
- batch_size: [16, 32, 64]

**Success criteria**: AUC >= 0.70 (comparable to PatchTST 0.718)

---

## Last Session Work (2026-01-26 12:00)

### Alternative Architecture HPO Script Created
1. Created `experiments/architectures/hpo_neuralforecast.py` (~400 lines)
   - Optuna-based HPO with TPE sampler (20 startup trials)
   - Supports both `--model itransformer` and `--model informer`
   - `--dry-run` flag for testing
   - `--resume` flag to continue interrupted studies
   - SQLite study storage for persistence
   - Incremental result saving (trials dir + best_params.json + study_summary.md)

2. Updated `experiments/architectures/common.py` with HPO utilities
   - `prepare_hpo_data()` - Shared data preparation for HPO
   - `format_hpo_results_table()` - Markdown table formatting

3. Created output directories
   - `outputs/hpo/architectures/itransformer/trials/`
   - `outputs/hpo/architectures/informer/trials/`

4. Fixed NeuralForecast early stopping
   - `val_size` must be passed to `nf.fit()`, not model constructor
   - Per NeuralForecast docs: https://nixtlaverse.nixtla.io/neuralforecast/models.itransformer.html

5. Ran smoke test (3 trials)
   - Script runs without crashing
   - Training completes (0.9 min for 3 trials)
   - AUC=0 in smoke test (probability collapse persists with random params)

6. Ran `make test` - 944 passed, 2 skipped

### Files Created/Modified
- `experiments/architectures/hpo_neuralforecast.py` - NEW (untracked)
- `experiments/architectures/common.py` - MODIFIED
- `outputs/hpo/architectures/` - NEW (directory structure)

---

## Files Owned/Modified
- `experiments/foundation/` - PRIMARY
- `experiments/architectures/` - PRIMARY
- `experiments/architectures/hpo_neuralforecast.py` - NEW (HPO script)
- `outputs/foundation/` - Results
- `outputs/architectures/` - Results
- `outputs/hpo/architectures/` - HPO results (NEW)
- `docs/foundation_model_results.md` - Documentation
- `docs/architecture_comparison_results.md` - Documentation

---

## Key Decisions (Workstream-Specific)

### HPO Investigation Approved (2026-01-26)
- **Context**: Original alternative architecture tests were unfair (1 config each vs 50+ for PatchTST)
- **Decision**: Run proper HPO with comparable rigor
- **Rationale**: dropout=0.5 was critical for PatchTST; never tested on alternatives

### NeuralForecast val_size Fix (2026-01-26)
- **Issue**: `early_stop_patience_steps` requires `val_size`
- **Fix**: Pass `val_size=100` to `nf.fit()`, not model constructor
- **Source**: NeuralForecast GitHub issue #435

---

## Session History

### 2026-01-26 12:00
- Created HPO script for NeuralForecast models
- Fixed early stopping (val_size to fit(), not constructor)
- Smoke test passed (3 trials in 0.9 min)
- make test passed (944/946)
- **Next**: Full 50-trial HPO runs

### 2026-01-25 15:00
- Fixed NeuralForecast bugs (loss API, early stopping, parameter naming)
- Ran iTransformer: AUC 0.517 (-28%)
- Ran Informer: AUC 0.587 (-18%)
- **Conclusion**: Both architectures FAILED (but unfair comparison)

---

## Next Session Should

1. **Run iTransformer HPO (50 trials)**
   ```bash
   caffeinate -i ./venv/bin/python experiments/architectures/hpo_neuralforecast.py --model itransformer --trials 50 2>&1 | tee outputs/hpo/architectures/itransformer/hpo.log
   ```
   Estimated time: 8-10 hours (overnight)

2. **Run Informer HPO (50 trials)**
   ```bash
   caffeinate -i ./venv/bin/python experiments/architectures/hpo_neuralforecast.py --model informer --trials 50 2>&1 | tee outputs/hpo/architectures/informer/hpo.log
   ```

3. **Analyze results** after both complete
   - Compare best configs to PatchTST baseline
   - Create `docs/architecture_hpo_results.md`

4. **Decision point**
   - If AUC >= 0.70: Consider horizon experiments (H3, H5)
   - If AUC < 0.65: Confirm architecture not viable, close investigation

---

## Memory Entities (Workstream-Specific)
- None created this session
