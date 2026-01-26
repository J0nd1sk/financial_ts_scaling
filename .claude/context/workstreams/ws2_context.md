# Workstream 2 Context: foundation
# Last Updated: 2026-01-25 15:00

## Identity
- **ID**: ws2
- **Name**: foundation
- **Focus**: Foundation model & alternative architecture investigation
- **Status**: INVESTIGATION COMPLETE - Both paths rejected

---

## Current Task
- **Working on**: Alternative Architecture Investigation (ARCH-01, ARCH-02)
- **Status**: COMPLETE - Both architectures failed vs PatchTST

---

## Investigation Summary

### Research Questions
1. Can pre-trained foundation models (Lag-Llama, TimesFM) beat task-specific PatchTST?
2. Can alternative transformer architectures (iTransformer, Informer) beat PatchTST?

### Final Results - Foundation Models

| Experiment | Val AUC | vs PatchTST | Status |
|------------|---------|-------------|--------|
| **PatchTST 200M** | **0.718** | Baseline | BEST |
| TimesFM TFM-01 | 0.364 | -49% | Anti-correlated |
| TimesFM TFM-07 (50 features) | 0.364 | -49% | **IDENTICAL to TFM-01** |
| TimesFM (inverted) | 0.636 | -11% | Still below baseline |
| Lag-Llama (all modes) | 0.499-0.576 | -20% to -30% | FAILED |

### Final Results - Alternative Architectures (NEW - 2026-01-25)

| Experiment | Val AUC | vs PatchTST | Status |
|------------|---------|-------------|--------|
| **PatchTST 200M** | **0.718** | Baseline | BEST |
| iTransformer (ARCH-01) | 0.517 | **-28%** | FAILED - barely above random |
| Informer (ARCH-02) | 0.587 | **-18%** | FAILED - probability collapse |

### Critical Discovery: Covariates Ignored (Foundation Models)
- TimesFM predictions identical with 1 vs 50 features (correlation 1.0000000000)
- Foundation models cannot use feature engineering

### Critical Discovery: Architecture Mismatch (Alternative Architectures)
- iTransformer's inverted attention loses temporal patterns
- Informer's forecasting→threshold approach causes probability collapse
- Both show narrow prediction ranges (collapsed to mean)

---

## Last Session Work (2026-01-25 15:00)

### Alternative Architecture Investigation
1. Fixed NeuralForecast loss API bug (`loss="MSE"` → `loss=MSE()`)
2. Fixed early stopping conflicts with cross_validation
3. Fixed Informer parameter naming (`e_layers` → `encoder_layers`)
4. Ran ARCH-01 (iTransformer): **AUC 0.517** (-28% vs baseline)
5. Ran ARCH-02 (Informer): **AUC 0.587** (-18% vs baseline)
6. Updated `docs/architecture_comparison_results.md` with full analysis

### Bugs Fixed
- NeuralForecast requires loss objects, not strings
- Early stopping conflicts with cross_validation (removed)
- Parameter naming differences between original papers and NeuralForecast

### Files Created/Modified
- `experiments/architectures/itransformer_forecast.py` - FIXED
- `experiments/architectures/informer_forecast.py` - FIXED
- `outputs/architectures/itransformer_forecast/results.json` - NEW
- `outputs/architectures/informer_forecast/results.json` - NEW
- `docs/architecture_comparison_results.md` - UPDATED with results

---

## Files Owned/Modified
- `experiments/foundation/` - PRIMARY
- `experiments/architectures/` - PRIMARY (NEW)
- `outputs/foundation/` - Results
- `outputs/architectures/` - Results (NEW)
- `docs/foundation_model_results.md` - Documentation
- `docs/architecture_comparison_results.md` - Documentation (UPDATED)

---

## Key Decisions (Workstream-Specific)

### Alternative Architecture Investigation Closed (2026-01-25)
- **Context**: Tested iTransformer and Informer via NeuralForecast
- **Finding**: Both significantly worse than PatchTST (17-28% lower AUC)
- **Root cause**: Task mismatch (forecasting→threshold) and attention mechanism unsuitability
- **Decision**: ABANDON architecture investigation

### Foundation Model Investigation Closed (2026-01-26)
- **Finding**: TimesFM ignores covariates entirely
- **Decision**: Foundation model path is not viable

---

## Session History

### 2026-01-25 15:00
- Implemented Alternative Architecture Investigation plan
- Fixed NeuralForecast bugs (loss API, early stopping, parameter naming)
- Ran iTransformer: AUC 0.517 (-28%)
- Ran Informer: AUC 0.587 (-18%)
- Documented results in architecture_comparison_results.md
- **Conclusion**: Both architectures FAILED

### 2026-01-26 09:00
- Analyzed TFM-07 results - discovered covariates completely ignored
- Created foundation_model_results.md with full analysis
- Computed prediction correlation (1.0000000000)

### 2026-01-25 23:55
- Created TimesFM_a50/a100 notebooks

---

## Next Session Should

1. **Discuss and analyze** what happened (per user request)
   - Why did iTransformer fail so badly?
   - Why did Informer show probability collapse?
   - What does this tell us about financial time series?

2. **Make final decision** on closing ws2
   - Both foundation models AND alternative architectures failed
   - Strong evidence to focus exclusively on PatchTST + feature scaling

3. **Archive** or **continue** (user choice)
   - If archive: Move to archive, update global context
   - If continue: Could test direct classification (Phase 2 of arch plan)

---

## Investigation Conclusions

### Foundation Models: NOT VIABLE
1. Lag-Llama: 20% below PatchTST
2. TimesFM: Anti-correlated AND ignores all covariates

### Alternative Architectures: NOT VIABLE
1. iTransformer: 28% below PatchTST (inverted attention loses temporal patterns)
2. Informer: 18% below PatchTST (probability collapse, all-negative predictions)

### Recommendation
**FOCUS ON PHASE 6C** with PatchTST. The baseline architecture is actually the best option discovered. Feature scaling (tier_a100 → tier_a200) is the most promising path forward.

---

## Memory Entities (Workstream-Specific)
- None created this session
