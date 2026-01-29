# Workstream 2 Context: foundation
# Last Updated: 2026-01-28 15:00

## Identity
- **ID**: ws2
- **Name**: foundation
- **Focus**: Foundation model & alternative architecture investigation
- **Status**: METHODOLOGY CORRECTION - v3 design ready, pending implementation

---

## Current Task
- **Working on**: Alternative Architecture HPO v3 Design
- **Status**: Documentation complete, awaiting implementation approval

---

## Investigation Summary

### Research Questions
1. Can pre-trained foundation models (Lag-Llama, TimesFM) beat task-specific PatchTST?
2. Can alternative transformer architectures (iTransformer, Informer) beat PatchTST?
3. **CRITICAL DISCOVERY**: v1/v2 had fundamental methodology flaw (regression vs classification)

### Foundation Models - Results (FINAL)

| Experiment | Val AUC | vs PatchTST | Status |
|------------|---------|-------------|--------|
| **PatchTST 200M** | **0.718** | Baseline | BEST |
| TimesFM TFM-01 | 0.364 | -49% | Anti-correlated |
| TimesFM TFM-07 (50 features) | 0.364 | -49% | IDENTICAL to TFM-01 |
| TimesFM (inverted) | 0.636 | -11% | Still below baseline |
| Lag-Llama (all modes) | 0.499-0.576 | -20% to -30% | FAILED |

**Conclusion**: Foundation models failed due to domain mismatch (pre-trained on non-financial data). Results stand - no rerun needed.

### Alternative Architectures - v2 Results (INVALID)

| Experiment | Val AUC | Recall | Prediction Range | Status |
|------------|---------|--------|------------------|--------|
| iTransformer v2 | 0.621 | **0%** | [0.004, 0.004] | INVALID |
| Informer v2 | 0.669 | **0%** | ~constant | INVALID |

**Root Cause**: Trained as regressors (MAE on returns), evaluated as classifiers. See `docs/methodology_lessons_v1_v2.md`.

---

## Critical Discovery: Methodology Flaw (2026-01-28)

### What Went Wrong

| Aspect | v1/v2 (Wrong) | PatchTST (Correct) | v3 (Planned) |
|--------|---------------|-------------------|--------------|
| Loss | MAE on returns | BCE on binary | Bernoulli |
| Target | Float returns | Binary (0/1) | Binary (0/1) |
| Output | ~0.005 range | [0, 1] probabilities | [0, 1] probabilities |
| Task | Regression | Classification | Classification |

### Why 0% Recall

1. MAE loss trains model to predict expected returns (~0.005)
2. All predictions cluster in [0.004, 0.006] range
3. Threshold at 0.5 → no positive predictions
4. Result: 0% recall despite moderate AUC (ranking vs calibration)

### v3 Correct Approach

```python
from neuralforecast.losses.pytorch import DistributionLoss
loss = DistributionLoss(distribution='Bernoulli')
```

---

## Last Session Work (2026-01-28 15:00)

### Methodology Correction Documentation
1. Created `docs/methodology_lessons_v1_v2.md` - What went wrong
2. Created `docs/architecture_hpo_v3_design.md` - Correct approach
3. Updated `docs/project_history.md` - Section 6.16
4. Updated `docs/research_paper/notes/project_journey.md` - Section 9
5. Updated `.claude/context/decision_log.md` - Entry 2026-01-28
6. Created Memory MCP entity for lesson

### Key Insight
**Always match training objective to evaluation objective.**

If you want classification (AUC, precision, recall), train with classification loss (BCE, Bernoulli).

---

## Files Owned/Modified
- `experiments/foundation/` - PRIMARY
- `experiments/architectures/` - PRIMARY
- `experiments/architectures/hpo_neuralforecast.py` - Needs v3 update
- `outputs/foundation/` - Results
- `outputs/architectures/` - Results (v2 invalid)
- `docs/methodology_lessons_v1_v2.md` - NEW
- `docs/architecture_hpo_v3_design.md` - NEW
- `docs/foundation_model_results.md` - Documentation
- `docs/architecture_comparison_results.md` - Needs update for v2 invalidity

---

## Key Decisions (Workstream-Specific)

### Methodology Correction (2026-01-28)
- **Context**: v1/v2 had 0% recall due to regression/classification mismatch
- **Decision**: Discard v2 results, design v3 with proper Bernoulli loss
- **Rationale**: Task mismatch caused invalid results
- **Documented**: methodology_lessons_v1_v2.md, architecture_hpo_v3_design.md

### Foundation Models Conclusion (2026-01-25)
- **Finding**: Domain mismatch is primary issue, not task alignment
- **Decision**: No rerun needed for foundation models
- **Status**: Investigation complete, PatchTST wins

---

## Session History

### 2026-01-28 15:00
- Discovered methodology flaw in v1/v2 experiments
- Created comprehensive documentation
- Updated project history and decision log
- v3 design ready for implementation

### 2026-01-26 12:00
- Created HPO script for NeuralForecast models
- Fixed early stopping (val_size to fit(), not constructor)
- Smoke test passed (3 trials in 0.9 min)
- **NOTE**: v2 results now known to be invalid

### 2026-01-25 15:00
- Fixed NeuralForecast bugs (loss API, early stopping, parameter naming)
- Ran iTransformer: AUC 0.517 → 0.621 (v2)
- Ran Informer: AUC 0.587 → 0.669 (v2)
- **NOTE**: Results invalid due to methodology flaw

---

## Next Session Should

1. **Review v3 design** with user
   - Confirm Bernoulli loss approach is acceptable
   - Approve implementation plan

2. **Implement v3 changes** to `hpo_neuralforecast.py`
   - Change `MAE()` to `DistributionLoss('Bernoulli')`
   - Verify binary targets passed correctly

3. **Run v3 smoke test** (3 trials)
   - Verify predictions in [0, 1] range
   - Verify prediction spread > 0.1
   - Verify recall > 0%

4. **If smoke test passes**, run full HPO
   - iTransformer first (50 trials)
   - Informer second (50 trials)

5. **Decision point after v3**
   - If AUC >= 0.65: Consider horizon experiments
   - If AUC < 0.65: Close investigation, PatchTST wins

---

## Memory Entities (Workstream-Specific)
- `Alternative_Architecture_Methodology_Lesson_20260128` - Critical lesson learned
