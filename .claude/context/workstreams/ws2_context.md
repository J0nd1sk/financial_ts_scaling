# Workstream 2 Context: foundation
# Last Updated: 2026-01-24 17:30

## Identity
- **ID**: ws2
- **Name**: foundation
- **Focus**: Foundation model investigation - testing pre-trained time series models
- **Status**: active

---

## Current Task
- **Working on**: TimesFM API v2.5 migration + Colab execution
- **Status**: Notebook updated, ready for Colab execution

---

## Investigation Summary

### Research Question
Can pre-trained foundation models (Lag-Llama, TimesFM) beat task-specific PatchTST on SPY direction prediction?

### Results So Far

| Experiment | AUC | Precision | Recall | Status |
|------------|-----|-----------|--------|--------|
| **PatchTST H1** | **0.718** | 0.58 | 0.45 | BASELINE |
| Lag-Llama (all modes) | 0.499-0.576 | ~0.50 | ~0.50 | FAILED |
| TimesFM TFM-01 | Pending | - | - | **Notebook fixed, ready** |

### Lag-Llama Findings (FAILED)
- Tested 4 configurations: forecast, encoder-mean, encoder-last, all-hidden
- All variants performed at or below random chance
- Hypothesis: Pre-trained on diverse domains, not specialized for financial patterns
- Decision: Move to TimesFM

---

## Last Session Work (2026-01-24 17:30)

### TimesFM API v2.5 Migration Complete
The original notebook used TimesFM 2.0 API which caused `AttributeError: module 'timesfm' has no attribute 'TimesFmHparams'`.

**Changes made to `TimesFM_SPY_Experiments.ipynb`:**

| Cell | Change |
|------|--------|
| Cell 0 (markdown) | Updated "500M" → "200M" model description |
| Cell 2 (install) | Added `pip install jax jaxlib -q` for covariates |
| Cell 6 (config) | Added `MODEL_NAME = "TimesFM-2.5-200M-PyTorch"` |
| Cell 10 (model init) | **Complete rewrite**: `TimesFM_2p5_200M_torch.from_pretrained()` + `ForecastConfig` + `compile()` |
| Cell 12 (inference) | Updated `tfm.forecast(horizon=HORIZON, inputs=inputs)` |
| Cell 14 (fine-tuning) | Rewritten with `TimesFMFinetuner`, `FinetuningConfig`, `TimeSeriesDataset` |
| Cell 16 (covariates) | Updated `forecast_with_covariates()` call signature |
| Cell 22 (multi-runner) | Updated model init to use new API |
| Cell 23 (troubleshooting) | Added AttributeError fix, updated version info |
| Cell 26 (HP sweep) | Updated `run_lr_sweep()` model init |

**API Change Summary:**
```python
# OLD (2.0, broken):
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(backend="gpu", ...),
    checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id="...")
)

# NEW (2.5, working):
tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
tfm.compile(timesfm.ForecastConfig(max_context=80, max_horizon=1, ...))
```

---

## Files Owned/Modified
- `experiments/foundation/` - PRIMARY
  - `TimesFM_SPY_Experiments.ipynb` - **Updated for API v2.5**
  - `train_lagllama_h1_forecast.py` - Lag-Llama experiment script
- `outputs/foundation/` - Results storage

---

## Key Decisions (Workstream-Specific)

### Lag-Llama Abandoned
- **Decision**: Stop investigating Lag-Llama after all configurations failed
- **Rationale**: AUC 0.499-0.576 across all modes = no signal
- **Alternative**: TimesFM may have different architecture better suited to financial data

### TimesFM via Colab
- **Decision**: Run TimesFM on Colab due to GPU requirements
- **Rationale**: M4 MacBook Pro MPS not optimal for TimesFM
- **Notebook**: `experiments/foundation/TimesFM_SPY_Experiments.ipynb`

### TimesFM 2.5 API (2026-01-24)
- **Decision**: Migrate from 2.0 to 2.5 API
- **Rationale**: 2.0 API deprecated, causes AttributeError
- **Model change**: 500M → 200M (2.5 only has 200M PyTorch)
- **Still valid test**: 200M still much larger than Lag-Llama's ~7M

---

## Session History

### 2026-01-24 17:30
- **Fixed TimesFM Colab notebook for API v2.5**
- Updated 9 cells with new API patterns
- Model: TimesFM-2.0-500M → TimesFM-2.5-200M
- Notebook ready for Colab execution

### 2026-01-24 09:00
- Completed Lag-Llama investigation
- Created TimesFM Colab notebook (with 2.0 API - now outdated)
- Paused workstream pending Colab execution

### 2026-01-23
- Set up foundation model investigation
- Created experiment structure
- Ran initial Lag-Llama tests

---

## Next Session Should

### Priority 1: Execute TimesFM Experiments in Colab
1. Upload `SPY_dataset_a20.parquet` to Colab
2. Run TFM-01 (zero-shot, 80-context)
3. **Verify no AttributeError** - new API should work
4. Check for prediction collapse (spread > 0.001)
5. Record AUC, precision, recall results

### Verification Checklist (in Colab)
- [ ] Cell 2 installs complete without errors
- [ ] Cell 10 prints "Model loaded successfully"
- [ ] Cell 12 produces varied predictions (not collapse)
- [ ] AUC metric computable (even if low)

### Based on Results
- If TimesFM > 0.718 AUC: Investigate further, try TFM-03 (fine-tuning)
- If TimesFM ~ 0.65-0.718: Try fine-tuning (TFM-03, TFM-04)
- If TimesFM < 0.60 with collapse: Foundation models not suitable, conclude FD-01

---

## Memory Entities (Workstream-Specific)
- No Memory entities created specifically for this workstream yet
