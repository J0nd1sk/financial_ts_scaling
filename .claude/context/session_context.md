# Session Handoff - 2026-01-23 (Late Session)

## Current State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `ea73990` feat: Implement tier a50 indicators (30 new, 50 total)
- **Uncommitted changes**:
  - `src/models/foundation/lag_llama.py` - **MODIFIED THIS SESSION**: Added forecast mode, get_distribution_params(), compute_nll_loss()
  - `tests/test_lag_llama.py` - **MODIFIED THIS SESSION**: Added 6 new tests for forecast mode (ALL PASSING NOW)
  - `docs/indicator_catalog.md` - From previous session
  - `experiments/foundation/train_lagllama_h1_close.py` - From previous session
  - `experiments/foundation/train_lagllama_h1_proj.py` - From previous session
  - `.claude/context/session_context.md` - This file
- **Untracked files**:
  - `experiments/foundation/train_lagllama_h1_headonly.py` - From previous session
  - `outputs/foundation/` - Experiment outputs

### Task Status
- **Working on**: FD-01d - Lag-Llama Forecast-Then-Threshold approach
- **Status**: Tasks 1-2 COMPLETE, Task 3 (experiment script) APPROVED but not yet created

---

## Test Status
- Last `make test`: 2026-01-23
- Result: **553 passed**, 2 skipped, 19 warnings
- **ALL TESTS PASSING** - Previous failing test fixed this session

---

## Completed This Session

### FD-01d Implementation (Tasks 1-2)

**Objective**: Use Lag-Llama for forecasting (its pretrained task) instead of classification, then threshold at inference.

**Task 1: Added 6 New Tests (TDD RED → GREEN)**
1. `TestLagLlamaForecastMode::test_forecast_mode_returns_raw_forecast`
2. `TestLagLlamaForecastMode::test_forecast_mode_output_varies`
3. `TestLagLlamaDistributionParams::test_get_distribution_params_shapes`
4. `TestLagLlamaNLLLoss::test_compute_nll_loss_scalar`
5. `TestLagLlamaNLLLoss::test_compute_nll_loss_has_gradient`
6. `TestLagLlamaTrainerIntegration::test_classification_mode_unchanged`

**Task 2: LagLlamaWrapper Modifications**

Changes to `src/models/foundation/lag_llama.py`:

1. **New parameter**: `mode: Literal["classification", "forecast"] = "classification"`
   - `"classification"`: Returns P(X > threshold) via CDF (existing behavior)
   - `"forecast"`: Returns raw `loc` (predicted value) without clamping

2. **New method**: `get_distribution_params(x) -> (df, loc, scale)`
   - Returns StudentT distribution parameters from backbone
   - df shape: `(batch, internal_context_len)` (58 for Lag-Llama)
   - loc/scale shape: `(batch, 1)` (forecast-specific)

3. **New method**: `compute_nll_loss(x, target) -> scalar`
   - Computes negative log-likelihood for StudentT distribution
   - Native loss function for Lag-Llama fine-tuning
   - Takes last position of df to match loc/scale shapes

4. **Modified**: `forward()` refactored
   - Uses `get_distribution_params()` internally
   - Handles shape mismatches correctly
   - Returns `loc` directly for forecast mode

**Bug Fixed**: Shape mismatch between df `(batch, 58)` and loc/scale `(batch, 1)` - now correctly uses last position of df.

---

## Pending (Approved, Not Started)

### Task 3: Create Experiment Script FD-01d
**File**: `experiments/foundation/train_lagllama_h1_forecast.py` (NEW)

**Status**: **APPROVED** by user just before handoff request

**Key Config**:
```python
MODE = "forecast"
CONTEXT_LENGTH = 1150
NUM_FEATURES = 1  # Univariate (close returns only)
THRESHOLD = 0.01
EPOCHS = 30
LR = 1e-4
BATCH_SIZE = 4
```

**Key Differences from FD-01b**:
1. **Target**: Returns (`close[t+1]/close[t] - 1`) instead of binary
2. **Loss**: NLL via `model.compute_nll_loss()` instead of BCE
3. **Eval**: Threshold forecasts at 1% for AUC comparison

### Task 4: Run Experiment and Analyze
- Compare to baselines: FD-01b AUC 0.576, PatchTST AUC 0.718
- Success: AUC >= 0.74 (5% improvement over PatchTST)
- Fallback: If fails, pivot to TimesFM

---

## Key Decisions Made

1. **Forecast-then-threshold approach**: Train Lag-Llama on its native forecasting task, then threshold predictions at inference for classification metrics.

2. **NLL loss for fine-tuning**: Use StudentT negative log-likelihood (native loss) instead of BCE.

3. **Shape handling**: df from backbone is `(batch, 58)`, loc/scale are `(batch, 1)` - use last position of df for consistency.

---

## FD-01d Plan Summary

**Key Insight**: Classification approach (FD-01b) produced near-constant predictions. Lag-Llama was pretrained for forecasting, not classification. Using its native task may give better results.

**Success Criteria**:
| Metric | Target | Rationale |
|--------|--------|-----------|
| Val AUC | >= 0.74 | 5% improvement over PatchTST (0.718) |
| Forecast range | > 0.05 spread | Must show variation (not near-constant) |
| Recall | > 0% | Must predict some positives |
| Val NLL | Decreasing | Model should improve with fine-tuning |

**Fallback**: If FD-01d fails, pivot to TimesFM investigation.

---

## Files Modified This Session
| File | Lines Changed | Description |
|------|---------------|-------------|
| `src/models/foundation/lag_llama.py` | ~80 added | mode param, get_distribution_params(), compute_nll_loss(), forward() refactor |
| `tests/test_lag_llama.py` | ~120 added | 6 new tests for forecast mode and NLL loss |

---

## Next Session Should

1. **Create experiment script** (Task 3 - APPROVED):
   ```bash
   # File: experiments/foundation/train_lagllama_h1_forecast.py
   ```

2. **Run FD-01d experiment**:
   ```bash
   ./venv/bin/python experiments/foundation/train_lagllama_h1_forecast.py
   ```

3. **Analyze results**:
   - Check `outputs/foundation/lagllama_h1_forecast/results.json`
   - Compare AUC to baselines (FD-01b: 0.576, PatchTST: 0.718)
   - Check forecast range (must vary, not constant)

4. **Decision point**:
   - If AUC >= 0.74: Lag-Llama viable, consider FD-01e
   - If AUC < 0.70 but forecast varies: Lag-Llama forecasts don't rank well
   - If forecast near-constant: Fundamental architecture mismatch, pivot to TimesFM

---

## Commands to Run First

```bash
source venv/bin/activate
make test
git status
make verify
```

---

## Experiment Baselines

| Experiment | AUC | Notes |
|------------|-----|-------|
| PatchTST H1 | 0.718 | Target baseline to beat |
| FD-01a (zero-shot) | 0.499 | Random performance |
| FD-01b (fine-tune classification) | 0.576 | Near-constant predictions |
| FD-01c (head-only) | 0.512 | Failed |
| **FD-01d (forecast)** | **TBD** | Current experiment |

---

## User Preferences (Authoritative)

### Development Approach
- TDD: tests first, always
- Planning sessions before implementation
- Uses tmux for long-running experiments

### Context Durability
- Document in multiple places: Memory MCP + context files + docs/
- Code comments are secondary, not primary durability

### Documentation Philosophy
- Flat docs/ structure (no subdirs except research_paper/, archive/)
- Precision in language - never reduce fidelity
- Consolidate rather than delete - preserve historical context

### Communication Standards
- Precision over brevity
- Never summarize away important details
- Evidence-based claims

### Hyperparameters (Fixed - Ablation-Validated)
Always use unless new ablation evidence supersedes:
- **Dropout**: 0.5
- **Learning Rate**: 1e-4
- **Context Length**: 80 days (PatchTST) / 1150 days (Lag-Llama)
- **Normalization**: RevIN only (no z-score) for PatchTST
- **Splitter**: SimpleSplitter (442 val samples) - except Lag-Llama needs custom splitting
- **Head dropout**: 0.0 (ablation showed no benefit)
- **Metrics**: AUC, accuracy, precision, recall, pred_range (all required)

### Foundation Model Investigation
- PatchTST baseline: H1 AUC 0.718 (target to beat: 0.74 = 5% improvement)
- Lag-Llama classification: FAILED (all experiments < 0.60 AUC)
- **Current**: FD-01d forecast approach - implementation complete, experiment pending
- **Fallback**: TimesFM via Colab (decoder-based)
