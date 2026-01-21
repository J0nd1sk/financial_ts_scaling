# Session Handoff - 2026-01-20 ~17:30 UTC

## Current State

### Branch & Git
- **Branch**: main
- **Last commit**: `10f6825` exp: Focal Loss breakthrough - PatchTST AUC 0.54→0.67
- **Uncommitted changes**: None
- **Ahead of origin**: 5 commits (not pushed)

### Task Status
- **Working on**: Model comparison (RF/XGBoost/GB vs PatchTST) and loss function investigation
- **Status**: MAJOR BREAKTHROUGH - Focal Loss fixes PatchTST probability collapse!

---

## 🚨 MAJOR FINDINGS THIS SESSION

### 1. XGBoost Baseline Established
| Threshold | XGBoost AUC | Prediction Range |
|-----------|-------------|------------------|
| 0.5% | 0.7336 | [0.028, 0.860] |
| 1.0% | **0.7555** | [0.045, 0.482] |
| 2.0% | 0.7523 | [0.032, 0.069] |

### 2. PatchTST Probability Collapse Diagnosed
- Standard BCE causes model to predict ~0.5 for EVERYTHING
- Loss settles at 0.693 (BCE for always predicting the mean)
- Prediction std drops to 0.0000 within 5-10 epochs
- This is a **local minimum trap** - "lazy prediction" is stable

### 3. Focal Loss BREAKTHROUGH 🎉
| Loss Function | Best AUC | Improvement |
|---------------|----------|-------------|
| FocalLoss(γ=2) | **0.6717** | +24.8% vs BCE |
| SoftAUCLoss | 0.6149 | +14.2% vs BCE |
| BCE_weighted | 0.6052 | +12.4% vs BCE |
| BCE (baseline) | 0.5383 | - |

**Gap to XGBoost reduced from 22% → 11%!**

### 4. Why Focal Loss Works
- Down-weights easy (already correct) predictions
- Forces model to keep learning on hard examples
- Prevents convergence to "always predict 0.5" local minimum
- Configuration: `FocalLoss(gamma=2.0, alpha=0.25)` with very_low_lr (3e-6), warmup=10

---

## Previous Findings (Earlier Terminals)

### Random Forest BEATS PatchTST on ALL thresholds
| Threshold | Pos Rate | PatchTST | Random Forest | Gradient Boost | Winner |
|-----------|----------|----------|---------------|----------------|--------|
| **0.5%** | 46% | 0.586 | **0.628** | 0.583 | **RF +7%** |
| **1.0%** | 20% | 0.695 | **0.716** | 0.624 | **RF +3%** |
| **2.0%** | 2% | 0.621 | 0.705 | **0.766** | **GB +23%** |

**Key insight**: Signal EXISTS - tree models find it, transformers need Focal Loss to find it.

---

## Test Status
- Last `make test`: 2026-01-20 ~17:25 UTC
- Result: **467 passed**, 2 warnings
- Failing: none

---

## Completed This Session

1. **XGBoost Testing** - established strong baseline (AUC 0.7555 @ 1%)
2. **PatchTST Diagnosis** - identified probability collapse issue
   - Tested 8 LR/warmup/scheduler configurations
   - All converged to Pred Std = 0.0000
3. **Loss Function Investigation** - tested 10 loss functions
   - FocalLoss(γ=2) achieves AUC 0.6717
   - SoftAUCLoss achieves AUC 0.6149 with better spread
4. **Memory Entities Created**:
   - `Finding_FocalLossFixesPatchTST_20260120`
   - `Pattern_TransformerProbabilityCollapse_20260120`
   - `Finding_RFBeatsPatchTST_20260120`
   - `Finding_SignalExistsAt0.5pct_20260120`

---

## Files Created/Modified This Session

| File | Change | Status |
|------|--------|--------|
| `scripts/test_xgboost_thresholds.py` | NEW - XGBoost comparison | COMMITTED |
| `scripts/diagnose_patchtst.py` | NEW - LR/warmup diagnosis | COMMITTED |
| `scripts/test_loss_functions.py` | NEW - Loss function comparison | COMMITTED |
| `outputs/patchtst_diagnosis/` | 8 history CSVs + summary | COMMITTED |
| `outputs/loss_function_comparison/summary.json` | Results | COMMITTED |

---

## Next Session Should

### Immediate
1. **Close the remaining 11% gap** to XGBoost:
   - Test Focal Loss + even lower LR + longer training
   - Try architectural changes (fewer layers, different d_model)
   - Test on other thresholds (0.5%, 2%)

2. **Validate findings on test set**:
   - Run best Focal Loss config on 2025 data
   - Compare to XGBoost on same test set

### Research Questions
1. Does Focal Loss help at all thresholds equally?
2. Can we combine Focal Loss with architectural changes to match XGBoost?
3. Is the remaining gap due to:
   - Temporal patterns that don't exist (favors point-in-time models)?
   - Insufficient model capacity?
   - Need for ensemble approach?

### Code Tasks
1. Update HPO framework to use Focal Loss by default
2. Document findings in research paper notes
3. Consider pushing commits to origin

---

## Memory Entities Updated This Session

- `Finding_FocalLossFixesPatchTST_20260120` - Focal Loss improves AUC from 0.54→0.67
- `Pattern_TransformerProbabilityCollapse_20260120` - BCE causes prediction collapse
- `Finding_RFBeatsPatchTST_20260120` - RF outperforms PatchTST on ALL thresholds
- `Finding_SignalExistsAt0.5pct_20260120` - Signal confirmed with RF AUC 0.628

---

## Commands to Run First
```bash
source venv/bin/activate
make test
git status
git log --oneline -5
```

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

### Current Focus
- Model comparison: RF/XGBoost vs PatchTST with Focal Loss
- Loss function optimization for transformers
- SimpleSplitter + RevIN + Focal Loss are the foundation for neural models
