# Architecture Exploration Plan

## Motivation

Tree-based models (XGBoost, RF) outperform PatchTST transformers on this task. The transformers appear to converge early, collapsing to majority-class prediction before uncovering the signal that tree models find.

**Research Question**: Can architectural changes prevent early convergence and allow transformers to find the signal?

## Hypotheses

1. **Shallow+Wide**: Fewer layers with more capacity per layer may behave more like ensemble methods
2. **Smaller Models**: 200K-500K params may have less capacity to memorize noise
3. **No Attention**: Self-attention may be overkill; simple MLP on patches might work better
4. **Tree Baseline**: XGBoost on same features proves signal strength for comparison

---

## Experiment 1: Shallow + Wide Ablation

**Hypothesis**: Shallow models (1-2 layers) with wide dimensions prevent early convergence by reducing compositional depth while maintaining capacity.

**Configurations to test**:

| Config | Layers | d_model | n_heads | d_ff | Est. Params |
|--------|--------|---------|---------|------|-------------|
| L1_d256 | 1 | 256 | 4 | 1024 | ~400K |
| L1_d512 | 1 | 512 | 8 | 2048 | ~1.5M |
| L1_d768 | 1 | 768 | 8 | 3072 | ~3.5M |
| L2_d256 | 2 | 256 | 4 | 1024 | ~700K |
| L2_d512 | 2 | 512 | 8 | 2048 | ~2.8M |
| L2_d768 | 2 | 768 | 8 | 3072 | ~6.5M |

**Baseline comparison**: Current best 2M config (L=4, d=64)

**Metrics**: AUC, accuracy, prediction spread, convergence epoch

**Expected runtime**: ~30 min (12 configs × ~2-3 min each)

---

## Experiment 2: Remove Attention (MLP-Only)

**Hypothesis**: Self-attention may be unnecessary overhead; a simple MLP on flattened patches might work as well or better.

**Approach**:
- Flatten patches into single vector
- Apply MLP layers (no attention mechanism)
- Compare to equivalent-param transformer

**Configurations**:

| Config | Architecture | Hidden dims | Est. Params |
|--------|-------------|-------------|-------------|
| MLP_small | Flatten → 256 → 128 → 1 | [256, 128] | ~200K |
| MLP_medium | Flatten → 512 → 256 → 1 | [512, 256] | ~800K |
| MLP_large | Flatten → 1024 → 512 → 256 → 1 | [1024, 512, 256] | ~2M |

**Implementation**: May need new model class or modify PatchTST to bypass attention.

---

## Experiment 3: XGBoost Baseline

**Hypothesis**: XGBoost on the exact same 20 features will significantly outperform transformers, proving the signal exists and the issue is transformer architecture/optimization.

**Approach**:
- Use same train/val/test splits (SimpleSplitter)
- Use same 20 features
- Use same horizons (h1, h3, h5)
- Standard XGBoost hyperparameters (can tune later)

**Configurations**:
- XGBoost with default params
- XGBoost with light tuning (max_depth, n_estimators, learning_rate)

**Expected outcome**: AUC significantly higher than transformer (~0.70-0.75 vs ~0.65-0.67)

---

## Experiment 4: Smaller Models (200K-500K params)

**Hypothesis**: Very small models cannot memorize noise and must learn generalizable patterns.

**Configurations**:

| Config | Layers | d_model | n_heads | d_ff | Est. Params |
|--------|--------|---------|---------|------|-------------|
| 200K_a | 2 | 32 | 2 | 128 | ~200K |
| 200K_b | 1 | 64 | 2 | 256 | ~200K |
| 500K_a | 2 | 64 | 2 | 256 | ~500K |
| 500K_b | 3 | 48 | 2 | 192 | ~500K |

**Rationale**:
- Current 2M model may be overparameterized for ~7K training samples
- Rule of thumb: params should be << 10× training samples
- 200K-500K params with 7K samples is more reasonable ratio

---

## Experiment 5: Training Dynamics (if time permits)

**Hypothesis**: Optimization settings cause early convergence, not architecture.

**Configurations**:
- Much lower LR: 1e-5, 1e-6 (vs current 1e-4)
- Higher dropout: 0.4, 0.5 (vs current 0.2)
- Early stopping on AUC (vs loss)
- Longer training: 30-50 epochs with patience

---

## Execution Order

1. **XGBoost baseline** (30 min) - Establishes target AUC to beat
2. **Smaller models** (30 min) - Tests overparameterization hypothesis
3. **Shallow+wide** (30 min) - Tests depth vs width tradeoff
4. **MLP-only** (1-2 hrs) - Requires implementation work

## Success Criteria

- Find configuration with AUC > 0.70 (matching tree models)
- OR conclusively show transformers cannot match trees on this task
- Document which architectural choices matter most

---

## Files to Create

- `scripts/test_shallow_wide.py` - Shallow+wide ablation
- `scripts/test_xgboost_baseline.py` - XGBoost comparison
- `scripts/test_small_models.py` - 200K-500K param models
- `src/models/mlp_baseline.py` - MLP-only model (if needed)

---

*Created: 2026-01-20*
*Status: Planning*
