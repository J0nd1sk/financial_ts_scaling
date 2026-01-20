# Phase 6A Best Practices Audit

**Created:** 2026-01-20
**Purpose:** Comprehensive audit of our experimental setup against best practices for financial time-series transformer models
**Goal:** Identify gaps and experiments needed to optimize the entire pipeline

---

## Executive Summary

We found one critical bug (feature normalization). But finding one bug doesn't mean the system is optimal. This audit systematically reviews every component against best practices to identify what experiments we need to determine the optimal setup.

**Key Question:** What experiments do we need to run to know we have the best possible setup?

---

## 1. Feature Engineering & Normalization

### Current State
- **Raw features used**: Close, Open, High, Low, Volume, OHLCV-derived indicators
- **Normalization**: ❌ NONE (critical bug found)
- **Features**: 20-25 technical indicators (RSI, MACD, Bollinger Bands, etc.)

### Best Practices for Financial Time-Series

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| Z-score (train stats) | `(x - μ_train) / σ_train` | Simple, preserves relationships | Requires storing params, OOD on regime change |
| Rolling Z-score | `(x - μ_window) / σ_window` | Adapts to regime changes | Different normalization per sample |
| Percent returns | `(x_t - x_{t-1}) / x_{t-1}` | Naturally stationary | Loses absolute levels |
| Log returns | `log(x_t / x_{t-1})` | Symmetric, additive | Same as above |
| Rank transform | Percentile within window | Bounded [0,1], robust | Loses magnitude info |
| Bounded indicators only | RSI, %B, percentiles | No normalization needed | Limited feature set |

### Gap Analysis

| Issue | Severity | Current | Best Practice |
|-------|----------|---------|---------------|
| No normalization | 🔴 CRITICAL | Raw values | Must normalize |
| Price levels used directly | 🔴 HIGH | Close, Open, etc. | Use returns or normalize |
| Non-stationary features | 🟡 MEDIUM | MACD, ATR scale with price | Use percent-based versions |
| Feature selection | 🟡 MEDIUM | Fixed 20 indicators | Should validate usefulness |

### Experiments Needed

1. **Normalization comparison**: Z-score vs rolling Z-score vs log returns
2. **Feature ablation**: Which features actually help? (RF feature importance as baseline)
3. **Bounded vs unbounded**: Compare using only bounded features (RSI, %B, etc.)
4. **Lookback sensitivity**: Does normalization window size matter?

---

## 2. Train/Validation/Test Splits

### Current State
- **Splitter**: ChunkSplitter with non-overlapping chunks
- **Mode**: "contiguous" - sequential chunks
- **Val ratio**: 15% → **only 19 samples!**
- **Time periods**: Train through 2020, Val 2021-2022, Test 2023+

### Best Practices for Financial Time-Series

| Approach | Description | When to Use |
|----------|-------------|-------------|
| Time-based holdout | Fixed date cutoffs | Standard, simple |
| Expanding window | Train grows over time | Walk-forward validation |
| Rolling window | Fixed train size, moves forward | Regime change detection |
| Purged K-fold | K-fold with gap to prevent leakage | More data efficiency |
| Combinatorial purged CV | Multiple train/test combinations | Research-grade validation |

### Gap Analysis

| Issue | Severity | Current | Best Practice |
|-------|----------|---------|---------------|
| Val set too small | 🔴 CRITICAL | 19 samples | 100+ minimum for reliable metrics |
| Non-overlapping chunks | 🟡 MEDIUM | Strict isolation | May be overly conservative |
| No embargo period | 🟡 MEDIUM | None | Gap between train/val to prevent leakage |
| Single validation set | 🟡 MEDIUM | One split | Consider walk-forward or CV |

### Experiments Needed

1. **Val set sizing**: 50 vs 100 vs 200 vs 500 samples - what's minimum for stable AUC?
2. **Sliding vs chunked**: Does strict chunk isolation help or hurt?
3. **Embargo period**: Add gap between train/val/test - does it change results?
4. **Walk-forward validation**: Compare to single holdout

---

## 3. Loss Function

### Current State
- **Loss**: BCELoss (binary cross-entropy)
- **Class imbalance**: ❌ Not handled (h1 has 10% positive rate)
- **Output**: Sigmoid in model, probabilities to loss

### Best Practices for Imbalanced Binary Classification

| Loss | Description | When to Use |
|------|-------------|-------------|
| BCE | Standard binary CE | Balanced classes |
| BCE + pos_weight | Weighted BCE | Moderate imbalance |
| Focal Loss | Down-weights easy examples | Hard example mining |
| AUC Loss (soft) | Differentiable AUC proxy | When ranking matters |
| Contrastive/Triplet | Margin-based separation | Embedding learning |
| Label smoothing | Soft targets | Calibration, regularization |

### Gap Analysis

| Issue | Severity | Current | Best Practice |
|-------|----------|---------|---------------|
| No class imbalance handling | 🔴 HIGH | BCE only | pos_weight or focal |
| BCE allows prior collapse | 🟡 MEDIUM | Can output constant | Use AUC loss or margin |
| Sigmoid in model | 🟡 LOW | Limits loss choices | Output logits, apply in loss |

### Experiments Needed

1. **BCE vs BCE+pos_weight vs Focal vs SoftAUC**: Which gives best test AUC?
2. **Combined losses**: BCE + AUC regularization?
3. **Label smoothing**: Does it help calibration?
4. **Logits vs probabilities**: Move sigmoid to loss function

---

## 4. Model Architecture

### Current State
- **Model**: PatchTST (patch-based transformer)
- **Patch size**: 16, stride 8
- **Context length**: 60 days
- **Prediction head**: Flatten + Linear (very simple)

### Best Practices for Time-Series Transformers

| Architecture | Key Feature | Best For |
|--------------|-------------|----------|
| PatchTST | Patch tokenization | Long sequences, efficiency |
| Informer | ProbSparse attention | Very long sequences |
| Autoformer | Auto-correlation | Seasonal patterns |
| FEDformer | Frequency domain | Periodic signals |
| Simple MLP | Baseline | Sanity check |
| LSTM/GRU | Recurrent | Sequential dependencies |

### Gap Analysis

| Issue | Severity | Current | Best Practice |
|-------|----------|---------|---------------|
| No baseline comparison | 🟡 HIGH | Only PatchTST | Need MLP/LSTM baseline |
| Simple prediction head | 🟡 MEDIUM | Linear only | Consider MLP head |
| Patch size not tuned | 🟡 MEDIUM | Fixed 16 | Should search |
| Positional encoding scale | 🟡 LOW | randn(std=1) | Typically smaller |

### Experiments Needed

1. **MLP baseline**: Does a simple MLP match/beat transformer?
2. **Prediction head**: Linear vs 2-layer MLP vs attention pooling
3. **Patch size search**: 8 vs 16 vs 32
4. **Positional encoding**: Standard init (std=0.02) vs current (std=1.0)
5. **Context length**: 30 vs 60 vs 120 days

---

## 5. Training Methodology

### Current State
- **Optimizer**: Adam
- **LR schedule**: None (constant)
- **Dropout**: 0.1
- **Early stopping**: By val_loss or val_auc
- **Batch size**: Dynamic based on model size

### Best Practices

| Component | Options | Typical Best |
|-----------|---------|--------------|
| Optimizer | Adam, AdamW, SGD+momentum | AdamW for transformers |
| LR schedule | Constant, cosine, warmup+decay | Warmup + cosine decay |
| Weight decay | 0, 0.01, 0.1 | 0.01-0.1 for transformers |
| Dropout | 0.0-0.3 | 0.1-0.2 typical |
| Gradient clipping | None, 1.0, 5.0 | 1.0 for stability |

### Gap Analysis

| Issue | Severity | Current | Best Practice |
|-------|----------|---------|---------------|
| No LR schedule | 🟡 MEDIUM | Constant | Warmup + decay |
| Adam vs AdamW | 🟡 LOW | Adam | AdamW has better regularization |
| No gradient clipping | 🟡 LOW | None | Add for stability |

### Experiments Needed

1. **LR schedule**: Constant vs warmup+cosine
2. **Weight decay**: 0 vs 0.01 vs 0.1
3. **Gradient clipping**: None vs 1.0
4. **Optimizer**: Adam vs AdamW

---

## 6. Evaluation Methodology

### Current State
- **Metrics**: val_loss, AUC-ROC
- **Test evaluation**: Single backtest on 2023+ data
- **Confidence intervals**: ❌ None

### Best Practices for Financial Model Evaluation

| Metric | What It Measures | When to Use |
|--------|------------------|-------------|
| AUC-ROC | Ranking ability | Threshold-agnostic evaluation |
| AUC-PR | Precision-recall tradeoff | Imbalanced classes |
| Brier score | Calibration | Probability accuracy |
| Sharpe ratio | Risk-adjusted return | Trading strategy |
| Max drawdown | Worst loss | Risk assessment |
| Hit rate | Directional accuracy | Simple interpretability |

### Gap Analysis

| Issue | Severity | Current | Best Practice |
|-------|----------|---------|---------------|
| No confidence intervals | 🟡 MEDIUM | Point estimates | Bootstrap CIs |
| No calibration check | 🟡 MEDIUM | None | Reliability diagram |
| No trading simulation | 🟡 LOW | Classification only | Simulated returns |
| Single test period | 🟡 MEDIUM | 2023+ only | Multiple periods |

### Experiments Needed

1. **Calibration analysis**: Are predicted probabilities reliable?
2. **Bootstrap confidence intervals**: How stable are AUC estimates?
3. **Multiple test periods**: Does performance vary by market regime?
4. **Trading simulation**: What are realistic returns after costs?

---

## 7. Financial-Specific Considerations

### Current State
- **Regime awareness**: ❌ None
- **Transaction costs**: ❌ Not modeled
- **Look-ahead bias**: ⚠️ Not audited
- **Survivorship bias**: N/A (SPY only)

### Best Practices

| Consideration | Issue | Mitigation |
|---------------|-------|------------|
| Non-stationarity | Market regimes change | Rolling normalization, regime detection |
| Look-ahead bias | Using future info | Strict temporal ordering, point-in-time data |
| Transaction costs | Erode returns | Model in evaluation |
| Slippage | Execution differs from signal | Conservative assumptions |
| Market impact | Large trades move price | Size-appropriate evaluation |

### Gap Analysis

| Issue | Severity | Current | Best Practice |
|-------|----------|---------|---------------|
| Look-ahead bias audit | 🔴 HIGH | Not done | Must verify |
| No regime handling | 🟡 MEDIUM | Single model | Regime-aware training |
| No transaction costs | 🟡 LOW | None | Include in backtest |

### Experiments Needed

1. **Look-ahead bias audit**: Trace all data paths for leakage
2. **Regime analysis**: Performance by market regime (bull/bear/sideways)
3. **Transaction cost sensitivity**: How do results change with 0.1% costs?

---

## Priority Matrix

### Must Fix Before Continuing (Blocking)

| Issue | Component | Effort | Impact |
|-------|-----------|--------|--------|
| Feature normalization | Data | Medium | Critical |
| Val set size | Splits | Low | Critical |
| Look-ahead bias audit | Data | Medium | Critical |

### Should Experiment With (High Value)

| Experiment | Component | Effort | Expected Value |
|------------|-----------|--------|----------------|
| Loss function comparison | Training | Medium | High |
| MLP baseline | Architecture | Low | High (sanity check) |
| Normalization method | Data | Medium | High |
| Prediction head variants | Architecture | Low | Medium |

### Nice to Have (Lower Priority)

| Experiment | Component | Effort | Expected Value |
|------------|-----------|--------|----------------|
| LR schedules | Training | Low | Low-Medium |
| Patch size search | Architecture | Medium | Low-Medium |
| Calibration analysis | Evaluation | Low | Low |
| Transaction costs | Evaluation | Low | Low |

---

## Recommended Experiment Plan

### Phase 1: Fix Critical Issues (Before Any More Training)
1. ✅ Implement feature normalization (Z-score on training set)
2. ✅ Fix validation set size (time-based split with 200+ samples)
3. ✅ Complete look-ahead bias audit

### Phase 2: Establish Baselines
4. Train MLP baseline with same features
5. Train single 2M PatchTST with fixes from Phase 1
6. Compare: Does transformer beat MLP?

### Phase 3: Systematic Ablations
7. Normalization: Z-score vs rolling vs returns
8. Loss function: BCE vs pos_weight vs Focal vs SoftAUC
9. Prediction head: Linear vs MLP
10. Architecture: Patch size, context length

### Phase 4: Full Scaling Study (Only After Phases 1-3)
11. Re-run 2M → 20M → 200M → 2B with optimized setup
12. Proper statistical analysis with confidence intervals

---

## Questions to Answer Before Proceeding

1. **Is our task even learnable?** MLP baseline will tell us if ANY model can beat random on this data.

2. **What normalization works best?** Z-score may not be optimal for non-stationary financial data.

3. **Does the transformer architecture add value?** Or is an MLP sufficient for this task?

4. **Are our features informative?** RF achieves 0.68-0.82 AUC - can we match that?

5. **Is 1-day ahead prediction the right task?** Maybe longer horizons are more predictable.

---

## Next Steps

1. **Review this audit** - Are there gaps I missed?
2. **Prioritize experiments** - What order?
3. **Implement fixes** - Normalization first
4. **Run baselines** - MLP before more transformer work
5. **Iterate** - Based on baseline results

