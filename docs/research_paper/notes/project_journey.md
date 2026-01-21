# Project Journey: Financial Time-Series Transformer Scaling Experiments

> **STATUS: LIVING DOCUMENT - WORK IN PROGRESS**
>
> This document captures the evolving story of this research project. It is not intended for the formal paper but serves as institutional knowledge—a lab notebook that preserves what we tried, what we learned, and why we made certain decisions. Update this document as the project progresses.
>
> *Last updated: 2026-01-20*

---

## Executive Summary

We began this project with a straightforward hypothesis: transformer models (specifically PatchTST) should outperform traditional methods on financial time-series prediction, and neural scaling laws should apply—bigger models achieving better results. The literature supported transformers for price prediction tasks, and scaling laws had proven robust across language and vision domains.

What we discovered was far more interesting. After extensive experimentation (600+ HPO trials, dozens of ablation studies), we found that transformers struggle with weak-signal classification tasks in ways that tree-based models do not. The core issue isn't model capacity—it's that neural networks converge to degenerate solutions (predicting class priors) before learning discriminative features. Random Forest achieves AUC 0.72 on the same data where our best transformer reaches 0.69. The gap isn't closed, but the investigation has revealed fundamental insights about loss functions, normalization, architecture, and what transformers actually need to outperform simpler methods. This journey continues.

---

## 1. Original Hypothesis

### What We Expected

Based on the literature, we entered with these beliefs:

1. **Transformers > Traditional Models**: Papers showed PatchTST and similar architectures outperforming LSTMs, CNNs, and classical methods on time-series forecasting tasks.

2. **Scaling Laws Apply**: Kaplan et al. (2020) and Hoffmann et al. (2022) demonstrated that model performance improves predictably with parameter count across language and vision. We expected similar behavior for financial prediction.

3. **More Parameters = Better Performance**: Given sufficient data, larger models should capture more complex patterns and achieve lower loss.

### The Research Question

> *Do neural scaling laws apply to transformer models trained on financial time-series data? Specifically, does increasing parameters (2M → 20M → 200M → 2B) improve prediction accuracy following a power law?*

### What We Planned to Test

- **Phase 6A**: Parameter scaling (hold features constant, vary model size)
- **Phase 6B**: Horizon scaling (test different prediction horizons)
- **Phase 6C**: Feature scaling (expand from 20 to 2000 indicators)
- **Phase 6D**: Data scaling (add more assets and history)

---

## 2. Experimental Infrastructure

### What We Built (Nov 2025 - Dec 2025)

Before running experiments, we built substantial infrastructure:

| Component | Description |
|-----------|-------------|
| **PatchTST from scratch** | Pure PyTorch implementation (~500 lines) for full control |
| **Data pipeline** | OHLCV download, 20 technical indicators, VIX integration |
| **HPO framework** | Optuna with architectural search, forced extreme testing |
| **Thermal monitoring** | M4 MacBook Pro temperature management |
| **Experiment tracking** | CSV logs, JSON configs, W&B integration |
| **Memory-safe batching** | Dynamic batch sizing based on architecture |
| **Early stopping** | Patience-based stopping on validation AUC |

### Key Infrastructure Decisions

1. **Hybrid data splits**: Random chunk assignment for val/test (not pure chronological) to avoid regime dependency
2. **Architectural HPO**: Search architecture (d_model, n_layers, n_heads) not just training params
3. **Forced extreme testing**: First 10 HPO trials test boundary conditions (min/max of each dimension)

---

## 3. Timeline of Discoveries

### Phase 1-5: Setup and Data (Nov-Dec 2025)

Standard setup work. Downloaded SPY data (1993-2025), built feature pipeline, established test infrastructure. 130+ tests passing.

### December 2025: Initial HPO Results

**Finding: "Inverse Scaling"**

| Budget | Best val_loss | Δ from 2M |
|--------|---------------|-----------|
| 2M | 0.2630 | baseline |
| 20M | 0.3191 | +21% worse |
| 200M | 0.3574 | +36% worse |
| 2B | 0.3716 | +41% worse |

We initially celebrated this as a significant finding—evidence that scaling laws don't apply in data-limited regimes. We wrote extensive documentation about "inverse scaling" and the "data-limited regime hypothesis."

**What we thought**: Insufficient training data (~4000 samples) couldn't support large models. The parameter-to-sample ratio was astronomically high (2B params / 4K samples = 500,000:1 vs. Chinchilla's optimal 20:1).

### January 2026: The Unraveling Begins

**Discovery 1: Double Sigmoid Bug**

During backtest evaluation, found that our evaluation script applied `torch.sigmoid()` to outputs that already had sigmoid applied in the model's forward pass. `sigmoid(sigmoid(x))` compresses everything to ~0.5-0.7 range.

*Impact*: Predictions looked collapsed, but some of it was measurement error.

**Discovery 2: ChunkSplitter Bug**

Our "hybrid chunk" splitter was giving only 19 validation samples (one per chunk), not the expected ~500. Training used sliding window (~5700 samples) but validation didn't.

*Impact*: All HPO decisions were based on 19 samples—statistically meaningless. Required SimpleSplitter rewrite.

**Discovery 3: Feature Lookahead Contamination**

Some technical indicators were computed with future data leaking in. Fixed during feature pipeline audit.

*Impact*: Unknown magnitude, but any lookahead invalidates results.

### January 2026: The Real Problem Emerges

**Discovery 4: Probability Collapse / Prior Prediction**

After fixing the above bugs and implementing SimpleSplitter (442 val samples), we observed:

- All models output near-constant probabilities (spread < 1%)
- Predictions clustered around the class prior (~0.5 for balanced, ~0.1 for imbalanced)
- Loss settles at BCE's theoretical minimum for "predict the mean"
- Prediction std drops to 0.0000 within 5-10 epochs

**Root Cause**: BCE loss has a local minimum at "predict class prior for all samples." The model finds this lazy solution before learning any discriminative features. Gradient vanishes when predictions are "safe" (near 0.5 for balanced classes).

**Reinterpretation of "Inverse Scaling"**: We weren't measuring predictive ability. We were measuring speed of convergence to a degenerate solution. Smaller models converged faster to the lazy solution, achieving lower loss—but no model was actually learning the signal.

### January 2026: Proof That Signal Exists

**Discovery 5: Random Forest Baseline**

| Threshold | RF AUC | PatchTST AUC | Gap |
|-----------|--------|--------------|-----|
| 0.5% | 0.628 | 0.586 | RF +7% |
| 1.0% | 0.716 | 0.695 | RF +3% |
| 2.0% | 0.766 | 0.621 | GB +23% |

**Critical insight**: The data HAS predictive signal. Tree models find it easily. Transformers fail to extract it.

Top RF features: RSI (13.6%), Bollinger %B (10.9%), MACD (7.2%), ATR (6.8%). These are exactly the indicators we fed to PatchTST—but RF uses them effectively while PatchTST doesn't.

### January 2026: Partial Solutions

**Discovery 6: Focal Loss Helps**

| Loss Function | AUC | Δ vs BCE |
|---------------|-----|----------|
| FocalLoss(γ=2) | 0.6717 | +24.8% |
| SoftAUCLoss | 0.6149 | +14.2% |
| BCE_weighted | 0.6052 | +12.4% |
| BCE (baseline) | 0.5383 | — |

Focal Loss down-weights easy (already correct) predictions, forcing the model to keep learning on hard examples. This prevents convergence to the "always predict 0.5" local minimum.

**Gap to XGBoost reduced from 22% to 11%**, but not closed.

**Discovery 7: RevIN Outperforms Z-Score**

| Normalization | AUC |
|---------------|-----|
| RevIN only | 0.667 |
| Z-score + RevIN | 0.515 |
| Z-score only | 0.476 |

Per-instance normalization (RevIN) works better than global z-score. This was surprising—we expected z-score to help.

**Discovery 8: Context Length Sweet Spot**

| Context | AUC | Δ vs 60d |
|---------|-----|----------|
| 60 days | 0.601 | baseline |
| **80 days** | **0.695** | **+15.5%** |
| 120 days | 0.688 | +14.4% |
| 180 days | 0.549 | -8.7% |
| 252 days | 0.477 | -20.7% |

**80 days (~4 months) is optimal**. Longer contexts introduce more noise than signal for short-term prediction. This suggests financial patterns are relatively short-lived.

**Discovery 9: Shallow-Wide Beats Deep-Narrow**

| Architecture | Params | AUC |
|--------------|--------|-----|
| L1, d=768 | 7.3M | 0.694 |
| L2, d=512 | 6.4M | 0.694 |
| L4, d=64 (baseline) | 0.2M | 0.694 |

All achieve similar AUC despite 30x parameter difference. Depth provides no benefit—and very shallow models (L=1-2) work just as well.

---

## 4. Current Understanding

### What We Know

1. **The data contains signal**: RF achieves AUC 0.72, proving predictability exists
2. **Transformers struggle with weak-signal classification**: BCE allows lazy prediction
3. **Focal Loss partially fixes collapse**: AUC improves from 0.54 to 0.67
4. **Architecture matters less than expected**: L=1-4 all perform similarly
5. **80-day context is optimal**: Longer hurts, shorter hurts
6. **RevIN > Z-score**: Per-instance normalization wins
7. **Gap to tree models remains**: Best transformer AUC 0.69 vs RF 0.72

### What We Don't Know

1. **Why transformers can't match RF**: Is it architectural? Loss function? Training dynamics?
2. **Whether attention helps at all**: MLP experiment in progress
3. **Whether feature scaling will help**: Phase 6C hypothesis untested
4. **Optimal regularization**: Dropout and weight decay not fully explored

### Current Best Configuration

```yaml
architecture:
  n_layers: 1-4 (doesn't matter much)
  d_model: 256-768
  n_heads: 4-8

training:
  loss: FocalLoss(gamma=2.0, alpha=0.25)
  lr: 3e-6 (very low)
  warmup: 10 epochs
  dropout: 0.20

data:
  context_length: 80 days
  normalization: RevIN (per-instance)
  splitter: SimpleSplitter (val=2023-2024, test=2025+)
```

---

## 5. Active Investigations

### Currently Testing (as of 2026-01-20)

1. **MLP-only model**: Remove attention entirely. If MLP performs equally well, attention is adding noise, not signal. Script: `scripts/test_mlp_only.py`

2. **Dropout variations**: Test higher dropout (0.3-0.5) to prevent early convergence

3. **Learning rate experiments**: Even lower LR with longer training—force slow learning

### Hypothesis

The core hypothesis remains: **Transformers SHOULD be able to outperform tree models** given:
- Deeper context windows (trend/momentum patterns)
- Wider feature sets (more relationships to learn)
- Correct training dynamics (prevent early convergence)

If current experiments don't close the gap, we'll proceed to **Phase 6C (feature scaling)**. More features = more relationships = more for attention to learn. With 2000 features, there are ~2M pairwise relationships vs ~190 with 20 features.

---

## 6. Lessons Learned for Financial ML

### Technical Lessons

| Lesson | Details |
|--------|---------|
| **BCE allows degenerate solutions** | For weak-signal classification, BCE has a local minimum at "predict class prior." Use Focal Loss or AUC-based losses. |
| **Validate on meaningful sample sizes** | 19 validation samples is useless. Caught ChunkSplitter bug because results didn't make sense. |
| **Tree models are strong baselines** | Always run RF/XGBoost first. If they can't find signal, neither will transformers. |
| **Per-instance normalization wins** | RevIN outperformed global z-score. Financial data varies too much across time for global stats. |
| **Context length has a sweet spot** | For daily equity data, 80 days works best. More history adds noise for short-term prediction. |
| **Depth provides little benefit** | L=1 performs as well as L=4. Financial patterns may not require deep compositional hierarchies. |

### Process Lessons

| Lesson | Details |
|--------|---------|
| **Verify outputs, not just execution** | Code ran successfully but outputs were incomplete (missing architecture in JSON). Always check final artifacts. |
| **Test the full pipeline** | Unit tests passed but integration output was wrong. Need end-to-end validation. |
| **Watch for Python falsy bugs** | `x = value or default` fails when value is 0. Use explicit `if value is not None`. |
| **Document decisions in real-time** | The decision_log.md captured every choice. Invaluable for understanding the journey. |
| **Preserve historical context** | Don't delete old plans—consolidate into history documents. "What we did and why" matters. |

### Research Lessons

| Lesson | Details |
|--------|---------|
| **"Inverse scaling" can be misleading** | We thought smaller models were better. Actually, all models were failing—smaller just failed faster. |
| **Signal existence must be proven separately** | Before blaming the model, prove signal exists with a simple baseline. |
| **Loss function > Architecture** | Changing BCE to Focal Loss improved AUC more than any architectural change. |
| **Financial ML is different** | Weak signals, non-stationarity, and regime changes make standard deep learning assumptions fail. |

---

## 7. Experiments Catalog

### HPO Experiments (600 trials)

| Experiment | Trials | Best | Finding |
|------------|--------|------|---------|
| 2M × h1,h3,h5 | 150 | 0.2630 | Smallest budget wins |
| 20M × h1,h3,h5 | 150 | 0.3191 | 21% worse than 2M |
| 200M × h1,h3,h5 | 150 | 0.3574 | 36% worse than 2M |
| 2B × h1,h3,h5 | 150 | 0.3716 | 41% worse than 2M |

*Note: These results measured convergence to degenerate solution, not predictive ability.*

### Ablation Studies

| Study | Location | Key Finding |
|-------|----------|-------------|
| Context length | `outputs/context_length_ablation/` | 80 days optimal (+15.5% vs 60) |
| RevIN comparison | `outputs/revin_comparison/` | RevIN alone > z-score |
| Loss function | `outputs/loss_function_comparison/` | Focal Loss best (+24.8% vs BCE) |
| Shallow/wide | `outputs/shallow_wide_experiment/` | L=1 matches L=4 |
| Small models | `outputs/small_models_experiment/` | 65K params achieves AUC 0.69 |
| Threshold comparison | `outputs/threshold_comparison/` | 1% threshold most predictable |
| Seed comparison | `outputs/seed_comparison/` | Results stable across seeds |

### Baseline Comparisons

| Model | Script | Best AUC | Notes |
|-------|--------|----------|-------|
| Random Forest | `scripts/test_xgboost_thresholds.py` | 0.716 | Strong baseline |
| XGBoost | same | 0.755 | Best overall |
| Gradient Boosting | same | 0.766 | Best at 2% threshold |
| PatchTST + Focal | various | 0.695 | Best transformer |

### In Progress

| Experiment | Script | Status |
|------------|--------|--------|
| MLP-only | `scripts/test_mlp_only.py` | Not started |

---

## 8. What's Next

### Immediate (Current Session)

1. Run MLP experiment - does removing attention help?
2. Test higher dropout (0.3, 0.4, 0.5)
3. Test even lower learning rates with longer training

### If Current Experiments Don't Close Gap

Proceed to **Phase 6C: Feature Scaling**
- Expand from 20 to 200 to 2000 indicators
- Hypothesis: More features = more relationships = more for attention to learn
- May unlock the transformer's potential

### Long-term

- Complete Phase 6B (horizon scaling) for publication completeness
- Consider Phase 6D (data scaling) - more assets
- Write formal paper with honest narrative about the journey

---

## Appendix: Key Memory Entities

These Memory MCP entities contain detailed context:

- `Finding_FocalLossFixesPatchTST_20260120`
- `Pattern_TransformerProbabilityCollapse_20260120`
- `Finding_RFBeatsPatchTST_20260120`
- `Finding_SignalExistsAt0.5pct_20260120`
- `Phase6A_PriorCollapse_RootCause`
- `MPS_GPU_Utilization_Finding`
- `Research_Narrative_Core_Thesis`

---

*This document will be updated as the project progresses. See `decision_log.md` for granular decision history.*
