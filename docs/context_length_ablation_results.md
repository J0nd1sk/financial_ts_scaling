# Context Length Ablation Study Results

**Date:** 2026-01-20
**Commit:** 67d057e

## Overview

This study tests the impact of context window length on PatchTST performance for 1-day threshold (>1%) prediction on SPY.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Architecture | d_model=64, n_layers=4, n_heads=4, d_ff=256 |
| Patch config | patch_length=16, stride=8 |
| Dropout | 0.20 |
| Normalization | RevIN (per-instance) |
| Splitter | SimpleSplitter (val=2023-2024, test=2025+) |
| Task | threshold_1pct, horizon=1 |
| Training | 50 epochs max, early stopping on val_auc (patience=10) |
| Data | SPY_dataset_a20.parquet (20 features) |

## Results

| Context Length | Val AUC | Δ vs Baseline | Val Loss | Val Samples | Train Samples |
|----------------|---------|---------------|----------|-------------|---------------|
| 60 days | 0.6011 | baseline | 0.3336 | 442 | 7277 |
| **80 days** | **0.6945** | **+15.5%** | 0.3424 | 422 | 7257 |
| 90 days | 0.6344 | +5.5% | 0.3292 | 412 | 7247 |
| 120 days | 0.6877 | +14.4% | 0.3093 | 382 | 7217 |
| 180 days | 0.5489 | -8.7% | 0.3821 | 322 | 7157 |
| 252 days | 0.4768 | -20.7% | 0.3895 | 250 | 7085 |

**Note:** ctx336 was excluded because the test region (2025+, ~261 days) is smaller than context_length + horizon.

## Key Findings

### 1. Optimal Context Length: 80 Days
- **80 days achieves the highest AUC (0.6945)**, representing a +15.5% improvement over the 60-day baseline
- This corresponds to approximately 4 months of trading history
- 120 days is a close second (0.6877, +14.4%)

### 2. Sweet Spot: 80-120 Days
- The optimal range appears to be 80-120 days (~4-6 months)
- Both significantly outperform the 60-day default
- Performance degrades sharply beyond 120 days

### 3. Longer Context Hurts Performance
- 180 days: AUC drops to 0.5489 (worse than baseline)
- 252 days (1 trading year): AUC 0.4768 (worse than random)
- Hypothesis: Longer contexts introduce more noise than signal for short-term (1-day) prediction

### 4. Sample Size Trade-off
- Longer contexts reduce available validation samples (442 → 250)
- This may affect reliability of results for longer contexts
- However, the performance degradation is consistent and significant

## Interpretation

The 80-day window likely captures:
- ~4 months of price action
- Approximately 1.5 quarters of market cycles
- Sufficient history for pattern recognition without excessive noise

Longer windows may fail because:
- Market regimes change over 6+ months
- Old patterns become irrelevant for short-term prediction
- Signal-to-noise ratio decreases with more historical data

## Implications for Phase 6A

1. **Update default context_length from 60 to 80** for future experiments
2. **Re-run loss comparison** with 80-day context to see if findings hold
3. **Consider context length as a hyperparameter** in HPO (range: 60-120)

## Reproduction

```bash
# Run all context length experiments
PYTHONPATH=. ./venv/bin/python experiments/context_length_ablation/run_context_ablation.py

# Or run individually
PYTHONPATH=. ./venv/bin/python experiments/context_length_ablation/train_ctx80.py
```

## Re-running with Different Hyperparameters

To test with different loss functions, learning rates, etc.:

1. Edit `train_ctx60.py` (template) with new settings
2. Regenerate other scripts:
   ```bash
   for ctx in 80 90 120 180 252; do
     sed "s/ctx_ablation_60/ctx_ablation_$ctx/; s/CONTEXT_LENGTH = 60/CONTEXT_LENGTH = $ctx/" \
       experiments/context_length_ablation/train_ctx60.py > experiments/context_length_ablation/train_ctx$ctx.py
   done
   ```
3. Run: `PYTHONPATH=. ./venv/bin/python experiments/context_length_ablation/run_context_ablation.py`

## Files

- Scripts: `experiments/context_length_ablation/train_ctx{60,80,90,120,180,252}.py`
- Runner: `experiments/context_length_ablation/run_context_ablation.py`
- Results: `outputs/context_length_ablation/ctx_ablation_*/results.json`
- Summary: `outputs/context_length_ablation/summary.json`

## Memory Entities

- `Finding_ContextLengthAblation_20260120`: Full results and interpretation
- `Plan_ContextLengthAblation`: Planning decisions

---

# Alternative Architecture Context Length Ablation

**Date:** 2026-01-31
**Commit:** TBD (pending)

## Overview

Extended context length ablation study to iTransformer and Informer architectures using v3 methodology (Focal Loss, proper classification training).

## Experimental Setup

| Parameter | iTransformer | Informer |
|-----------|--------------|----------|
| Loss | Focal (γ=0.5, α=0.9) | Focal (γ=0.5, α=0.9) |
| hidden_size | 32 | 256 |
| learning_rate | 1e-5 | 1e-4 |
| max_steps | 3000 | 1000 |
| dropout | 0.4 | 0.4 |
| n_layers | 6 | 2 |
| n_heads | 4 | 2 |
| Data | SPY_dataset_a20.parquet | SPY_dataset_a20.parquet |

**Note:** Hyperparameters transferred from 80d HPO (v3). Context lengths tested: 60d, 80d, 120d, 180d, 220d.

## Results

### iTransformer

| Context | Val AUC | Precision | Recall | Dir Acc | Pred Range |
|---------|---------|-----------|--------|---------|------------|
| 60d | 0.552 | 0.202 | 0.980 | 0.574 | 0.39-0.87 |
| **80d** | **0.590** | 0.203 | 0.941 | 0.542 | 0.30-0.87 |
| 120d | 0.503 | 0.205 | 0.990 | 0.574 | 0.35-0.90 |
| 180d | 0.548 | 0.204 | 0.911 | 0.560 | 0.28-0.90 |
| 220d | 0.583 | 0.212 | 0.911 | 0.572 | 0.24-0.90 |

**Optimal: 80d** (AUC 0.590)

### Informer

| Context | Val AUC | Precision | Recall | Dir Acc | Pred Range |
|---------|---------|-----------|--------|---------|------------|
| 60d | 0.539 | 0.209 | 0.624 | 0.552 | 0.00-1.00 |
| 80d | 0.554 | 0.227 | 0.505 | 0.508 | 0.00-1.00 |
| 120d | 0.512 | 0.218 | 0.564 | 0.492 | 0.00-1.00 |
| **180d** | **0.585** | 0.228 | 0.634 | 0.520 | 0.00-1.00 |
| 220d | 0.557 | 0.242 | 0.495 | 0.472 | 0.00-1.00 |

**Optimal: 180d** (AUC 0.585)

## Cross-Architecture Comparison

| Architecture | Best Context | Best AUC | Δ vs PatchTST |
|--------------|--------------|----------|---------------|
| **PatchTST** | **80d** | **0.718** | baseline |
| iTransformer | 80d | 0.590 | **-17.8%** |
| Informer | 180d | 0.585 | **-18.5%** |

## Key Findings

### 1. Architecture-Specific Optimal Context
- **iTransformer**: Optimal at 80d (same as PatchTST)
- **Informer**: Optimal at 180d (benefits from longer context)
- ProbSparse attention in Informer may handle longer sequences better

### 2. Non-Monotonic Context-Performance Relationship
- Both architectures show a dip at 120d
- Performance is not a simple function of context length
- May indicate regime-specific or noise-related effects

### 3. Persistent Performance Gap
- Neither architecture approaches PatchTST performance
- Gap of ~12-18% AUC regardless of context tuning
- Context optimization alone cannot close this gap

### 4. Behavioral Differences
| Metric | iTransformer | Informer |
|--------|--------------|----------|
| Recall | 0.91-0.99 (predicts almost all +) | 0.50-0.63 (balanced) |
| Pred Range | 0.24-0.90 (narrower) | 0.00-1.00 (full) |
| Calibration | Poor (over-predicts positive) | Better spread |

## Implications

1. **For a200 training**: Use architecture-specific optimal contexts
   - iTransformer: 80d
   - Informer: 180d

2. **For research**: PatchTST's patching mechanism appears fundamentally superior for financial time series

3. **Architectural insight**: Informer's ProbSparse attention benefits from longer context, but this doesn't translate to competitive performance

## Files

- Script: `experiments/architectures/context_ablation_nf.py`
- Runner: `scripts/run_context_ablation.sh`
- Results: `outputs/architectures/context_ablation/{model}/ctx{N}/results.json`
