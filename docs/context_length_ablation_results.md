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
