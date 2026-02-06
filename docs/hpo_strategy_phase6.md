# Two-Phase Budget-Aware HPO Strategy

**Date**: 2026-02-01
**Status**: Implemented
**Workstream**: ws2 (foundation)

## Overview

This document describes the two-phase HPO strategy for systematically exploring transformer architectures across parameter scales (750k → 2M → 20M → 200M).

## Motivation

Previous HPO focused on a single parameter budget (2M). To understand scaling laws and identify optimal architectures across scales, we need:

1. **Forced extreme configurations** to explore the full design space
2. **Budget-aware architecture sizing** (shallow-wide vs deep-narrow per budget)
3. **Early stopping** to avoid wasting compute on converged trials
4. **Supplementary trials** on top-performing budgets

## Phase 1: Forced Extremes (18 configs + TPE)

### Group 1: Budget × Architecture (8 configs)

Each budget has shallow-wide and deep-narrow variants:

| # | Budget | Style | d_model | n_layers | n_heads | ~Params |
|---|--------|-------|---------|----------|---------|---------|
| 1 | 750k | shallow | 192 | 2 | 4 | ~884k |
| 2 | 750k | deep | 128 | 4 | 4 | ~786k |
| 3 | 2M | shallow | 320 | 2 | 4 | ~2.5M |
| 4 | 2M | deep | 192 | 5 | 4 | ~2.2M |
| 5 | 20M | shallow | 768 | 3 | 8 | ~21M |
| 6 | 20M | deep | 384 | 12 | 8 | ~21M |
| 7 | 200M | shallow | 1536 | 8 | 16 | ~226M |
| 8 | 200M | deep | 768 | 28 | 16 | ~197M |

**Parameter formula**: params ≈ 12 × n_layers × d_model²

### Group 2: Dropout Extremes (4 configs)

| # | Budget | Dropout | Notes |
|---|--------|---------|-------|
| 9 | 2M | 0.1 | Low regularization |
| 10 | 2M | 0.3 | Mid-low |
| 11 | 2M | 0.7 | High regularization |
| 12 | 200M | 0.1 | Cross-budget validation |

### Group 3: Learning Rate Extremes (3 configs)

| # | Budget | LR | Notes |
|---|--------|-----|-------|
| 13 | 2M | 1e-5 | Low LR |
| 14 | 2M | 1e-3 | High LR |
| 15 | 20M | 1e-5 | Cross-budget validation |

### Group 4: Weight Decay Extremes (3 configs)

| # | Budget | WD | Notes |
|---|--------|-----|-------|
| 16 | 2M | 0.0 | No regularization |
| 17 | 2M | 1e-2 | High regularization |
| 18 | 200M | 0.0 | Cross-budget validation |

### Default Regularization

For Group 1 configs:
- dropout: 0.5
- learning_rate: 1e-4
- weight_decay: 1e-3

### TPE Trials

After 18 forced configs, TPE explores the space with ~50 additional trials.

### Early Stopping

Phase 1 stops early when:
1. At least 20 trials completed (configurable via `--early-stop-patience`)
2. Top 5 trials differ by ≤0.02 AUC (configurable via `--early-stop-threshold`)

## Phase 2: Supplementary Trials

**Trigger**: After Phase 1, analyze results to identify top 2 budgets.

**Dimensions to explore** (only on top budgets):
- n_heads: [1, 2, 4, 8, 16]
- learning_rate: [5e-6, 2e-3] (more extreme)
- weight_decay: [5e-2, 0.1] (higher)
- batch_size: [8, 16, 32, 64, 128, 256]

**Trials per budget**: ~20-30

## Usage

### Phase 1: Forced Extremes

```bash
# Dry run (verify configs)
./venv/bin/python experiments/architectures/hpo_neuralforecast.py \
    --model itransformer \
    --data-tier a200 \
    --trials 70 \
    --forced-extremes \
    --dry-run

# Full run
./venv/bin/python experiments/architectures/hpo_neuralforecast.py \
    --model itransformer \
    --data-tier a200 \
    --trials 70 \
    --forced-extremes

# Subset of budgets
./venv/bin/python experiments/architectures/hpo_neuralforecast.py \
    --model itransformer \
    --data-tier a200 \
    --trials 40 \
    --forced-extremes \
    --budgets 2M 20M
```

### Phase 2: Supplementary

```bash
# After analyzing Phase 1 results, focus on best budget
./venv/bin/python experiments/architectures/hpo_neuralforecast.py \
    --model itransformer \
    --data-tier a200 \
    --trials 30 \
    --supplementary \
    --param-budget 20M
```

### Early Stopping Configuration

```bash
# Stricter early stopping
./venv/bin/python experiments/architectures/hpo_neuralforecast.py \
    --model itransformer \
    --forced-extremes \
    --early-stop-patience 15 \
    --early-stop-threshold 0.01
```

## Architecture Sizing Rationale

### Shallow-Wide (larger d_model, fewer layers)

- Better for simple patterns
- Lower compute per token
- Less prone to gradient issues
- Good for smaller datasets

### Deep-Narrow (smaller d_model, more layers)

- Better for complex hierarchical patterns
- More representational capacity
- Higher compute per token
- May benefit from skip connections

### Scaling Formula

```
params ≈ 12 × n_layers × d_model²

Where 12 comes from:
- Self-attention: 4 × d_model² (Q, K, V, O projections)
- FFN: 8 × d_model² (assuming ff_dim = 4 × d_model)
```

## Implementation Details

### Module: `src/training/hpo_budget_extremes.py`

Key exports:
- `BUDGET_CONFIGS`: Architecture configs per budget
- `DEFAULT_REGULARIZATION`: Default dropout/LR/WD values
- `generate_forced_configs()`: Generate all 18 forced configs
- `check_early_stopping_convergence()`: Early stopping logic
- `compute_budget_aware_extremes()`: Get config for budget/style

### Integration: `experiments/architectures/hpo_neuralforecast.py`

New CLI arguments:
- `--forced-extremes`: Enable forced extreme configs
- `--budgets`: Subset of budgets to include
- `--early-stop-patience`: Min trials before early stop check
- `--early-stop-threshold`: AUC range for convergence
- `--supplementary`: Phase 2 mode
- `--param-budget`: Target budget for Phase 2

## Future Work

### Full Experimentation Phase

When ready for full experiments:
1. Run HPO on all exchanges separately
2. Run HPO on aggregated cross-exchange data
3. May require fresh HPO rounds per data type
4. Document final scaling law findings

This current implementation establishes methodology; full experiments will rerun HPO with production data.

## Related Documents

- `CLAUDE.md`: Project constraints and experimental protocol
- `.claude/context/workstreams/ws2_context.md`: Workstream context
- `.claude/context/decision_log.md`: Decision log entry
