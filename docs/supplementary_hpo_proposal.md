# Supplementary HPO Trials Proposal

Based on analysis of 250 HPO trials across a50/a100 tiers and 2M/20M/200M budgets.

---

## Observed Trends

### 1. Dropout is Critical
| Dropout | Mean AUC | Max AUC | % of Top 60 |
|---------|----------|---------|-------------|
| 0.1 | 0.7048 | 0.7236 | 3.3% |
| 0.3 | 0.7016 | 0.7315 | 13.3% |
| **0.5** | **0.7150** | 0.7302 | **56.7%** |
| 0.7 | 0.7099 | 0.7244 | 26.7% |

**Finding**: Dropout=0.5 dominates. But we never tested 0.4 or 0.6!

### 2. Learning Rate Sweet Spot
| LR | Mean AUC | Max AUC | % of Top 60 |
|----|----------|---------|-------------|
| 1e-5 | 0.7061 | 0.7233 | 3.3% |
| 5e-5 | 0.7059 | 0.7315 | 20.0% |
| **1e-4** | **0.7146** | 0.7300 | **61.7%** |
| 5e-4 | 0.7054 | 0.7302 | 15.0% |

**Finding**: LR=1e-4 is optimal. Gap between 5e-5 and 1e-4 is large - should try 7e-5, 8e-5.

### 3. Architecture: Bimodal Depth
| n_layers | % of Top 60 |
|----------|-------------|
| 2 | 26.7% |
| 6 | 35.0% |
| 7 | 15.0% |

**Finding**: Shallow (2) OR mid-deep (6-7) - nothing in between. Correlation is negative (-0.211).

### 4. d_model: 128 Dominates
| d_model | % of Top 60 |
|---------|-------------|
| 48 | 23.3% |
| 128 | **63.3%** |

**Finding**: d_model=128 is the clear winner, but 48 works for 2M budget. Never tried 112 or 144.

### 5. Weight Decay: Higher is Better
| weight_decay | % of Top 60 |
|--------------|-------------|
| 0.0 | 13.3% |
| 1e-5 | 5.0% |
| **1e-4** | **36.7%** |
| **1e-3** | **45.0%** |

**Finding**: Higher weight decay works better. Should try 5e-4 range.

### 6. d_ff_ratio: Slight Edge to 2
| d_ff_ratio | Mean AUC | % of Top 60 |
|------------|----------|-------------|
| 2 | 0.7082 | 58.3% |
| 4 | 0.7083 | 41.7% |

**Finding**: Essentially tied. Not a key lever.

---

## Supplementary Trials Proposal

### Phase 1: Fine-Tune the Winners (a50 tier, 20M budget)

Since a50-20M achieved the best AUC (0.7315), let's explore its neighborhood.

**Base Config** (Trial 5):
- d_model=128, n_layers=6, n_heads=8
- dropout=0.3, lr=5e-5, wd=1e-4

#### Trial Set A: Dropout Exploration (6 trials)
| Trial | dropout | Notes |
|-------|---------|-------|
| A1 | 0.35 | Between current 0.3 and optimal 0.5 |
| A2 | 0.40 | Halfway |
| A3 | 0.45 | Just below sweet spot |
| A4 | 0.50 | Sweet spot - verify with this config |
| A5 | 0.55 | Slightly above sweet spot |
| A6 | 0.60 | Explore upper range |

#### Trial Set B: Learning Rate Fine-Tuning (5 trials)
| Trial | lr | Notes |
|-------|-----|-------|
| B1 | 7e-5 | Between 5e-5 and 1e-4 |
| B2 | 8e-5 | Closer to 1e-4 |
| B3 | 9e-5 | Very close to sweet spot |
| B4 | 1.2e-4 | Slightly above |
| B5 | 1.5e-4 | Higher |

#### Trial Set C: Weight Decay Exploration (4 trials)
| Trial | weight_decay | Notes |
|-------|--------------|-------|
| C1 | 3e-4 | Between 1e-4 and 1e-3 |
| C2 | 5e-4 | Midpoint |
| C3 | 7e-4 | Closer to 1e-3 |
| C4 | 2e-3 | Above current best |

### Phase 2: Architecture Variants (a50 tier)

#### Trial Set D: Shallow Architecture Optimization (2M budget)
Best shallow config was Trial 15: d_model=96, n_layers=2, dropout=0.5, lr=5e-4

| Trial | d_model | n_layers | n_heads | Notes |
|-------|---------|----------|---------|-------|
| D1 | 80 | 2 | 8 | Slightly smaller |
| D2 | 112 | 2 | 8 | Between 96 and 128 |
| D3 | 96 | 3 | 8 | One more layer |
| D4 | 96 | 2 | 4 | Fewer heads |

#### Trial Set E: Deep Architecture Optimization (20M budget)
| Trial | d_model | n_layers | n_heads | Notes |
|-------|---------|----------|---------|-------|
| E1 | 128 | 5 | 8 | One less layer |
| E2 | 128 | 7 | 8 | One more layer |
| E3 | 144 | 6 | 8 | Larger d_model (not tested) |
| E4 | 128 | 6 | 4 | Fewer heads (4 was good in 2M) |

### Phase 3: Combined Optimization (6 trials)

Apply best findings from Phase 1+2:

| Trial | d_model | n_layers | dropout | lr | wd | Notes |
|-------|---------|----------|---------|------|------|-------|
| F1 | 128 | 6 | 0.45 | 8e-5 | 5e-4 | Combine best from each |
| F2 | 128 | 6 | 0.50 | 1e-4 | 5e-4 | Higher dropout |
| F3 | 128 | 5 | 0.50 | 1e-4 | 1e-3 | Shallower + high wd |
| F4 | 144 | 6 | 0.45 | 8e-5 | 5e-4 | Larger d_model |
| F5 | 128 | 7 | 0.50 | 8e-5 | 1e-3 | Deeper + high wd |
| F6 | 96 | 2 | 0.50 | 1e-4 | 1e-3 | Shallow sweet spot |

---

## Implementation: Supplementary HPO Script

```python
#!/usr/bin/env python3
"""Supplementary HPO trials targeting promising hyperparameter regions."""

SUPPLEMENTARY_CONFIGS = {
    # Phase 1: Fine-tune a50-20M winner
    "A1": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.35, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "A2": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.40, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "A3": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.45, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "A4": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.50, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "A5": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.55, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "A6": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.60, "learning_rate": 5e-5, "weight_decay": 1e-4},

    # Phase 1: Learning rate fine-tuning
    "B1": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 7e-5, "weight_decay": 1e-4},
    "B2": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 8e-5, "weight_decay": 1e-4},
    "B3": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 9e-5, "weight_decay": 1e-4},
    "B4": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 1.2e-4, "weight_decay": 1e-4},
    "B5": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 1.5e-4, "weight_decay": 1e-4},

    # Phase 1: Weight decay exploration
    "C1": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 1e-4, "weight_decay": 3e-4},
    "C2": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 1e-4, "weight_decay": 5e-4},
    "C3": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 1e-4, "weight_decay": 7e-4},
    "C4": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 1e-4, "weight_decay": 2e-3},

    # Phase 2: Architecture variants
    "D1": {"d_model": 80, "n_layers": 2, "n_heads": 8, "d_ff_ratio": 2,
           "dropout": 0.5, "learning_rate": 5e-4, "weight_decay": 1e-5},
    "D2": {"d_model": 112, "n_layers": 2, "n_heads": 8, "d_ff_ratio": 2,
           "dropout": 0.5, "learning_rate": 5e-4, "weight_decay": 1e-5},
    "D3": {"d_model": 96, "n_layers": 3, "n_heads": 8, "d_ff_ratio": 2,
           "dropout": 0.5, "learning_rate": 5e-4, "weight_decay": 1e-5},
    "D4": {"d_model": 96, "n_layers": 2, "n_heads": 4, "d_ff_ratio": 2,
           "dropout": 0.5, "learning_rate": 5e-4, "weight_decay": 1e-5},

    "E1": {"d_model": 128, "n_layers": 5, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "E2": {"d_model": 128, "n_layers": 7, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "E3": {"d_model": 144, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 5e-5, "weight_decay": 1e-4},
    "E4": {"d_model": 128, "n_layers": 6, "n_heads": 4, "d_ff_ratio": 4,
           "dropout": 0.5, "learning_rate": 5e-5, "weight_decay": 1e-4},

    # Phase 3: Combined optimization
    "F1": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.45, "learning_rate": 8e-5, "weight_decay": 5e-4},
    "F2": {"d_model": 128, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.50, "learning_rate": 1e-4, "weight_decay": 5e-4},
    "F3": {"d_model": 128, "n_layers": 5, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.50, "learning_rate": 1e-4, "weight_decay": 1e-3},
    "F4": {"d_model": 144, "n_layers": 6, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.45, "learning_rate": 8e-5, "weight_decay": 5e-4},
    "F5": {"d_model": 128, "n_layers": 7, "n_heads": 8, "d_ff_ratio": 4,
           "dropout": 0.50, "learning_rate": 8e-5, "weight_decay": 1e-3},
    "F6": {"d_model": 96, "n_layers": 2, "n_heads": 8, "d_ff_ratio": 2,
           "dropout": 0.50, "learning_rate": 1e-4, "weight_decay": 1e-3},
}
```

---

## Expected Outcomes

| If Trial | Shows | Then |
|----------|-------|------|
| A3-A4 beat A1-A2 | Dropout ~0.45-0.50 optimal | Confirms sweet spot |
| B2-B3 beat B1 | LR ~8e-5 to 9e-5 optimal | Narrow LR further |
| C2-C3 beat C1 | WD ~5e-4 to 7e-4 optimal | Higher regularization helps |
| E1 beats E2 | n_layers=5 better than 7 | Shallow wins for 20M |
| F1-F2 beat current best | Combined optimization works | New champion config |

---

## Resource Estimate

- **Total Trials**: 27
- **Time per Trial**: ~30-60 seconds (a50, 20M budget)
- **Total Time**: ~15-30 minutes
- **Output**: `outputs/phase6c_a50/supplementary_trials/`

---

## Next Steps

1. **Approve** this proposal
2. **Implement** `scripts/run_supplementary_hpo.py`
3. **Run** all 27 trials with verbose=True
4. **Analyze** which direction improved results
5. **Iterate** with refined search if promising
