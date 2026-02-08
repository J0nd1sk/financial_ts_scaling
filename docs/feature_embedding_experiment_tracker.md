# Feature Embedding Experiment Tracker

**Last Updated**: 2025-02-07
**Status**: LF-P7 In Progress

---

## Experiment Overview

| Category | Count | Description |
|----------|-------|-------------|
| FE (P1-P12) | 92 | Feature embedding architecture experiments |
| LF (LF-P1 to LF-P11) | 70 | Loss function experiments |
| AE (AE-P1 to AE-P5) | 21 | Advanced embedding architecture experiments |
| **Total** | **183** | |

---

## Run Status

### Completed Priorities

| Priority | Experiments | Status | Best Result |
|----------|-------------|--------|-------------|
| P1 | FE-01 to FE-06 | ✅ Complete | FE-05: 44.4% precision |
| P2 | FE-07 to FE-09 | ✅ Complete | FE-07: 36.7% precision |
| P3 | FE-10 to FE-12 | ✅ Complete | FE-11: 55.6% precision |
| P4 | FE-13 to FE-17 | ✅ Complete | FE-16: 50.0% precision |
| P5 | FE-18 to FE-36 | ✅ Complete | FE-26: 60.0% precision |
| P6 | FE-37 to FE-44 | ✅ Complete | FE-40: 47.4% precision |
| P7 | FE-45 to FE-50 | ✅ Complete | - |
| P8 | FE-51 to FE-58 | ✅ Complete | - |
| P9 | FE-59 to FE-64 | ✅ Complete | - |
| P10 | FE-65 to FE-70 | ✅ Complete | FE-68: 50.0% precision |
| P11 | FE-71 to FE-84 | ✅ Complete | FE-80: 53.3% precision |
| P12 | FE-85 to FE-92 | ✅ Complete | FE-89: 40.0% precision |
| LF-P1 | LF-01 to LF-08 | ✅ Complete | - |
| LF-P2 | LF-09 to LF-20 | ✅ Complete | - |
| LF-P3 | LF-21 to LF-28 | ✅ Complete | - |
| LF-P4 | LF-29 to LF-34 | ✅ Complete | - |
| LF-P5 | LF-35 to LF-40 | ✅ Complete | LF-40: **85.7% precision** |
| LF-P6 | LF-41 to LF-48 | ✅ Complete | LF-48: 60.0% precision |

### In Progress

| Priority | Experiments | Status | Notes |
|----------|-------------|--------|-------|
| LF-P7 | LF-49 to LF-54 | 🔄 Running | MildFocal loss (LF-51 done: 60% prec) |

### Pending

| Priority | Experiments | Type | Command |
|----------|-------------|------|---------|
| LF-P8 | LF-55 to LF-60 | AsymmetricFocal | `caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P8` |
| LF-P9 | LF-61 to LF-64 | EntropyReg | `caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P9` |
| LF-P10 | LF-65 to LF-68 | VarianceReg | `caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P10` |
| LF-P11 | LF-69 to LF-70 | CalibratedFocal | `caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P11` |
| AE-P1 | AE-01 to AE-04 | Progressive | `caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority AE-P1` |
| AE-P2 | AE-05 to AE-08 | Bottleneck | `caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority AE-P2` |
| AE-P3 | AE-09 to AE-12 | MultiHead | `caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority AE-P3` |
| AE-P4 | AE-13 to AE-16 | GatedResidual | `caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority AE-P4` |
| AE-P5 | AE-17 to AE-21 | Attention | `caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority AE-P5` |

**Remaining: 37 experiments** (after LF-P7 completes)

---

## Top Results (by Precision)

| Rank | Exp ID | Tier | d_embed | Precision | Recall | AUC | Loss Function |
|------|--------|------|---------|-----------|--------|-----|---------------|
| 1 | **LF-40** | a500 | 16 | **0.857** | 0.079 | 0.725 | LabelSmoothing e=0.20 |
| 2 | LF-48 | a200 | 64 | 0.600 | 0.079 | 0.723 | LabelSmoothing e=0.10 |
| 3 | FE-26 | a200 | 128 | 0.600 | 0.158 | 0.710 | BCE |
| 4 | LF-51 | a100 | 32 | 0.600 | 0.118 | 0.726 | MildFocal g=1.0 |
| 5 | FE-35 | a200 | 32 | 0.571 | 0.105 | 0.724 | BCE |
| 6 | FE-11 | a100 | 128 | 0.556 | 0.132 | 0.700 | BCE |
| 7 | FE-80 | a100 | 24 | 0.533 | 0.105 | 0.734 | BCE |
| 8 | FE-16 | a100 | 64 | 0.500 | 0.184 | 0.680 | BCE |
| 9 | FE-68 | a20 | None | 0.500 | 0.053 | 0.724 | BCE |
| 10 | FE-40 | a200 | 8 | 0.474 | 0.118 | 0.689 | BCE |

---

## Key Findings

### 1. Label Smoothing is Dominant
- **LF-40** (LabelSmoothing e=0.20 on a500/d_embed=16) achieves **85.7% precision**
- This is the highest precision observed across all experiments
- Label smoothing prevents overconfident predictions and improves calibration

### 2. Smaller Feature Tiers Can Work
- a20 baseline (FE-68) achieves 50% precision with good AUC (0.724)
- Suggests simpler feature sets may avoid overfitting

### 3. d_embed Compression Helps
- Best results often use smaller d_embed (16, 32) rather than large expansions
- Aggressive compression forces learning of essential representations

### 4. Recall vs Precision Trade-off
- High precision comes at cost of low recall (typical ~0.08-0.15)
- Models are conservative, only predicting positive when highly confident

---

## Observations on Probability Collapse

Several experiments show probability collapse (predictions clustered near 0.5):
- FE-69, FE-70, FE-72: 0% precision, 0% recall, predictions in [0.01, 0.50]
- These typically occur with extreme compression (d_embed=8, 16) on small tiers

Mitigation strategies being tested:
- Variance regularization (LF-P10)
- Entropy regularization (LF-P9)
- Asymmetric focal loss (LF-P8)

---

## Next Steps

1. Complete LF-P7 (MildFocal) - currently running
2. Run LF-P8 through LF-P11 (subtle loss functions)
3. Run AE-P1 through AE-P5 (advanced embeddings)
4. Analyze results and identify best overall configuration
5. Consider follow-up experiments combining best loss + best embedding

---

## Commands Quick Reference

```bash
# Check what's running
ps aux | grep run_experiments

# Run next priority after LF-P7
caffeinate ./venv/bin/python experiments/feature_embedding/run_experiments.py --priority LF-P8

# List all experiments
./venv/bin/python experiments/feature_embedding/run_experiments.py --list

# Dry run to see what would run
./venv/bin/python experiments/feature_embedding/run_experiments.py --priority AE-P1 --dry-run

# Run single experiment
./venv/bin/python experiments/feature_embedding/run_experiments.py --exp-id AE-01
```

---

## File Locations

- Experiment runner: `experiments/feature_embedding/run_experiments.py`
- Results: `outputs/feature_embedding/{tier}_dembed_{d_embed}/results.json`
- Loss functions: `src/training/losses.py`
- Embeddings: `src/models/feature_embeddings.py`
- This tracker: `docs/feature_embedding_experiment_tracker.md`
