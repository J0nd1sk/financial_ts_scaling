# Draft Notes: Financial TS Scaling Experiments

Working notes and analysis chunks for the research paper. Each section represents a discrete analysis session.

---

## 2025-12-31: Cross-Horizon Architecture Transfer Analysis

### Context
- Phase 6A HPO complete for 2M (all horizons) and 20M (all horizons)
- 200M h1 HPO at trial 47/50
- Supplementary experiments: tested h3-optimal config (d=64, L=32, h=2) on h1 and h5 horizons
- **Important caveat**: Training data limited to pre-2021 (~8K rows for SPY). Parameter scaling benefits may require more data to manifest.

### Supplementary Experiment Results

Tested the h3-optimal architecture on other horizons:

| Horizon | HPO-Optimal | With h3-config | Degradation |
|---------|-------------|----------------|-------------|
| h1 (1-day) | 0.3199 | 0.3840 | +20% worse |
| h3 (3-day) | 0.2630 | 0.2630 | (baseline) |
| h5 (5-day) | 0.3371 | 0.7815 | +132% worse |

The h3-optimal config catastrophically fails on h5, suggesting architecture must be tuned per-horizon.

### Optimal Architectures by Horizon (2M Budget)

| Horizon | d_model | n_layers | n_heads | dropout | val_loss |
|---------|---------|----------|---------|---------|----------|
| h1 | 64 | 48 | 2 | 0.30 | 0.3199 |
| h3 | 64 | 32 | 2 | 0.10 | 0.2630 |
| h5 | 64 | 64 | 16 | 0.24 | 0.3371 |

Observation: All prefer narrow models (d=64) but depth varies significantly.

### Sensitivity Analysis (Wrong Architecture)

When using h3-optimal (L=32) on other horizons:

**n_heads sensitivity (d=64, L=32, dropout=0.10):**
- h1: h=2, h=8, h=16 all give val=0.4012 (no difference)
- h5: h=2, h=8, h=16 all give val=0.7815 (no difference)

**Conclusion**: When depth is wrong, tuning n_heads has zero effect.

**dropout sensitivity (d=64, L=32, h=2):**
- h1: drop=0.10→0.4012, drop=0.20→0.3986, drop=0.30→0.3840 (helps partially)
- h5: drop=0.10→0.7815, drop=0.20→0.7869, drop=0.30→0.7827 (no help)

**Conclusion**: Dropout can partially compensate for wrong depth on h1, but not on h5 where the mismatch is severe.

### Cross-Scale Comparison (2M vs 20M)

| Budget | Horizon | val_loss | d_model | n_layers | n_heads |
|--------|---------|----------|---------|----------|---------|
| 2M | h1 | 0.3199 | 64 | 48 | 2 |
| 2M | h3 | 0.2630 | 64 | 32 | 2 |
| 2M | h5 | 0.3371 | 64 | 64 | 16 |
| 20M | h1 | 0.3483 | 128 | 180 | 16 |
| 20M | h3 | 0.3191 | 256 | 32 | 2 |
| 20M | h5 | 0.3458 | 384 | 12 | 4 |

**Key observations:**
1. 20M performs WORSE than 2M on h1 (0.3483 vs 0.3199)
2. 20M performs WORSE than 2M on h5 (0.3458 vs 0.3371)
3. 20M performs worse than 2M on h3 (0.3191 vs 0.2630)
4. h3 maintains L=32 preference at both scales
5. 20M h5 prefers shallow (L=12) vs 2M h5 deep (L=64) — contradictory pattern

### Preliminary Hypotheses

1. **Architecture-horizon coupling is strong**: Configs do not transfer across horizons. Each horizon needs dedicated HPO.

2. **Depth (n_layers) is the primary lever**: More important than n_heads or d_model for matching to prediction horizon.

3. **Parameter scaling alone does not improve performance** (at current data scale): 20M consistently underperforms 2M. This could indicate:
   - Overfitting due to limited training data (~4K samples after splits)
   - Financial time series may have an optimal model capacity ceiling
   - Need to scale data before scaling parameters (chinchilla-style)

4. **h3 (3-day) is the "easiest" prediction horizon**: Achieves best val_loss (0.2630) with shallowest model (L=32). May align with natural market cycles.

### Open Questions for Further Investigation

- Does 200M show same pattern? (Currently at trial 47/50 for h1)
- What happens with more training data (multi-asset, longer history)?
- Is there a data:params ratio threshold where scaling helps?
- Why does h3 prefer shallow while h1/h5 prefer deeper?

---

*Next analysis chunk will be added below this line.*

---

