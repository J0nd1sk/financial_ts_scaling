# Discussion Section Draft: Phase 6A Findings

## Why Scaling Laws Fail for Financial Time-Series

### The Data-Limited Regime

Our experiments place financial time-series prediction firmly in what Hoffmann et al. (2022) term the "data-limited regime." Consider the parameter-to-sample ratios:

| Budget | Parameters | Training Samples | Ratio |
|--------|------------|------------------|-------|
| 2M | 2,065,217 | ~4,000 | 516:1 |
| 20M | 20,742,060 | ~4,000 | 5,185:1 |
| 200M | 178,374,913 | ~4,000 | 44,594:1 |
| 2B | 1,646,530,476 | ~4,000 | 411,633:1 |

For comparison, the Chinchilla scaling law suggests optimal ratio of ~20:1 (parameters:tokens). Our smallest model already violates this by 25×; our largest by 20,000×.

### Effective Data Hypothesis

We propose that effective data for learning is not simply sample count, but:

$$D_{eff} = S \times F \times R(F)$$

Where:
- $S$ = number of samples
- $F$ = number of features
- $R(F)$ = relationship complexity, approximately $O(F^2)$

With 25 features, $R(F) \approx 300$ pairwise relationships. Even with 4,000 samples, $D_{eff} \approx 1.2M$ effective training signals. This cannot support models with hundreds of millions of parameters.

### Feature Richness as Prerequisite

This analysis motivates our core hypothesis: **feature richness is a prerequisite for scaling laws to emerge**. Phase 6C will test this by scaling features from 25 to 2000:

| Feature Count | $R(F)$ | $D_{eff}$ |
|---------------|--------|-----------|
| 25 | 300 | 1.2M |
| 200 | 19,900 | 80M |
| 2000 | 1,999,000 | 8B |

At 2000 features, effective data approaches the scale needed for 200M+ parameter models.

## Comparison to Prior Work

### Scaling Laws (Kaplan et al., 2020)

Kaplan et al. established that language model loss follows:
$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}$$

with $\alpha_N \approx 0.076$. Our results show **negative** $\alpha_N$ (inverse scaling), indicating this relationship does not hold in our domain.

### Chinchilla (Hoffmann et al., 2022)

Hoffmann et al. found compute-optimal training requires balanced scaling of data and parameters. Our experiments confirm this indirectly: with fixed (insufficient) data, parameter scaling is not just suboptimal—it's harmful.

### Financial ML Literature

Traditional quantitative finance uses small models (logistic regression, random forests) with rich features. Our results support this practice: the complexity bottleneck is in features, not model capacity. This aligns with Rasekhschaffe & Jones (2019), who found ensemble methods with 100+ features outperform deep networks in equity prediction.

## Architectural Insights

### The Width-Depth Tradeoff

Our architecture analysis reveals budget-dependent optima:

| Budget | Optimal Style | d_model | n_layers |
|--------|---------------|---------|----------|
| 2M | Balanced | 64 | 32-48 |
| 20M | Wide-shallow | 256 | 32 |
| 200M | Wide-shallow | 384-512 | 48-96 |
| 2B | Balanced | 1024 | 180 |

At 2M, balanced architectures excel. As parameters increase, width becomes more important than depth. At 2B, balanced returns as optimal, possibly because extreme width or depth both cause stability issues.

### Attention Heads: Minimal Impact

Perhaps our most surprising finding: n_heads has negligible effect. This contrasts sharply with NLP, where multi-head attention is essential for capturing diverse linguistic phenomena.

We speculate that financial time series have simpler attention patterns. The relevant dependencies (e.g., recent vs. historical prices, trend vs. volatility) may be capturable with just two attention perspectives.

### Horizon-Specific Architectures

Architectures do not transfer across prediction horizons. The catastrophic +132% degradation when using h3-optimal architecture on h5 indicates fundamental differences in optimal temporal aggregation.

This has practical implications: deployment systems predicting multiple horizons need separate models, not a single multi-horizon architecture.

## Limitations

### Single Asset

Our experiments use only SPY. While this provides clean isolation, generalization to other assets, asset classes, or market regimes is unknown. Phase 6D will address this.

### Binary Classification Task

We predict >1% price movements (binary). Regression tasks or different thresholds might show different scaling behavior.

### Limited Historical Period

Training data spans 1993-2020. Pre-1993 data might show different patterns due to different market structure (algorithmic trading was minimal before 2000).

### Single Architecture Family

We test only PatchTST. Other architectures (Informer, Autoformer, LSTM) might exhibit different scaling behavior.

## Implications for Practitioners

### Model Selection

For equity prediction with limited features (~25):
- Use 2M parameter models (d=64, L=32, h=2)
- Train for ~50 epochs with patience=10 early stopping
- Expect validation loss ~0.26-0.34 depending on horizon
- Larger models will overfit and perform worse

### Feature Engineering Priority

Our results suggest that resources are better invested in feature engineering than model scaling. Adding meaningful features (fundamentals, sentiment, cross-asset correlations) may be prerequisite to benefiting from larger models.

### Horizon-Specific Models

Do not assume one model serves all horizons. Train separate models for 1-day, 3-day, and 5-day predictions. The 3-day horizon appears most predictable.

### Compute Efficiency

2M models train in ~40 seconds; 2B models take ~25 minutes. Given 2M achieves better performance, there is no justification for larger models in this regime.

## Future Directions

### Phase 6B: Horizon Scaling
Test whether longer horizons (weekly, monthly) benefit from different architectural patterns or larger models.

### Phase 6C: Feature Scaling
The critical test of our "feature richness prerequisite" hypothesis. Scale features from 25 to 2000 and retest parameter scaling.

### Phase 6D: Data Scaling
Add more assets (ETFs, stocks, indices) and test whether data diversity enables larger models.

### Cross-Domain Validation
Test whether similar inverse scaling occurs in other domains: weather prediction, medical time-series, IoT sensor data.

---

*This discussion section is a draft. Final version will incorporate Phase 6B-6D results and additional statistical analysis.*
