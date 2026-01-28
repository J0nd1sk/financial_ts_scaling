# Comprehensive HPO Performance Analysis: Phase 6C (a50 + a100)

**Generated**: 2026-01-27
**Task**: 1% threshold classification (will price rise >1% within horizon?)
**Horizon**: 1 day
**Validation Period**: 2023-01-01 to 2024-12-31
**Models Evaluated**: Top 3 per budget × 2 tiers = 18 models

---

## Executive Summary

| Tier | Features | Best Budget | Best AUC | Avg Precision | Avg Recall | Collapse? |
|------|----------|-------------|----------|---------------|------------|-----------|
| **a50** | 55 | 20M | **0.7315** | 50.8% | 9.65% | No |
| **a100** | 105 | 20M | 0.7189 | 47.5% | 11.4% | No |

**Key Findings**:
1. **More features HURT performance**: a50 (55 features) outperforms a100 (105 features) by 1.3% AUC
2. **Scaling laws VIOLATED**: 200M parameters perform WORSE than 20M
3. **Recall is critically low**: ~10% means missing 90% of opportunities
4. **No probability collapse**: Wide prediction ranges [0.01-0.88] show models are learning

---

## Section 1: Complete Performance Tables

### 1.1 a50 Tier (55 Features)

| Budget | Trial | AUC | Precision | Recall | Accuracy | TP | FP | FN | TN | Pred Range |
|--------|-------|-----|-----------|--------|----------|----|----|----|----|------------|
| **2M** | 15 | 0.7302 | 40.9% | 11.8% | 81.0% | 9 | 13 | 67 | 333 | [0.005, 0.881] |
| 2M | 45 | 0.7300 | 53.8% | 9.2% | 82.2% | 7 | 6 | 69 | 340 | [0.006, 0.672] |
| 2M | 48 | 0.7300 | 53.8% | 9.2% | 82.2% | 7 | 6 | 69 | 340 | [0.006, 0.672] |
| **20M** | 5 | **0.7315** | 53.3% | 10.5% | 82.2% | 8 | 7 | 68 | 339 | [0.012, 0.721] |
| 20M | 4 | 0.7294 | 50.0% | 9.2% | 82.0% | 7 | 7 | 69 | 339 | [0.012, 0.719] |
| 20M | 40 | 0.7244 | 41.7% | 13.2% | 81.0% | 10 | 14 | 66 | 332 | [0.004, 0.835] |
| **200M** | 8 | 0.7294 | 54.5% | 7.9% | 82.2% | 6 | 5 | 70 | 341 | [0.011, 0.729] |
| 200M | 25 | 0.7294 | 54.5% | 7.9% | 82.2% | 6 | 5 | 70 | 341 | [0.011, 0.729] |
| 200M | 26 | 0.7294 | 54.5% | 7.9% | 82.2% | 6 | 5 | 70 | 341 | [0.011, 0.729] |

### 1.2 a100 Tier (105 Features)

| Budget | Trial | AUC | Precision | Recall | Accuracy | TP | FP | FN | TN | Pred Range |
|--------|-------|-----|-----------|--------|----------|----|----|----|----|------------|
| **2M** | 30 | 0.7173 | 50.0% | 6.6% | 82.0% | 5 | 5 | 71 | 341 | [0.016, 0.686] |
| 2M | 31 | 0.7173 | 50.0% | 6.6% | 82.0% | 5 | 5 | 71 | 341 | [0.016, 0.686] |
| 2M | 32 | 0.7173 | 50.0% | 6.6% | 82.0% | 5 | 5 | 71 | 341 | [0.016, 0.686] |
| **20M** | 30 | **0.7189** | 50.0% | 9.2% | 82.0% | 7 | 7 | 69 | 339 | [0.001, 0.850] |
| 20M | 31 | 0.7189 | 50.0% | 9.2% | 82.0% | 7 | 7 | 69 | 339 | [0.001, 0.850] |
| 20M | 32 | 0.7189 | 50.0% | 9.2% | 82.0% | 7 | 7 | 69 | 339 | [0.001, 0.850] |
| **200M** | 37 | 0.7152 | 40.0% | **21.1%** | 80.1% | 16 | 24 | 60 | 322 | [0.010, 0.874] |
| 200M | 0 | 0.7151 | 47.6% | 13.2% | 81.8% | 10 | 11 | 66 | 335 | [0.003, 0.852] |
| 200M | 41 | 0.7150 | 40.0% | **21.1%** | 80.1% | 16 | 24 | 60 | 322 | [0.009, 0.881] |

---

## Section 2: Scaling Law Analysis

### 2.1 Parameter Scaling (Fixed Feature Count)

| Tier | 2M AUC | 20M AUC | 200M AUC | Trend | Law Holds? |
|------|--------|---------|----------|-------|------------|
| a50 | 0.7302 | **0.7315** | 0.7294 | Peak at 20M | **NO** |
| a100 | 0.7173 | **0.7189** | 0.7152 | Peak at 20M | **NO** |

**Finding**: Neither tier shows monotonic improvement with parameters. 20M is optimal.

### 2.2 Feature Scaling (Fixed Parameter Budget)

| Budget | a50 AUC | a100 AUC | Diff | Winner |
|--------|---------|----------|------|--------|
| 2M | 0.7302 | 0.7173 | -1.3% | **a50** |
| 20M | 0.7315 | 0.7189 | -1.3% | **a50** |
| 200M | 0.7294 | 0.7152 | -1.4% | **a50** |

**Finding**: More features consistently HURT performance. 55 features beats 105 features.

---

## Section 3: Precision-Recall Tradeoff Analysis

### 3.1 Precision vs Recall by Model

| Model | Precision | Recall | F1 | Strategy |
|-------|-----------|--------|-----|----------|
| a50-2M-T15 | 40.9% | 11.8% | 18.4% | Higher recall, lower precision |
| a50-20M-T5 | 53.3% | 10.5% | 17.6% | Balanced |
| a50-200M-T8 | 54.5% | 7.9% | 13.8% | High precision, low recall |
| a100-2M-T30 | 50.0% | 6.6% | 11.6% | Very low recall |
| a100-20M-T30 | 50.0% | 9.2% | 15.5% | Moderate |
| a100-200M-T37 | 40.0% | **21.1%** | 27.6% | **Best recall** |

### 3.2 Key Observations

1. **Recall ceiling**: Even the best model (a100-200M-T37) only catches 21% of opportunities
2. **Precision floor**: Models with higher recall have ~40% precision (random baseline ~18%)
3. **F1 range**: 12-28% indicates poor overall classification quality
4. **Inverse relationship**: Models with best AUC often have worst recall (too conservative)

---

## Section 4: Probability Collapse Detection

### 4.1 Prediction Range Analysis

| Model | Pred Min | Pred Max | Range Width | Collapsed? |
|-------|----------|----------|-------------|------------|
| a50-2M-T15 | 0.005 | 0.881 | 0.876 | No |
| a50-20M-T5 | 0.012 | 0.721 | 0.709 | No |
| a50-200M-T8 | 0.011 | 0.729 | 0.718 | No |
| a100-2M-T30 | 0.016 | 0.686 | 0.670 | No |
| a100-20M-T30 | 0.001 | 0.850 | 0.849 | No |
| a100-200M-T37 | 0.010 | 0.874 | 0.864 | No |

**Finding**: No probability collapse detected. All models produce diverse predictions.

### 4.2 Collapse Definition

- **Collapsed**: Range width < 0.2 (all predictions clustered around 0.4-0.6)
- **Healthy**: Range width > 0.5 with predictions spanning most of [0, 1]

---

## Section 5: Optimal Hyperparameters

### 5.1 Best Configuration per Tier/Budget

| Tier | Budget | d_model | n_layers | n_heads | dropout | lr | weight_decay |
|------|--------|---------|----------|---------|---------|-----|--------------|
| a50 | 2M | 96 | 2 | 8 | **0.50** | 5e-4 | 1e-5 |
| a50 | 20M | 128 | 6 | 8 | 0.30 | 5e-5 | 1e-4 |
| a50 | 200M | 128 | 6 | 16 | **0.50** | 1e-4 | 1e-4 |
| a100 | 2M | 96 | 2 | 8 | 0.10 | 1e-5 | 1e-5 |
| a100 | 20M | 64 | 4 | 8 | **0.70** | 5e-4 | 1e-3 |
| a100 | 200M | 320 | 12 | 8 | **0.50** | 1e-5 | 1e-5 |

### 5.2 Hyperparameter Patterns

1. **Dropout**: High values (0.5-0.7) dominate best models - strong regularization needed
2. **Learning rate**: 1e-5 to 5e-4 range, no clear pattern
3. **Model depth**: Shallow works for 2M (2 layers), deeper for 200M (12 layers)
4. **d_model**: 64-128 optimal, even for 200M budget (not fully utilizing capacity)

---

## Section 6: Class Imbalance Analysis

### 6.1 Validation Set Statistics

- **Total samples**: 422
- **Positive samples (>1% rise)**: 76 (18.0%)
- **Negative samples**: 346 (82.0%)
- **Imbalance ratio**: 4.6:1

### 6.2 Model Behavior on Imbalanced Data

| Model | Positive Preds | Coverage | Precision | Comment |
|-------|---------------|----------|-----------|---------|
| a50-2M-T15 | 22 | 5.2% | 40.9% | Conservative |
| a50-20M-T5 | 15 | 3.6% | 53.3% | Very conservative |
| a100-200M-T37 | 40 | 9.5% | 40.0% | More aggressive |

**Finding**: Best AUC models are overly conservative, predicting positive for only 3-5% of samples.

---

## Section 7: Conclusions & Recommendations

### 7.1 Key Conclusions

1. **Scaling laws do NOT apply** to this financial prediction task
2. **Feature bloat hurts**: 55 features > 105 features consistently
3. **20M is optimal**: Neither smaller (2M) nor larger (200M) improves performance
4. **High regularization critical**: Dropout 0.5+ consistently wins
5. **Recall is the bottleneck**: All models miss 80-90% of opportunities

### 7.2 Recommendations for Future Work

1. **Feature selection**: Investigate which 55 features in a50 provide signal
2. **Threshold optimization**: Default 0.5 threshold may not be optimal
3. **Class weighting**: Address 4.6:1 imbalance with weighted loss
4. **Ensemble methods**: Combine high-recall and high-precision models
5. **Task reformulation**: Consider regression or multi-class targets

### 7.3 HPO Process Improvements (Implemented)

Fixed `experiments/templates/hpo_template.py` to use `verbose=True` so all future HPO runs will capture:
- Precision and Recall
- Prediction range (min, max)
- Confusion matrix (TP, TN, FP, FN)
- Accuracy

---

## Appendix: Raw Data

### A.1 Validation Set Composition

| Metric | Value |
|--------|-------|
| Date range | 2023-01-01 to 2024-12-31 |
| Trading days | 422 |
| Positive class | 76 (18.0%) |
| Negative class | 346 (82.0%) |

### A.2 Baseline Performance

| Model | Description | Expected AUC |
|-------|-------------|--------------|
| Random | 50/50 guess | 0.500 |
| Always negative | Predict all 0 | 0.500 |
| Prior-informed | Predict 18% positive | ~0.500 |
| **Our best** | a50-20M | **0.7315** |

Our best model beats random by 23% AUC.
