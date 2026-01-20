# Phase 6A Research Gap Analysis: Our Implementation vs. Best Practices

**Created:** 2026-01-20
**Purpose:** Compare our PatchTST implementation against published best practices for time-series transformers on financial data

---

## Executive Summary

Research confirms that transformer models CAN outperform LSTMs on financial time-series, but require specific techniques we're missing. The most critical gap is **RevIN (Reversible Instance Normalization)**, which is standard in PatchTST implementations but completely absent from ours.

---

## Critical Gap #1: RevIN Not Implemented

### What RevIN Is

RevIN (Reversible Instance Normalization) is a normalization technique specifically designed for non-stationary time series. It:
1. Normalizes each input instance by its own mean/std at model input
2. Processes the normalized data through the model
3. Denormalizes the output using the same statistics

```python
# From official PatchTST implementation
class RevIN(nn.Module):
    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean = torch.mean(x, dim=1, keepdim=True)
            x = x - self.mean
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True) + eps)
            x = x / self.stdev
        elif mode == 'denorm':
            x = x * self.stdev + self.mean
        return x
```

### Why It Matters

> "RevIN normalizes time-series data instance-wise at the input, then restores the original distribution in the output layer, making it possible to leverage the benefits of normalization without losing essential non-stationary information."
> — [RevIN Paper (ICLR 2022)](https://openreview.net/forum?id=cGDAkQo1C0p)

> "RevIN is a key technique enabling simple linear models to achieve state-of-the-art performance in time series forecasting."
> — [RevIN GitHub](https://github.com/ts-kim/RevIN)

### Our Implementation

❌ **We have NO RevIN.** Our model receives raw, unnormalized features.

### Impact

This is likely the PRIMARY cause of our distribution shift problem. Without RevIN:
- Model trains on Close≈88, sees Close≈576 at test time
- Features have completely different scales across time periods
- Model outputs "I don't know" (sigmoid midpoint) for OOD inputs

### Fix Required

Add RevIN layer to PatchTST:
```python
if self.revin:
    x = self.revin_layer(x, 'norm')  # Before model
# ... model processing ...
# For classification, skip denorm (we want normalized predictions)
```

---

## Critical Gap #2: No Input Normalization

### Best Practice (from official PatchTST data loading)

```python
# From official implementation
self.scaler = StandardScaler()
train_data = df_data[border1s[0]:border2s[0]]
self.scaler.fit(train_data.values)  # Fit on TRAINING data only
data = self.scaler.transform(df_data.values)  # Apply to all
```

### Our Implementation

❌ **No normalization at all.** Raw features passed directly to model.

### Why Standard Scaling Isn't Enough

From research on financial time series:
> "Z-score normalization is unable to efficiently handle non-stationary time series since the statistics are fixed during training and inference."
> — [Deep Adaptive Input Normalization](https://arxiv.org/pdf/1902.07892)

This is why RevIN (per-instance) is preferred over global StandardScaler for financial data.

---

## Critical Gap #3: Architecture Differences

### Official PatchTST Settings

| Parameter | Official | Ours | Issue |
|-----------|----------|------|-------|
| context_length | 336-512 | **60** | ⚠️ 5-8x shorter |
| n_layers | 2-4 | **48** | ⚠️ 12-24x more layers! |
| n_heads | 4-16 | 2-32 | ✓ Overlapping range |
| d_model | 16-256 | 64-1024 | ✓ Overlapping range |
| dropout | 0.2 | 0.1 | ⚠️ Half the dropout |
| patch_len | 16 | 16 | ✓ Match |
| stride | 8 | 8 | ✓ Match |

### Critical Issues

**1. Context Length (60 vs 336-512)**
> "PatchTST/42 uses look-back window L=336, while PatchTST/64 uses L=512."
> — [PatchTST Paper](https://arxiv.org/abs/2211.14730)

With only 60 days, we have ~6 patches. Official has 42-64 patches. We may not have enough temporal context.

**2. Number of Layers (48 vs 2-4)**
> "For encoder layers: 3 for Electricity, 4 for Traffic, and 2 for other datasets."
> — [PatchTST Hugging Face](https://huggingface.co/blog/patchtst)

We're using 12-24x more layers than recommended. This is MASSIVE overkill for:
- Our small context length (6 patches)
- Our limited training data (~6000 samples)

More layers with limited data = overfitting and degenerate solutions.

**3. Dropout (0.1 vs 0.2)**
With our excessive layer count, lower dropout compounds overfitting risk.

---

## Critical Gap #4: Positional Encoding Initialization

### Our Implementation (patchtst.py:98)

```python
self.position_embedding = nn.Parameter(torch.randn(1, max_patches, d_model))
```

This uses `randn` with std=1.0.

### Standard Practice

Typical transformer positional embeddings use much smaller initialization:
- BERT: std=0.02
- GPT: std=0.02
- Most implementations: std=0.01-0.02

### Measured Impact (from our diagnostics)

| Component | L2 Norm |
|-----------|---------|
| Patch embeddings | 4.63 |
| Positional encoding | 8.26 |
| **Ratio** | **0.56** |

The positional encoding is ~2x larger than the content signal. The model may be attending more to position than content.

---

## Critical Gap #5: Classification vs Forecasting

### PatchTST Design

PatchTST was designed for **forecasting** (predicting future values), not **classification** (predicting direction/threshold).

> "The correlation between channels is crucial for stock price data, and models like PatchTST, which convert the input into univariate series, fail to perform well because they omit these essential inter-channel correlations."
> — [Tokenizing Stock Prices Paper](https://arxiv.org/html/2504.17313)

### Implications

1. Our classification head (flatten + linear) may be too simple
2. Channel-independence may lose important cross-feature correlations
3. The model may need different training dynamics for classification

### Possible Mitigations

- Use a more complex classification head (MLP instead of linear)
- Consider channel-mixing variants
- Use AUC-based loss instead of BCE

---

## Critical Gap #6: Loss Function for Imbalanced Classification

### Research Finding

> "The Galformer model (transformer with generative decoding and **hybrid loss function**) outperforms ARIMA, LSTM, Transformer, and Informer models."
> — [Galformer Paper](https://www.nature.com/articles/s41598-024-72045-3)

> "A carefully constructed vanilla LSTM consistently outperforms more sophisticated attention-driven models when trained under default hyperparameters and in **data-limited financial settings**."
> — [StockBot 2.0 Paper](https://arxiv.org/html/2601.00197)

### Our Implementation

- BCELoss with no class weighting
- 10% positive rate for h1 (9:1 imbalance)
- No focal loss, no AUC loss

### Recommendations

For imbalanced binary classification:
1. BCE + pos_weight (simple fix)
2. Focal Loss (down-weight easy examples)
3. SoftAUC Loss (optimize ranking directly)
4. Hybrid losses (BCE + AUC regularization)

---

## Gap Summary Table

| Component | Official/Best Practice | Our Implementation | Severity |
|-----------|------------------------|-------------------|----------|
| **RevIN** | Yes, built into model | ❌ None | 🔴 CRITICAL |
| **Input normalization** | StandardScaler on train | ❌ None | 🔴 CRITICAL |
| **Context length** | 336-512 | 60 | 🟠 HIGH |
| **Encoder layers** | 2-4 | 48 | 🟠 HIGH |
| **Positional encoding init** | std=0.02 | std=1.0 | 🟡 MEDIUM |
| **Dropout** | 0.2 | 0.1 | 🟡 MEDIUM |
| **Loss function** | Hybrid/weighted | Plain BCE | 🟡 MEDIUM |
| **Patch length** | 16 | 16 | ✅ OK |
| **Stride** | 8 | 8 | ✅ OK |

---

## Recommended Fix Priority

### Tier 1: Must Fix (Blocking)

1. **Add RevIN layer to PatchTST**
   - Most impactful single change
   - Handles distribution shift elegantly
   - Standard in all modern time-series transformers

2. **Reduce encoder layers to 2-4**
   - 48 layers is extreme overkill
   - Causes degenerate solutions with limited data
   - Match official recommendations

### Tier 2: Should Fix (High Impact)

3. **Increase context length to 336+**
   - Gives model more temporal context
   - More patches = richer attention patterns
   - May require data pipeline changes

4. **Add class weighting to loss**
   - BCE + pos_weight at minimum
   - Consider Focal or SoftAUC loss

### Tier 3: Should Consider (Medium Impact)

5. **Fix positional encoding initialization**
   - Change to std=0.02
   - Minor compared to RevIN

6. **Increase dropout to 0.2**
   - Especially if keeping more layers

---

## Validation Experiments Needed

Before re-running full scaling experiments:

1. **RevIN A/B test**: Same model with/without RevIN
2. **Layer count ablation**: 2 vs 4 vs 8 vs 16 layers
3. **Context length ablation**: 60 vs 120 vs 240 vs 336
4. **Loss function comparison**: BCE vs pos_weight vs Focal vs SoftAUC
5. **Baseline comparison**: MLP vs PatchTST (does transformer help?)

---

## Sources

- [PatchTST Paper (ICLR 2023)](https://arxiv.org/abs/2211.14730)
- [PatchTST Official Implementation](https://github.com/yuqinie98/PatchTST)
- [RevIN Paper (ICLR 2022)](https://openreview.net/forum?id=cGDAkQo1C0p)
- [RevIN Implementation](https://github.com/ts-kim/RevIN)
- [PatchTST Hugging Face Blog](https://huggingface.co/blog/patchtst)
- [PatchTST Nixtla Documentation](https://nixtlaverse.nixtla.io/neuralforecast/models.patchtst.html)
- [Deep Adaptive Input Normalization](https://arxiv.org/pdf/1902.07892)
- [Transformer vs LSTM for Stock Prediction](https://www.sciencedirect.com/science/article/pii/S2666827025001136)
- [StockBot 2.0: LSTMs vs Transformers](https://arxiv.org/html/2601.00197)
- [Galformer: Hybrid Loss](https://www.nature.com/articles/s41598-024-72045-3)

