# Foundation Model & Decoder Architecture Investigation

**Status:** Planning
**Branch:** `experiment/foundation-decoder-investigation`
**Created:** 2026-01-22
**Version:** 1.0

---

## Motivation

Phase 6A concluded that PatchTST operates in a **data-limited regime**: increasing parameters from 2M to 200M provided only +1.7% AUC improvement. This raises a fundamental question:

**Is the limitation due to:**
1. Insufficient signal in 25 features? (→ Feature scaling, Phase 6C)
2. Wrong architectural inductive bias? (→ This investigation)
3. Both?

PatchTST is an **encoder-only** transformer. While encoders excel at representation learning, **decoder** architectures (or encoder-decoder hybrids) may:
- Extract different temporal patterns through causal attention
- Benefit from pre-training on billions of diverse time points (foundation models)
- Be more sample-efficient through transfer learning

---

## Research Questions

### Primary Questions
1. **Foundation Model Transfer**: Can foundation models pre-trained on general time series improve financial prediction over from-scratch PatchTST?
2. **Decoder vs Encoder**: Does causal (decoder) attention extract different signal than bidirectional (encoder) attention?
3. **Probabilistic Outputs**: Do probabilistic forecasters (Lag-Llama) naturally suit threshold classification tasks?

### Secondary Questions
4. Is self-attention even necessary? (TimeMixer is pure MLP)
5. Does "inverted" attention (across features, not time) help? (iTransformer)

---

## Hypotheses

| ID | Hypothesis | Test |
|----|------------|------|
| H1 | Foundation models outperform from-scratch training due to transfer learning | Compare fine-tuned Lag-Llama/TimesFM vs PatchTST |
| H2 | Decoder attention is NOT inherently better than encoder for fixed-horizon classification | Compare decoder-only vs encoder-only architectures |
| H3 | Probabilistic outputs improve threshold classification | Compare Lag-Llama distribution-based predictions vs point forecasts |
| H4 | Attention may be unnecessary for this task | Compare TimeMixer (MLP) vs transformer-based models |

---

## Models to Evaluate

### Tier 1: Foundation Models (Highest Priority)

| Model | Type | Pre-training | Fine-tuning | Notes |
|-------|------|--------------|-------------|-------|
| **Lag-Llama** | Decoder-only | General TS | ✅ Supported | Probabilistic, best fit for threshold tasks |
| **TimesFM** | Decoder-only | General TS | ✅ Supported | Google, 200M params, newest (2.5) |
| **Timer** | Generative | General TS | ✅ Supported | Tsinghua, checkpoints available |

### Tier 2: Architecture Comparison (Trained from Scratch)

These models contribute to **parameter scaling law research** - they are NOT foundation models.

| Model | Type | Key Innovation | Notes |
|-------|------|----------------|-------|
| **PatchTST** | Encoder-only | Patch embeddings | Our baseline |
| **iTransformer** | "Inverted" | Feature-wise attention | ICLR 2024 Spotlight |
| **TimeMixer** | MLP (no attention) | No attention needed? | Tests attention necessity |
| **Informer** | Encoder-Decoder | ProbSparse attention | Memory-efficient long sequences |
| **Autoformer** | Encoder-Decoder | Autocorrelation | Replaces attention with autocorr |
| **FEDformer** | Encoder-Decoder | Frequency domain | Fourier/Wavelet decomposition |
| **ETSformer** | Encoder-Decoder | Exponential smoothing | Stats + neural hybrid |
| **Crossformer** | Encoder-Decoder | Cross-variate | Explicit cross-dimension attention |
| **ARMA-Attention** | Hybrid | Statistical + neural | Conditional on implementation |

### Tier 3: Financial-Specific (If Available)

| Model | Status | Notes |
|-------|--------|-------|
| **LENS** | ❌ No public weights | Paper only (Aug 2024) |
| **MarketGPT** | ✅ HuggingFace | Smaller, less tested |

---

## Experimental Design

### Control Variables (Same as Phase 6A)
- **Data**: SPY OHLCV + 20 indicators (25 features)
- **Target**: >1% threshold within horizon (using HIGH prices)
- **Splits**: SimpleSplitter (train <2023, val 2023-2024, test 2025+)
- **Metrics**: AUC-ROC, Accuracy, Precision, Recall, Prediction Range
- **Context**: 80 days (ablation-validated)

### Experimental Matrix

**Phase 1: Foundation Model Fine-Tuning**
| Experiment | Model | Horizon | Approach |
|------------|-------|---------|----------|
| FD-01 | Lag-Llama | H1 | Fine-tune, use distribution for threshold P |
| FD-02 | Lag-Llama | H3 | Fine-tune, use distribution for threshold P |
| FD-03 | TimesFM | H1 | Fine-tune, forecast → threshold |
| FD-04 | TimesFM | H3 | Fine-tune, forecast → threshold |

**Phase 2: Architecture Comparison (Trained from Scratch)**
| Experiment | Model | Horizon | Notes |
|------------|-------|---------|-------|
| FD-05 | iTransformer | H1 | Train from scratch, classification head |
| FD-06 | iTransformer | H3 | Train from scratch, classification head |
| FD-07 | TimeMixer | H1 | Train from scratch, classification head |
| FD-08 | TimeMixer | H3 | Train from scratch, classification head |
| FD-09 | Informer | H1 | ProbSparse attention |
| FD-10 | Informer | H3 | ProbSparse attention |
| FD-11 | Autoformer | H1 | Autocorrelation-based |
| FD-12 | Autoformer | H3 | Autocorrelation-based |
| FD-13 | FEDformer | H1 | Frequency domain |
| FD-14 | FEDformer | H3 | Frequency domain |
| FD-15 | ETSformer | H1 | Exponential smoothing |
| FD-16 | ETSformer | H3 | Exponential smoothing |
| FD-17 | Crossformer | H1 | Cross-variate attention |
| FD-18 | Crossformer | H3 | Cross-variate attention |

**Baseline (from Phase 6A)**
| Model | H1 AUC | H3 AUC |
|-------|--------|--------|
| PatchTST 2M | 0.706 | 0.618 |
| PatchTST 200M | 0.718 | 0.622 |

### Success Criteria

| Outcome | Implication |
|---------|-------------|
| Foundation model > PatchTST by ≥5% AUC | Transfer learning helps; pursue fine-tuning path |
| Decoder ≈ Encoder (within ±2%) | Architecture doesn't matter; focus on features |
| TimeMixer ≈ Transformers | Attention unnecessary; simpler models sufficient |
| Foundation model >> from-scratch | Data-limited regime confirmed; pre-training essential |

---

## Implementation Plan

### Task 1: Environment Setup
- [ ] Install Lag-Llama dependencies (GluonTS)
- [ ] Install TimesFM dependencies
- [ ] Verify GPU/MPS compatibility
- [ ] Create `src/models/foundation/` module

### Task 2: Lag-Llama Integration
- [ ] Download pre-trained weights
- [ ] Create fine-tuning script
- [ ] Adapt for classification (distribution → threshold probability)
- [ ] Run FD-01, FD-02

### Task 3: TimesFM Integration
- [ ] Download pre-trained weights (HuggingFace)
- [ ] Create fine-tuning script
- [ ] Adapt for classification (forecast → threshold check)
- [ ] Run FD-03, FD-04

### Task 4: iTransformer Implementation
- [ ] Port architecture to our codebase
- [ ] Add classification head
- [ ] Integrate with our Trainer
- [ ] Run FD-05, FD-06

### Task 5: TimeMixer Implementation
- [ ] Port architecture to our codebase
- [ ] Add classification head
- [ ] Integrate with our Trainer
- [ ] Run FD-07, FD-08

### Task 6: Analysis & Documentation
- [ ] Compare all results against PatchTST baseline
- [ ] Statistical significance tests
- [ ] Document findings
- [ ] Decision: roll into main project or discard

### Task 7: Informer Implementation
- [ ] Port Informer architecture (ProbSparse attention)
- [ ] Add classification head
- [ ] Integrate with Trainer
- [ ] Run FD-09, FD-10

### Task 8: Autoformer Implementation
- [ ] Port Autoformer architecture (autocorrelation)
- [ ] Add classification head
- [ ] Integrate with Trainer
- [ ] Run FD-11, FD-12

### Task 9: FEDformer Implementation
- [ ] Port FEDformer architecture (frequency domain)
- [ ] Add classification head
- [ ] Integrate with Trainer
- [ ] Run FD-13, FD-14

### Task 10: ETSformer Implementation
- [ ] Port ETSformer architecture (exponential smoothing)
- [ ] Add classification head
- [ ] Integrate with Trainer
- [ ] Run FD-15, FD-16

### Task 11: Crossformer Implementation
- [ ] Port Crossformer architecture (cross-variate)
- [ ] Add classification head
- [ ] Integrate with Trainer
- [ ] Run FD-17, FD-18

### Task 12: Extended Analysis
- [ ] Compare all 9 architectures
- [ ] Identify patterns (encoder vs decoder, attention vs MLP)
- [ ] Final recommendations for Phase 6C

---

## Dependencies & Risks

### Dependencies
| Dependency | Version | Purpose |
|------------|---------|---------|
| gluonts | ≥0.14 | Lag-Llama base |
| timesfm | latest | TimesFM |
| torch | 2.9+ | All models |

### Risks
| Risk | Likelihood | Mitigation |
|------|------------|------------|
| GluonTS conflicts with existing deps | Medium | Use separate venv or careful version pinning |
| Foundation models too slow for HPO | Medium | Use fixed hyperparameters from papers |
| MPS incompatibility | Low | Fall back to CPU or cloud GPU |
| Weights not downloadable | Low | Use alternative models |

---

## TimesFM Execution Strategy

**Issue**: TimesFM requires JAX, which has ARM64/Apple Silicon incompatibilities. Native installation fails.

**Approved Approaches**:

| Approach | Role | Description |
|----------|------|-------------|
| **Docker x86** | Primary (Local) | `--platform linux/amd64` with Rosetta 2 emulation |
| **Google Colab** | Secondary (Cloud) | Agent produces `.ipynb` notebooks for user execution |

### Docker Requirements
```bash
# Pull and run with x86 emulation
docker run --platform linux/amd64 -v $(pwd):/workspace -it python:3.11 bash
pip install timesfm torch pandas
```

### Colab Notebook Requirements
Notebooks must be:
- **Self-contained**: No local imports from project
- **Data download cells**: Include MD5 verification
- **Results export**: JSON format compatible with `runner.py` schema

Example notebook structure:
```
1. Install dependencies (pip install timesfm)
2. Download data (parquet from project or regenerate)
3. Load/fine-tune TimesFM
4. Evaluate (AUC, accuracy, precision, recall)
5. Export results as JSON
```

---

## Timeline Estimate

| Phase | Tasks | Estimate |
|-------|-------|----------|
| Setup | 1 | 2-4 hours |
| Lag-Llama | 2 | 4-8 hours |
| TimesFM | 3 | 4-8 hours |
| iTransformer | 4 | 4-6 hours |
| TimeMixer | 5 | 4-6 hours |
| Initial Analysis | 6 | 2-4 hours |
| Informer | 7 | 3-5 hours |
| Autoformer | 8 | 3-5 hours |
| FEDformer | 9 | 3-5 hours |
| ETSformer | 10 | 3-5 hours |
| Crossformer | 11 | 3-5 hours |
| Extended Analysis | 12 | 3-5 hours |
| **Total** | | **40-70 hours** |

---

## Decision Criteria for Rollback

This investigation is a **detour**. Roll findings into main project if:
- ✅ Any model beats PatchTST baseline by ≥5% AUC
- ✅ Clear architectural insight emerges
- ✅ Foundation model path shows promise for Phase 6C/6D

Abandon and return to feature scaling (Phase 6C) if:
- ❌ All models within ±2% of PatchTST
- ❌ Integration complexity too high for marginal gains
- ❌ Dependency conflicts intractable

---

## References

- [Lag-Llama Paper](https://arxiv.org/abs/2310.08278)
- [Lag-Llama GitHub](https://github.com/time-series-foundation-models/lag-llama)
- [TimesFM GitHub](https://github.com/google-research/timesfm)
- [TimesFM HuggingFace](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
- [iTransformer GitHub](https://github.com/thuml/iTransformer)
- [TimeMixer GitHub](https://github.com/kwuking/TimeMixer)
- [Timer GitHub](https://github.com/thuml/Large-Time-Series-Model)

---

*Document Version: 1.0*
*Created: 2026-01-22*
