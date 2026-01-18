# Product Requirements Document: Financial Time-Series Transformer Scaling Laws

> **ARCHIVED VERSION** - This document reflects the project state at completion of Phase 6A HPO (January 2026). Some approaches described here (e.g., scattered chunk-based data splits, Meta-Module consideration) were appropriate for HPO exploration but were revised for final training experiments. See `docs/project_prd.md` for current specification.

---

## 1. Research Objective

Empirically test whether neural scaling laws apply to transformer models trained on financial time-series data, with findings suitable for publication by a non-academic researcher.

## 2. Core Hypothesis

Transformer model performance on financial prediction tasks scales predictably with:
- Parameter count (compute)
- Feature dimensionality
- Dataset size and quality

## 3. Architecture Specification

**Base Model:** PatchTST
- Clean parameter scaling without architectural confounds
- Proven time-series performance
- Modular design enables systematic testing

**Output Heads:**
- Binary classification: Sigmoid activation with probability scoring
- Regression: Continuous prediction with conformal prediction for uncertainty quantification

**Meta-Module (Conditional):**
- CatBoost stacking layer to combine transformer outputs
- **Status:** Tentative, contingent on experimental results
- Purpose: Test if ensemble improves scaling behavior

## 4. Experimental Design

### 4.1 Scaling Dimensions

**Parameter Budgets:** 2M, 20M, 200M parameters
- Systematic 10x scaling intervals
- Batch size re-tuning required per budget change

**Feature Tiers:** 20 → 200 → 2000 technical indicators
- From pandas-ta and TA-Lib libraries
- Batch size re-tuning recommended per tier change

**Dataset Matrix (2D):**

*Asset Scaling (horizontal):*
1. SPY only
2. SPY + DIA
3. SPY + DIA + QQQ
4. SPY + DIA + QQQ + sector ETFs/stocks
5. Full: Above + economic indicators (FRED API)

*Quality Scaling (vertical):*
1. OHLCV + technical indicators
2. + sentiment data (SF Fed)
3. + VIX/volatility indicators
4. + trend/momentum features

### 4.2 Temporal Configuration

**8 Timescales:**
- Daily, 2-day, 3-day, weekly, 2-week, 3-week, monthly
- Daily+multi-resolution (includes weekly/monthly as features)

**6 Tasks per Timescale (48 models per dataset):**

*Binary Classification (5):*
- Direction prediction (up/down)
- Threshold predictions: >1%, >2%, >3%, >5%

*Regression (1):*
- Price prediction with conformal prediction uncertainty

**Total Models:** 48 models × N dataset configurations

### 4.3 Data Splits

- Training: Through 2020
- Validation: 2021-2022
- Test: 2023+

### 4.4 Experimental Phases

1. **Parameter Scaling:** Fixed features/data, vary parameters
2. **Feature Scaling:** Fixed parameters/data, vary features
3. **Data Scaling:** Fixed parameters/features, vary dataset size
4. **Quality Scaling:** Fixed parameters/features, vary data quality
5. **Interaction Effects:** Test coupled scaling dimensions
6. **Full Expansion:** Complete parameter space exploration

## 5. Technical Stack

**Framework:** PyTorch with MPS acceleration
**Hardware:** M4 MacBook Pro, 128GB RAM
**Thermal Management:**
- Normal: <70°C
- Acceptable: 70-85°C
- Warning: 85-95°C
- Critical stop: >95°C

**Optimization:** Optuna for hyperparameter search

**Experiment Tracking:**
- Primary: Weights & Biases
- Secondary: Local MLflow (model registry)

**Data Sources:**
- Market data: Yahoo Finance
- Economic data: FRED API
- Sentiment: SF Fed

**Storage:** Parquet (CSV samples for LLM compatibility)

**Environment:** Python 3.12

## 6. Development Protocols

### 6.1 Quality Controls

**Mandatory:**
- Test-driven development for all code
- RACI matrix enforcement (Human: planning/review; LLM: execution)
- Approval gates for major changes
- Manual context handoff at low utilization

**Code Treatment:**
- All samples are propositions requiring approval
- No implementation without planning session
- Separation of static rules from dynamic state

### 6.2 Terminology Standards

- "Module" for stacking architecture (Base Module, Meta-Module)
- Never "Layer" for architecture levels (reserve for neural network layers)

### 6.3 Documentation Requirements

**Systematic tracking of:**
- Experimental controls per phase
- Statistical analysis protocols
- Hyperparameter configurations
- Scaling law emergence patterns

**Version Control:**
- `.gitignore`: `*.pt`, `*.pth`, `*.parquet`, `outputs/checkpoints/`, `mlruns/`, `wandb/`, `data/raw/`, `data/processed/`

## 7. Success Criteria

**Primary:** Publishable evidence supporting or refuting scaling law applicability to financial transformers

**Secondary:** 
- Clean experimental protocol enabling replication
- Statistical rigor sufficient for non-academic publication
- Career advancement potential through novel findings

## 8. Current Baseline

Existing ML model: >70% accuracy on next-day SPY direction prediction

**Goal:** Determine if transformer scaling improves upon this baseline and follows predictable laws

## 9. Future Research Directions

**Post-Phase 6:**
- Encoder vs decoder architecture comparison
- Causal PatchTST (masked attention)
- Lag-Llama implementation from scratch

## 10. Risk Factors

- Thermal throttling limiting training capacity
- Scaling laws may not apply to financial data (regime changes, non-stationarity)
- Dataset size constraints on scaling dimension
- Publication acceptance given non-academic researcher status