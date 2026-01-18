# Financial TS Transformer Scaling: Phase Plans

**Status:** Active Development Plan
**Version:** 3.0
**Last Updated:** 2026-01-18

> For completed phases (0-5.5) and detailed methodology history, see `docs/archive/project_phase_plans_full_history.md`

---

## Methodology Reference

Development methodology is defined in:
- **CLAUDE.md** - Project rules, approval gates, TDD requirements
- **.claude/rules/** - Detailed rule files (testing.md, development-discipline.md, etc.)
- **.claude/context/phase_tracker.md** - Current phase status

---

# Phase 6: Scaling Experiments

**Research Question:** Do neural scaling laws apply to transformer models trained on financial time-series data?

## Experimental Design Overview

### Constants (All Experiments)
- **Architecture:** PatchTST (from-scratch implementation)
- **Task:** Threshold 1% (>1% gain within horizon)
- **Data:** SPY (Phase 6A-6C), expanded in 6D

### Scaling Dimensions
1. **Parameters:** 2M → 20M → 200M → 2B
2. **Horizons:** 1-day, 3-day, 5-day (+ 2-day in 6B)
3. **Features:** 25 → 100 → 500 → 2000 indicators
4. **Data:** SPY → multi-asset → cross-domain (Phase 6D)

### Data Splits
- **HPO:** Scattered chunk-based splits for regime diversity
- **Final Training:** Contiguous splits (train through Sept 2024, val Oct-Dec 2024, test 2025+)

---

# Phase 6A: Parameter Scaling 🔄 IN PROGRESS

**Objective:** Establish whether error follows power law with model parameters

**Status:** HPO complete, final training in progress

## Experimental Matrix

**Hold Constant:**
- Features: 25 (5 OHLCV + 20 indicators)
- Data: SPY only

**Vary:**
- Parameter budget: 2M, 20M, 200M, 2B
- Horizon: h1 (1-day), h3 (3-day), h5 (5-day)

### HPO Results Summary (Completed)

| Budget | Horizon | Best Architecture | val_loss |
|--------|---------|-------------------|----------|
| 2M | h1 | d=64, L=48, h=2 | 0.3136 |
| 2M | h3 | d=64, L=32, h=2 | 0.2538 |
| 2M | h5 | d=64, L=64, h=16 | 0.3368 |
| 20M | h1 | d=128, L=180, h=16 | 0.3461 |
| 20M | h3 | d=256, L=32, h=2 | 0.3035 |
| 20M | h5 | d=384, L=12, h=4 | 0.3457 |
| 200M | h1 | d=384, L=96, h=4 | 0.3488 |
| 200M | h3 | d=768, L=24, h=16 | 0.3281 |
| 200M | h5 | d=256, L=256, h=16 | 0.3521 |
| 2B | h1 | d=1024, L=128, h=2 | 0.3599 |
| 2B | h3 | d=768, L=256, h=32 | 0.3716 |
| 2B | h5 | d=1024, L=180, h=4 | 0.3575 |

**Key Finding:** Larger models did NOT improve over smaller models — data-limited regime.

### Final Training (16 experiments)

4 budgets × 4 horizons (h1, h2, h3, h5) = 16 final training runs

**Tasks Remaining:**
- [ ] Interpolate H2 architectures from H1/H3
- [ ] Implement best checkpoint saving in Trainer
- [ ] Create final training script template
- [ ] Generate 16 final training scripts
- [ ] Run final training with contiguous splits
- [ ] Analyze results and generate scaling curves

## Analysis Outputs

1. **Parameter Scaling Curve:** Plot log(error) vs log(params) for each horizon
2. **Power Law Fit:** error ∝ N^(-α), report α and R²
3. **Horizon Effect:** Does scaling effectiveness vary with prediction horizon?

## Success Criteria

- [ ] All 16 final training experiments complete
- [ ] Parameter scaling curves plotted
- [ ] Power law exponent (α) computed with confidence intervals
- [ ] Draft finding on scaling law applicability

---

# Phase 6B: Horizon Scaling ⏸️ PENDING

**Objective:** Expand horizon coverage using Phase 6A architectures

**Depends on:** Phase 6A completion

## Experimental Matrix

**Hold Constant:**
- Features: 25 (tier a20)
- Data: SPY only

**Vary:**
- Parameter budget: 2M, 20M, 200M, 2B
- Horizon: 2-day (new), weekly (new)

**Note:** Reuse best architectures from Phase 6A HPO

### Experiment Count

2 new horizons × 4 params = 8 runs (reusing HPO results)

## Success Criteria

- [ ] All horizon experiments complete
- [ ] Scaling curves computed for each horizon
- [ ] Horizon effect quantified

---

# Phase 6C: Feature Scaling ⏸️ PENDING

**Objective:** Test whether feature richness unlocks scaling laws

**Hypothesis:** With 2000 features (~2M pairwise relationships), larger models may show scaling benefits that weren't visible with only 25 features.

## Experimental Matrix

**Hold Constant:**
- Data: SPY only
- Horizon: Best from 6A/6B

**Vary:**
- Features: 100, 500, 2000 indicators
- Parameters: 2M, 20M, 200M, 2B

### Experiment Count

**HPO runs:** 3 tiers × 4 params = 12 (at best horizon)
**Final runs:** 3 tiers × 4 params = 12

**Total: 24 runs**

### Feature Tiers

| Tier | Features | Indicator Categories |
|------|----------|---------------------|
| a25 | 25 | Basic (current: OHLCV + SMA, EMA, RSI, MACD, BBands, ATR, OBV) |
| a100 | 100 | + Extended momentum, volatility, volume |
| a500 | 500 | + Pattern recognition, multi-timeframe |
| a2000 | 2000 | + All indicators, interaction features |

## Analysis Outputs

1. **Feature × Parameter Surface:** Error vs features vs params
2. **Feature Scaling Curves:** Does more features help larger models?
3. **Interaction Effects:** Feature × Parameter interactions

## Success Criteria

- [ ] All 24 experiments complete
- [ ] Feature scaling effect quantified
- [ ] Determine if feature richness unlocks scaling laws

---

# Phase 6D: Data Scaling ⏸️ GATED

**Gate Condition:** Proceed only after Phase 6A-6C results analyzed

**Objective:** Test whether data diversity improves scaling

## Scope (Preliminary)

**Additional Data Sources:**
- DIA, QQQ (ETFs)
- AAPL, MSFT, GOOGL, AMZN, TSLA (stocks)
- VIX volatility indicators
- FRED economic indicators

**Approach:**
- Use best configuration from Phase 6C
- Test data scaling with fixed features/params/horizon
- Quantify data diversity effect on scaling

## Success Criteria

- [ ] Data scaling effect quantified
- [ ] Cross-asset generalization tested
- [ ] Publication-ready comprehensive scaling analysis

---

# Runtime Estimates

| Phase | Experiments | Est. Hours |
|-------|-------------|------------|
| 6A Final Training | 16 | 50-100 |
| 6B Horizon | 8 | 20-40 |
| 6C Feature | 24 | 100-200 |
| 6D Data | TBD | TBD |
| **Total (6A-6C)** | **48** | **170-340** |

---

# Appendix: Code Proposition Policy

**ALL code changes require:**

1. Planning session (for non-trivial changes)
2. Test-first development
3. Explicit approval before execution
4. `make test` passing before commit

See `.claude/rules/development-discipline.md` for full approval gate requirements.

---

*Document version: v3.0*
*Streamlined from v2.0 (2374 lines → ~200 lines)*
*Full history preserved in: docs/archive/project_phase_plans_full_history.md*
