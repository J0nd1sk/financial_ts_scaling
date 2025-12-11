# Phase 6A Execution Plan

**Objective:** Parameter scaling baseline - test if error ∝ N^(-α)

**Secondary Research Question:** Do optimal hyperparameters vary significantly with prediction horizon? This will determine whether we need separate HPO per horizon in future phases.

## Experiment Matrix

**Hold Constant:**
- Features: 25 (5 OHLCV + 20 indicators) - **NO VIX** (VIX is Phase 6D only)
- Data: SPY only (`data/processed/v1/SPY_dataset_a25.parquet`)
- Timescale: daily

**Vary:**
- Parameter budget: 2M, 20M, 200M, 2B
- Prediction horizon: 1-day, 3-day, 5-day (to test if HPO params vary)

## Data Configuration

| Dataset | Path | Features | Notes |
|---------|------|----------|-------|
| Phase 6A-6C | `SPY_dataset_a25.parquet` | 25 | 5 OHLCV + 20 indicators |
| Phase 6D | `SPY_dataset_c.parquet` | 33 | Adds 8 VIX features |

## Execution Strategy

### Stage 1: Validate Pipeline (First HPO Test)

Run a short 2-3 trial HPO to validate everything works end-to-end:

```bash
# Validation test (2-3 trials, ~30-45 min)
python experiments/phase6a/hpo_2M_threshold_1pct.py  # Modified for 2-3 trials
```

**Validates:**
- Data loading from `SPY_dataset_a25.parquet`
- ChunkSplitter creates proper train/val/test splits
- HPO optimizes val_loss (not train_loss)
- Results logged to `docs/experiment_results.csv`
- Best params saved to `outputs/hpo/`

### Stage 2: Horizon Variance Test (2M Only)

Before running full matrix, test if HPO params vary by horizon using cheapest budget (2M):

| Script | Budget | Task | Horizon | Purpose |
|--------|--------|------|---------|---------|
| `hpo_2M_threshold_1pct_1d.py` | 2M | 1% | 1-day | Baseline |
| `hpo_2M_threshold_1pct_3d.py` | 2M | 1% | 3-day | Compare params |
| `hpo_2M_threshold_1pct_5d.py` | 2M | 1% | 5-day | Compare params |

**Decision Point:** After Stage 2 completes, compare best_params across horizons:
- If params vary significantly (>20% difference in key hyperparams): Run separate HPO per horizon
- If params similar: Can borrow params across horizons (significant time savings)

### Stage 3: Full HPO Matrix

Depending on Stage 2 results:

**Option A: Params Vary by Horizon (36 HPO runs)**
- 4 budgets × 3 tasks × 3 horizons = 36 runs
- Skip threshold_2pct (borrow interpolated params from 1% and 3%)

**Option B: Params Similar Across Horizons (12 HPO runs)**
- 4 budgets × 3 tasks × 1 horizon (1-day) = 12 runs
- Borrow params for 3-day and 5-day horizons

### Stage 4: Final Training

After HPO completes, run final training with best params.

## HPO Scripts - Stage 1 & 2 (Initial)

| # | Script | Budget | Task | Horizon | Status |
|---|--------|--------|------|---------|--------|
| 1 | `hpo_2M_threshold_1pct.py` | 2M | 1% | 1-day | ✅ EXISTS |
| 2 | `hpo_2M_threshold_1pct_3d.py` | 2M | 1% | 3-day | ⏳ GENERATE after Stage 1 |
| 3 | `hpo_2M_threshold_1pct_5d.py` | 2M | 1% | 5-day | ⏳ GENERATE after Stage 1 |

## Estimated Runtime

| Budget | HPO per run (50 trials) | Training per run |
|--------|-------------------------|------------------|
| 2M | ~12-15 hours | ~30 min |
| 20M | ~25-35 hours | ~1.5 hr |
| 200M | ~50-100 hours | ~4 hr |
| 2B | ~200-300 hours | ~12 hr |

**Stage 1 (Validation):** ~30-45 min (2-3 trials)
**Stage 2 (Horizon Test):** ~36-45 hours (3 × 2M HPO)
**Stage 3 (Full Matrix):** ~300-450 hours (Option A) or ~100-150 hours (Option B)

## Monitoring Commands

```bash
# Watch HPO progress
tail -f outputs/hpo/phase6a_*/hpo.log 2>/dev/null || echo "No logs yet"

# Check thermal status
sudo powermetrics --samplers smc -i 1000 -n 1 | grep -i temp

# View results
cat docs/experiment_results.csv | column -t -s,

# Check for completed experiments
ls -la outputs/hpo/phase6a_*/best_params.json

# Compare hyperparameters across horizons (after Stage 2)
for f in outputs/hpo/phase6a_2M_threshold_1pct*/best_params.json; do
  echo "=== $f ===" && cat "$f" | jq '.best_params'
done
```

## Output Files

| Type | Path |
|------|------|
| HPO Best Params | `outputs/hpo/phase6a_{budget}_{task}_{horizon}/best_params.json` |
| HPO Study DB | `outputs/hpo/phase6a_{budget}_{task}_{horizon}/study.db` |
| Training Checkpoints | `outputs/training/phase6a_{budget}_{task}_{horizon}/` |
| Experiment Log | `docs/experiment_results.csv` |
| Results Report | `docs/experiment_results.md` |

## Success Criteria

### Stage 1 (Validation)
- [ ] Short HPO test (2-3 trials) completes without errors
- [ ] Results logged to `docs/experiment_results.csv`
- [ ] Best params saved correctly

### Stage 2 (Horizon Test)
- [ ] All 3 horizon HPO runs complete for 2M budget
- [ ] Horizon variance analysis documented
- [ ] Decision made: separate HPO per horizon or borrow params

### Stage 3+ (Full Execution)
- [ ] All HPO runs complete without thermal abort
- [ ] All training runs complete
- [ ] Scaling curve plotted: log(error) vs log(params)
- [ ] Power law fit: α and R² computed

## Key Decisions (2025-12-11)

1. **Feature Count:** Phase 6A uses 25 features (5 OHLCV + 20 indicators). VIX features (8) are reserved for Phase 6D data scaling only.

2. **Data File:** Use `SPY_dataset_a25.parquet` (not `SPY_dataset_c.parquet` which has VIX).

3. **Horizon Testing:** Test 1-day, 3-day, 5-day horizons to determine if HPO params vary significantly by horizon. This informs whether future phases need separate HPO per horizon.

4. **Staged Execution:** Validate pipeline first (2-3 trials), then test horizon variance (2M only), then decide on full matrix strategy.

---

*Updated: 2025-12-11*
