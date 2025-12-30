# Phase 6A Execution Plan

**Status: Stage 4 in progress** | Updated: 2025-12-29 | v2.0

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

### Stage 1: Validate Pipeline (First HPO Test) ✅ COMPLETE

**Completed: 2025-12-11**

Ran 3-trial HPO to validate end-to-end pipeline:

```bash
python experiments/phase6a/hpo_2M_threshold_1pct.py  # 3 trials, ~2.5 min
```

**Validated:**
- [x] Data loading from `SPY_dataset_a25.parquet`
- [x] ChunkSplitter creates proper train/val/test splits
- [x] HPO optimizes val_loss (not train_loss)
- [x] Results logged to `docs/experiment_results.csv`
- [x] Best params saved to `outputs/hpo/`

### Stage 2: Horizon Variance Test (2M Only) ✅ COMPLETE

**Completed: 2025-12-13**

Tested if HPO params vary by horizon using 2M budget:

| Script | Budget | Horizon | Best Val Loss | Best Architecture |
|--------|--------|---------|---------------|-------------------|
| `hpo_2M_h1_threshold_1pct.py` | 2M | 1-day | 0.337 | d=64, L=48, h=8 |
| `hpo_2M_h3_threshold_1pct.py` | 2M | 3-day | 0.262 | d=64, L=32, h=32 |
| `hpo_2M_h5_threshold_1pct.py` | 2M | 5-day | — | ⏳ Not run |

**Decision: Option A selected — horizon variance IS significant**

Key findings:
- h3 achieves significantly lower loss than h1 (0.262 vs 0.337)
- Optimal architecture varies by horizon (different depth/heads configurations)
- Conclusion: Separate HPO per horizon required for valid scaling analysis

### Stage 3: Full HPO Matrix ✅ COMPLETE (2M, 20M, 200M)

**Completed: 2025-12-21** (2M, 20M, 200M complete; 2B started)

**Option A Selected: 12 HPO runs per budget** (4 budgets × 3 horizons × 1 task)

Rationale:
- Horizon variance confirmed in Stage 2 → separate HPO per horizon
- Focus on threshold_1pct task only (simplest binary classification)
- Skip 2%, 3%, 5% tasks → can borrow/interpolate from 1% results later

| Budget | h1 Status | h3 Status | h5 Status |
|--------|-----------|-----------|-----------|
| 2M | ✅ 50 trials | ✅ 50 trials | ⏳ Pending |
| 20M | ✅ 50 trials | ✅ 50 trials | ✅ 50 trials |
| 200M | ✅ 46 trials | ✅ 50 trials | ✅ 50 trials |
| 2B | 🔄 3 trials (smoke) | ⏳ Pending | ⏳ Pending |

### Stage 4: Final Training 🔄 IN PROGRESS

**Status: 2B HPO in progress, final training pending**

After all HPO completes:
1. Run final training with best params per budget/horizon
2. Compute test set metrics
3. Plot scaling curves: log(error) vs log(params)
4. Fit power law: error = C × N^(-α)

**Current blockers:**
- 2B HPO not complete (memory optimization stage completed 2025-12-29)
- Missing runs: 2M_h5, 2B_h1 (full 50), 2B_h3, 2B_h5

## HPO Scripts (12 Total)

| # | Script | Budget | Horizon | Trials | Best Val Loss | Status |
|---|--------|--------|---------|--------|---------------|--------|
| 1 | `hpo_2M_h1_threshold_1pct.py` | 2M | 1-day | 50 | 0.337 | ✅ Complete |
| 2 | `hpo_2M_h3_threshold_1pct.py` | 2M | 3-day | 50 | 0.262 | ✅ Complete |
| 3 | `hpo_2M_h5_threshold_1pct.py` | 2M | 5-day | 0 | — | ⏳ Pending |
| 4 | `hpo_20M_h1_threshold_1pct.py` | 20M | 1-day | 50 | 0.363 | ✅ Complete |
| 5 | `hpo_20M_h3_threshold_1pct.py` | 20M | 3-day | 50 | 0.274 | ✅ Complete |
| 6 | `hpo_20M_h5_threshold_1pct.py` | 20M | 5-day | 50 | 0.347 | ✅ Complete |
| 7 | `hpo_200M_h1_threshold_1pct.py` | 200M | 1-day | 46 | 0.363 | ✅ Complete |
| 8 | `hpo_200M_h3_threshold_1pct.py` | 200M | 3-day | 50 | 0.308 | ✅ Complete |
| 9 | `hpo_200M_h5_threshold_1pct.py` | 200M | 5-day | 50 | 0.351 | ✅ Complete |
| 10 | `hpo_2B_h1_threshold_1pct.py` | 2B | 1-day | 3 | 0.363 | 🔄 Smoke test |
| 11 | `hpo_2B_h3_threshold_1pct.py` | 2B | 3-day | 0 | — | ⏳ Pending |
| 12 | `hpo_2B_h5_threshold_1pct.py` | 2B | 5-day | 0 | — | ⏳ Pending |

**Scripts location:** `experiments/phase6a/`
**Runner:** `scripts/run_phase6a_hpo.sh`

## Estimated Runtime

| Budget | Est. HPO (50 trials) | Actual HPO | Notes |
|--------|---------------------|------------|-------|
| 2M | ~12-15 hours | ~4 hours | Faster than expected |
| 20M | ~25-35 hours | ~4-5 hours | Faster than expected |
| 200M | ~50-100 hours | ~48-82 hours | h5 took longest (narrow-deep arch) |
| 2B | ~200-300 hours | TBD | Memory optimizations applied |

**Actual Stage Durations:**
- Stage 1 (Validation): ~2.5 min (3 trials)
- Stage 2 (Horizon Test): ~6 hours (2M × 2 horizons)
- Stage 3 (Full Matrix): ~200 hours cumulative (2M/20M/200M complete)

**2B Memory Challenges:**
- Original 2B trials consumed 115GB+ memory
- Deep+wide architectures (d=1024, L=256) caused swap thrashing
- Optimization stage (Dec 26-29) added: dynamic batch sizing, gradient accumulation, early stopping
- Post-optimization: 2B smoke test (3 trials) completed successfully in ~4.3 hours

## Monitoring Commands

```bash
# Watch HPO progress
tail -f outputs/hpo/phase6a_*/hpo.log 2>/dev/null || echo "No logs yet"

# Check thermal status
sudo powermetrics --samplers smc -i 1000 -n 1 | grep -i temp

# View results
cat docs/experiment_results.csv | column -t -s,

# Check for completed experiments
ls -la outputs/hpo/phase6a_*/*_best.json

# Compare hyperparameters across horizons (after Stage 2)
for f in outputs/hpo/phase6a_*/*_best.json; do
  echo "=== $f ===" && cat "$f" | jq '{best_params, architecture}'
done
```

## Output Files

| Type | Path |
|------|------|
| HPO Best Params | `outputs/hpo/phase6a_{budget}_h{horizon}_{task}/{experiment}_{budget}_best.json` |
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

## Key Decisions

### 2025-12-11 (Initial Design)

1. **Feature Count:** Phase 6A uses 25 features (5 OHLCV + 20 indicators). VIX features (8) reserved for Phase 6D.

2. **Data File:** Use `SPY_dataset_a25.parquet` (not `SPY_dataset_c.parquet` which has VIX).

3. **Horizon Testing:** Test 1-day, 3-day, 5-day horizons to determine if HPO params vary significantly.

4. **Staged Execution:** Validate → horizon variance test → full matrix decision.

### 2025-12-13 (HPO Fixes)

5. **n_heads Logging:** Fixed missing n_heads in architecture logging.

6. **Grid Gaps:** Added L=160, L=180 to n_layers grid for better coverage.

7. **Forced Extremes:** First 6 trials force extreme architectures to ensure exploration of design space boundaries.

### 2025-12-21 (200M Complete)

8. **Horizon Variance Confirmed:** h3 consistently achieves lowest loss; h5 prefers narrow-deep architectures.

9. **Optimal Heads:** h=16 consistently outperforms h=8 across budgets.

10. **Architecture Patterns:** Wide-medium (d=768-1024, L=12-24) optimal for h1/h3; narrow-deep (d=256, L=256) optimal for h5.

### 2025-12-26 (2B Memory Issues)

11. **Memory Exhaustion:** 2B trials with d=1024, L=256 consumed 115GB+, caused multi-day swap thrashing.

12. **Optimization Stage:** Launched detour to add dynamic batch sizing, gradient accumulation, early stopping.

### 2025-12-29 (Optimization Complete)

13. **Dynamic Batch Sizing:** `get_memory_safe_batch_config()` selects batch size based on architecture memory footprint.

14. **Gradient Accumulation:** Maintains effective batch size of 256 even with small actual batches.

15. **Early Stopping:** Patience=10 epochs prevents wasted compute on diverging trials.

16. **Dropout Search:** Added dropout (0.1-0.3) to HPO search space (was hardcoded at 0.1).

---

## HPO Results Summary

Best validation loss per budget/horizon (lower is better):

| Budget | h1 (1-day) | h3 (3-day) | h5 (5-day) | Best Architecture |
|--------|------------|------------|------------|-------------------|
| 2M | 0.337 | **0.262** | — | d=64, L=32-48, h=8-32 |
| 20M | 0.363 | **0.274** | 0.347 | d=192-768, L=4-64, h=2-4 |
| 200M | 0.363 | **0.308** | 0.351 | d=256-1024, L=12-256, h=16 |
| 2B | 0.363 | — | — | d=1024, L=180, h=2 (smoke test) |

**Key Observations:**
- h3 (3-day horizon) consistently achieves lowest loss across all budgets
- 200M_h3 achieved best overall: val_loss=0.308 (d=768, L=24, h=16)
- h5 prefers narrow-deep: 200M_h5 best with d=256, L=256, h=16
- Larger models don't always beat smaller ones (20M_h1 ≈ 200M_h1)

**Full results:** `docs/experiment_results.csv`

---

## Remaining Work

### HPO (4 runs remaining)
- [ ] 2M_h5: 50 trials
- [ ] 2B_h1: Complete 50 trials (3 smoke test done)
- [ ] 2B_h3: 50 trials
- [ ] 2B_h5: 50 trials

### Final Training
- [ ] Run best architecture per budget/horizon with full epochs
- [ ] Compute test set accuracy/loss

### Analysis
- [ ] Plot scaling curve: log(val_loss) vs log(param_count)
- [ ] Fit power law: loss = C × N^(-α)
- [ ] Compute R² for scaling law fit
- [ ] Document findings in research paper

---

## Lessons Learned

See `docs/phase6a_implementation_history.md` for detailed history of:
- Architectural HPO implementation (8 tasks)
- HPO fixes (6 tasks)
- Time optimization stage (6 tasks)

**Key Takeaways:**
1. **Architectural HPO is essential** — training-only HPO misses the most important scaling variable
2. **Horizon affects optimal architecture** — can't assume same arch works for all prediction lengths
3. **Memory management critical for 2B** — dynamic batch sizing + gradient accumulation required
4. **Early stopping saves significant time** — prevents wasted epochs on diverging trials

---

*Updated: 2025-12-29*
*Version: 2.0*
*Supersedes: v1.0 (2025-12-11)*
