# Global Project Context - 2026-01-26

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | active | 2026-01-26 13:30 | tier_a200 ALL CHUNKS COMPLETE (uncommitted) - 106 new indicators, 206 total |
| ws2 | foundation | **COMPLETE** | 2026-01-25 15:00 | Foundation models AND alt architectures FAILED vs PatchTST |
| ws3 | phase6c | **HPO 2M Done** | 2026-01-26 09:30 | HPO 2M h1 complete (AUC 0.7178), planning 20M/200M HPO |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `eee3c27` feat: Add tier_a200 Chunks 3-4 with 40 new features (ranks 141-180)
- **Uncommitted**:
  - `src/features/tier_a200.py` - Chunks 1-5 implementation (106 new features, 206 total)
  - `tests/features/test_tier_a200.py` - Chunks 1-5 tests (252 tests)

### Test Status
- Last `make test`: 2026-01-26 13:30 - **944 passed**, 2 skipped
- All tests pass

### Data Versions
- Raw: SPY/DIA/QQQ OHLCV (v1)
- Processed: a20, a50, a100 (v1) - both features-only and _combined versions
- tier_a200: Module COMPLETE (all 5 chunks, 106 additions, 206 total), no processed data yet

## Cross-Workstream Coordination

### Blocking Dependencies
- [ws3 HPO 2M Done]: Best config found (d_model=96, layers=2, heads=8, lr=1e-5, dropout=0.1, wd=0.001)
- [ws3 Next]: Need 20M and 200M HPO runs
- [ws1 Complete]: tier_a200 ready to commit, can generate processed data for Phase 6C

### File Ownership
| Files | Owner |
|-------|-------|
| `src/features/tier_a200.py` | ws1 (MODIFIED - Chunks 1-5 COMPLETE) |
| `tests/features/test_tier_a200.py` | ws1 (MODIFIED - Chunks 1-5 COMPLETE) |
| `experiments/phase6c_a100/*` | ws3 |
| `outputs/phase6c_a100/hpo_*` | ws3 |

---

## tier_a200 Completion Summary (ws1)

### All Chunks Complete
| Chunk | Ranks | Features | Categories |
|-------|-------|----------|------------|
| 1 | 101-120 | 20 | TEMA, WMA, KAMA, HMA, VWMA, derived |
| 2 | 121-140 | 20 | Duration Counters, MA Cross Recency, Proximity |
| 3 | 141-160 | 20 | BB Extension, RSI Duration, Mean Reversion, Patterns |
| 4 | 161-180 | 20 | MACD Extensions, Volume Dynamics, Calendar, Candle |
| 5 | 181-206 | 26 | Ichimoku, Donchian, Divergence, Entropy/Regime |

**Total**: 106 new features + 100 from tier_a100 = **206 features**

---

## HPO 2M Analysis Summary (ws3)

### Best Configuration Found
```
d_model=96, n_layers=2, n_heads=8, d_ff_ratio=4
learning_rate=1e-5, dropout=0.1, weight_decay=0.001
AUC-ROC: 0.7178 (vs baseline 0.7049 = +1.29%)
```

### Key Findings

1. **HPO beats baseline**: 0.7178 > 0.7049 (+1.29 AUC points)
2. **20M still better**: Reference 20M model achieves 0.7342 AUC
3. **Probability collapse**: Predictions in [0.505, 0.662] range (narrow!)
4. **Max precision ~41%**: At threshold 0.60 with only 16% recall
5. **Dropout 0.1 >> 0.5**: Contradicts previous ablation (tier_a100 features different?)
6. **Weight decay helps**: 0.001 > 0.0001 > 1e-5 > 0.0
7. **Slower LR better**: 1e-5 optimal, may benefit from even slower

### Gaps in Search Space
- 2-head NOT tested with best architecture config
- No dropout values 0.05, 0.15, 0.2 tested
- No weight decay values 0.005, 0.01 tested
- No LR values 5e-6, 2e-6 tested

---

## User Priorities (from recent sessions)

1. **Run 20M and 200M HPO** with same expanded search space
2. **Test more dropout values**: 0.05, 0.15, 0.2, 0.3 on top configs
3. **Test more weight decay**: 0.005, 0.01 for stronger regularization
4. **Address probability collapse**: Slowing learning and adding regularization helps
5. **After HPO exploration**: Apply calibration techniques (Platt scaling, isotonic regression, temperature scaling)
6. **Final step**: Run full experiments on best configs from all HPO

---

## User Preferences (Authoritative)

### Development Approach
- TDD: tests first, always
- Planning sessions before implementation
- Uses tmux for long-running experiments

### Context Durability
- Multiple places: Memory MCP + context files + docs/
- Code comments are secondary

### Documentation Philosophy
- Flat docs/ (no subdirs except research_paper/, archive/)
- Precision - never reduce fidelity
- Consolidate rather than delete

### Hyperparameters (HPO-Validated, Replacing Ablation)
- **Dropout**: 0.1 (HPO found better than 0.5 for tier_a100)
- **Learning Rate**: 1e-5 (slower is better)
- **Weight Decay**: 0.001 (regularization helps)
- **Context**: 80d (unchanged)
- **Normalization**: RevIN only (unchanged)
- **Splitter**: SimpleSplitter (unchanged)

---

## Next Session Notes

### ws1 (feature_generation) - Ready to Commit
1. Commit tier_a200 files:
   ```bash
   git add -A && git commit -m "feat: Add tier_a200 Chunk 5 with 26 new features (ranks 181-206)

   Implements Ichimoku Cloud (6), Donchian Channel (5), Momentum Divergence (4),
   and Entropy/Regime (11) indicators. Total tier_a200: 106 features, 206 overall."
   ```
2. Generate processed tier_a200 parquet files if needed for experiments

### ws3 (phase6c) - HPO Continuation
1. Consider expanding search space before 20M HPO:
   - dropout: add 0.05, 0.15, 0.2
   - weight_decay: add 0.005, 0.01
   - learning_rate: add 5e-6, 2e-6

2. Run 20M HPO overnight:
   ```bash
   caffeinate -i ./venv/bin/python experiments/phase6c_a100/hpo_20m_h1.py 2>&1 | tee outputs/phase6c_a100/hpo_20m_h1.log
   ```

3. Then run 200M HPO

### Key Insight
**Probability collapse is the core issue** - models achieve decent AUC (ranking) but poor calibration (probability meaningfulness). User notes slowing down learning and adding regularization helps. This aligns with HPO finding lr=1e-5 and wd=0.001 optimal.
