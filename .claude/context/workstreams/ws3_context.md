# Workstream 3 Context: Phase 6C Experiments
# Last Updated: 2026-01-26 09:30

## Identity
- **ID**: ws3
- **Name**: phase6c
- **Focus**: Phase 6C HPO and feature scaling experiments
- **Status**: **HPO 2M Complete** - Analysis done, planning next HPO runs

## Current Task
- **Working on**: HPO Analysis & Planning Next Steps
- **Status**: HPO 2M h1 complete, comprehensive analysis done, planning 20M/200M HPO

---

## Session 2026-01-26 09:30: HPO Analysis Complete

### Key Results

#### HPO 2M h1 Results (50 trials, 20 min)
**Best Config Found**:
```
d_model=96, n_layers=2, n_heads=8, d_ff_ratio=4
learning_rate=1e-5, dropout=0.1, weight_decay=0.001
AUC-ROC: 0.7178
```

**Optuna Convergence**: 10 trials hit identical AUC=0.7178 (TPE found optimum)

**HPO vs Baseline Comparison**:
- HPO 2M: 0.7178 AUC (+1.29% vs baseline)
- Baseline 2M: 0.7049 AUC (d_model=64, layers=4, dropout=0.5, lr=1e-4)
- 20M reference: 0.7342 AUC (d_model=512, layers=6)
- **Conclusion**: HPO found better 2M config, but 20M still outperforms → need 20M/200M HPO

#### CRITICAL FINDING: Probability Collapse
Model predictions collapsed to narrow range:
- **Probability range**: [0.505, 0.662] (very narrow!)
- At threshold 0.5: 100% recall, 18% precision (predicts everything positive)
- **Max achievable precision**: ~41% at threshold 0.60 (with 16% recall)
- Best F1: 0.44 at threshold 0.54 (precision 35%, recall 61%)

#### Hyperparameter Insights

**Weight Decay** (tested 0.0, 1e-5, 1e-4, 1e-3):
- wd=0.001: Mean AUC 0.7037 (28 trials) - BEST
- wd=0.0001: Mean AUC 0.7006 (7 trials)
- wd=1e-5: Mean AUC 0.6993 (10 trials)
- wd=0.0: Mean AUC 0.6889 (5 trials) - WORST
- **Conclusion**: Weight decay helps, try higher values (0.005, 0.01)

**Dropout** (tested 0.1, 0.3, 0.5, 0.7):
- dropout=0.1: Mean AUC 0.7055 - BEST (contradicts earlier ablation!)
- dropout=0.3: Mean AUC 0.6970
- dropout=0.5: Mean AUC 0.6905
- dropout=0.7: Mean AUC 0.6894
- **Conclusion**: Low dropout better for tier_a100, test 0.05/0.15/0.2

**Learning Rate** (tested 1e-5, 5e-5, 1e-4, 5e-4):
- lr=1e-5: Mean AUC 0.7042 - BEST but narrow range
- lr=5e-5: Mean AUC 0.7086 (fewer trials)
- lr=1e-4: Mean AUC 0.6916
- lr=5e-4: Mean AUC 0.6822 - WORST
- **Conclusion**: Slower learning helps, try even slower (5e-6, 2e-6)

**n_heads** (tested 2, 4, 8):
- 8-head best: 0.7178 AUC
- 4-head: 0.7167 AUC (very close!)
- 2-head: 0.7152 AUC
- **Gap**: 2-head NOT tested with best config (d_model=96, layers=2, lr=1e-5, dropout=0.1)

### User Priorities for Next Steps

1. **Run 20M and 200M HPO** with same expanded search space
2. **Test more dropout values**: 0.05, 0.15, 0.2, 0.3 on top configs
3. **Test more weight decay**: 0.005, 0.01 for stronger regularization
4. **Address probability collapse**: User notes this is persistent issue, slowing learning helps
5. **After HPO exploration**: Apply calibration (Platt scaling, isotonic regression, temperature scaling)
6. **Final**: Run full experiments on best configs from all HPO

---

## Files Modified This Session
- None (analysis only, commit already done at session start)

## Git Status
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `268e328` feat: Expand HPO search space to include training hyperparameters
- **Clean**: Working tree clean

---

## Outputs Generated
```
outputs/phase6c_a100/hpo_2m_h1/
├── all_trials.json (50 trials with full params)
├── best_params.json
└── trial_000/ through trial_049/ (checkpoints)
```

---

## Next Session Should

1. **Plan extended HPO search**:
   - Add dropout values: 0.05, 0.15, 0.2
   - Add weight decay values: 0.005, 0.01
   - Add slower learning rates: 5e-6, 2e-6
   - Test 2-head and 4-head with best config explicitly

2. **Run 20M HPO overnight**:
   ```bash
   caffeinate -i ./venv/bin/python experiments/phase6c_a100/hpo_20m_h1.py 2>&1 | tee outputs/phase6c_a100/hpo_20m_h1_$(date +%Y%m%d).log
   ```

3. **Run 200M HPO** (after 20M completes)

4. **After all HPO complete**:
   - Analyze results across parameter budgets
   - Implement probability calibration if still collapsed
   - Run full experiments on best configurations

---

## Key Decisions This Session

1. **HPO is working**: Found 2M config (0.7178) that beats baseline (0.7049)
2. **Probability collapse is the core issue**: Not just AUC, need calibrated predictions
3. **Parameter scaling matters**: 20M reference (0.7342) > all 2M configs
4. **Dropout finding contradicts ablation**: 0.1 >> 0.5 for tier_a100 (feature quality difference?)
5. **Weight decay beneficial**: 0.001 optimal, may benefit from more

---

## Session History

### 2026-01-26 09:30 (HPO Analysis)
- Analyzed HPO 2M h1 results (50 trials complete)
- Found best config: d_model=96, layers=2, heads=8, lr=1e-5, dropout=0.1, wd=0.001
- Identified probability collapse issue [0.505, 0.662]
- Max precision ~41% achievable
- Compared HPO vs baseline: +1.29% improvement
- Planned next steps: 20M/200M HPO, more regularization, calibration

### 2026-01-26 00:30 (Deep Code Audit - Phase 1)
- Completed Phase 1 of deep code audit
- Added recall/precision/pred_range metrics to Trainer
- Fixed exception handling in all 6 HPO scripts
- Fixed experiment paths and naming conventions

### 2026-01-25 (Phase 2 Implementation)
- Expanded HPO search space with training hyperparameters
- Added learning_rate, dropout, weight_decay to all 6 HPO scripts
- Committed changes
