# Global Project Context - 2026-02-07

## Active Workstreams

| ID | Name | Status | Last Update | Summary |
|----|------|--------|-------------|---------|
| ws1 | feature_generation | **COMPLETE** | 2026-01-31 16:45 | tier_a500 DONE - 500 features committed + data regenerated |
| ws2 | feature_embedding | **ACTIVE** | 2026-02-06 | Budget-aware HPO implemented, ready to run |
| ws3 | feature_embedding_experiments | **ACTIVE** | 2026-02-07 18:30 | Phase 3 IMPLEMENTED - 183 experiments (92 FE + 70 LF + 21 AE), LF-P7 running |

## Shared State

### Branch & Git
- **Branch**: `experiment/foundation-decoder-investigation`
- **Last commit**: `e51d4e5` feat: Add HPO v3 (precision-first), loss functions, and context ablation results
- **Uncommitted** (pending commit):
  - `experiments/feature_embedding/run_experiments.py` - 183 experiments (92 FE + 70 LF + 21 AE)
  - `tests/test_feature_embedding_experiments.py` - 25 tests
  - `docs/feature_embedding_experiment_tracker.md` - NEW tracking document
  - `Makefile` - test-ws3 target updated
  - Modified: `src/features/tier_a500.py`, `src/models/patchtst.py`, `src/models/arch_grid.py`
  - New: `experiments/feature_embedding/`, `outputs/feature_embedding/`
  - Context files updated

### Test Status
- **ws3 tests (make test-ws3)**: 251 passed (includes feature_embedding tests)
- **Full suite**: Some feature tests may have pre-existing issues

### Data Versions
- **Raw**: SPY/DIA/QQQ/VIX OHLCV (v1)
- **Processed**:
  - a20: 20 features, 8100 rows
  - a50: 50 features, 8022 rows
  - a100: 100 features, 8022 rows
  - a200: 206 features, 7977 rows
  - a500: 500 features, 7977 rows (**FIXED**: was 906 rows, now 7977 with ffill/bfill)

---

## Cross-Workstream Coordination

### Blocking Dependencies
- None currently

### File Ownership

| Files | Owner | Status |
|-------|-------|--------|
| `src/features/tier_a500.py` | ws1 | **COMPLETE** |
| `src/models/patchtst.py` | ws3 | d_embed parameter added |
| `experiments/feature_embedding/*` | ws3 | 140 experiments defined |
| `src/training/hpo_budget_extremes.py` | ws2 | Budget-aware HPO |

---

## Session Summary (2026-02-07 - ws3)

### Phase 3 IMPLEMENTED: 22 LF + 21 AE Experiments

**Total experiments now: 183** (92 FE + 70 LF + 21 AE)

#### New Loss Function Experiments (LF-49 to LF-70)
| Priority | Loss Function | Experiments | Count |
|----------|--------------|-------------|-------|
| LF-P7 | MildFocal | LF-49 to LF-54 | 6 |
| LF-P8 | AsymmetricFocal | LF-55 to LF-60 | 6 |
| LF-P9 | EntropyRegularized | LF-61 to LF-64 | 4 |
| LF-P10 | VarianceRegularized | LF-65 to LF-68 | 4 |
| LF-P11 | CalibratedFocal | LF-69 to LF-70 | 2 |

#### Advanced Embedding Experiments (AE-01 to AE-21)
| Priority | Embedding Type | Experiments | Count |
|----------|---------------|-------------|-------|
| AE-P1 | Progressive | AE-01 to AE-04 | 4 |
| AE-P2 | Bottleneck | AE-05 to AE-08 | 4 |
| AE-P3 | MultiHead | AE-09 to AE-12 | 4 |
| AE-P4 | GatedResidual | AE-13 to AE-16 | 4 |
| AE-P5 | Attention | AE-17 to AE-21 | 5 |

**Implementation Details**:
- Extended ExperimentSpec with embedding_type/embedding_params fields
- Added get_embedding_layer() helper function
- Added 5 new loss function types to get_loss_function()
- Updated get_early_stop_metric() for calibration-focused losses
- 25 tests passing

### Current Run Status
- **LF-P7 RUNNING** (MildFocal experiments)
- **37 experiments remaining** after LF-P7

### Best Result So Far
- **LF-40**: 85.7% precision using LabelSmoothing (e=0.20) on a500/d_embed=16

### Previous: Feature Embedding Architecture Results (P1-P7)

| Tier | Best Exp | Config | Precision | Params |
|------|----------|--------|-----------|--------|
| a100 | **FE-06** | d_embed=32, d=128, L=4, h=8 | **69.2%** | 0.87M |
| a200 | FE-04 | d_embed=64 | 46.7% | 0.94M |
| a500 | **FE-50** | d_embed=16, d=64, L=4, h=4 | **60.0%** | 0.23M |

**Key Findings**:
1. **Inverse scaling laws**: Smaller d_embed and d_model generally better
2. **FE-50 breakthrough**: a500 (worst tier) becomes competitive with tiny model + aggressive compression
3. **Large models hurt**: d_model=256/512/1024 all worse than d_model=128

---

## User Priorities

### ws1 (feature_generation) - COMPLETE
tier_a500 done. Committed and data regenerated.

### ws2 (foundation) - Ready
1. Commit budget-aware HPO changes
2. Run validation with optimal params

### ws3 (feature_embedding) - Active
1. **Run Phase 2 LF-P1** (SoftAUC experiments, 8 experiments)
2. Analyze results - does loss function improve precision?
3. Continue with LF-P2 through LF-P6 based on results
4. Optionally run P8-P12 architecture experiments in parallel

---

## User Preferences (Authoritative)

### Metric Priority for Analysis (CRITICAL)
| Priority | Metric | Notes |
|----------|--------|-------|
| **#1** | **PRECISION** | When model predicts "buy", how often correct? |
| **#2** | **RECALL** | Of all opportunities, how many caught? |
| Secondary | AUC-ROC | Ranking only, not primary |
| **NEVER** | F1, Accuracy | Hides tradeoffs, irrelevant for imbalanced data |

### Development Approach
- TDD: tests first, always
- Planning sessions before implementation
- Uses tmux for long-running experiments
- **Workstream-specific testing**: `make test-ws{N}` for fast iteration

### Context Durability
- Multiple places: Memory MCP + context files + docs/
- Code comments are secondary

### Documentation Philosophy
- Flat docs/ (no subdirs except research_paper/, archive/)
- Precision - never reduce fidelity
- Consolidate rather than delete

### Hyperparameters (Validated)
- **d_embed**: 16-32 optimal (tier-dependent)
- **d_model**: 64-128 optimal (smaller is better)
- **Dropout**: 0.5 (default), may vary
- **Learning Rate**: 1e-4
- **Context**: 80d (standard, but sweeps planned)
- **Normalization**: RevIN only
- **Splitter**: SimpleSplitter

---

## Key Insights

### Feature Embedding Results (2026-02-07)

**Inverse Scaling Laws Confirmed**:
- Smaller d_embed generally better (32 > 64 > 128)
- Smaller d_model generally better (128 > 256 > 512 > 1024)
- Fewer features not always better (a100 > a200 > a500 with standard arch)

**FE-50 Breakthrough**:
- a500 + d_embed=16 + d_model=64 + h=4 = **60% precision, 0.23M params**
- Previously a500 maxed at ~43%
- Suggests: noisy features need extreme compression + tiny model

**What Doesn't Work**:
- Large d_model (256, 512, 1024)
- Feature expansion (d_embed > num_features)
- Too aggressive compression (d_embed=8 on a100 → useless)

### Future Research Questions
1. Does SoftAUC/Focal loss improve precision?
2. Does longer context need more d_embed?
3. Does longer context need more layers?
4. Interaction between context length and feature tier?
5. Do loss functions change optimal architecture?
