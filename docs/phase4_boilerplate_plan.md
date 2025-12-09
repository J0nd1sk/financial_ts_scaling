# Phase 4: Boilerplate - Training Infrastructure Plan

**Status:** Approved for Implementation
**Date:** 2025-12-08
**Execution Strategy:** Option A - Sequential TDD (7 sub-tasks with individual approval gates)

---

## Objective

Implement the complete training infrastructure to enable running PatchTST experiments on SPY financial time-series data.

**Outcome:** Can run `python scripts/train.py --config configs/daily/threshold_1pct.yaml` and train a model with thermal monitoring and experiment tracking.

---

## Scope

### In Scope

1. PatchTST model config presets (2M/20M/200M parameters)
2. YAML config loader with dataclass validation (`src/config/training.py`)
3. PyTorch Dataset class for features + binary threshold targets
4. Training script with thermal monitoring (`scripts/train.py`)
5. Batch size discovery script (`scripts/find_batch_size.py`)
6. Example training configs (`configs/daily/*.yaml`)
7. W&B + MLflow integration

### Out of Scope

- Multi-horizon configs (2d/3d/5d) - create after daily works
- Regression task configs - binary thresholds first
- HPO/Optuna integration - separate phase
- Multi-asset datasets (DIA, QQQ) - Phase 5
- Full experiment runs - just infrastructure

---

## Reproducibility

### Seed Handling

All training runs must be reproducible for valid scaling law comparisons.

- **Config field:** `seed: int` (default: 42)
- **Training script sets:**
  ```python
  torch.manual_seed(config.seed)
  torch.mps.manual_seed(config.seed)
  ```
- **Dataloader:** `generator=torch.Generator().manual_seed(config.seed)`

### Determinism Notes

- MPS backend does not fully support `torch.use_deterministic_algorithms(True)`
- Known non-deterministic ops will be documented in experiment results
- For exact reproducibility, use CPU backend (slower but deterministic)

### Reproducibility Test

- `test_reproducible_batch_with_fixed_seed`: Same seed → identical first batch tensor values

---

## Sub-Task Breakdown

### Task 1: Config System

**Purpose:** Load and validate training configuration from YAML files

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `src/config/__init__.py` | Package init | 5 |
| `src/config/training.py` | Config dataclasses + loader | 150 |
| `tests/test_config.py` | Config tests | 80 |

**Tests:**
- `test_load_valid_config_returns_dataclass`: Valid YAML → TrainingConfig dataclass
- `test_load_config_missing_required_field_raises`: Missing required field → ValueError
- `test_load_config_invalid_param_budget_raises`: Invalid budget (5M) → ValueError
- `test_load_config_validates_paths_exist`: Non-existent path → ValueError

**Dependencies:** None

---

### Task 2: Dataset Class

**Purpose:** PyTorch Dataset that loads features and generates binary threshold targets

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `src/data/__init__.py` | Package init | 5 |
| `src/data/dataset.py` | PyTorch Dataset class | 120 |
| `tests/test_dataset.py` | Dataset tests | 100 |

**Target Construction Rule:**
```python
future_max = max(close[t+1 : t+horizon])
label = 1 if future_max >= close[t] * (1 + threshold) else 0
```

**Tests:**
- `test_dataset_returns_correct_shapes`: Input (context_length, n_features), target (1,)
- `test_dataset_binary_label_threshold_1pct`: Known sequence → correct label
- `test_dataset_handles_horizon_correctly`: horizon=5 uses next 5 days
- `test_dataset_excludes_samples_near_end`: Last `horizon` rows excluded
- `test_dataset_length_matches_expected`: Total - warmup - horizon = length
- `test_dataset_raises_on_nan_in_close`: NaN in close prices → ValueError
- `test_dataset_rejects_short_sequences`: len < warmup + horizon → ValueError
- `test_dataset_warmup_excludes_initial_rows`: First `warmup` rows not in dataset

**Dependencies:** Task 1 (needs config dataclasses)

---

### Task 3: Model Configs

**Purpose:** YAML configuration files for PatchTST at each parameter budget

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `configs/model/patchtst_2m.yaml` | ~2M params | 20 |
| `configs/model/patchtst_20m.yaml` | ~20M params | 20 |
| `configs/model/patchtst_200m.yaml` | ~200M params | 20 |
| `src/models/__init__.py` | Package init | 5 |
| `src/models/utils.py` | Parameter counting helper | 30 |
| `tests/test_model_config.py` | Model config tests | 80 |

**Rationale for YAML:** Standardize on single config format (YAML) across training and model configs. Avoids dual schema/loader complexity.

**Tests:**
- `test_count_parameters_helper_accurate`: Known model → exact param count
- `test_patchtst_2m_config_params_within_budget`: ≤2.5M params (uses helper)
- `test_patchtst_20m_config_params_within_budget`: ≤25M params (uses helper)
- `test_patchtst_200m_config_params_within_budget`: ≤250M params (uses helper)
- `test_patchtst_forward_pass_output_shape`: Correct output dimensions

**Dependencies:** None

---

### Task 4: Thermal Callback

**Purpose:** Monitor CPU temperature during training, warn/stop as needed

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `src/training/thermal.py` | Thermal monitoring callback | 80 |
| `tests/test_thermal.py` | Thermal callback tests | 80 |

**Thresholds:**
- `<70°C`: Normal - full operation
- `70-85°C`: Acceptable - monitor closely
- `85-95°C`: Warning - log warning, consider pause
- `>95°C`: Critical - STOP immediately, save checkpoint

**Tests:**
- `test_thermal_callback_logs_temperature`: Mock powermetrics → log created
- `test_thermal_callback_warns_at_85c`: 85°C → warning logged
- `test_thermal_callback_stops_at_95c`: 95°C → stop flag set
- `test_thermal_callback_graceful_when_unavailable`: No powermetrics → warning + continue

**Dependencies:** None

---

### Task 5: Tracking Integration

**Purpose:** Log metrics and artifacts to W&B and MLflow

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `src/training/__init__.py` | Package init | 5 |
| `src/training/tracking.py` | W&B + MLflow integration | 100 |
| `tests/test_tracking.py` | Tracking tests | 80 |

**Tests:**
- `test_tracking_logs_to_wandb`: wandb.log called with metrics
- `test_tracking_logs_to_mlflow`: mlflow.log_metric called
- `test_tracking_saves_config_artifact`: Config saved as artifact
- `test_tracking_handles_disabled_trackers`: Can disable either tracker

**Dependencies:** None

---

### Task 6: Training Script

**Purpose:** Main training loop integrating all components

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `src/training/trainer.py` | Training loop + callbacks | 200 |
| `scripts/train.py` | CLI entry point | 80 |
| `configs/daily/threshold_1pct.yaml` | Example config | 30 |
| `tests/test_training.py` | Training integration tests | 100 |

**Tests:**
- `test_train_one_epoch_completes`: Micro dataset → no errors
- `test_train_logs_metrics`: Loss logged to trackers
- `test_train_saves_checkpoint`: Checkpoint file created
- `test_train_respects_thermal_stop`: Mock 95°C → training stops
- `test_training_verifies_manifest_before_start`: Invalid/missing manifest → fails before epoch 1
- `test_training_logs_data_version`: Data MD5 hash logged to trackers for reproducibility
- `test_reproducible_batch_with_fixed_seed`: Same seed → identical first batch values

**Dependencies:** Tasks 1-5 (all components)

---

### Task 7: Batch Size Discovery

**Purpose:** Find largest viable batch size for each parameter budget

**Files:**
| Path | Purpose | ~Lines |
|------|---------|--------|
| `scripts/find_batch_size.py` | Batch size discovery CLI | 100 |
| `tests/test_batch_size.py` | Batch size tests | 60 |

**Algorithm:**
1. Start with batch_size = 8
2. Try forward + backward pass
3. If success, double batch_size
4. If OOM, halve and return previous successful value

**Tests:**
- `test_find_batch_size_returns_power_of_two`: Result in [8, 16, 32, ...]
- `test_find_batch_size_respects_memory`: Doesn't OOM

**Dependencies:** Tasks 1-3 (config, dataset, model)

---

## Execution Order

```
Week 1: Foundation (Parallel)
├── Task 1: Config System
├── Task 3: Model Configs
├── Task 4: Thermal Callback
└── Task 5: Tracking Integration

Week 2: Data
└── Task 2: Dataset Class (requires Task 1)

Week 3: Integration
├── Task 6: Training Script (requires Tasks 1-5)
└── Task 7: Batch Size Discovery (requires Tasks 1-3, partial 6)
```

---

## Assumptions

1. Feature data exists at `data/processed/v1/SPY_features_a20.parquet`
2. Raw SPY data exists at `data/raw/SPY.parquet` (for Close prices)
3. PatchTST from Hugging Face transformers works with our data format
4. MPS backend supports all required operations
5. Context length and patch length compatible with ~20 features

### Testing Environment

6. **W&B:** Tests use `WANDB_MODE=disabled` (no API key required). Production uses `wandb login`.
7. **MLflow:** Tests use `mlflow.set_tracking_uri("file:///tmp/mlruns")` (local filesystem). Production uses configured server.
8. **Thermal:** Tests mock `powermetrics` subprocess calls. CI environments use graceful fallback (warning logged, training continues).
9. **Manifest verification:** Training script validates manifest entries and MD5 checksums before starting. Tests use fixture data with known checksums.

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| PatchTST input format incompatible | Medium | Test with minimal example first |
| MPS memory limits for larger models | High | Batch size discovery mandatory |
| Thermal monitoring unreliable | Low | Test callback in isolation |
| W&B/MLflow conflicts | Low | Test each tracker independently |
| Config schema too rigid | Medium | Keep schema simple, iterate |
| Target generation off-by-one | Medium | Extensive unit tests |
| Training too slow for testing | Medium | Use small subset for unit tests |

---

## Success Criteria

- [ ] `make test` passes with all new tests
- [ ] PatchTST configs exist for 2M/20M/200M param budgets
- [ ] Config loader validates YAML against dataclass schema
- [ ] Dataset class generates correct binary threshold labels
- [ ] Training script runs for 1 epoch without errors
- [ ] Thermal callback logs temperature and stops at >95°C
- [ ] W&B run created with metrics logged
- [ ] MLflow run created with artifacts saved
- [ ] `find_batch_size.py` discovers viable batch size for 2M model

---

## Estimated Effort

| Task | Estimated Time |
|------|---------------|
| Task 1: Config System | 2-3 hours |
| Task 2: Dataset Class | 3-4 hours |
| Task 3: Model Configs | 2-3 hours |
| Task 4: Thermal Callback | 2-3 hours |
| Task 5: Tracking Integration | 2-3 hours |
| Task 6: Training Script | 4-6 hours |
| Task 7: Batch Size Discovery | 2-3 hours |
| **Total** | **17-25 hours** |

---

## Approval Gates

Each sub-task requires:
1. Planning confirmation before starting
2. Test plan review
3. RED phase verification (tests fail)
4. GREEN phase verification (tests pass)
5. Commit approval

---

*Document Version: 1.1*
*Approved: 2025-12-08*
*Updated: 2025-12-08 (GPT5-Codex review feedback)*
*Execution Strategy: Option A (Sequential TDD)*
