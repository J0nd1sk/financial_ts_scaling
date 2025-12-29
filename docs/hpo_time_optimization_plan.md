# HPO Time Optimization Plan

**Parent Phase**: 6A (Parameter Scaling)
**Stage Type**: Cleanup/Optimization (temporary detour within Phase 6A)
**Date**: 2025-12-26
**Status**: In Progress — Task 4 of 6

> **Note**: This is a stage within Phase 6A, not a separate phase.
> Once complete, Phase 6A HPO runs resume.

### Stage Progress (Revised 2025-12-27)
- ✅ Task 1: Memory-safe batch config (`get_memory_safe_batch_config()`) — 6 tests
- ✅ Task 2: Gradient accumulation in Trainer — 3 tests
- ✅ Task 3: Early stopping in Trainer — 5 tests
- ⏳ **Task 4: Wire HPO to use new training features** ← CURRENT (consolidated from old Tasks 4-6)
- ⏳ Task 5: Regenerate 12 HPO scripts + runner 'q' quit
- ⏳ Task 6: Integration smoke test

> **Revision Note (2025-12-27)**: Original Tasks 4-6 consolidated into single Task 4.
> Discovery: PatchTST already has dropout support (added at model creation).
> The actual gap was that HPO hardcodes `dropout=0.1` instead of searching it.

---

## Memory Findings

Relevant knowledge from previous sessions:

| Entity | Key Insight |
|--------|-------------|
| `MPS_GPU_Utilization_Finding` | MPS only wins for matrices ≥4000×4000; batch≥256 for GPU benefit |
| `Batch_Size_Regularization_Tradeoff` | Large batches OK for GPU efficiency; add explicit regularization to compensate |

**Session context findings:**
- GPU was 86% idle during 2B training with batch_size=64
- Trial 4 (2B_h1) stuck at 115GB memory, 0% CPU for days
- 2B trials with L=256 + d≥1024 + batch=128 exceed memory capacity

---

## Problem Statement

Current HPO scripts for 2B budget:
1. **Memory exhaustion**: Large architectures (d≥1024, L≥192) with batch_size≥64 exceed 128GB unified memory
2. **No early stopping**: Trials run full 50-100 epochs even when val_loss plateaus at epoch 10
3. **Fixed batch size**: Same batch_size search space [64, 128, 256, 512] regardless of architecture size
4. **No gradient accumulation**: Cannot simulate large effective batches with memory-safe micro-batches

**Evidence**: 2B_h1 Trial 4 (d=1024, L=256, batch=128) consumed 115GB memory, triggered swap thrashing, and stalled for 2+ days with 0% CPU.

---

## Objective

Implement 4 optimizations to enable 2B HPO to complete successfully:

1. **Dynamic batch sizing**: Smaller physical batches for larger architectures
2. **Gradient accumulation**: Maintain effective batch size for training dynamics
3. **Early stopping**: Terminate trials that plateau, saving ~33%+ time
4. **Higher regularization**: Increased dropout/weight_decay for large-batch training

**In Scope:**
- Modify `src/models/arch_grid.py` with batch config function
- Modify `src/training/trainer.py` with gradient accumulation and early stopping
- Modify `configs/hpo/architectural_search.yaml` with higher regularization ranges
- Modify `src/experiments/templates.py` to use dynamic batching
- Regenerate all 12 Phase 6A HPO scripts (especially 2B)

**Out of Scope:**
- Changing the architecture search space itself
- Modifying the ChunkSplitter or data pipeline
- Adding new parameter budgets (2M, 20M, 200M, 2B remain fixed)
- Gradient checkpointing (more complex, deferred if needed)

---

## Success Criteria

- [ ] 2B HPO trials with d=1024, L=256 complete without memory exhaustion
- [ ] 2B HPO trials with d=2048 complete without memory exhaustion
- [ ] Trials plateau detection: stops training within 5-10 epochs of no improvement
- [ ] Effective batch size maintained at 128-256 via gradient accumulation
- [ ] All 342 existing tests pass
- [ ] New tests for dynamic batching, gradient accumulation, early stopping

---

## Technical Design

### Task 1: Memory-Safe Batch Config Function

Add to `src/models/arch_grid.py`:

```python
def get_memory_safe_batch_config(
    d_model: int,
    n_layers: int,
    target_effective_batch: int = 256,
) -> dict[str, int]:
    """Return memory-safe batch configuration based on architecture size.

    Returns:
        dict with 'micro_batch' and 'accumulation_steps'
    """
    # Estimate memory usage (rough heuristic)
    # Memory ∝ d_model² × n_layers (attention + FFN activations)
    memory_score = (d_model ** 2) * n_layers / 1e9  # Normalize to ~1.0 for d=1024, L=256

    if memory_score <= 0.1:    # Small models (d≤512 or shallow)
        micro_batch = 256
    elif memory_score <= 0.5:  # Medium models
        micro_batch = 128
    elif memory_score <= 1.5:  # Large models
        micro_batch = 64
    elif memory_score <= 3.0:  # XLarge models
        micro_batch = 32
    else:                      # Massive models (2B+)
        micro_batch = 16

    accumulation_steps = max(1, target_effective_batch // micro_batch)

    return {
        'micro_batch': micro_batch,
        'accumulation_steps': accumulation_steps,
        'effective_batch': micro_batch * accumulation_steps,
    }
```

**Memory estimates:**
| d_model | n_layers | memory_score | micro_batch | accum | effective |
|---------|----------|--------------|-------------|-------|-----------|
| 256     | 192      | 0.013        | 256         | 1     | 256       |
| 768     | 32       | 0.019        | 256         | 1     | 256       |
| 768     | 256      | 0.151        | 128         | 2     | 256       |
| 1024    | 256      | 0.268        | 128         | 2     | 256       |
| 1536    | 64       | 0.151        | 128         | 2     | 256       |
| 2048    | 32       | 0.134        | 128         | 2     | 256       |

### Task 2: Gradient Accumulation in Trainer

Modify `src/training/trainer.py`:

```python
def __init__(
    self,
    ...
    accumulation_steps: int = 1,  # NEW
) -> None:
    self.accumulation_steps = accumulation_steps
    ...

def _train_epoch(self, epoch: int) -> float:
    self.model.train()
    total_loss = 0.0
    num_batches = 0

    self.optimizer.zero_grad()  # Zero once at epoch start

    for batch_idx, (batch_x, batch_y) in enumerate(self.dataloader):
        batch_x = batch_x.to(self.device)
        batch_y = batch_y.to(self.device)

        # Forward pass
        outputs = self.model(batch_x)
        loss = self.criterion(outputs, batch_y)

        # Scale loss by accumulation steps for proper averaging
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()

        # Accumulate gradients, step every N batches
        if (batch_idx + 1) % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        total_loss += loss.item()  # Track unscaled loss
        num_batches += 1

    # Handle leftover batches (if total batches not divisible by accum_steps)
    if num_batches % self.accumulation_steps != 0:
        self.optimizer.step()
        self.optimizer.zero_grad()

    return total_loss / max(num_batches, 1)
```

### Task 3: Early Stopping in Trainer

Modify `src/training/trainer.py`:

```python
def __init__(
    self,
    ...
    early_stopping_patience: int | None = None,  # NEW: epochs without improvement
    early_stopping_min_delta: float = 0.001,     # NEW: minimum improvement threshold
) -> None:
    self.early_stopping_patience = early_stopping_patience
    self.early_stopping_min_delta = early_stopping_min_delta
    self._best_val_loss = float('inf')
    self._epochs_without_improvement = 0
    ...

def _check_early_stopping(self, val_loss: float) -> bool:
    """Check if training should stop early.

    Returns True if val_loss hasn't improved by min_delta for patience epochs.
    """
    if self.early_stopping_patience is None:
        return False

    if val_loss < self._best_val_loss - self.early_stopping_min_delta:
        self._best_val_loss = val_loss
        self._epochs_without_improvement = 0
        return False

    self._epochs_without_improvement += 1
    return self._epochs_without_improvement >= self.early_stopping_patience

def train(self, verbose: bool = False) -> dict[str, Any]:
    ...
    for epoch in range(self.epochs):
        epoch_loss = self._train_epoch(epoch)

        if self.val_dataloader is not None:
            val_loss = self._evaluate_val()

            # Check early stopping
            if self._check_early_stopping(val_loss):
                stopped_early = True
                stop_reason = "early_stopping"
                break
        ...
```

### Task 4: Wire HPO to Use New Training Features (CONSOLIDATED)

> **Context**: Original Tasks 4-6 were tightly coupled and have been consolidated.
>
> **Key Discovery**: PatchTST already has `dropout` and `head_dropout` parameters
> in `PatchTSTConfig` (lines 34-35 of patchtst.py). The model was never "missing"
> dropout — the gap is that `create_architectural_objective()` in hpo.py hardcodes
> `dropout=0.1` instead of sampling it from the config.

#### Objective

Wire the HPO objective function to use:
1. Dynamic batch sizing (from Task 1's `get_memory_safe_batch_config()`)
2. Gradient accumulation (from Task 2)
3. Early stopping (from Task 3)
4. Searchable dropout (already in model, needs to be sampled)

#### 4A: Update `configs/hpo/architectural_search.yaml`

```yaml
training_search_space:
  learning_rate:
    type: log_uniform
    low: 1.0e-4
    high: 1.0e-3

  epochs:
    type: categorical
    choices: [50, 75, 100]

  # REMOVED: batch_size (now dynamic based on architecture)

  weight_decay:
    type: log_uniform
    low: 1.0e-4      # Increased from 1.0e-5
    high: 5.0e-3     # Increased from 1.0e-3

  warmup_steps:
    type: categorical
    choices: [100, 200, 300, 500]

  dropout:           # NEW - was hardcoded at 0.1
    type: uniform
    low: 0.1
    high: 0.3

# Early stopping configuration (read by HPO objective)
early_stopping:
  patience: 10       # Stop if no improvement for 10 epochs
  min_delta: 0.001   # Minimum improvement threshold
```

#### 4B: Update `src/training/hpo.py` `create_architectural_objective()`

Current code (lines 266-313) has these gaps:

```python
# CURRENT (hardcoded):
model_config = PatchTSTConfig(
    ...
    dropout=0.1,        # ← Should sample from config
    head_dropout=0.0,   # ← Keep fixed (not searched)
)

trainer = Trainer(
    ...
    batch_size=batch_size,  # ← Should use dynamic sizing
    # MISSING: accumulation_steps
    # MISSING: early_stopping_patience
    # MISSING: early_stopping_min_delta
)
```

**Required changes:**

1. Import `get_memory_safe_batch_config` from `src.models.arch_grid`
2. Sample `dropout` from `training_search_space` (with fallback to 0.1)
3. Call `get_memory_safe_batch_config(d_model, n_layers)` to get batch config
4. Read `early_stopping` section from config (or use defaults)
5. Pass new params to Trainer:
   - `batch_size=batch_config['micro_batch']`
   - `accumulation_steps=batch_config['accumulation_steps']`
   - `early_stopping_patience=early_stopping_config.get('patience')`
   - `early_stopping_min_delta=early_stopping_config.get('min_delta', 0.001)`

```python
# AFTER (dynamic):
from src.models.arch_grid import get_memory_safe_batch_config

# In objective function:
dropout = sampled_params.get("dropout", 0.1)

batch_config = get_memory_safe_batch_config(
    d_model=arch["d_model"],
    n_layers=arch["n_layers"],
    target_effective_batch=256,
)

model_config = PatchTSTConfig(
    ...
    dropout=dropout,      # ← Sampled from config
    head_dropout=0.0,
)

trainer = Trainer(
    ...
    batch_size=batch_config['micro_batch'],
    accumulation_steps=batch_config['accumulation_steps'],
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
)
```

**4B Note**: May need to update function signature to pass early_stopping config:
- Add `early_stopping_config: dict | None = None` parameter
- Or read from YAML path directly inside objective

#### 4C: Add tests to `tests/test_hpo.py`

Six new tests (TDD approach - write tests first):
- `test_architectural_search_config_has_dropout`: Verify dropout in training_search_space
- `test_architectural_search_config_no_batch_size`: Verify batch_size removed
- `test_architectural_search_config_has_early_stopping`: Verify early_stopping section exists
- `test_architectural_objective_samples_dropout`: Mock trial, verify dropout sampled and used
- `test_architectural_objective_uses_dynamic_batch`: Verify get_memory_safe_batch_config called
- `test_architectural_objective_passes_early_stopping`: Verify Trainer receives early stopping params

#### Files Affected

| File | Changes |
|------|---------|
| `configs/hpo/architectural_search.yaml` | Remove batch_size, add dropout, add early_stopping section |
| `src/training/hpo.py` | Import arch_grid, sample dropout, use dynamic batch, pass new Trainer params |
| `tests/test_hpo.py` | Add 6 tests for config structure and objective behavior |

#### Success Criteria

- [ ] `architectural_search.yaml` has dropout in training_search_space
- [ ] `architectural_search.yaml` has early_stopping section
- [ ] `architectural_search.yaml` does NOT have batch_size
- [ ] `create_architectural_objective()` samples dropout
- [ ] `create_architectural_objective()` uses `get_memory_safe_batch_config()`
- [ ] `Trainer()` call includes `accumulation_steps`, `early_stopping_patience`, `early_stopping_min_delta`
- [ ] All 356+ tests pass

#### Scope Estimate

- Files: 3 (1 config, 1 source, 1 test)
- Lines: ~50-70 changed
- New tests: 6
- Complexity: Medium

---

### Task 5: Regenerate HPO Scripts + Runner 'q' Quit

After all code changes, regenerate all 12 HPO scripts with:
- Dynamic batch sizing
- Gradient accumulation
- Early stopping
- Dropout in search space

Also add 'q' keystroke handling to runner script for graceful tmux exit.

### Task 6: Integration Smoke Test

Run 2B HPO with 3 trials to verify:
- Memory stays under control with dynamic batching
- Early stopping triggers appropriately
- Dropout is being sampled (check logs)
- Gradient accumulation working (check effective batch in logs)

---

## Assumptions

1. Memory pressure is primarily from activations, not model weights
2. Gradient accumulation provides equivalent training dynamics to large batches
3. Early stopping at patience=10 won't prematurely terminate good trials
4. 128GB unified memory can handle d=2048, L=32 with micro_batch=64

---

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Memory heuristic not accurate enough | Medium | Start conservative, monitor actual memory usage, adjust thresholds |
| Gradient accumulation changes training dynamics | Low | Effective batch size maintained; well-established technique |
| Early stopping too aggressive | Medium | Use patience=10 (not 5); track stopped_early in results |
| Dropout changes model behavior | Low | Dropout is standard for transformers; already in TransformerEncoderLayer |
| Breaking existing tests | Medium | Run full test suite after each task; TDD approach |

---

## Test Plan

### Existing Tests to Modify

- `tests/test_training.py::TestTrainOneEpoch`: Add test for accumulation_steps parameter
- `tests/test_training.py::TestTrainerWithSplits`: Add test for early stopping

### New Tests to Add

**Task 1 - Batch Config (tests/test_arch_grid.py)**:
- `test_get_memory_safe_batch_config_small_model`: d=256, L=16 → batch=256
- `test_get_memory_safe_batch_config_medium_model`: d=768, L=32 → batch=128 or 256
- `test_get_memory_safe_batch_config_large_model`: d=1024, L=256 → batch=64
- `test_get_memory_safe_batch_config_xlarge_model`: d=2048, L=64 → batch=32
- `test_effective_batch_maintained`: Verify micro_batch × accum_steps = target

**Task 2 - Gradient Accumulation (tests/test_training.py)**:
- `test_gradient_accumulation_steps_1`: Default behavior unchanged
- `test_gradient_accumulation_steps_4`: Loss computed correctly with accumulation
- `test_gradient_accumulation_optimizer_step_frequency`: Verify step called every N batches

**Task 3 - Early Stopping (tests/test_training.py)**:
- `test_early_stopping_disabled_by_default`: patience=None means no early stop
- `test_early_stopping_triggers_after_patience`: Stops when val_loss plateaus
- `test_early_stopping_resets_on_improvement`: Counter resets when val_loss improves
- `test_early_stopping_respects_min_delta`: Small improvements don't reset counter
- `test_early_stopping_result_includes_stop_reason`: Result dict has stop_reason="early_stopping"

**Task 4 - Wire HPO (tests/test_hpo.py)** (REVISED):
- `test_architectural_search_config_has_dropout`: Verify dropout in training_search_space
- `test_architectural_search_config_no_batch_size`: Verify batch_size removed
- `test_architectural_search_config_has_early_stopping`: Verify early_stopping section
- `test_architectural_objective_samples_dropout`: Verify dropout sampled from config
- `test_architectural_objective_uses_dynamic_batch`: Verify get_memory_safe_batch_config used
- `test_architectural_objective_passes_early_stopping`: Verify Trainer gets early stopping params

> **Note**: Task 6 dropout tests removed — PatchTST already has dropout.
> See `tests/test_patchtst.py` line 40-41 for existing coverage.

### Edge Cases

- Accumulation with single batch: handled correctly
- Early stopping with no validation set: disabled (returns False)
- Memory score edge cases: boundary values tested

---

## Files Affected (Revised 2025-12-27)

| File | Changes | Status |
|------|---------|--------|
| `src/models/arch_grid.py` | Add `get_memory_safe_batch_config()` | ✅ Task 1 |
| `src/training/trainer.py` | Add accumulation_steps, early_stopping params | ✅ Tasks 2-3 |
| `tests/test_arch_grid.py` | Add batch config tests (6 tests) | ✅ Task 1 |
| `tests/test_training.py` | Add accumulation and early stopping tests (8 tests) | ✅ Tasks 2-3 |
| `configs/hpo/architectural_search.yaml` | Remove batch_size, add dropout, add early_stopping | ⏳ Task 4 |
| `src/training/hpo.py` | Wire dynamic batch, dropout sampling, early stopping | ⏳ Task 4 |
| `tests/test_hpo.py` | Add tests for new HPO behavior (6 tests) | ⏳ Task 4 |
| `experiments/phase6a/hpo_*.py` (12 files) | Regenerate all | ⏳ Task 5 |
| `scripts/run_phase6a_hpo.sh` | Add 'q' keystroke quit | ⏳ Task 5 |

> **Removed**: `src/models/patchtst.py` — already has dropout support

---

## Scope Estimate

- **Files**: 10+ (4 source, 4 test, 12 scripts)
- **Lines**: ~300-400 new/modified
- **Complexity**: Medium-High (multiple interacting features)
- **Tests**: ~25 new tests

---

## Task Breakdown (Revised 2025-12-27)

| Task | Description | Est. Time | Dependencies | Status |
|------|-------------|-----------|--------------|--------|
| 1 | Add `get_memory_safe_batch_config()` to arch_grid.py + tests | 30 min | None | ✅ Complete |
| 2 | Add gradient accumulation to Trainer + tests | 45 min | None | ✅ Complete |
| 3 | Add early stopping to Trainer + tests | 30 min | None | ✅ Complete |
| 4 | Wire HPO to use new training features (config + hpo.py + tests) | 45 min | Tasks 1-3 | ⏳ Current |
| 5 | Regenerate 12 HPO scripts + runner 'q' quit | 20 min | Task 4 | ⏳ Pending |
| 6 | Integration smoke test: 2B with 3 trials | 30 min | Task 5 | ⏳ Pending |

**Total estimated time**: ~3.5 hours (unchanged)
**Completed**: Tasks 1-3 (~1.75 hours)
**Remaining**: Tasks 4-6 (~1.5 hours)

> **Note**: Original Tasks 4-6 consolidated into new Task 4 after discovering
> PatchTST already has dropout. See Task 4 section above for detailed breakdown.

---

## Approval Required

Proceed with this plan? (yes/no/modify)

Changes requested:
- [ ] Adjust batch config thresholds
- [ ] Different early stopping patience value
- [ ] Additional regularization techniques
- [ ] Other: _______________
