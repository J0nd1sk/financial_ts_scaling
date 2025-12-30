# Architectural HPO Design

**Created:** 2025-12-11
**Status:** Approved, ready for implementation
**Phase:** 6A (Parameter Scaling)

## Problem Statement

The current HPO implementation only searches **training parameters** (learning rate, epochs, weight decay, warmup steps, dropout) while keeping the **model architecture fixed**. This means we're finding the best way to train a pre-chosen architecture, not finding the best architecture for a given parameter budget.

For scaling law research, we need to answer: **"What is the best architecture at each parameter budget?"**

A 2M parameter budget can be achieved by many different architectures:
- *Example:* Deep & narrow: 24 layers × 64 dimensions
- *Example:* Shallow & wide: 4 layers × 256 dimensions
- *Example:* Balanced: 8 layers × 128 dimensions

Each may perform very differently. We need to search this architectural space.

---

## Design Overview

**Approach:** Pre-compute all valid architecture combinations per budget, then run ~50 trials exploring both architecture and training params.

**Key insight:** Instead of sampling architectures randomly (which might exceed budget), we pre-generate a list of valid architectures that definitely fit within each budget. Then Optuna samples from this list.

---

## Architectural Parameters

### Parameters to Search (DEFINITIVE values)

| Parameter | What it controls | Search values |
|-----------|------------------|---------------|
| `d_model` | Embedding dimension | 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048 |
| `n_layers` | Transformer depth | 2, 3, 4, 6, 8, 12, 16, 24, 32, 48 |
| `n_heads` | Attention heads | 2, 4, 8, 16, 32 |
| `d_ff` | Feedforward width | 2× or 4× d_model (only these two ratios) |

### Fixed Architectural Parameters (DEFINITIVE)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `patch_len` | 10 | Temporal resolution - keep constant to isolate scaling effects |
| `stride` | 5 | Half of patch_len for overlap |
| `context_length` | 60 | Fixed input window |
| `num_classes` | 1 | Binary classification |

---

## Training Parameters

### Search Ranges (DEFINITIVE - narrower than before)

| Parameter | Range | Sampling method |
|-----------|-------|-----------------|
| `learning_rate` | 1e-4 to 1e-3 | Log-uniform (continuous) |
| `epochs` | 50, 75, 100 | Categorical (pick one of these three) |
| `weight_decay` | 1e-5 to 1e-3 | Log-uniform (continuous) |
| `warmup_steps` | 100, 200, 300, 500 | Categorical (pick one of these four) |
| `batch_size` | 32, 64, 128, 256 | Categorical (pick one of these four) |

**Rationale for narrower ranges:** With only 1-2 trials per architecture, we can't afford to waste trials on obviously bad training params. These ranges represent "sensible defaults" that are unlikely to catastrophically fail.

---

## Architecture Grid Generation

### Process (DEFINITIVE)

**Step 1: Generate all combinations**
- Cross-product of all d_model × n_layers × n_heads × d_ff_ratio values
- *Example:* 10 × 10 × 5 × 2 = 1,000 raw combinations

**Step 2: Filter by constraints**
- `n_heads` must evenly divide `d_model`
- *Example:* n_heads=8 works with d_model=128, but not d_model=192

**Step 3: Filter by parameter budget**
- Estimate parameter count using formula below
- Keep only combinations within ±25% of target budget

**Step 4: Ensure extremes are included**
- The final list MUST include at least one config with:
  - Minimum valid d_model (tests "deep & narrow")
  - Maximum valid d_model (tests "shallow & wide")
  - Minimum valid n_layers
  - Maximum valid n_layers
- This allows isolating the effect of each architectural dimension

### Budget Tolerances (DEFINITIVE)

| Budget | Target | Acceptable Range (±25%) |
|--------|--------|-------------------------|
| 2M | 2,000,000 | 1,500,000 - 2,500,000 |
| 20M | 20,000,000 | 15,000,000 - 25,000,000 |
| 200M | 200,000,000 | 150,000,000 - 250,000,000 |
| 2B | 2,000,000,000 | 1,500,000,000 - 2,500,000,000 |

### Expected Output

~25-35 valid architectures per budget (actual count depends on what passes filters)

---

## Parameter Count Estimation

### Formula (DEFINITIVE)

```python
def estimate_param_count(
    d_model: int,
    n_layers: int,
    n_heads: int,  # Used for validation only, doesn't affect count
    d_ff: int,
    num_features: int,
    context_length: int = 60,
    patch_len: int = 10,
    stride: int = 5,
    num_classes: int = 1,
) -> int:
    """Estimate total parameter count for a PatchTST configuration."""
    num_patches = (context_length - patch_len) // stride + 1

    # Patch embedding: project (patch_len * features) to d_model
    patch_embedding = (patch_len * num_features) * d_model

    # Position embedding: learnable positions for each patch
    position_embedding = num_patches * d_model

    # Per encoder layer:
    #   - Self-attention: Q, K, V, O projections = 4 * d_model^2
    #   - FFN: two linear layers = 2 * d_model * d_ff
    #   - Layer norms: 2 norms * 2 params each * d_model = 4 * d_model
    per_layer = (4 * d_model * d_model) + (2 * d_model * d_ff) + (4 * d_model)
    encoder = n_layers * per_layer

    # Prediction head: flatten patches and project to classes
    prediction_head = (num_patches * d_model) * num_classes

    return patch_embedding + position_embedding + encoder + prediction_head
```

---

## Trial Allocation Strategy

### Overview (DEFINITIVE: 50 trials per budget/horizon combo)

**Target:** 25-35 architectures explored with 50 total trials

### Allocation Approach

1. **First ~20 architectures**: 2 trials each (40 trials)
   - Each trial samples different training params from the narrow ranges
   - Purpose: Learn which training param combinations work well across architectures

2. **Remaining ~10-15 architectures**: 1 trial each (10-15 trials)
   - Apply training params that performed well in early trials
   - Purpose: Maximize architectural coverage with limited remaining trials

### What "Light Sampling" Means

For each trial:
1. Select an architecture from the pre-computed valid list
2. Randomly sample training params from the narrow ranges defined above
3. Train the model and record val_loss

**We are NOT doing:**
- Grid search over training params
- Full HPO optimization per architecture
- Exhaustive training param combinations

**We ARE doing:**
- Random sampling from narrow, sensible ranges
- 1-2 random training param combinations per architecture
- Relying on narrow ranges to avoid obviously bad settings

---

## Output Format

### experiment_results.csv (DEFINITIVE columns)

```
experiment, trial, budget, horizon,
d_model, n_layers, n_heads, d_ff, param_count,
learning_rate, epochs, batch_size, weight_decay, warmup_steps,
val_loss, train_loss, duration_s, timestamp
```

### Result JSON Format

Output filename: `{experiment}_{budget}_best.json`

```json
{
  "experiment": "<experiment_name>",
  "budget": "<budget>",
  "best_params": {
    "arch_idx": "<int>",
    "learning_rate": "<float>",
    "epochs": "<int>",
    "batch_size": "<int>",
    "weight_decay": "<float>",
    "warmup_steps": "<int>"
  },
  "best_value": "<float>",
  "n_trials_completed": "<int>",
  "n_trials_pruned": "<int>",
  "timestamp": "<ISO-8601>",
  "study_name": "<study_name>",
  "optuna_version": "<version>",
  "architecture": {
    "d_model": "<int>",
    "n_layers": "<int>",
    "n_heads": "<int>",
    "d_ff": "<int>",
    "param_count": "<int>"
  }
}
```

**Example** (from actual h1 run):
```json
{
  "experiment": "phase6a_2M_h1_threshold_1pct",
  "budget": "2M",
  "best_params": {
    "arch_idx": 57,
    "learning_rate": 0.0001,
    "epochs": 50,
    "batch_size": 32,
    "weight_decay": 0.0002,
    "warmup_steps": 200
  },
  "best_value": 0.337,
  "n_trials_completed": 50,
  "n_trials_pruned": 0,
  "timestamp": "2025-12-13T03:51:46.140871+00:00",
  "study_name": "phase6a_2M_h1_threshold_1pct_2M",
  "optuna_version": "4.6.0",
  "architecture": {
    "d_model": 64,
    "n_layers": 48,
    "n_heads": 8,
    "d_ff": 256,
    "param_count": 2414273
  }
}
```

---

## Implementation Plan

See `docs/phase6a_implementation_history.md` for detailed task breakdown and execution history.

---

## Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture search approach | Pre-computed grid | Guarantees budget compliance, no wasted trials |
| Fixed patch_len/stride | 10/5 | Isolate scaling effects from temporal resolution |
| Training param ranges | Narrow | With 1-2 trials per arch, can't afford bad settings |
| Trial allocation | 25-35 archs, 1-2 trials each | Breadth over depth for scaling law research |
| Extreme configs | Required | Isolate effect of each architectural dimension |
| Batch size | Search 32-256 | 128GB RAM allows larger batches |
| Budget tolerance | ±25% | Balance precision with architectural diversity |
