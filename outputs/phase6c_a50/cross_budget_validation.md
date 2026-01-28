# Cross-Budget Validation Report

**Tier**: a50
**Horizon**: 1
**Timestamp**: 2026-01-27T14:53:21.440997

**Configs Found**: 2M, 20M, 200M

## Validation Matrix

Each cell shows AUC when using row's config on column's budget.

| Config \ Budget | 2M | 20M | 200M |
|---|---|---|---|
| **2M** | 0.5000 | 0.5200 | 0.5400 |
| **20M** | 0.5000 | 0.5200 | 0.5400 |
| **200M** | 0.5000 | 0.5200 | 0.5400 |

## Analysis

- **Diagonal average** (matched config/budget): 0.5200
- **Off-diagonal average** (cross-budget): 0.5200
- **Transfer gap**: 0.0000