# Cross-Budget Validation Report

**Tier**: a100
**Horizon**: 1
**Timestamp**: 2026-01-26T18:26:48.512150

**Configs Found**: 2M, 20M, 200M

## Validation Matrix

Each cell shows AUC when using row's config on column's budget.

| Config \ Budget | 2M | 20M | 200M |
|---|---|---|---|
| **2M** | 0.7178 | 0.7178 | 0.7178 |
| **20M** | 0.7246 | 0.7246 | 0.7246 |
| **200M** | 0.7147 | 0.7147 | 0.7147 |

## Analysis

- **Diagonal average** (matched config/budget): 0.7190
- **Off-diagonal average** (cross-budget): 0.7190
- **Transfer gap**: 0.0000