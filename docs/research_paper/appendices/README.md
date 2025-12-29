# Appendices

This directory contains appendix materials for the research paper.

## Planned Contents

### Feature Dictionaries
- `features_tier_20.csv` - Base 20 indicators
- `features_tier_200.csv` - Extended 200 indicators
- `features_tier_2000.csv` - Full 2000 indicators

**CSV Structure:**
| Column | Description |
|--------|-------------|
| feature_name | Short identifier (e.g., `rsi_14`) |
| feature_long_name | Full name (e.g., `Relative Strength Index (14-period)`) |
| description | What the indicator measures |
| calculation_method | Formula or algorithm (verified from code) |
| library | Implementation source (e.g., `pandas_ta`) |
| parameters | Configuration values (e.g., `length=14`) |
| category | Feature family (momentum, volatility, trend, volume, etc.) |
| data_source | Origin of raw data |
| tier | Which tier introduced this feature (20, 200, 2000) |

### HPO Documentation
- `hpo_methodology.md` - HPO approach and search spaces
- `hpo_results_summary.md` - Aggregated HPO results

### Architecture Documentation
- `architecture_grid.md` - Valid architectures per parameter budget

## Notes

- Feature dictionary entries must be verified against actual code implementation
- All calculation methods should match what's in `src/features/`
