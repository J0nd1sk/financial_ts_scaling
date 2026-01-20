# Research Paper Materials

This directory contains all materials for the research paper on neural scaling laws for financial time-series transformers.

## Directory Structure

```
research_paper/
├── README.md                          # This file
├── paper_outline.md                   # High-level paper structure
├── draft_notes.md                     # Working notes and analysis chunks
├── phase6a_results_analysis.md        # Comprehensive Phase 6A analysis
│
├── appendices/
│   ├── appendix_b1_hpo_methodology.md    # HPO development and corrections
│   ├── appendix_b2_architecture_analysis.md  # Architecture patterns
│   ├── appendix_b3_training_parameters.md    # Training param analysis
│   └── appendix_c_statistical_analysis.md    # Statistical tests
│
├── figures/
│   ├── fig1_scaling_curve_data.csv       # Scaling curve data
│   ├── fig2_architecture_style_data.csv  # Style performance by budget
│   ├── fig3_horizon_comparison_data.csv  # Horizon effects
│   ├── fig4_final_training_data.csv      # Final training results
│   └── fig5_lr_by_budget_data.csv        # Learning rate patterns
│
├── tables/
│   ├── table1_hpo_summary.csv            # HPO best results
│   ├── table2_final_training.csv         # Final training results
│   └── table3_scaling_comparison.csv     # Scaling law comparison
│
└── notes/
    ├── phase6a_conclusions.md            # Detailed justifications
    └── discussion_draft.md               # Discussion section draft
```

## Current Status

### Phase 6A: Parameter Scaling (COMPLETE)
- 600 HPO trials across 12 studies
- 16 final training runs
- **Core finding**: Inverse scaling - smaller models outperform larger ones

### Pending Phases
- **Phase 6B**: Horizon scaling (not started)
- **Phase 6C**: Feature scaling (not started)
- **Phase 6D**: Data scaling (not started)

## Key Documents

### For Understanding Results
1. Start with `phase6a_results_analysis.md` for comprehensive analysis
2. Read `notes/phase6a_conclusions.md` for detailed justifications
3. Consult `appendices/appendix_c_statistical_analysis.md` for statistical support

### For Paper Writing
1. `paper_outline.md` - Overall structure
2. `notes/discussion_draft.md` - Discussion section draft
3. `tables/*.csv` - Publication-ready tables
4. `figures/*.csv` - Data for generating figures

### For Methodology
1. `appendices/appendix_b1_hpo_methodology.md` - HPO approach
2. `appendices/appendix_b2_architecture_analysis.md` - Architecture insights
3. `appendices/appendix_b3_training_parameters.md` - Training params

## Core Findings Summary

| Finding | Evidence | Strength |
|---------|----------|----------|
| Inverse scaling | 2M beats 2B by 21% | Strong |
| Data-limited regime | 4000 samples / 2B params = severe overfit | Strong |
| h3 most predictable | Best loss across all budgets | Moderate |
| n_heads minimal impact | r=0.047 correlation | Strong |
| Architectures horizon-specific | +132% degradation on transfer | Strong |
| Feature richness hypothesis | Theoretical | Hypothesis |

## Figure Generation

The CSV files in `figures/` are designed for easy plotting with Python/matplotlib or R/ggplot2:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Example: Scaling curve
df = pd.read_csv('figures/fig1_scaling_curve_data.csv', comment='#')
plt.figure(figsize=(10, 6))
plt.semilogx(df['params_M'], df['mean_val_loss'], 'o-')
plt.xlabel('Parameters (Millions)')
plt.ylabel('Validation Loss')
plt.title('Inverse Scaling in Financial Time-Series')
plt.savefig('fig1_scaling_curve.pdf')
```

## LaTeX Integration

Tables can be converted to LaTeX format:

```bash
# Using csvtolatex or similar tool
csvtolatex tables/table1_hpo_summary.csv > table1.tex
```

Or use pandas:
```python
df = pd.read_csv('tables/table1_hpo_summary.csv')
print(df.to_latex(index=False))
```

## Contributing

When adding new analysis:
1. Add timestamped section to `draft_notes.md`
2. Update relevant appendix if methodology changes
3. Add/update CSV files for new data
4. Update this README if structure changes

---

*Last updated: 2026-01-19*
*Phase 6A analysis complete*
