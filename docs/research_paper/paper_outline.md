# Research Paper Outline

**THIS IS JUST ONE POSSIBLE OUTLINE AND MAY NEED MAJOR REVISION**
*Consider this a draft outline and definitely not necessarily a permanent structure for our research paper*I

## Working Title

"Scaling Laws for Financial Time-Series Transformers: The Role of Feature Richness"

## Core Thesis

Input richness (feature dimensionality) matters as much as model size for neural scaling laws to emerge in financial time-series prediction.

## Key Arguments

1. **Parameter scaling alone fails with limited features**
   - Evidence: Phase 6A results (200M → 2B shows no improvement with 20 features)
   - Interpretation: Insufficient input complexity starves larger models

2. **Feature richness unlocks parameter scaling**
   - Evidence: Phase 6C results (expected - 2000 features should show scaling)
   - Interpretation: More features = more learnable relationships = O(n²) growth

3. **Sample scaling provides additional gains**
   - Evidence: Phase 6D results
   - Interpretation: Classic scaling law behavior with sufficient features

## Proposed Sections

1. Abstract
2. Introduction
   - Neural scaling laws background
   - Financial ML challenges
   - Research questions
3. Related Work
   - Scaling laws (Kaplan et al., Hoffmann et al.)
   - Financial ML with transformers
   - Technical analysis and feature engineering
4. Methodology
   - PatchTST architecture
   - Feature tiers (20/200/2000)
   - Experimental protocol
   - HPO approach
5. Results
   - Phase 6A: Parameter scaling with limited features
   - Phase 6B: Horizon effects
   - Phase 6C: Feature × Parameter interaction
   - Phase 6D: Data scaling
6. Discussion
   - Why feature richness matters
   - Effective data = samples × features × relationships
   - Practical implications
7. Conclusion
8. Appendices
   - A: Feature dictionaries (20/200/2000)
   - B: HPO methodology and results
   - C: Architecture configurations
   - D: Hardware and reproducibility

## Format

- Primary: Markdown with LaTeX math notation
- Conversion: pandoc to LaTeX for final submission
- Target venues: NeurIPS, ICML, or quantitative finance journals
