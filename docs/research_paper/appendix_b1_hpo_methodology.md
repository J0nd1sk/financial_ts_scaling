# Appendix B.1: Hyperparameter Optimization Methodology

## Overview

This appendix documents the development of the hyperparameter optimization (HPO) methodology used in this study. The infrastructure was developed iteratively over approximately four weeks (December 11, 2025 – January 5, 2026) using an AI coding assistant for implementation. This section describes the methodology evolution, emphasizing corrections made during development that were essential for scientific validity.

## Initial Approach and Critical Correction

The initial HPO implementation searched only training parameters: learning rate, epochs, weight decay, warmup steps, and dropout. Architecture parameters (model dimension, layer count, attention heads) were fixed.

This approach was fundamentally flawed for scaling law research. The research question requires identifying the optimal architecture within each parameter budget, not merely optimizing training dynamics for an arbitrary architecture. I identified this gap during code review and redirected the implementation toward architectural search.

The corrected approach pre-computes a grid of valid architectures per budget (within ±25% of target parameter count), then jointly searches architecture and training parameters. This ensures that reported results reflect the best achievable performance at each scale, not the performance of a potentially suboptimal fixed architecture.

## Methodology Refinements

Six significant corrections were made during development:

**1. Architecture Search (Dec 11-12).** Added architectural parameters to the search space: model dimension (64–2048), layer count (2–256), and attention heads (2–32). Implemented parameter count estimation to filter architectures by budget.

**2. Forced Extreme Testing (Dec 13).** Initial random sampling risked missing boundary configurations. Modified the search to explicitly test minimum and maximum values of each architectural dimension in the first six trials before Bayesian optimization proceeds.

**3. Search Space Expansion (Dec 13).** Audit revealed gaps in the layer count grid for the 20M budget. Added intermediate values (L=160, 180) to ensure coverage of valid configurations near budget boundaries.

**4. Memory-Aware Batch Sizing (Dec 26-29).** Large architectures (d≥1024, L≥192) exceeded available memory with standard batch sizes. Implemented dynamic batch sizing based on architecture dimensions, with gradient accumulation to maintain effective batch size for training stability.

**5. Diversity Enhancement (Jan 3).** The Bayesian optimizer converged prematurely to similar configurations. Increased random exploration trials from 10 to 20 and added logic to force hyperparameter variation when the same architecture is sampled repeatedly.

**6. Process Management (Jan 5).** Background processes on macOS were throttled during extended HPO runs, causing 12× slowdown. Implemented process management to prevent operating system interference with compute-intensive trials.

## Final Methodology

### HPO Structure

Separate HPO studies are conducted for each combination of parameter budget and prediction horizon: 4 budgets (2M, 20M, 200M, 2B) × 3 horizons (1-day, 3-day, 5-day) = 12 independent studies. This structure allows investigation of whether optimal architectures vary across budgets and whether they transfer across prediction horizons.

### Data Splits

Data is divided using hybrid chunk-based splits rather than pure chronological splits. Validation and test sets consist of non-overlapping temporal chunks (61 days each, equal to context length plus prediction horizon), randomly assigned. The training set uses sliding windows over remaining data to maximize sample count while ensuring no overlap with validation or test chunks. The split ratio is 70% train, 15% validation, 15% test.

To accelerate HPO iteration, each trial trains on 30% of the training data. Final model training after HPO uses 100% of training data; performance improvement from the additional data is expected.

### Configuration

The production HPO configuration incorporates all corrections:

| Component | Specification |
|-----------|---------------|
| Studies | 12 (4 budgets × 3 horizons) |
| Trials per study | 50 |
| Training data per trial | 30% of training set |
| Search algorithm | TPE (Tree-structured Parzen Estimator) with 20 startup trials |
| Architecture search | Pre-computed grid filtered by budget (±25% tolerance) |
| Training parameters | Learning rate (1e-5 to 1e-3), epochs (25–100), dropout (0.1–0.3) |
| Extreme testing | First 6 trials test min/max of d_model, n_layers, n_heads |
| Early stopping | Patience=10 epochs, min_delta=0.001 |
| Batch sizing | Dynamic based on architecture memory footprint |

Validation loss serves as the optimization objective. The test set is reserved for final evaluation and not used during HPO.

## Lessons Learned

Three observations from this development process merit discussion:

First, code that executes successfully is not necessarily scientifically valid. The initial training-parameter-only HPO ran without errors and produced results, but those results would have been meaningless for scaling law analysis. Human oversight during code review was essential for identifying this methodological flaw.

Second, scaling experiments surface implementation challenges absent at smaller scales. Memory constraints, search space coverage, and optimizer convergence became significant issues only at the 200M and 2B parameter budgets. Methodology must be validated across the full experimental range, not just at convenient small scales.

Third, iterative refinement is expected. The six corrections described above were not failures but rather the natural result of applying systematic validation to an evolving methodology. Documenting these corrections provides transparency about the research process.

## Reproducibility

All HPO scripts, architecture grids, and configuration files are included in the supplementary materials. The implementation history, including commit hashes for each correction, is preserved in the project repository.
