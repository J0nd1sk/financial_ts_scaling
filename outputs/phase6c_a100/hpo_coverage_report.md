# HPO Coverage Analysis Report

*Generated from Phase 6C A100 HPO experiments*

## Cross-Budget Comparison

### Summary
| Budget | Best AUC | Best d_model | Best n_layers | Best n_heads | Best LR | Best dropout |
|--------|----------|--------------|---------------|--------------|---------|--------------|
| 2M | 0.7178 | 96 | 2 | 8 | 1e-05 | 0.1 |
| 20M | 0.7246 | 64 | 4 | 8 | 0.0001 | 0.7 |
| 200M | 0.7147 | 128 | 6 | 16 | 1e-05 | 0.3 |

### Scaling Law Analysis
- 2M: 0.7178
- 20M: 0.7246
- 200M: 0.7147

**Scaling law VIOLATED**: 20M > 2M > 200M
This suggests larger models may be overfitting or require different training regimes.

### Cross-Budget Pattern Consistency
- **Dropout**: Inconsistent across budgets {'2M': 0.1, '20M': 0.7, '200M': 0.3}
- **Learning rate**: Inconsistent across budgets {'2M': 1e-05, '20M': 0.0001, '200M': 1e-05}

### Trial Efficiency
- **2M**: 9/50 wasted (18.0%)
- **20M**: 16/50 wasted (32.0%)
- **200M**: 18/50 wasted (36.0%)
## 2M Budget Analysis

### Best Configuration
- **Best AUC**: 0.7178
- **Best Trial**: 19
  - d_model: 96
  - n_layers: 2
  - n_heads: 8
  - d_ff_ratio: 4
  - learning_rate: 1e-05
  - dropout: 0.1
  - weight_decay: 0.001

### Convergence Analysis
- Total trials: 50
- Unique configurations: 41
- Repeated configurations: 1
- Wasted trials (repeats): 9

**Most Repeated Configurations:**
  1. 10x: d=96, L=2, h=8

### Value Frequencies
- **d_model**: 32:5, 48:6, 64:4, 80:6, 96:29
- **n_layers**: 2:29, 3:4, 4:7, 5:6, 6:4
- **n_heads**: 2:8, 4:10, 8:32
- **learning_rate**: 1e-05:32, 5e-05:7, 0.0001:5, 0.0005:6
- **dropout**: 0.1:32, 0.3:7, 0.5:6, 0.7:5
- **weight_decay**: 0.0:5, 1e-05:10, 0.0001:7, 0.001:28

### Coverage Gaps
- **Architecture combinations**: 26/75 (34.7% coverage)
  - Untested examples (d_model, n_layers, n_heads): [(80, 4, 8), (64, 2, 8), (48, 3, 4), (64, 5, 4), (32, 3, 4)]
- **Training HP combinations**: 25/64 (39.1% coverage)

### Architecture Patterns
- Width↔AUC correlation: 0.3752
- Depth↔AUC correlation: -0.2736
- Width/Depth ratio↔AUC: 0.5002

**Average AUC by d_model:**
  - d_model=32: 0.6870
  - d_model=48: 0.7028
  - d_model=64: 0.7010
  - d_model=80: 0.6933
  - d_model=96: 0.7045

**Average AUC by n_layers:**
  - n_layers=2: 0.7046
  - n_layers=3: 0.6981
  - n_layers=4: 0.6938
  - n_layers=5: 0.7005
  - n_layers=6: 0.6905

### Training HP Impact

**learning_rate:**
  - 5e-05: mean=0.7086 (n=7)
  - 1e-05: mean=0.7042 (n=32)
  - 0.0001: mean=0.6916 (n=5)
  - 0.0005: mean=0.6822 (n=6)

**dropout:**
  - 0.1: mean=0.7055 (n=32)
  - 0.3: mean=0.6970 (n=7)
  - 0.5: mean=0.6905 (n=6)
  - 0.7: mean=0.6894 (n=5)

**weight_decay:**
  - 0.001: mean=0.7037 (n=28)
  - 0.0001: mean=0.7006 (n=7)
  - 1e-05: mean=0.6993 (n=10)
  - 0.0: mean=0.6889 (n=5)

## 20M Budget Analysis

### Best Configuration
- **Best AUC**: 0.7246
- **Best Trial**: 47
  - d_model: 64
  - n_layers: 4
  - n_heads: 8
  - d_ff_ratio: 2
  - learning_rate: 0.0001
  - dropout: 0.7
  - weight_decay: 0.0

### Convergence Analysis
- Total trials: 50
- Unique configurations: 34
- Repeated configurations: 4
- Wasted trials (repeats): 16

**Most Repeated Configurations:**
  1. 12x: d=64, L=4, h=8
  2. 4x: d=64, L=4, h=8
  3. 2x: d=128, L=4, h=8

### Value Frequencies
- **d_model**: 64:30, 96:6, 128:5, 160:4, 192:5
- **n_layers**: 4:28, 5:4, 6:5, 7:7, 8:6
- **n_heads**: 4:11, 8:39
- **learning_rate**: 1e-05:5, 5e-05:5, 0.0001:30, 0.0005:10
- **dropout**: 0.1:6, 0.3:5, 0.5:7, 0.7:32
- **weight_decay**: 0.0:8, 1e-05:30, 0.0001:6, 0.001:6

### Coverage Gaps
- **Architecture combinations**: 20/50 (40.0% coverage)
  - Untested examples (d_model, n_layers, n_heads): [(128, 8, 8), (64, 5, 4), (96, 7, 4), (160, 8, 4), (192, 5, 8)]
- **Training HP combinations**: 24/64 (37.5% coverage)

### Architecture Patterns
- Width↔AUC correlation: -0.6127
- Depth↔AUC correlation: -0.5282
- Width/Depth ratio↔AUC: -0.3446

**Average AUC by d_model:**
  - d_model=64: 0.7143
  - d_model=96: 0.7016
  - d_model=128: 0.6994
  - d_model=160: 0.6899
  - d_model=192: 0.6955

**Average AUC by n_layers:**
  - n_layers=4: 0.7155
  - n_layers=5: 0.6951
  - n_layers=6: 0.7013
  - n_layers=7: 0.6950
  - n_layers=8: 0.6979

### Training HP Impact

**learning_rate:**
  - 0.0001: mean=0.7114 (n=30)
  - 1e-05: mean=0.7059 (n=5)
  - 5e-05: mean=0.7011 (n=5)
  - 0.0005: mean=0.6998 (n=10)

**dropout:**
  - 0.7: mean=0.7144 (n=32)
  - 0.5: mean=0.6999 (n=7)
  - 0.1: mean=0.6995 (n=6)
  - 0.3: mean=0.6834 (n=5)

**weight_decay:**
  - 1e-05: mean=0.7125 (n=30)
  - 0.0: mean=0.7032 (n=8)
  - 0.001: mean=0.7028 (n=6)
  - 0.0001: mean=0.6925 (n=6)

## 200M Budget Analysis

### Best Configuration
- **Best AUC**: 0.7147
- **Best Trial**: 40
  - d_model: 128
  - n_layers: 6
  - n_heads: 16
  - d_ff_ratio: 4
  - learning_rate: 1e-05
  - dropout: 0.3
  - weight_decay: 0.0001

### Convergence Analysis
- Total trials: 50
- Unique configurations: 32
- Repeated configurations: 6
- Wasted trials (repeats): 18

**Most Repeated Configurations:**
  1. 8x: d=256, L=6, h=16
  2. 5x: d=192, L=8, h=16
  3. 4x: d=128, L=6, h=16

### Value Frequencies
- **d_model**: 128:20, 192:10, 256:14, 320:3, 384:3
- **n_layers**: 6:31, 8:6, 10:8, 12:5
- **n_heads**: 8:9, 16:41
- **learning_rate**: 1e-05:32, 5e-05:5, 0.0001:7, 0.0005:6
- **dropout**: 0.1:9, 0.3:26, 0.5:10, 0.7:5
- **weight_decay**: 0.0:10, 1e-05:5, 0.0001:29, 0.001:6

### Coverage Gaps
- **Architecture combinations**: 17/40 (42.5% coverage)
  - Untested examples (d_model, n_layers, n_heads): [(128, 8, 8), (256, 12, 8), (320, 12, 16), (384, 6, 8), (384, 8, 8)]
- **Training HP combinations**: 22/64 (34.4% coverage)

### Architecture Patterns
- Width↔AUC correlation: -0.5101
- Depth↔AUC correlation: -0.2491
- Width/Depth ratio↔AUC: -0.3999

**Average AUC by d_model:**
  - d_model=128: 0.7061
  - d_model=192: 0.7052
  - d_model=256: 0.7024
  - d_model=320: 0.6939
  - d_model=384: 0.6863

**Average AUC by n_layers:**
  - n_layers=6: 0.7055
  - n_layers=8: 0.7029
  - n_layers=10: 0.6944
  - n_layers=12: 0.7014

### Training HP Impact

**learning_rate:**
  - 1e-05: mean=0.7098 (n=32)
  - 5e-05: mean=0.6965 (n=5)
  - 0.0005: mean=0.6895 (n=6)
  - 0.0001: mean=0.6880 (n=7)

**dropout:**
  - 0.5: mean=0.7050 (n=10)
  - 0.3: mean=0.7047 (n=26)
  - 0.7: mean=0.7003 (n=5)
  - 0.1: mean=0.6972 (n=9)

**weight_decay:**
  - 0.0001: mean=0.7067 (n=29)
  - 0.0: mean=0.6987 (n=10)
  - 0.001: mean=0.6982 (n=6)
  - 1e-05: mean=0.6955 (n=5)

## Methodology Recommendations

### Issues Identified
1. **High trial redundancy**: 43 total wasted trials across budgets
2. **Missing metrics**: Precision, recall, pred_range not captured in HPO
3. **No forced extreme trials**: Pure TPE converges to local optima quickly
4. **Inconsistent patterns**: Different optimal HPs per budget suggests search space issues

### Recommended Improvements
1. **Two-phase HPO**: Forced extremes (6 trials) + TPE exploration (44 trials)
2. **Capture all metrics**: Save precision, recall, pred_range per trial
3. **Coverage-aware sampling**: Ensure parameter combinations are tested
4. **Cross-budget validation**: Test best configs from smaller budgets on larger