# Cross-Budget HPO Comparison

## Best Results by Budget

| Budget | Best AUC | Best d_model | Best n_layers | Best LR | Best dropout | Best WD |
|--------|----------|--------------|---------------|---------|--------------|---------|
| 2M | 0.7302 | 96 | 2 | 0.0005 | 0.5 | 1e-05 |
| 20M | 0.7315 | N/A | N/A | N/A | N/A | N/A |
| 200M | 0.7294 | 128 | 6 | 0.0001 | 0.5 | 0.0001 |

## Scaling Law Analysis

- 2M: 0.7302
- 20M: 0.7315
- 200M: 0.7294

**Scaling law VIOLATED**: 20M > 2M > 200M
This suggests larger models may be overfitting or need different regularization.

## Parameter Consistency Across Budgets

- **dropout**: Consistent - {'2M': 0.5, '200M': 0.5}
- **learning_rate**: Inconsistent - {'2M': 0.0005, '200M': 0.0001}
- **weight_decay**: Inconsistent - {'2M': 1e-05, '200M': 0.0001}

## Probability Collapse by Budget

| Budget | Collapse? | Pred Range | Avg Recall |
|--------|-----------|------------|------------|
| 2M | No | N/A | N/A |
| 20M | No | N/A | N/A |
| 200M | No | N/A | N/A |

---

# HPO Analysis: 2M (a50)

## Summary Statistics

- Total trials: 50
- Best AUC: 0.7302
- Mean AUC: 0.7128
- Std AUC: 0.0214

**Best Configuration:**
  - d_model: 96
  - n_layers: 2
  - n_heads: 8
  - d_ff_ratio: 2
  - learning_rate: 0.0005
  - dropout: 0.5
  - weight_decay: 1e-05

## Probability Collapse Analysis

No probability collapse detected.

## Parameter Importance

(Fraction of AUC variance explained by each parameter)

  d_model         | #################                                  | 0.3584
  learning_rate   | ###############                                    | 0.3040
  dropout         | ###########                                        | 0.2258
  n_layers        | ##########                                         | 0.2102
  n_heads         | ########                                           | 0.1604
  weight_decay    | #####                                              | 0.1046
  d_ff_ratio      |                                                    | 0.0003

## Parameter Trends

### d_model

- Trend: **increasing**
- Best value: 48 (mean AUC: 0.7236)
- Worst value: 32 (mean AUC: 0.6779)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 32 | 0.6779 | 0.0499 | 5 |
| 48 | 0.7236 | 0.0046 | 18 |
| 64 | 0.7119 | 0.0108 | 8 |
| 80 | 0.7144 | 0.0031 | 5 |
| 96 | 0.7113 | 0.0155 | 14 |

### n_layers

- Trend: **decreasing**
- Best value: 2 (mean AUC: 0.72)
- Worst value: 4 (mean AUC: 0.6946)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 2 | 0.7200 | 0.0129 | 22 |
| 3 | 0.7160 | 0.0090 | 5 |
| 4 | 0.6946 | 0.0379 | 10 |
| 5 | 0.7165 | 0.0094 | 9 |
| 6 | 0.7061 | 0.0089 | 4 |

### n_heads

- Trend: **increasing**
- Best value: 4 (mean AUC: 0.7179)
- Worst value: 2 (mean AUC: 0.6899)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 2 | 0.6899 | 0.0485 | 6 |
| 4 | 0.7179 | 0.0117 | 18 |
| 8 | 0.7145 | 0.0138 | 26 |

### learning_rate

- Trend: **increasing**
- Best value: 0.0001 (mean AUC: 0.7214)
- Worst value: 1e-05 (mean AUC: 0.6822)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 1e-05 | 0.6822 | 0.0469 | 6 |
| 5e-05 | 0.7123 | 0.0089 | 11 |
| 0.0001 | 0.7214 | 0.0068 | 17 |
| 0.0005 | 0.7154 | 0.0141 | 16 |

### dropout

- Trend: **decreasing**
- Best value: 0.5 (mean AUC: 0.7199)
- Worst value: 0.7 (mean AUC: 0.6854)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.1 | 0.7140 | 0.0074 | 7 |
| 0.3 | 0.7091 | 0.0134 | 13 |
| 0.5 | 0.7199 | 0.0119 | 25 |
| 0.7 | 0.6854 | 0.0535 | 5 |

### weight_decay

- Trend: **flat**
- Best value: 0.001 (mean AUC: 0.7197)
- Worst value: 0.0001 (mean AUC: 0.7006)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.0 | 0.7153 | 0.0153 | 16 |
| 1e-05 | 0.7125 | 0.0102 | 9 |
| 0.0001 | 0.7006 | 0.0372 | 11 |
| 0.001 | 0.7197 | 0.0124 | 14 |

## Coverage Matrices

(Count of trials testing each combination)

### d_model × n_layers

| d_model | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- |
| 32 | 1 | 0 | 4 | 0 | 0 |
| 48 | 12 | 2 | 1 | 2 | 1 |
| 64 | 3 | 1 | 2 | 0 | 2 |
| 80 | 1 | 1 | 1 | 2 | 0 |
| 96 | 5 | 1 | 2 | 5 | 1 |

### d_model × n_heads

| d_model | 2 | 4 | 8 |
| --- | --- | --- | --- |
| 32 | 1 | 2 | 2 |
| 48 | 0 | 8 | 10 |
| 64 | 1 | 1 | 6 |
| 80 | 3 | 1 | 1 |
| 96 | 1 | 6 | 7 |

### learning_rate × dropout

| learning_rate | 0.1 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| 1e-05 | 2 | 2 | 1 | 1 |
| 5e-05 | 2 | 7 | 1 | 1 |
| 0.0001 | 2 | 3 | 10 | 2 |
| 0.0005 | 1 | 1 | 13 | 1 |

## Forced Extreme Trials

| Trial | Extreme Type | AUC | d_model | n_layers | n_heads |
|-------|--------------|-----|---------|----------|---------|
| 0 | min_d_model | 0.7117 | 32 | 4 | 8 |
| 1 | max_d_model | 0.7156 | 96 | 4 | 8 |
| 2 | min_n_layers | 0.7273 | 64 | 2 | 8 |
| 3 | max_n_layers | 0.7123 | 64 | 6 | 8 |
| 4 | min_n_heads | 0.7078 | 64 | 4 | 2 |
| 5 | max_n_heads | 0.7099 | 64 | 4 | 8 |

---

# HPO Analysis: 20M (a50)

## Summary Statistics

- Total trials: 50
- Best AUC: 0.7315
- Mean AUC: 0.7115
- Std AUC: 0.0172

## Probability Collapse Analysis

No probability collapse detected.

## Parameter Importance

(Fraction of AUC variance explained by each parameter)

  d_model         | ###################                                | 0.3828
  weight_decay    | ###########                                        | 0.2236
  learning_rate   | #######                                            | 0.1463
  n_layers        | #####                                              | 0.1017
  dropout         | ####                                               | 0.0891
  n_heads         | #                                                  | 0.0225
  d_ff_ratio      |                                                    | 0.0065

## Parameter Trends

### d_model

- Trend: **flat**
- Best value: 128 (mean AUC: 0.7211)
- Worst value: 160 (mean AUC: 0.6908)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 64 | 0.7045 | 0.0086 | 7 |
| 96 | 0.7118 | 0.0106 | 7 |
| 128 | 0.7211 | 0.0112 | 24 |
| 160 | 0.6908 | 0.0087 | 5 |
| 192 | 0.6998 | 0.0274 | 7 |

### n_layers

- Trend: **decreasing**
- Best value: 4 (mean AUC: 0.7174)
- Worst value: 5 (mean AUC: 0.7022)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 4 | 0.7174 | 0.0119 | 12 |
| 5 | 0.7022 | 0.0167 | 5 |
| 6 | 0.7156 | 0.0119 | 10 |
| 7 | 0.7106 | 0.0219 | 17 |
| 8 | 0.7028 | 0.0164 | 6 |

### n_heads

- Trend: **flat**
- Best value: 8 (mean AUC: 0.7139)
- Worst value: 4 (mean AUC: 0.7088)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 4 | 0.7088 | 0.0210 | 24 |
| 8 | 0.7139 | 0.0126 | 26 |

### learning_rate

- Trend: **decreasing**
- Best value: 5e-05 (mean AUC: 0.7155)
- Worst value: 0.0005 (mean AUC: 0.698)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 1e-05 | 0.7099 | 0.0139 | 7 |
| 5e-05 | 0.7155 | 0.0140 | 18 |
| 0.0001 | 0.7153 | 0.0156 | 16 |
| 0.0005 | 0.6980 | 0.0227 | 9 |

### dropout

- Trend: **increasing**
- Best value: 0.7 (mean AUC: 0.7168)
- Worst value: 0.1 (mean AUC: 0.7037)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.1 | 0.7037 | 0.0198 | 7 |
| 0.3 | 0.7078 | 0.0244 | 13 |
| 0.5 | 0.7087 | 0.0105 | 7 |
| 0.7 | 0.7168 | 0.0116 | 23 |

### weight_decay

- Trend: **increasing**
- Best value: 0.001 (mean AUC: 0.7187)
- Worst value: 0.0 (mean AUC: 0.6941)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.0 | 0.6941 | 0.0143 | 6 |
| 1e-05 | 0.7053 | 0.0172 | 7 |
| 0.0001 | 0.7102 | 0.0216 | 14 |
| 0.001 | 0.7187 | 0.0102 | 23 |

## Coverage Matrices

(Count of trials testing each combination)

### d_model × n_layers

| d_model | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| 64 | 2 | 0 | 1 | 3 | 1 |
| 96 | 1 | 0 | 2 | 2 | 2 |
| 128 | 8 | 1 | 3 | 11 | 1 |
| 160 | 0 | 3 | 1 | 0 | 1 |
| 192 | 1 | 1 | 3 | 1 | 1 |

### d_model × n_heads

| d_model | 4 | 8 |
| --- | --- | --- |
| 64 | 3 | 4 |
| 96 | 3 | 4 |
| 128 | 13 | 11 |
| 160 | 2 | 3 |
| 192 | 3 | 4 |

### learning_rate × dropout

| learning_rate | 0.1 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| 1e-05 | 3 | 1 | 2 | 1 |
| 5e-05 | 2 | 7 | 0 | 9 |
| 0.0001 | 2 | 2 | 2 | 10 |
| 0.0005 | 0 | 3 | 3 | 3 |

## Forced Extreme Trials

| Trial | Extreme Type | AUC | d_model | n_layers | n_heads |
|-------|--------------|-----|---------|----------|---------|
| 0 | min_d_model | 0.7123 | 64 | 6 | 8 |
| 1 | max_d_model | 0.7175 | 192 | 6 | 8 |
| 2 | min_n_layers | 0.7244 | 128 | 4 | 8 |
| 3 | max_n_layers | 0.7203 | 128 | 8 | 8 |
| 4 | min_n_heads | 0.7294 | 128 | 6 | 4 |
| 5 | max_n_heads | 0.7315 | 128 | 6 | 8 |

---

# HPO Analysis: 200M (a50)

## Summary Statistics

- Total trials: 50
- Best AUC: 0.7294
- Mean AUC: 0.7077
- Std AUC: 0.0204

**Best Configuration:**
  - d_model: 128
  - n_layers: 6
  - n_heads: 16
  - d_ff_ratio: 4
  - learning_rate: 0.0001
  - dropout: 0.5
  - weight_decay: 0.0001

## Probability Collapse Analysis

No probability collapse detected.

## Parameter Importance

(Fraction of AUC variance explained by each parameter)

  d_model         | ##########################                         | 0.5353
  dropout         | #######################                            | 0.4724
  learning_rate   | #############                                      | 0.2625
  weight_decay    | #######                                            | 0.1584
  n_layers        | ######                                             | 0.1347
  n_heads         | ######                                             | 0.1301
  d_ff_ratio      |                                                    | 0.0039

## Parameter Trends

### d_model

- Trend: **decreasing**
- Best value: 128 (mean AUC: 0.7216)
- Worst value: 256 (mean AUC: 0.6877)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 128 | 0.7216 | 0.0112 | 26 |
| 192 | 0.7012 | 0.0126 | 6 |
| 256 | 0.6877 | 0.0139 | 8 |
| 320 | 0.6879 | 0.0147 | 4 |
| 384 | 0.6940 | 0.0260 | 6 |

### n_layers

- Trend: **decreasing**
- Best value: 6 (mean AUC: 0.7141)
- Worst value: 10 (mean AUC: 0.6977)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 6 | 0.7141 | 0.0220 | 29 |
| 8 | 0.7015 | 0.0092 | 4 |
| 10 | 0.6977 | 0.0167 | 11 |
| 12 | 0.6997 | 0.0150 | 6 |

### n_heads

- Trend: **increasing**
- Best value: 16 (mean AUC: 0.7123)
- Worst value: 8 (mean AUC: 0.6959)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 8 | 0.6959 | 0.0177 | 14 |
| 16 | 0.7123 | 0.0198 | 36 |

### learning_rate

- Trend: **decreasing**
- Best value: 0.0001 (mean AUC: 0.7161)
- Worst value: 5e-05 (mean AUC: 0.6911)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 1e-05 | 0.7099 | 0.0078 | 6 |
| 5e-05 | 0.6911 | 0.0171 | 11 |
| 0.0001 | 0.7161 | 0.0200 | 27 |
| 0.0005 | 0.6984 | 0.0163 | 6 |

### dropout

- Trend: **increasing**
- Best value: 0.5 (mean AUC: 0.7181)
- Worst value: 0.1 (mean AUC: 0.6757)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.1 | 0.6757 | 0.0165 | 5 |
| 0.3 | 0.6950 | 0.0164 | 11 |
| 0.5 | 0.7181 | 0.0158 | 27 |
| 0.7 | 0.7109 | 0.0074 | 7 |

### weight_decay

- Trend: **increasing**
- Best value: 0.001 (mean AUC: 0.7128)
- Worst value: 1e-05 (mean AUC: 0.6917)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.0 | 0.6931 | 0.0164 | 6 |
| 1e-05 | 0.6917 | 0.0202 | 5 |
| 0.0001 | 0.7119 | 0.0209 | 33 |
| 0.001 | 0.7128 | 0.0058 | 6 |

## Coverage Matrices

(Count of trials testing each combination)

### d_model × n_layers

| d_model | 6 | 8 | 10 | 12 |
| --- | --- | --- | --- | --- |
| 128 | 18 | 2 | 3 | 3 |
| 192 | 3 | 1 | 1 | 1 |
| 256 | 3 | 1 | 2 | 2 |
| 320 | 2 | 0 | 2 | 0 |
| 384 | 3 | 0 | 3 | 0 |

### d_model × n_heads

| d_model | 8 | 16 |
| --- | --- | --- |
| 128 | 4 | 22 |
| 192 | 4 | 2 |
| 256 | 4 | 4 |
| 320 | 0 | 4 |
| 384 | 2 | 4 |

### learning_rate × dropout

| learning_rate | 0.1 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| 1e-05 | 0 | 2 | 3 | 1 |
| 5e-05 | 3 | 6 | 1 | 1 |
| 0.0001 | 1 | 2 | 21 | 3 |
| 0.0005 | 1 | 1 | 2 | 2 |

## Forced Extreme Trials

| Trial | Extreme Type | AUC | d_model | n_layers | n_heads |
|-------|--------------|-----|---------|----------|---------|
| 0 | min_d_model | 0.7219 | 128 | 10 | 8 |
| 1 | max_d_model | 0.6725 | 384 | 10 | 8 |
| 2 | min_n_layers | 0.6908 | 256 | 6 | 8 |
| 3 | max_n_layers | 0.6711 | 256 | 12 | 8 |
| 4 | min_n_heads | 0.6939 | 256 | 10 | 8 |
| 5 | max_n_heads | 0.6891 | 256 | 10 | 16 |
