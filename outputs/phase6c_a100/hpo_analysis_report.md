# Cross-Budget HPO Comparison

## Best Results by Budget

| Budget | Best AUC | Best d_model | Best n_layers | Best LR | Best dropout | Best WD |
|--------|----------|--------------|---------------|---------|--------------|---------|
| 2M | 0.7178 | 96 | 2 | 1e-05 | 0.1 | 0.001 |
| 20M | 0.7246 | 64 | 4 | 0.0001 | 0.7 | 0.0 |
| 200M | 0.7147 | 128 | 6 | 1e-05 | 0.3 | 0.0001 |

## Scaling Law Analysis

- 2M: 0.7178
- 20M: 0.7246
- 200M: 0.7147

**Scaling law VIOLATED**: 20M > 2M > 200M
This suggests larger models may be overfitting or need different regularization.

## Parameter Consistency Across Budgets

- **dropout**: Inconsistent - {'2M': 0.1, '20M': 0.7, '200M': 0.3}
- **learning_rate**: Inconsistent - {'2M': 1e-05, '20M': 0.0001, '200M': 1e-05}
- **weight_decay**: Inconsistent - {'2M': 0.001, '20M': 0.0, '200M': 0.0001}

## Probability Collapse by Budget

| Budget | Collapse? | Pred Range | Avg Recall |
|--------|-----------|------------|------------|
| 2M | No | N/A | N/A |
| 20M | No | N/A | N/A |
| 200M | No | N/A | N/A |

---

# HPO Analysis: 2M (a100)

## Summary Statistics

- Total trials: 50
- Best AUC: 0.7178
- Mean AUC: 0.7009
- Std AUC: 0.0152

**Best Configuration:**
  - d_model: 96
  - n_layers: 2
  - n_heads: 8
  - d_ff_ratio: 4
  - learning_rate: 1e-05
  - dropout: 0.1
  - weight_decay: 0.001

## Probability Collapse Analysis

No probability collapse detected.

## Parameter Importance

(Fraction of AUC variance explained by each parameter)

  learning_rate   | ##############                                     | 0.2851
  dropout         | #########                                          | 0.1807
  d_model         | #######                                            | 0.1472
  n_layers        | #####                                              | 0.1038
  weight_decay    | ####                                               | 0.0833
  d_ff_ratio      | ##                                                 | 0.0579
  n_heads         |                                                    | 0.0099

## Parameter Trends

### d_model

- Trend: **increasing**
- Best value: 96 (mean AUC: 0.7045)
- Worst value: 32 (mean AUC: 0.687)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 32 | 0.6870 | 0.0127 | 5 |
| 48 | 0.7028 | 0.0114 | 6 |
| 64 | 0.7010 | 0.0151 | 4 |
| 80 | 0.6933 | 0.0133 | 6 |
| 96 | 0.7045 | 0.0156 | 29 |

### n_layers

- Trend: **decreasing**
- Best value: 2 (mean AUC: 0.7046)
- Worst value: 6 (mean AUC: 0.6905)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 2 | 0.7046 | 0.0160 | 29 |
| 3 | 0.6981 | 0.0183 | 4 |
| 4 | 0.6938 | 0.0108 | 7 |
| 5 | 0.7005 | 0.0130 | 6 |
| 6 | 0.6905 | 0.0122 | 4 |

### n_heads

- Trend: **flat**
- Best value: 8 (mean AUC: 0.7019)
- Worst value: 2 (mean AUC: 0.6977)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 2 | 0.6977 | 0.0160 | 8 |
| 4 | 0.7005 | 0.0114 | 10 |
| 8 | 0.7019 | 0.0163 | 32 |

### learning_rate

- Trend: **decreasing**
- Best value: 5e-05 (mean AUC: 0.7086)
- Worst value: 0.0005 (mean AUC: 0.6822)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 1e-05 | 0.7042 | 0.0147 | 32 |
| 5e-05 | 0.7086 | 0.0065 | 7 |
| 0.0001 | 0.6916 | 0.0143 | 5 |
| 0.0005 | 0.6822 | 0.0079 | 6 |

### dropout

- Trend: **decreasing**
- Best value: 0.1 (mean AUC: 0.7055)
- Worst value: 0.7 (mean AUC: 0.6894)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.1 | 0.7055 | 0.0162 | 32 |
| 0.3 | 0.6970 | 0.0105 | 7 |
| 0.5 | 0.6905 | 0.0064 | 6 |
| 0.7 | 0.6894 | 0.0090 | 5 |

### weight_decay

- Trend: **increasing**
- Best value: 0.001 (mean AUC: 0.7037)
- Worst value: 0.0 (mean AUC: 0.6889)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.0 | 0.6889 | 0.0085 | 5 |
| 1e-05 | 0.6993 | 0.0155 | 10 |
| 0.0001 | 0.7006 | 0.0125 | 7 |
| 0.001 | 0.7037 | 0.0161 | 28 |

## Coverage Matrices

(Count of trials testing each combination)

### d_model × n_layers

| d_model | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- |
| 32 | 4 | 0 | 1 | 0 | 0 |
| 48 | 2 | 0 | 1 | 2 | 1 |
| 64 | 0 | 3 | 1 | 0 | 0 |
| 80 | 2 | 0 | 1 | 3 | 0 |
| 96 | 21 | 1 | 3 | 1 | 3 |

### d_model × n_heads

| d_model | 2 | 4 | 8 |
| --- | --- | --- | --- |
| 32 | 1 | 1 | 3 |
| 48 | 0 | 3 | 3 |
| 64 | 0 | 1 | 3 |
| 80 | 2 | 1 | 3 |
| 96 | 5 | 4 | 20 |

### learning_rate × dropout

| learning_rate | 0.1 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| 1e-05 | 22 | 2 | 4 | 4 |
| 5e-05 | 4 | 2 | 1 | 0 |
| 0.0001 | 3 | 2 | 0 | 0 |
| 0.0005 | 3 | 1 | 1 | 1 |

---

# HPO Analysis: 20M (a100)

## Summary Statistics

- Total trials: 50
- Best AUC: 0.7246
- Mean AUC: 0.7075
- Std AUC: 0.0153

**Best Configuration:**
  - d_model: 64
  - n_layers: 4
  - n_heads: 8
  - d_ff_ratio: 2
  - learning_rate: 0.0001
  - dropout: 0.7
  - weight_decay: 0.0

## Probability Collapse Analysis

No probability collapse detected.

## Parameter Importance

(Fraction of AUC variance explained by each parameter)

  dropout         | ######################                             | 0.4454
  n_layers        | ##################                                 | 0.3626
  d_model         | ################                                   | 0.3323
  weight_decay    | ##########                                         | 0.2028
  n_heads         | #######                                            | 0.1493
  d_ff_ratio      | #####                                              | 0.1096
  learning_rate   | #####                                              | 0.1080

## Parameter Trends

### d_model

- Trend: **decreasing**
- Best value: 64 (mean AUC: 0.7143)
- Worst value: 160 (mean AUC: 0.6899)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 64 | 0.7143 | 0.0115 | 30 |
| 96 | 0.7016 | 0.0080 | 6 |
| 128 | 0.6994 | 0.0156 | 5 |
| 160 | 0.6899 | 0.0109 | 4 |
| 192 | 0.6955 | 0.0230 | 5 |

### n_layers

- Trend: **decreasing**
- Best value: 4 (mean AUC: 0.7155)
- Worst value: 7 (mean AUC: 0.695)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 4 | 0.7155 | 0.0109 | 28 |
| 5 | 0.6951 | 0.0118 | 4 |
| 6 | 0.7013 | 0.0059 | 5 |
| 7 | 0.6950 | 0.0209 | 7 |
| 8 | 0.6979 | 0.0130 | 6 |

### n_heads

- Trend: **increasing**
- Best value: 8 (mean AUC: 0.7106)
- Worst value: 4 (mean AUC: 0.6963)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 4 | 0.6963 | 0.0170 | 11 |
| 8 | 0.7106 | 0.0134 | 39 |

### learning_rate

- Trend: **flat**
- Best value: 0.0001 (mean AUC: 0.7114)
- Worst value: 0.0005 (mean AUC: 0.6998)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 1e-05 | 0.7059 | 0.0108 | 5 |
| 5e-05 | 0.7011 | 0.0135 | 5 |
| 0.0001 | 0.7114 | 0.0141 | 30 |
| 0.0005 | 0.6998 | 0.0189 | 10 |

### dropout

- Trend: **increasing**
- Best value: 0.7 (mean AUC: 0.7144)
- Worst value: 0.3 (mean AUC: 0.6834)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.1 | 0.6995 | 0.0188 | 6 |
| 0.3 | 0.6834 | 0.0172 | 5 |
| 0.5 | 0.6999 | 0.0078 | 7 |
| 0.7 | 0.7144 | 0.0097 | 32 |

### weight_decay

- Trend: **flat**
- Best value: 1e-05 (mean AUC: 0.7125)
- Worst value: 0.0001 (mean AUC: 0.6925)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.0 | 0.7032 | 0.0142 | 8 |
| 1e-05 | 0.7125 | 0.0130 | 30 |
| 0.0001 | 0.6925 | 0.0203 | 6 |
| 0.001 | 0.7028 | 0.0119 | 6 |

## Coverage Matrices

(Count of trials testing each combination)

### d_model × n_layers

| d_model | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| 64 | 21 | 1 | 1 | 5 | 2 |
| 96 | 0 | 1 | 3 | 0 | 2 |
| 128 | 4 | 0 | 0 | 1 | 0 |
| 160 | 0 | 2 | 1 | 0 | 1 |
| 192 | 3 | 0 | 0 | 1 | 1 |

### d_model × n_heads

| d_model | 4 | 8 |
| --- | --- | --- |
| 64 | 4 | 26 |
| 96 | 2 | 4 |
| 128 | 2 | 3 |
| 160 | 1 | 3 |
| 192 | 2 | 3 |

### learning_rate × dropout

| learning_rate | 0.1 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| 1e-05 | 3 | 0 | 1 | 1 |
| 5e-05 | 2 | 0 | 1 | 2 |
| 0.0001 | 1 | 3 | 4 | 22 |
| 0.0005 | 0 | 2 | 1 | 7 |

---

# HPO Analysis: 200M (a100)

## Summary Statistics

- Total trials: 50
- Best AUC: 0.7147
- Mean AUC: 0.7030
- Std AUC: 0.0124

**Best Configuration:**
  - d_model: 128
  - n_layers: 6
  - n_heads: 16
  - d_ff_ratio: 4
  - learning_rate: 1e-05
  - dropout: 0.3
  - weight_decay: 0.0001

## Probability Collapse Analysis

No probability collapse detected.

## Parameter Importance

(Fraction of AUC variance explained by each parameter)

  learning_rate   | ############################                       | 0.5705
  d_model         | ########                                           | 0.1748
  weight_decay    | ######                                             | 0.1313
  n_heads         | #####                                              | 0.1095
  n_layers        | #####                                              | 0.1029
  dropout         | ##                                                 | 0.0599
  d_ff_ratio      | #                                                  | 0.0367

## Parameter Trends

### d_model

- Trend: **decreasing**
- Best value: 128 (mean AUC: 0.7061)
- Worst value: 384 (mean AUC: 0.6863)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 128 | 0.7061 | 0.0105 | 20 |
| 192 | 0.7052 | 0.0062 | 10 |
| 256 | 0.7024 | 0.0154 | 14 |
| 320 | 0.6939 | 0.0057 | 3 |
| 384 | 0.6863 | 0.0172 | 3 |

### n_layers

- Trend: **flat**
- Best value: 6 (mean AUC: 0.7055)
- Worst value: 10 (mean AUC: 0.6944)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 6 | 0.7055 | 0.0134 | 31 |
| 8 | 0.7029 | 0.0079 | 6 |
| 10 | 0.6944 | 0.0117 | 8 |
| 12 | 0.7014 | 0.0027 | 5 |

### n_heads

- Trend: **increasing**
- Best value: 16 (mean AUC: 0.7049)
- Worst value: 8 (mean AUC: 0.6942)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 8 | 0.6942 | 0.0136 | 9 |
| 16 | 0.7049 | 0.0114 | 41 |

### learning_rate

- Trend: **decreasing**
- Best value: 1e-05 (mean AUC: 0.7098)
- Worst value: 0.0001 (mean AUC: 0.688)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 1e-05 | 0.7098 | 0.0048 | 32 |
| 5e-05 | 0.6965 | 0.0127 | 5 |
| 0.0001 | 0.6880 | 0.0151 | 7 |
| 0.0005 | 0.6895 | 0.0091 | 6 |

### dropout

- Trend: **flat**
- Best value: 0.5 (mean AUC: 0.705)
- Worst value: 0.1 (mean AUC: 0.6972)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.1 | 0.6972 | 0.0154 | 9 |
| 0.3 | 0.7047 | 0.0137 | 26 |
| 0.5 | 0.7050 | 0.0041 | 10 |
| 0.7 | 0.7003 | 0.0082 | 5 |

### weight_decay

- Trend: **flat**
- Best value: 0.0001 (mean AUC: 0.7067)
- Worst value: 1e-05 (mean AUC: 0.6955)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.0 | 0.6987 | 0.0130 | 10 |
| 1e-05 | 0.6955 | 0.0174 | 5 |
| 0.0001 | 0.7067 | 0.0112 | 29 |
| 0.001 | 0.6982 | 0.0071 | 6 |

## Coverage Matrices

(Count of trials testing each combination)

### d_model × n_layers

| d_model | 6 | 8 | 10 | 12 |
| --- | --- | --- | --- | --- |
| 128 | 14 | 1 | 3 | 2 |
| 192 | 4 | 5 | 1 | 0 |
| 256 | 11 | 0 | 0 | 3 |
| 320 | 1 | 0 | 2 | 0 |
| 384 | 1 | 0 | 2 | 0 |

### d_model × n_heads

| d_model | 8 | 16 |
| --- | --- | --- |
| 128 | 4 | 16 |
| 192 | 1 | 9 |
| 256 | 2 | 12 |
| 320 | 1 | 2 |
| 384 | 1 | 2 |

### learning_rate × dropout

| learning_rate | 0.1 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| 1e-05 | 4 | 20 | 8 | 0 |
| 5e-05 | 1 | 2 | 1 | 1 |
| 0.0001 | 3 | 2 | 1 | 1 |
| 0.0005 | 1 | 2 | 0 | 3 |
