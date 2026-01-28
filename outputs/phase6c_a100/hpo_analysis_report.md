# Cross-Budget HPO Comparison

## Best Results by Budget

| Budget | Best AUC | Best d_model | Best n_layers | Best LR | Best dropout | Best WD |
|--------|----------|--------------|---------------|---------|--------------|---------|
| 2M | 0.7173 | 96 | 2 | 1e-05 | 0.1 | 1e-05 |
| 20M | 0.7189 | 64 | 4 | 0.0005 | 0.7 | 0.001 |
| 200M | 0.7152 | 320 | 12 | 1e-05 | 0.5 | 1e-05 |

## Scaling Law Analysis

- 2M: 0.7173
- 20M: 0.7189
- 200M: 0.7152

**Scaling law VIOLATED**: 20M > 2M > 200M
This suggests larger models may be overfitting or need different regularization.

## Parameter Consistency Across Budgets

- **dropout**: Inconsistent - {'2M': 0.1, '20M': 0.7, '200M': 0.5}
- **learning_rate**: Inconsistent - {'2M': 1e-05, '20M': 0.0005, '200M': 1e-05}
- **weight_decay**: Inconsistent - {'2M': 1e-05, '20M': 0.001, '200M': 1e-05}

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
- Best AUC: 0.7173
- Mean AUC: 0.7063
- Std AUC: 0.0139

**Best Configuration:**
  - d_model: 96
  - n_layers: 2
  - n_heads: 8
  - d_ff_ratio: 4
  - learning_rate: 1e-05
  - dropout: 0.1
  - weight_decay: 1e-05

## Probability Collapse Analysis

No probability collapse detected.

## Parameter Importance

(Fraction of AUC variance explained by each parameter)

  dropout         | #################                                  | 0.3592
  n_layers        | #################                                  | 0.3513
  d_model         | #################                                  | 0.3413
  d_ff_ratio      | ###############                                    | 0.3020
  learning_rate   | ###########                                        | 0.2342
  weight_decay    | ###########                                        | 0.2241
  n_heads         | ##########                                         | 0.2068

## Parameter Trends

### d_model

- Trend: **increasing**
- Best value: 64 (mean AUC: 0.7108)
- Worst value: 32 (mean AUC: 0.6847)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 32 | 0.6847 | 0.0230 | 5 |
| 48 | 0.7003 | 0.0138 | 5 |
| 64 | 0.7108 | 0.0077 | 10 |
| 80 | 0.7022 | 0.0062 | 5 |
| 96 | 0.7107 | 0.0104 | 25 |

### n_layers

- Trend: **flat**
- Best value: 6 (mean AUC: 0.7143)
- Worst value: 4 (mean AUC: 0.6917)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 2 | 0.7111 | 0.0108 | 25 |
| 3 | 0.7064 | 0.0020 | 3 |
| 4 | 0.6917 | 0.0184 | 10 |
| 5 | 0.6996 | 0.0061 | 5 |
| 6 | 0.7143 | 0.0019 | 7 |

### n_heads

- Trend: **increasing**
- Best value: 4 (mean AUC: 0.7093)
- Worst value: 2 (mean AUC: 0.6892)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 2 | 0.6892 | 0.0230 | 6 |
| 4 | 0.7093 | 0.0092 | 16 |
| 8 | 0.7082 | 0.0114 | 28 |

### learning_rate

- Trend: **decreasing**
- Best value: 1e-05 (mean AUC: 0.7107)
- Worst value: 0.0005 (mean AUC: 0.6882)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 1e-05 | 0.7107 | 0.0148 | 28 |
| 5e-05 | 0.7043 | 0.0072 | 12 |
| 0.0001 | 0.7039 | 0.0026 | 5 |
| 0.0005 | 0.6882 | 0.0131 | 5 |

### dropout

- Trend: **decreasing**
- Best value: 0.1 (mean AUC: 0.7124)
- Worst value: 0.7 (mean AUC: 0.6849)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.1 | 0.7124 | 0.0086 | 29 |
| 0.3 | 0.7003 | 0.0114 | 12 |
| 0.5 | 0.7017 | 0.0108 | 5 |
| 0.7 | 0.6849 | 0.0257 | 4 |

### weight_decay

- Trend: **flat**
- Best value: 1e-05 (mean AUC: 0.7119)
- Worst value: 0.001 (mean AUC: 0.6953)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.0 | 0.7017 | 0.0128 | 5 |
| 1e-05 | 0.7119 | 0.0086 | 28 |
| 0.0001 | 0.6995 | 0.0180 | 12 |
| 0.001 | 0.6953 | 0.0160 | 5 |

## Coverage Matrices

(Count of trials testing each combination)

### d_model × n_layers

| d_model | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- |
| 32 | 0 | 0 | 4 | 1 | 0 |
| 48 | 2 | 0 | 1 | 2 | 0 |
| 64 | 1 | 0 | 2 | 0 | 7 |
| 80 | 1 | 1 | 1 | 2 | 0 |
| 96 | 21 | 2 | 2 | 0 | 0 |

### d_model × n_heads

| d_model | 2 | 4 | 8 |
| --- | --- | --- | --- |
| 32 | 1 | 1 | 3 |
| 48 | 0 | 3 | 2 |
| 64 | 1 | 6 | 3 |
| 80 | 3 | 1 | 1 |
| 96 | 1 | 5 | 19 |

### learning_rate × dropout

| learning_rate | 0.1 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| 1e-05 | 24 | 1 | 2 | 1 |
| 5e-05 | 3 | 7 | 1 | 1 |
| 0.0001 | 1 | 3 | 0 | 1 |
| 0.0005 | 1 | 1 | 2 | 1 |

## Forced Extreme Trials

| Trial | Extreme Type | AUC | d_model | n_layers | n_heads |
|-------|--------------|-----|---------|----------|---------|
| 0 | min_d_model | 0.7094 | 32 | 4 | 8 |
| 1 | max_d_model | 0.7063 | 96 | 4 | 8 |
| 2 | min_n_layers | 0.7148 | 64 | 2 | 8 |
| 3 | max_n_layers | 0.7101 | 64 | 6 | 8 |
| 4 | min_n_heads | 0.6971 | 64 | 4 | 2 |
| 5 | max_n_heads | 0.6961 | 64 | 4 | 8 |

---

# HPO Analysis: 20M (a100)

## Summary Statistics

- Total trials: 50
- Best AUC: 0.7189
- Mean AUC: 0.7030
- Std AUC: 0.0151

**Best Configuration:**
  - d_model: 64
  - n_layers: 4
  - n_heads: 8
  - d_ff_ratio: 2
  - learning_rate: 0.0005
  - dropout: 0.7
  - weight_decay: 0.001

## Probability Collapse Analysis

No probability collapse detected.

## Parameter Importance

(Fraction of AUC variance explained by each parameter)

  dropout         | ###########################                        | 0.5432
  d_model         | ##########################                         | 0.5380
  n_layers        | ##############                                     | 0.2938
  weight_decay    | ########                                           | 0.1697
  n_heads         | ######                                             | 0.1389
  learning_rate   | #####                                              | 0.1046
  d_ff_ratio      | ##                                                 | 0.0591

## Parameter Trends

### d_model

- Trend: **decreasing**
- Best value: 64 (mean AUC: 0.7115)
- Worst value: 192 (mean AUC: 0.6802)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 64 | 0.7115 | 0.0086 | 28 |
| 96 | 0.6988 | 0.0060 | 5 |
| 128 | 0.6985 | 0.0102 | 8 |
| 160 | 0.6874 | 0.0134 | 3 |
| 192 | 0.6802 | 0.0191 | 6 |

### n_layers

- Trend: **decreasing**
- Best value: 4 (mean AUC: 0.7106)
- Worst value: 5 (mean AUC: 0.6898)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 4 | 0.7106 | 0.0118 | 26 |
| 5 | 0.6898 | 0.0093 | 3 |
| 6 | 0.6979 | 0.0128 | 9 |
| 7 | 0.6932 | 0.0221 | 6 |
| 8 | 0.6938 | 0.0094 | 6 |

### n_heads

- Trend: **increasing**
- Best value: 8 (mean AUC: 0.7059)
- Worst value: 4 (mean AUC: 0.6923)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 4 | 0.6923 | 0.0155 | 11 |
| 8 | 0.7059 | 0.0138 | 39 |

### learning_rate

- Trend: **flat**
- Best value: 0.0005 (mean AUC: 0.7065)
- Worst value: 0.0001 (mean AUC: 0.6917)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 1e-05 | 0.6975 | 0.0089 | 4 |
| 5e-05 | 0.7005 | 0.0105 | 11 |
| 0.0001 | 0.6917 | 0.0112 | 5 |
| 0.0005 | 0.7065 | 0.0168 | 30 |

### dropout

- Trend: **increasing**
- Best value: 0.7 (mean AUC: 0.7118)
- Worst value: 0.1 (mean AUC: 0.6787)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.1 | 0.6787 | 0.0146 | 5 |
| 0.3 | 0.6935 | 0.0152 | 11 |
| 0.5 | 0.6991 | 0.0070 | 6 |
| 0.7 | 0.7118 | 0.0077 | 28 |

### weight_decay

- Trend: **increasing**
- Best value: 1e-05 (mean AUC: 0.7096)
- Worst value: 0.0 (mean AUC: 0.6886)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.0 | 0.6886 | 0.0116 | 4 |
| 1e-05 | 0.7096 | 0.0114 | 13 |
| 0.0001 | 0.6965 | 0.0149 | 11 |
| 0.001 | 0.7048 | 0.0155 | 22 |

## Coverage Matrices

(Count of trials testing each combination)

### d_model × n_layers

| d_model | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- |
| 64 | 20 | 2 | 2 | 3 | 1 |
| 96 | 1 | 0 | 2 | 0 | 2 |
| 128 | 3 | 0 | 2 | 2 | 1 |
| 160 | 0 | 1 | 1 | 0 | 1 |
| 192 | 2 | 0 | 2 | 1 | 1 |

### d_model × n_heads

| d_model | 4 | 8 |
| --- | --- | --- |
| 64 | 3 | 25 |
| 96 | 2 | 3 |
| 128 | 3 | 5 |
| 160 | 1 | 2 |
| 192 | 2 | 4 |

### learning_rate × dropout

| learning_rate | 0.1 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| 1e-05 | 1 | 0 | 2 | 1 |
| 5e-05 | 1 | 7 | 0 | 3 |
| 0.0001 | 1 | 2 | 2 | 0 |
| 0.0005 | 2 | 2 | 2 | 24 |

## Forced Extreme Trials

| Trial | Extreme Type | AUC | d_model | n_layers | n_heads |
|-------|--------------|-----|---------|----------|---------|
| 0 | min_d_model | 0.7101 | 64 | 6 | 8 |
| 1 | max_d_model | 0.6854 | 192 | 6 | 8 |
| 2 | min_n_layers | 0.7052 | 128 | 4 | 8 |
| 3 | max_n_layers | 0.6954 | 128 | 8 | 8 |
| 4 | min_n_heads | 0.7043 | 128 | 6 | 4 |
| 5 | max_n_heads | 0.7036 | 128 | 6 | 8 |

---

# HPO Analysis: 200M (a100)

## Summary Statistics

- Total trials: 50
- Best AUC: 0.7152
- Mean AUC: 0.6974
- Std AUC: 0.0187

**Best Configuration:**
  - d_model: 320
  - n_layers: 12
  - n_heads: 8
  - d_ff_ratio: 2
  - learning_rate: 1e-05
  - dropout: 0.5
  - weight_decay: 1e-05

## Probability Collapse Analysis

No probability collapse detected.

## Parameter Importance

(Fraction of AUC variance explained by each parameter)

  learning_rate   | ######################                             | 0.4412
  dropout         | ##################                                 | 0.3796
  weight_decay    | #######                                            | 0.1586
  n_heads         | #####                                              | 0.1019
  n_layers        | ####                                               | 0.0831
  d_model         | ##                                                 | 0.0441
  d_ff_ratio      |                                                    | 0.0175

## Parameter Trends

### d_model

- Trend: **flat**
- Best value: 384 (mean AUC: 0.7031)
- Worst value: 192 (mean AUC: 0.6934)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 128 | 0.7023 | 0.0082 | 6 |
| 192 | 0.6934 | 0.0206 | 5 |
| 256 | 0.6947 | 0.0066 | 8 |
| 320 | 0.6949 | 0.0251 | 20 |
| 384 | 0.7031 | 0.0139 | 11 |

### n_layers

- Trend: **decreasing**
- Best value: 8 (mean AUC: 0.7036)
- Worst value: 12 (mean AUC: 0.6917)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 6 | 0.7031 | 0.0149 | 19 |
| 8 | 0.7036 | 0.0044 | 4 |
| 10 | 0.6940 | 0.0130 | 9 |
| 12 | 0.6917 | 0.0245 | 18 |

### n_heads

- Trend: **increasing**
- Best value: 16 (mean AUC: 0.7034)
- Worst value: 8 (mean AUC: 0.6914)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 8 | 0.6914 | 0.0223 | 25 |
| 16 | 0.7034 | 0.0118 | 25 |

### learning_rate

- Trend: **decreasing**
- Best value: 1e-05 (mean AUC: 0.7092)
- Worst value: 5e-05 (mean AUC: 0.6831)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 1e-05 | 0.7092 | 0.0053 | 26 |
| 5e-05 | 0.6831 | 0.0233 | 12 |
| 0.0001 | 0.6832 | 0.0226 | 6 |
| 0.0005 | 0.6891 | 0.0054 | 6 |

### dropout

- Trend: **increasing**
- Best value: 0.5 (mean AUC: 0.7057)
- Worst value: 0.1 (mean AUC: 0.6708)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.1 | 0.6708 | 0.0278 | 6 |
| 0.3 | 0.6909 | 0.0167 | 12 |
| 0.5 | 0.7057 | 0.0115 | 26 |
| 0.7 | 0.7010 | 0.0064 | 6 |

### weight_decay

- Trend: **increasing**
- Best value: 0.001 (mean AUC: 0.7069)
- Worst value: 0.0 (mean AUC: 0.6866)

| Value | Mean AUC | Std | Count |
|-------|----------|-----|-------|
| 0.0 | 0.6866 | 0.0197 | 6 |
| 1e-05 | 0.6929 | 0.0201 | 7 |
| 0.0001 | 0.6935 | 0.0217 | 19 |
| 0.001 | 0.7069 | 0.0093 | 18 |

## Coverage Matrices

(Count of trials testing each combination)

### d_model × n_layers

| d_model | 6 | 8 | 10 | 12 |
| --- | --- | --- | --- | --- |
| 128 | 1 | 1 | 1 | 3 |
| 192 | 3 | 0 | 1 | 1 |
| 256 | 3 | 1 | 2 | 2 |
| 320 | 7 | 1 | 2 | 10 |
| 384 | 5 | 1 | 3 | 2 |

### d_model × n_heads

| d_model | 8 | 16 |
| --- | --- | --- |
| 128 | 5 | 1 |
| 192 | 3 | 2 |
| 256 | 5 | 3 |
| 320 | 10 | 10 |
| 384 | 2 | 9 |

### learning_rate × dropout

| learning_rate | 0.1 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| 1e-05 | 1 | 3 | 21 | 1 |
| 5e-05 | 4 | 6 | 1 | 1 |
| 0.0001 | 0 | 2 | 2 | 2 |
| 0.0005 | 1 | 1 | 2 | 2 |

## Forced Extreme Trials

| Trial | Extreme Type | AUC | d_model | n_layers | n_heads |
|-------|--------------|-----|---------|----------|---------|
| 0 | min_d_model | 0.7151 | 128 | 10 | 8 |
| 1 | max_d_model | 0.6832 | 384 | 10 | 8 |
| 2 | min_n_layers | 0.6914 | 256 | 6 | 8 |
| 3 | max_n_layers | 0.6923 | 256 | 12 | 8 |
| 4 | min_n_heads | 0.6943 | 256 | 10 | 8 |
| 5 | max_n_heads | 0.6959 | 256 | 10 | 16 |
