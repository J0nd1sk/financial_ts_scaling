# Detailed Backtest Report: Threshold 0.30

Generated: 2026-01-21T09:35:59.121204

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | PatchTST |
| d_model | 512 |
| n_layers | 6 |
| n_heads | 2 |
| dropout | 0.5 |
| Parameters | 19,134,977 |
| Context Length | 80 days |
| Decision Threshold | 0.30 |

## Target Definition

**Predict:** Will tomorrow's High price reach >= 1% above today's Close?

```
Label = 1 if High[t+1] >= Close[t] * 1.01
Label = 0 otherwise
```

## Test Period

- **Period:** 2025 (out-of-sample holdout)
- **Trading Days:** 181
- **Actual Positive Days:** 24 (13.3%)

## Results Summary

| Metric | Value |
|--------|-------|
| Trades Triggered | 5 |
| Winning Trades | 5 |
| Losing Trades | 0 |
| Win Rate | 100.0% |

## Trade Details

| Date | Probability | Close | Target (1%) | Next High | Result |
|------|-------------|-------|-------------|-----------|--------|
| 2025-08-01 | 0.3006 | $618.17 | $624.36 | $627.62 | WIN |
| 2025-11-18 | 0.3154 | $658.14 | $664.72 | $665.37 | WIN |
| 2025-11-19 | 0.3144 | $660.68 | $667.28 | $673.57 | WIN |
| 2025-11-20 | 0.3236 | $650.61 | $657.11 | $662.59 | WIN |
| 2025-11-21 | 0.3159 | $657.09 | $663.66 | $668.09 | WIN |

## Interpretation

When the model outputs a probability >= 0.30, it signals that tomorrow's
High price is likely to reach at least 1% above today's Close.

In the 2025 out-of-sample backtest:
- The model triggered 5 trade signals
- All 5 signals were correct (100% precision)
- The model is conservative but highly accurate
