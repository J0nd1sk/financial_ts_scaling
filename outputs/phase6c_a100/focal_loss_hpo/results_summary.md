# Focal Loss HPO Results - Tier a100

Generated: 2026-01-29T10:03:30.524415

## Configuration
- Loss: Focal (gamma=0.0, alpha=0.8)
- Tier: a100

## Summary
- Total trials: 5
- Completed: 5
- Failed: 0
- Best AUC: 0.7097 (FL02)
- Best Precision: 0.3656716417910448
- Best Recall: 0.6447368421052632

## Best Config
```json
{
  "d_model": 64,
  "n_layers": 4,
  "n_heads": 8,
  "d_ff_ratio": 2,
  "dropout": 0.6,
  "learning_rate": 0.0005,
  "weight_decay": 0.001
}
```

## All Results (sorted by AUC)

| Config | AUC | Precision | Recall | Pred Range |
|--------|-----|-----------|--------|------------|
| FL02 | 0.7097 | 0.366 | 0.645 | [0.024, 0.968] |
| FL04 | 0.7079 | 0.398 | 0.566 | [0.018, 0.949] |
| FL05 | 0.7056 | 0.353 | 0.539 | [0.039, 0.944] |
| FL01 | 0.7034 | 0.368 | 0.605 | [0.018, 0.973] |
| FL03 | 0.7032 | 0.366 | 0.632 | [0.019, 0.959] |