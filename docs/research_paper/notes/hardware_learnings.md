# Hardware Utilization Learnings

## Platform

- **Machine**: M4 Max MacBook Pro
- **Memory**: 128GB unified memory
- **CPU**: 16 cores (4 efficiency + 12 performance)
- **GPU**: Apple Silicon integrated (MPS)

## Key Finding: MPS vs CPU Crossover Point

Benchmarking revealed that Apple MPS only outperforms CPU for **large matrix operations**:

| Matrix Size | MPS Time | CPU Time | Winner |
|-------------|----------|----------|--------|
| 1000×1000   | 0.016s   | 0.011s   | CPU    |
| 2000×2000   | 0.401s   | 0.062s   | CPU    |
| 4000×4000   | 0.135s   | 0.416s   | **MPS** |
| 8000×8000   | 0.798s   | 3.451s   | **MPS** |

**Crossover point**: ~4000×4000 matrices

## Implication for Training

With typical training configurations:
- batch_size=64, seq_len=60, d_model=768
- Effective matrix size: 3840×768 (below crossover)
- Result: **GPU 86% idle, CPU doing most work**

### Batch Size vs GPU Utilization

| Batch Size | Effective Matrix | GPU Benefit |
|------------|------------------|-------------|
| 64         | 3840 × 768       | None (CPU wins) |
| 128        | 7680 × 768       | Marginal |
| 256        | 15360 × 768      | Significant |
| 512        | 30720 × 768      | Strong |

**Recommendation**: Use batch_size >= 256 to utilize GPU effectively.

## Observed Hardware State (2B HPO Trial)

From `powermetrics` during 2B model training with batch_size=64:

- **GPU Active**: 13.92% (should be 80%+)
- **GPU Idle**: 86.08%
- **GPU Frequency**: 338 MHz (minimum, should be 1000-1500 MHz)
- **GPU Power**: 130 mW (should be 20-40W under load)
- **CPU (P-cores)**: 98% active at 4.5 GHz

**Diagnosis**: Training is CPU-bound due to small batch sizes.

## Recommendations for Future Experiments

1. **Minimum batch_size**: 256 (preferably 512)
2. **Early stopping**: Reduces wasted epochs, allows more trials
3. **Higher regularization**: Compensate for large-batch overfitting risk
4. **Monitor GPU utilization**: Use `powermetrics` to verify MPS is active

## Date

2025-12-22
