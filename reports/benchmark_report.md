# PulseNet Benchmark Results

## Inference Latency

| Metric | Value |
|--------|-------|
| mean_ms | 2.685 |
| median_ms | 2.524 |
| p95_ms | 3.943 |
| p99_ms | 4.269 |
| min_ms | 2.185 |
| max_ms | 4.389 |
| target_met | True |

## Throughput (samples/sec)

| Batch Size | Throughput |
|-----------|-----------|
| batch_1 | 329 |
| batch_8 | 2779 |
| batch_32 | 13429 |
| batch_64 | 20351 |
| batch_128 | 31424 |
| batch_256 | 52368 |

## Network Resilience

| Packet Loss | Data Integrity | Target Met |
|------------|---------------|------------|
| loss_10pct | 89.94% | [FAIL] |
| loss_20pct | 80.05% | [FAIL] |
| loss_30pct | 69.89% | [FAIL] |

## Encryption Overhead

| Operation | Mean (ms) | P95 (ms) |
|-----------|----------|---------|
| Encrypt | 0.0185 | 0.0268 |
| Decrypt | 0.0178 | 0.0238 |

## Detection Quality

| Metric | Value |
|--------|-------|
| precision | 0.2293 |
| recall | 1.0 |
| f1 | 0.373 |
| roc_auc | 0.3486 |
| pr_auc | 0.1655 |
| avg_precision | 0.1663 |

## Lead Time

| Metric | Value |
|--------|-------|
| avg_lead_time | 195.1 |
| median_lead_time | 194.0 |
| engines_detected | 10 |
| total_engines | 10 |
| detection_rate | 1.0 |

## Robustness (F1 Score)

**Baseline F1:** 0.373

### Noise Sensitivity
| Noise (Sigma) | F1 | Degradation |
|---------------|----|-------------|
| sigma_0.01 | 0.373 | 0.0% |
| sigma_0.05 | 0.373 | 0.0% |
| sigma_0.1 | 0.373 | 0.0% |

### Missing Data Sensitivity (Dropout)
| Rate | F1 | Degradation |
|------|----|-------------|
| rate_0.05 | 0.373 | 0.0% |
| rate_0.1 | 0.373 | 0.0% |
| rate_0.2 | 0.373 | 0.0% |
