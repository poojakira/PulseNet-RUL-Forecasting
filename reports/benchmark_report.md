# PulseNet Benchmark Suite

**Reproducibility:** All benchmarks use seeded random number generation (`random_seed=42`) and configurable thresholds. Results below are placeholders until you run the suite locally.

**Dataset:** NASA C-MAPSS FD001 (100 test engines, 14 selected sensor features after noise filtering)
**Model:** Isolation Forest (unsupervised, trained on healthy data only, contamination=0.05)
**Failure threshold:** RUL ≤ 30 cycles classified as "approaching failure" (standard C-MAPSS convention)

---

## How to Reproduce

```bash
# 1. Place C-MAPSS data in ./data/
#    train_FD001.txt, test_FD001.txt, RUL_FD001.txt
#    Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

# 2. Run full pipeline (trains model + generates test_features.csv)
python main_pipeline.py --mode full

# 3. Run benchmark suite
python main_pipeline.py --mode benchmark
#    or directly:
python scripts/verify_benchmarks.py

# Results saved to:
#   outputs/benchmarks/benchmark_results.json
#   outputs/benchmarks/benchmark_plots.png
```

---

## What the Benchmark Suite Measures

### 1. Inference Latency
Single-sample model `.predict()` call timing. Measures pure model overhead, not end-to-end API latency.
- Captures mean, median, p95, p99, min, max
- Target: median < 50ms (configurable in `config.yaml`)

### 2. Throughput
Batch inference throughput at multiple batch sizes (1, 8, 32, 64, 128, 256).
- Reports samples/second per batch size
- This is **model microbenchmark** — actual API throughput is lower due to JWT validation, serialization, and audit logging

### 3. Detection Quality (Unsupervised Anomaly Detection)
Standard classification metrics computed against RUL-derived ground truth:
- Precision, Recall, F1
- ROC-AUC, PR-AUC, Average Precision

> **Important:** The model is purely unsupervised — no labeled failures used during training. Metrics are precision-limited by design (favoring recall to avoid missed failures in safety-critical domains).

### 4. Lead Time (Early Warning Capability)
For each engine where the model predicted failure, measures how many cycles before actual failure the first prediction occurred.
- Average and median lead time in cycles
- Detection rate (engines flagged / total engines)

### 5. Network Resilience
Simulates random packet loss at 10%, 20%, 30% rates over 1000 trials.
- Measures data survivability (statistical baseline, not algorithmic resilience)
- Reproducible via `random_seed=42`

### 6. Encryption Overhead
AES-256-Fernet encrypt/decrypt timing per operation.
- Mean and P95 latency in milliseconds
- Validates that encryption is not a performance bottleneck

### 7. Robustness (Model Stability)
Measures F1 degradation under:
- Gaussian noise injection (sigma = 0.01, 0.05, 0.10)
- Random feature dropout (rate = 5%, 10%, 20%)
- Reproducible via `random_seed=42`

---

## Reading the Results

Once you've run the benchmark, results are written to `outputs/benchmarks/benchmark_results.json` with this schema:

```json
{
  "inference_latency": {"mean_ms": ..., "median_ms": ..., "p95_ms": ..., "target_met": true},
  "throughput": {"batch_1": ..., "batch_256": ...},
  "detection_quality": {"precision": ..., "recall": ..., "f1": ..., "roc_auc": ...},
  "lead_time": {"avg_lead_time": ..., "engines_detected": ..., "detection_rate": ...},
  "network_resilience": {"loss_10pct": {...}, "loss_20pct": {...}, "loss_30pct": {...}},
  "encryption": {"encrypt_mean_ms": ..., "decrypt_mean_ms": ...},
  "robustness": {"baseline_f1": ..., "noise": {...}, "dropout": {...}}
}
```

The accompanying `benchmark_plots.png` visualizes throughput, network resilience, encryption overhead, detection quality, robustness curves, and lead time summary.

---

## Methodology Notes

**Why unsupervised over supervised RUL regression?**
Production predictive maintenance systems often face cold-start problems — labeled failure data is scarce and expensive. Unsupervised methods can deploy with only "normal" operating data and detect novel failure modes not present in training.

**Why precision is intentionally low and that's correct:**
In aerospace maintenance, the cost matrix is asymmetric:
- A missed failure (false negative) → unplanned downtime, safety risk
- A false alarm (false positive) → an inspection cycle

The model is tuned to favor recall. A precision-optimized variant is on the roadmap and would require labeled failure windows.

**Why benchmark numbers vary between runs:**
The detection metrics are deterministic given fixed data and model. The robustness and network resilience benchmarks now use seeded random generation (default `seed=42`) to ensure reproducibility across runs.
