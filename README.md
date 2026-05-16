# PulseNet — Production ML System for Predictive Maintenance

[![CI](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.1](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**End-to-end ML systems engineering project: anomaly detection on NASA C-MAPSS turbofan engine data, deployed as a production-grade microservice with JWT authentication, async telemetry streaming, cryptographic audit trail, and containerized MLOps infrastructure.**

> This project demonstrates production ML engineering — not just model accuracy. It covers the full lifecycle: data ingestion, model training, real-time inference API, security hardening, observability, and deployment automation.

---

## Why This Project Exists

Unplanned equipment failures in aerospace and industrial settings cost an estimated [$630B+ annually worldwide](https://www.globenewswire.com/news-release/2026/02/04/3232190/0/en/predictive-maintenance-market-to-reach-us-91-04-billion-by-2033-as-ai-iot-and-downtime-costs-reshape-industrial-operations-astute-analytica.html). The predictive maintenance market was valued at ~$14.3B in 2025 and is projected to exceed $91B by 2033 ([Grand View Research](https://www.grandviewresearch.com/industry-analysis/predictive-maintenance-market)).

PulseNet addresses the **engineering challenge**: how do you take an ML model and deploy it as a reliable, secure, auditable service that maintenance engineers can actually use in production?

**Target users and workflows:**

| Role | How they use it | Risk mitigated |
|------|----------------|----------------|
| Maintenance Engineers | Receive anomaly alerts via REST API before scheduled inspections | Missed degradation leading to unplanned downtime |
| Operations Teams | Monitor real-time sensor streams on the dashboard | Late detection of compressor faults |
| Compliance/Audit | Query the tamper-evident blockchain ledger for prediction history | Untraceable maintenance decisions in regulated environments |

---

## System Architecture

```
[C-MAPSS Sensor Data]
        |
        v
[Data Ingestion + Validation] --> [Feature Engineering (rolling stats, normalization)]
        |
        v
[Model Training Pipeline]
   ├── Isolation Forest (unsupervised anomaly detection)
   ├── LSTM Autoencoder (sequence reconstruction)
   └── Transformer Autoencoder (attention-based reconstruction)
        |
        v
[FastAPI Inference Service]
   ├── JWT Auth + RBAC (admin/engineer/operator roles)
   ├── Dynamic Request Batching (GPU throughput optimization)
   ├── Rate Limiting + Request Correlation IDs
   ├── Prometheus Metrics (/metrics)
   └── Kubernetes Probes (/healthz, /readyz)
        |
        v
[Async Streaming Engine]              [Cryptographic Audit Ledger]
   ├── asyncio producer/consumer         ├── SHA-256 hash-chained blocks
   ├── Backpressure management           ├── Merkle tree integrity verification
   └── Batch drain optimization          └── Multi-tenant isolation
        |
        v
[Observability Stack]
   ├── Structured JSON logging
   ├── Prometheus counters/histograms
   └── GPU telemetry (pynvml)
```

> See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detailed design decisions and component interactions.

---

## Measured Results

All numbers below were produced by running `python main_pipeline.py --mode benchmark` on the actual codebase against [NASA C-MAPSS FD001](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) test data. Raw outputs are in [`reports/benchmark_results.json`](reports/benchmark_results.json).

### Anomaly Detection Performance

The primary model (Isolation Forest) performs **unsupervised** anomaly detection — it is trained only on healthy engine data and flags deviations without labeled failure examples.

| Metric | Value | Notes |
|--------|-------|-------|
| Recall (Sensitivity) | **1.00** | Detects 100% of engines approaching failure |
| Precision | 0.23 | Expected for unsupervised methods with aggressive threshold |
| F1 Score | 0.37 | Typical for unsupervised anomaly detection on C-MAPSS |
| Engine Detection Rate | 10/10 (100%) | All test engines flagged before failure |
| Avg Lead Time | **195 cycles** before failure | Provides significant advance warning |

**Why precision is low and that's okay:** The Isolation Forest is configured with a failure threshold of 125 cycles (RUL ≤ 125 = "approaching failure"). This is intentionally aggressive — in aerospace, **missing a failure (false negative) is catastrophic**, while a false alarm only triggers an inspection. The system prioritizes recall over precision by design.

### Inference Performance (sklearn `.predict()` microbenchmark)

| Metric | Value |
|--------|-------|
| Median Latency (single sample) | 2.52 ms |
| P95 Latency (single sample) | 3.94 ms |
| P99 Latency (single sample) | 4.27 ms |
| Throughput (batch=32) | 13,429 samples/sec |
| Throughput (batch=256) | 52,368 samples/sec |
| Target (<50ms) | Met |

> These are **model inference** benchmarks (CPU, sklearn). End-to-end API latency includes serialization, auth, and network overhead — typically 10-30ms additional depending on deployment.

### Security & Resilience

| Metric | Value |
|--------|-------|
| AES-256 Encrypt | 0.019 ms/operation |
| AES-256 Decrypt | 0.018 ms/operation |
| Robustness: F1 degradation at 10% noise | 0.0% |
| Robustness: F1 degradation at 20% dropout | 0.0% |

---

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| ML Models | PyTorch 2.1, scikit-learn 1.3 | LSTM/Transformer autoencoders + Isolation Forest |
| API | FastAPI 0.104, uvicorn | Async, OpenAPI docs auto-generated |
| Auth | python-jose (JWT), bcrypt, AES-256-Fernet | Industry-standard token auth + encryption at rest |
| Streaming | Python asyncio | Bounded queue with backpressure, no external broker dependency |
| Observability | Prometheus, structured JSON logging, pynvml | Production monitoring without vendor lock-in |
| Infrastructure | Docker, docker-compose, GitHub Actions CI | Reproducible builds, automated testing |
| Data | NASA C-MAPSS FD001 (public domain) | Standard benchmark for turbofan engine degradation |

---

## Quick Start

```bash
# Clone
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting.git
cd PulseNet-RUL-Forecasting

# Install
pip install -r requirements.txt

# Run full pipeline (requires C-MAPSS data in ./data/)
python main_pipeline.py --mode full

# Start API server
make serve
# API at http://localhost:8000 | Swagger UI at http://localhost:8000/docs

# Run tests
make test

# Docker (recommended for deployment)
docker-compose up -d
```

### Download C-MAPSS Data

The dataset is publicly available from NASA:
1. Download from [NASA Data Portal](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
2. Place `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` in `./data/`

---

## Project Structure

```
.
├── src/pulsenet/
│   ├── api/              # FastAPI app, JWT auth, RBAC, rate limiting, Prometheus
│   ├── models/           # IsolationForest, LSTM, Transformer, Ensemble, Registry
│   ├── pipeline/         # Orchestrator, ingestion, preprocessing, feature registry
│   ├── streaming/        # Async producer/consumer with backpressure queue
│   ├── security/         # AES encryption, blockchain audit ledger, access logging
│   ├── evaluation/       # Metrics calculation, ROC analysis
│   ├── benchmarks/       # Latency, throughput, robustness benchmarks
│   ├── mlops/            # MLflow tracking, data drift detection
│   ├── dashboard/        # Streamlit real-time monitoring UI
│   └── core/             # Exceptions, threshold optimization
├── tests/                # Unit + integration tests (pytest)
├── reports/              # Benchmark results (reproducible)
├── docs/                 # Architecture docs, design decisions
├── .github/workflows/    # CI: lint → test → typecheck → docker build
├── Dockerfile            # NVIDIA NGC base image for GPU inference
├── docker-compose.yml    # Full stack: API + Dashboard + MLflow
├── Makefile              # Developer shortcuts (make test, make serve, etc.)
└── config.yaml           # Runtime configuration (env var overrides supported)
```

---

## Production Engineering Decisions

These are the decisions that differentiate this from a Jupyter notebook:

| Decision | Implementation | Rationale |
|----------|---------------|-----------|
| Config as code | Pydantic-validated YAML + env var overrides | No magic constants; reproducible across environments |
| Feature registry | Centralized `FeatureRegistry` class | Eliminates training-serving skew (same transforms in train + inference) |
| Model versioning | Timestamped artifacts + model cards (YAML) | Track what's deployed, rollback if needed |
| Dynamic batching | `DynamicBatcher` groups concurrent API requests | Maximizes GPU utilization under load |
| Graceful shutdown | SIGTERM handler + lifespan context | Clean container restarts in K8s/ECS |
| Structured logging | JSON format in production, colored text in dev | Machine-parseable for log aggregation (CloudWatch, Datadog) |
| Multi-tenancy | `X-Tenant-ID` header → isolated audit trails | Data segregation for SaaS deployment |
| Blockchain audit | SHA-256 hash-chained blocks + Merkle root | Tamper-evident prediction history for compliance |
| Key rotation | Automated rotation with backup + age tracking | Limits blast radius of compromised encryption keys |
| Shadow model inference | Primary + shadow model predictions logged | Safe A/B testing before model promotion |

---

## Security Model

| Control | Threat Mitigated |
|---------|-----------------|
| JWT tokens (HS256, 60min expiry) | Unauthorized API access |
| bcrypt password hashing | Credential theft from DB compromise |
| AES-256-Fernet encryption | Data exposure at rest and in transit between pipeline stages |
| RBAC (admin/engineer/operator) | Privilege escalation |
| Rate limiting (100 req/min per IP) | DoS attacks |
| Request correlation IDs | Distributed tracing for incident response |
| Blockchain ledger | Prediction history tampering |

---

## Testing

```bash
make test          # Full suite with coverage
make test-fast     # Without coverage (faster iteration)
make lint          # Ruff linter + formatter check
make typecheck     # Pyright static analysis
```

Test coverage spans:
- **Model tests:** Train/predict/save/load, threshold optimization, hyperparameter tuning
- **Pipeline tests:** Rolling features, normalization, sequence windowing, label creation
- **API tests:** Auth flow, RBAC enforcement, health endpoint, input validation
- **Security tests:** Encrypt/decrypt roundtrip, key rotation, blockchain integrity, tamper detection
- **Streaming tests:** Queue backpressure, batch drain, metrics tracking

---

## Limitations & Future Work

**Current limitations (honest assessment):**
- Anomaly detection precision is low (0.23) — expected for unsupervised methods. A supervised approach with labeled failures would improve this but requires domain expert annotation.
- Streaming pipeline uses in-memory asyncio queues — suitable for single-node deployment but would need Kafka/Redis Streams for multi-node horizontal scaling.
- No live deployed demo URL yet (roadmap item).
- Tested only on C-MAPSS FD001 (single operating condition, single fault mode). FD002-FD004 would validate multi-condition generalization.

**Roadmap:**
- [ ] Live deployment on AWS ECS Fargate with public Swagger UI
- [ ] Expand to FD002/FD003/FD004 for multi-condition evaluation
- [ ] Grafana dashboard connected to Prometheus metrics
- [ ] Supervised model variant using labeled failure windows for higher precision

---

## Dataset

This project uses the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset, a standard benchmark for predictive maintenance research.

| Property | Value |
|----------|-------|
| Source | [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) |
| Subset used | FD001 |
| Training engines | 100 (run-to-failure) |
| Test engines | 100 |
| Sensors | 21 measurements per cycle |
| Operating conditions | 1 (sea level) |
| Fault mode | 1 (HPC degradation) |
| License | Public domain (US Government work) |

---

## Author

**Pooja Kiran** — designed and implemented the full system: data pipeline, ML models, FastAPI backend, security layer, streaming engine, Docker deployment, CI/CD, and test suite.

- M.S. Information Technology (Security Focus) — Arizona State University
- B.E. Computer Science & Engineering — M.S. Ramaiah University of Applied Sciences
- Certifications: AWS Cloud Architecting, AWS Cloud Security Foundations, Honeywell Aerospace & ASU Technology Innovation Lab
- Published: IEEE INDICON 2023

GitHub: [@poojakira](https://github.com/poojakira) | LinkedIn: [Pooja Kiran](https://www.linkedin.com/in/poojakiran/)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
