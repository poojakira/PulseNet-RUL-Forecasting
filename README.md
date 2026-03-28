# PulseNet — Predictive Maintenance Platform

**Anomaly detection for aerospace engine health monitoring — academic/personal project**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Multi-Model ML** · **Ensemble Voting** · **AES-256 Encryption** · **Blockchain Audit** · **Real-Time Streaming** · **Prometheus Metrics** · **MLOps**

---

## 1. Overview

PulseNet is a predictive maintenance project built to explore aerospace engine health monitoring using the NASA C-MAPSS dataset. The system implements multiple ML models, a REST API, and a monitoring dashboard as a hands-on learning project in ML systems design, security, and MLOps.

### Key Features

- **4 ML Models** — Isolation Forest, LSTM Autoencoder, Transformer Autoencoder, and Ensemble (majority vote / weighted score)
- **Real-Time Streaming** — Async producer/consumer pipeline with backpressure control
- **Security Layer** — AES-256 Fernet encryption, JWT + RBAC (3-tier), blockchain audit trail with Merkle tree
- **Monitoring** — Prometheus `/metrics` endpoint, MLflow experiment tracking, data drift detection
- **Deployment** — Docker Compose with FastAPI, Streamlit dashboard, and streaming worker

---

## 2. Architecture

📄 **[Read the Full System Design Document](docs/design_doc.md)**

### Pipeline Flow

```
python main_pipeline.py --mode full
┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐  ┌───────────┐
│  Ingest  │─▶│  Preprocess  │─▶│  Train   │─▶│  Evaluate  │─▶│ Inference │
│ C-MAPSS  │  │  Features    │  │  Models  │  │  F1/AUC    │  │ + Logging │
└──────────┘  └──────────────┘  └──────────┘  └────────────┘  └───────────┘
    │               │                │               │               │
 AES-256        Rolling Mean    IF / LSTM / TF    Comparison    Blockchain
 Encrypt        Normalize       Ensemble           Multi-Model    Audit Log
```

---

## 3. Quick Start

### Option 1: Docker

```bash
git clone https://github.com/poojakira/PulseNet.git && cd PulseNet
cp .env.example .env
# Place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in project root
docker-compose up --build
```

| Service | URL |
|---|---|
| **API** (Swagger UI) | http://localhost:8000/docs |
| **Dashboard** | http://localhost:8501 |
| **Prometheus Metrics** | http://localhost:8000/metrics |

### Option 2: Local

```bash
pip install -r requirements.txt
cp .env.example .env
python main_pipeline.py --mode full
python main.py
streamlit run src/pulsenet/dashboard/app.py
```

---

## 4. ML Models

| Model | Type | Approach | Use Case |
|---|---|---|---|
| **Isolation Forest** | Tree ensemble | Anomaly isolation depth | Baseline, fast inference |
| **LSTM Autoencoder** | RNN | Reconstruction error | Temporal patterns |
| **Transformer AE** | Attention | Positional + reconstruction | Long-range dependencies |
| **Ensemble** | Meta-model | Majority vote / weighted score | Combined accuracy |

---

## 5. Benchmark Results

| Metric | Result | Target |
|---|---|---|
| Inference Latency (median) | <5 ms | <50 ms ✅ |
| Throughput (batch=64) | >10,000 samples/sec | >1,000 ✅ |
| Data Integrity (30% loss) | 99.8% | >95% ✅ |
| Encryption Overhead | <0.5 ms | <10 ms ✅ |
| Blockchain Block Add | <1 ms | <5 ms ✅ |

> Note: Benchmarks measured in a local development environment. Results may vary in production.

---

## 6. API Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | ❌ | System status |
| `/token` | POST | ❌ | JWT login |
| `/predict` | POST | ✅ | Single inference |
| `/predict/batch` | POST | ✅ | Batch inference |
| `/train` | POST | ✅ | Retrain model |
| `/audit` | GET | ✅ | Blockchain logs |
| `/verify-chain` | GET | ✅ | Chain integrity |
| `/metrics` | GET | ❌ | Prometheus metrics |

---

## 7. Security

- AES-256 Fernet encryption with key rotation
- JWT authentication with 3-tier RBAC (admin/engineer/operator)
- SHA-256 hash chaining + Merkle tree verification
- Access audit logging

---

## 8. Testing

```bash
PYTHONPATH=src pytest tests/ -v --cov=src/pulsenet --cov-report=term-missing
```

| Suite | Test Count |
|---|---|
| Model tests | 52 total cases |
| API tests | Included in 52 |
| Security tests | Included in 52 |
| Pipeline tests | Included in 52 |

---

## 9. CI/CD

| Job | Tool |
|---|---|
| Lint | Ruff |
| Test | Pytest + Coverage |
| Type Check | Pyright |
| Docker | Docker Build |

---

## 10. Deployment

```bash
docker-compose up --build
```

---

## 11. References

- **Dataset**: [NASA C-MAPSS Turbofan Engine Degradation (FD001)](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)
- **Isolation Forest**: Liu et al., Isolation Forest, ICDM 2008
- **AES Cryptography**: NIST FIPS 197
- **Blockchain**: SHA-256 hash chaining (Nakamoto, 2008)

---

## 12. Team Contributions

> This is a personal/academic project built to learn ML systems design, security, and DevOps. Neither contributor has professional industry experience — all work was done as self-directed learning.

### Pooja Kiran

| # | What I Worked On | What I Built / Learned | Outcome |
|---|---|---|---|
| 1 | Multi-model ML pipeline | Implemented 4 anomaly detection models (Isolation Forest, LSTM Autoencoder, Transformer Autoencoder, Ensemble) using PyTorch and scikit-learn on the NASA C-MAPSS dataset | 4 working models trained and evaluated on FD001 dataset |
| 2 | GPU training setup | Set up PyTorch training with DDP and AMP; used NGC container as base Docker image to learn GPU-accelerated training workflows | Learned distributed training concepts; containerized training environment |
| 3 | AES-256 encryption layer | Implemented Fernet symmetric encryption for data at rest and in transit; added basic key rotation logic | Encryption applied to all sensor records; overhead measured at <0.5 ms locally |
| 4 | Blockchain audit log | Implemented SHA-256 hash chaining and a simple Merkle tree for audit trail; persisted to ledger.json | Tamper-evident log of all model predictions |
| 5 | FastAPI REST backend | Built a REST API with 8 endpoints using FastAPI; added JWT authentication and 3-tier role-based access control | API runs locally at localhost:8000; Swagger docs auto-generated |
| 6 | MLflow + drift detection | Integrated MLflow for experiment tracking; added KL-divergence based drift detection as a learning exercise | Experiments tracked in mlruns/; drift detection triggers a retrain flag |
| 7 | Streaming pipeline | Built an async producer/consumer queue with backpressure control to simulate streaming sensor data | Streaming worker runs as a separate Docker service |
| 8 | Prometheus metrics | Added /metrics endpoint using Prometheus middleware in FastAPI for learning observability concepts | Metrics endpoint returns HTTP request counts and latency histograms |

### Rhutvik Pachghare

| # | What I Worked On | What I Built / Learned | Outcome |
|---|---|---|---|
| 1 | Hardware telemetry simulation | Wrote `scripts/robotics_telemetry_bridge.py` to simulate an edge controller reading 14 sensor inputs at 1 Hz; added a health-index threshold check that triggers a mock safe-shutdown | Learned edge-to-cloud telemetry pipeline concepts; script runs as a local simulation |
| 2 | Pytest test suite | Wrote 52 test cases across 4 test files (models, API, security, pipeline) to learn unit and integration testing | 52 tests passing; coverage report generated |
| 3 | Docker Compose setup | Containerized the project with a 4-service Docker Compose file (FastAPI + Streamlit + MLflow + streaming worker) | One-command deployment with `docker-compose up --build` |
| 4 | Streamlit dashboard | Built a real-time Streamlit dashboard showing live sensor data, anomaly scores, and blockchain audit entries | Dashboard accessible at localhost:8501 |
| 5 | CI/CD with GitHub Actions | Set up a 4-job GitHub Actions pipeline (lint, test, type-check, Docker build) to learn CI/CD workflows | Pipeline runs on every push to main |

---

**Version**: 2.1.0 | **License**: [Apache 2.0](LICENSE)
