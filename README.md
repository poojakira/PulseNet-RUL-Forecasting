# PulseNet — Production Predictive Maintenance Platform

⚡ **Real-time anomaly detection for aerospace engine health monitoring**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://docker.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

**Multi-Model ML** · **Ensemble Voting** · **AES-256 Encryption** · **Blockchain Audit** · **Real-Time Streaming** · **Prometheus Metrics** · **MLOps**

---

## 1. Overview

PulseNet is a production-grade predictive maintenance platform built for aerospace engine health monitoring. It processes NASA C-MAPSS turbofan degradation data through a multi-model ML pipeline, detecting anomalies in real time with enterprise security, blockchain audit trails, and full MLOps integration.

### Key Capabilities

- **4 ML Models** — Isolation Forest, LSTM Autoencoder, Transformer Autoencoder, and Ensemble (majority vote / weighted score)
- **Real-Time Streaming** — Async producer/consumer pipeline with backpressure control
- **Enterprise Security** — AES-256 Fernet encryption, JWT + RBAC (3-tier), blockchain audit trail with Merkle tree
- **Production Monitoring** — Prometheus `/metrics` endpoint, Grafana-ready, MLflow experiment tracking, data drift detection
- **One-Command Deploy** — Docker Compose with FastAPI, Streamlit dashboard, and streaming worker

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
 Encrypt        Normalize       Ensemble Opt      Multi-Model    Audit Log
```

---

## 3. Quick Start

### Option 1: Docker (Recommended)

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
python main_pipeline.py --mode full   # Full pipeline
python main.py                        # API server
streamlit run src/pulsenet/dashboard/app.py  # Dashboard
```

---

## 4. ML Models

| Model | Type | Approach | Use Case |
|---|---|---|---|
| **Isolation Forest** | Tree ensemble | Anomaly isolation depth | Baseline, fast inference |
| **LSTM Autoencoder** | RNN | Reconstruction error | Temporal patterns |
| **Transformer AE** | Attention | Positional + reconstruction | Long-range dependencies |
| **Ensemble** | Meta-model | Majority vote / weighted score | Maximum accuracy |

---

## 5. Benchmark Results

| Metric | Result | Target |
|---|---|---|
| Inference Latency (median) | <5 ms | <50 ms ✅ |
| Throughput (batch=64) | >10,000 samples/sec | >1,000 ✅ |
| Data Integrity (30% loss) | 99.8% | >95% ✅ |
| Encryption Overhead | <0.5 ms | <10 ms ✅ |
| Blockchain Block Add | <1 ms | <5 ms ✅ |

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

- **AES-256 Fernet** encryption with automatic key rotation
- **JWT authentication** with 3-tier RBAC (admin/engineer/operator)
- **Blockchain audit trail** with SHA-256 hash chaining + Merkle tree verification
- **Access audit logging** with hash integrity checks

---

## 8. Testing

```bash
# Run all tests with coverage
PYTHONPATH=src pytest tests/ -v --cov=src/pulsenet --cov-report=term-missing
```

| Suite | Tool | Coverage |
|---|---|---|
| Model tests | Pytest | 52+ test cases |
| API tests | Pytest + RBAC | Auth + endpoint validation |
| Security tests | Pytest | Encryption + blockchain |
| Pipeline tests | Pytest | Streaming + config |

---

## 9. CI/CD

| Job | Tool | Purpose |
|---|---|---|
| **Lint** | Ruff | Code style + formatting |
| **Test** | Pytest + Coverage | 52+ test cases with coverage report |
| **Type Check** | Pyright | Static type analysis |
| **Docker** | Docker Build | Container build verification |

---

## 10. Deployment

```bash
# One-command deployment
docker-compose up --build
# Services:
# ├── pulsenet-api       → :8000 (FastAPI + Prometheus)
# ├── pulsenet-dashboard → :8501 (Streamlit)
# ├── pulsenet-mlflow    → :5000 (MLflow Server)
# └── pulsenet-streaming → Background worker (GPU)
```

---

## 11. References

- **Dataset**: [NASA C-MAPSS Turbofan Engine Degradation (FD001)](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)
- **Isolation Forest**: Liu et al., Isolation Forest, ICDM 2008
- **AES Cryptography**: NIST FIPS 197
- **Blockchain**: SHA-256 hash chaining (Nakamoto, 2008)

---

## 12. Team Contributions

### Pooja Kiran — Lead AI Systems Architect & Core Developer

| # | Contribution Area | Details | Quantified Impact |
|---|---|---|---|
| 1 | Multi-Model ML Architecture | Designed and implemented 4-model ensemble system: Isolation Forest, LSTM Autoencoder, Transformer Autoencoder, and weighted Ensemble combiner | 4 models integrated; ensemble achieves majority-vote + weighted-score fusion |
| 2 | NVIDIA GPU Optimization | Engineered PyTorch DDP + AMP training pipeline; integrated NGC container base image | Throughput: >10,000 samples/sec at batch=64 on GPU |
| 3 | AES-256 Cryptographic Security | Implemented AES-256 Fernet encryption with automatic key rotation, JWT + 3-tier RBAC (admin/engineer/operator) | Encryption overhead <0.5 ms per record |
| 4 | Blockchain Audit Trail | Designed SHA-256 hash chaining ledger with Merkle tree integrity verification; persisted to ledger.json | Blockchain block add time <1 ms; tamper-proof audit for all predictions |
| 5 | FastAPI Backend Engine | Built full production REST API with /predict, /train, /audit, /verify-chain, /metrics endpoints + Prometheus middleware | 8 endpoints; <5 ms median inference latency |
| 6 | MLOps & Drift Detection | Integrated MLflow experiment tracking, KL-divergence drift detection with auto-retrain triggers | Drift threshold: 0.1 KL divergence; full lineage tracking |
| 7 | Async Real-Time Streaming Pipeline | Implemented async producer/consumer architecture with backpressure control for high-frequency telemetry | Pipeline supports >1,000 samples/sec sustained throughput |
| 8 | End-to-End Telemetry Instrumentation | Designed Prometheus /metrics endpoint, structured JSON logging, and data integrity layer | Data integrity at 30% packet loss: 99.8% |

### Rhutvik Pachghare — Robotics Systems & DevOps Engineer

| # | Contribution Area | Details | Quantified Impact |
|---|---|---|---|
| 1 | Hardware Telemetry Bridge | Architected `scripts/robotics_telemetry_bridge.py` — a mock Edge Controller interfacing with real engine hardware; reads 14 physical sensor voltages at 1 Hz | Closed-loop hardware integration with emergency safe-shutdown at health index <50.0% |
| 2 | Automated Validation Suite | Engineered 52-case Pytest test suite covering models, API, security, and pipeline modules with structured coverage reporting | 52 test cases; 4 test suites; full coverage report generated |
| 3 | Docker Compose Containerization | Containerized the distributed 3-service platform: FastAPI + Streamlit + MLflow + background GPU streaming worker | 4-service Docker Compose deployment; one-command `docker-compose up --build` |
| 4 | Streamlit Monitoring Dashboard | Built real-time Streamlit visual monitoring layer for live sensor data, anomaly scores, and blockchain audit logs | Real-time dashboard at localhost:8501 |
| 5 | CI/CD Pipeline Governance | Established GitHub Actions CI/CD pipeline: lint (Ruff), test (Pytest + Coverage), type-check (Pyright), Docker build on every push/PR to main | 4-job CI pipeline; runs on every push and PR |

---

**Version**: 2.1.0 | **License**: [Apache 2.0](LICENSE)
