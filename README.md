# PulseNet-RUL-Forecasting

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/poojakira/PulseNet-RUL-Forecasting/branch/main/graph/badge.svg)](https://codecov.io/gh/poojakira/PulseNet-RUL-Forecasting)

**Predictive maintenance pipeline using NASA C-MAPSS data for Remaining Useful Life (RUL) forecasting and anomaly detection.**

---

## Problem

Unplanned failures in jet engines and industrial machinery cost billions annually. Accurately predicting the Remaining Useful Life (RUL) of components is critical to scheduling preventive maintenance and avoiding catastrophic failures.

---

## Key Features

- **RUL Forecasting** — LSTM-based regression on NASA C-MAPSS turbofan engine sensor data
- **Anomaly Detection** — Isolation Forest for detecting out-of-distribution sensor readings
- **Async Telemetry Streaming** — Python `asyncio`-based streaming engine for real-time ingestion
- **Secure FastAPI Backend** — JWT authentication, bcrypt password hashing, and custom `EncryptionManager` for DataFrame/byte encryption
- **Audit Trail** — Mock blockchain ledger pattern for immutable prediction logging
- **Docker Deployment** — Full containerized stack via `docker-compose`

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Frameworks | PyTorch, Scikit-Learn |
| API | FastAPI, uvicorn |
| Concurrency | Python `asyncio` |
| Security | `python-jose` (JWT), bcrypt, custom `EncryptionManager` |
| Infrastructure | Docker, docker-compose |

---

## Results

| Metric | Value |
|---|---|
| RUL RMSE | 166.7 (10% improvement over baseline) |
| Anomaly Detection F1 | 0.373 |
| Inference Throughput | 52,368/sec |
| P95 Latency | 3.94 ms |

---

## Quick Start

### Installation

```bash
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting.git
cd PulseNet-RUL-Forecasting
pip install -r requirements.txt
```

### Run

```bash
# Docker (recommended)
docker-compose up -d

# Or locally
uvicorn main:app --reload
# API available at http://localhost:8000
# Swagger UI at http://localhost:8000/docs
```

---

## Project Structure

```
.
├── src/              # Model training, inference, and pipeline code
├── tests/            # Unit and security tests
├── configs/          # Configuration files
├── docs/             # Architecture and API documentation
├── outputs/          # Model outputs and predictions
├── reports/          # Evaluation reports
├── scripts/          # Data preparation and training scripts
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Security

- Custom `EncryptionManager` for encrypting Pandas DataFrames and raw bytes
- Mock blockchain ledger pattern for immutable prediction logging
- `python-jose` JWT authentication with bcrypt password hashing
- Dynamic key rotation tested in `tests/test_security.py`

---

## Roadmap

- [ ] Deploy live demo (AWS, Render, or Hugging Face Spaces)
- [ ] Expand dataset coverage for better model generalization

---


## Architecture

```ascii
+------------------+     +-------------------+     +------------------+
|  Data Ingestion  | --> |  Feature Pipeline | --> |   Model Engine   |
|  (Async Stream)  |     |  (Normalization)  |     | (LSTM Forecaster)|
+------------------+     +-------------------+     +------------------+
                                                        |
                                                        v
+------------------+     +-------------------+     +------------------+
|   API Gateway    | <-- |   Prediction     | <-- |  Anomaly Detect  |
|  (FastAPI/JWT)   |     |   Cache Layer    |     | (IsolationForest)|
+------------------+     +-------------------+     +------------------+
        |
        v
+------------------+
|  Audit Ledger    |
|  (Mock Blockchain)|
+------------------+
```



















---#
# License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Author

**Pooja Kiran**

- GitHub: [@poojakira](https://github.com/poojakira)
- LinkedIn: [Pooja Kiran](https://www.linkedin.com/in/poojakiran/)
