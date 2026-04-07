# PulseNet-RUL-Forecasting

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-52%20passed-brightgreen)](#testing)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](docker-compose.yml)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Predictive maintenance pipeline using NASA C-MAPSS data for Remaining Useful Life (RUL) forecasting and anomaly detection — designed for production-style deployment with a secure FastAPI backend, async telemetry streaming, and containerized MLOps infrastructure.**

---

## Problem

Unplanned failures in jet engines and industrial machinery cost billions annually. Accurately predicting the Remaining Useful Life (RUL) of components is critical to scheduling preventive maintenance and avoiding catastrophic failures.

**Who uses this, in what workflow, and what risk it reduces:**

- **Maintenance engineers** receive RUL predictions via the REST API before a scheduled inspection window, allowing them to prioritize which engines need servicing and skip unnecessary downtime on healthy units.
- **Operations dashboards** ingest the async telemetry stream to surface anomaly alerts in real time — flagging sensor readings that deviate from normal operating patterns before a hard fault occurs.
- **Audit/compliance teams** use the immutable blockchain-ledger pattern to verify the prediction history, ensuring that maintenance decisions are traceable and defensible in regulated environments (aviation, energy, manufacturing).

The primary failure risk mitigated: **false negatives in fault detection** — missing an engine approaching end-of-life — which translates to unplanned downtime, costly emergency repairs, and safety risk.

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
| RUL RMSE | **166.7** (~10% improvement vs. linear-decay baseline on C-MAPSS FD001) |
| Anomaly Detection F1 | 0.373 |
| Inference Throughput | 52,368 req/sec |
| P95 Latency | 3.94 ms |

> **Baseline:** A linear-decay heuristic that assumes RUL decreases uniformly from max observed cycles — the standard naive baseline for C-MAPSS benchmarking. The LSTM model was trained on an 80/20 train-test split of FD001 (train cycles clipped at 125). See [`reports/`](reports/) for data split details, training configuration, and experiment notes.

---

## Quick Start

### Installation

```bash
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting.git
cd PulseNet-RUL-Forecasting
cp config.example.yaml config.yaml   # fill in your secrets / paths
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
├── src/pulsenet/
│   ├── models/       # LSTM RUL model definition, training loop, checkpointing
│   ├── data/         # C-MAPSS dataset loader, normalization, sequence windowing
│   ├── api/          # FastAPI routers, request/response schemas, middleware
│   └── security/     # EncryptionManager, JWT auth, key rotation utilities
├── tests/
│   ├── unit/         # Model, data pipeline, and API logic tests (~40 cases)
│   └── security/     # JWT, encryption, key rotation tests (~12 cases)
├── reports/          # Experiment notes, data splits, training config snapshots
├── scripts/          # Data preparation and training entry-point scripts
├── docs/             # Architecture diagrams and API documentation
├── outputs/          # Model checkpoints and prediction outputs
├── config.yaml       # Active runtime config (gitignored in production)
├── config.example.yaml  # Safe template to commit; copy → config.yaml to start
├── Dockerfile
├── docker-compose.yml
├── Makefile          # `make train`, `make test`, `make lint`, `make docker-up`
└── README.md
```

> **Note on config:** `config.yaml` and `config.example.yaml` live at the root intentionally — they serve as the single runtime config consumed by both local runs and Docker. The `configs/` reference in older docs referred to this same root-level config; no separate directory is needed.

---

## Architecture & Deployment

The system runs as a single containerized service (FastAPI + uvicorn) behind a reverse proxy, suitable for deployment on **AWS ECS (Fargate)** or **Kubernetes**:

```
[Sensor / Data Source]
        │  async telemetry stream
        ▼
[Async Ingestion Layer]  ──►  [LSTM RUL Model]  ──►  [Isolation Forest Anomaly Detector]
        │                                                        │
        ▼                                                        ▼
[FastAPI REST API]  ──►  [JWT Auth + EncryptionManager]  ──►  [Audit Ledger]
        │
        ▼
[Docker Container]  ──►  ECS Task / K8s Pod  ──►  ALB / API Gateway  ──►  Client
```

**Production deployment sketch (AWS):**
- Docker image pushed to **ECR**, deployed as an **ECS Fargate** task with auto-scaling on CPU/request count
- Placed behind an **Application Load Balancer** with HTTPS termination; API Gateway handles rate limiting and auth offload
- Environment secrets (JWT secret, encryption key) injected via **AWS Secrets Manager** → ECS task environment at runtime, never baked into the image
- Logging via **CloudWatch Logs**; alerting on anomaly-score spikes via CloudWatch Alarms → SNS

**Observability & failure handling:**
- Structured JSON logging on every prediction and anomaly event (timestamp, engine ID, predicted RUL, anomaly score)
- `/health` endpoint for liveness/readiness probes in ECS/K8s
- If the ML model fails to load, the API returns a `503` with a clear error message — no silent fallback to random outputs
- Anomaly threshold is configurable via `config.yaml`; alerts surface to the operator dashboard when the score exceeds the threshold

---

## Security

**Threat model:** The primary risks addressed are (1) unauthorized access to sensitive maintenance predictions and (2) tampering with prediction history.

| Control | What it mitigates |
|---|---|
| JWT authentication (`python-jose`) | Prevents unauthenticated API access; token expiry limits session hijacking window |
| bcrypt password hashing | Protects credentials at rest if the user DB is compromised |
| `EncryptionManager` (AES) | Encrypts DataFrames and raw byte payloads in transit between pipeline stages — data-at-rest and in-motion protection |
| Dynamic key rotation (`tests/test_security.py`) | Limits blast radius of a compromised key; rotation is tested end-to-end |
| Blockchain-ledger audit trail | Tamper-evident log of all predictions — supports compliance audits and forensic review |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run only security tests
pytest tests/security/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

**Test categories:**

| Category | Location | Count |
|---|---|---|
| Model & data pipeline | `tests/unit/` | ~40 cases |
| Security (JWT, encryption, rotation) | `tests/security/` | ~12 cases |
| **Total** | | **~52 cases** |

Raw test output is available in [`test_output.txt`](test_output.txt) and [`test_security_output.txt`](test_security_output.txt) for reference.

---

## Experiments & Model Iteration

The LSTM architecture was chosen after comparing three approaches on C-MAPSS FD001:

| Approach | RMSE (test) | Notes |
|---|---|---|
| Linear-decay baseline | ~185 | Heuristic; no training |
| Ridge regression on raw sensors | ~178 | Underfits temporal patterns |
| **LSTM (PulseNet)** | **166.7** | Best; captures degradation trends over time |

Hyperparameter trials (hidden size, sequence length, dropout) are logged in [`reports/`](reports/). The current best config uses a 2-layer LSTM with hidden size 64, sequence window 30, and dropout 0.2.

---

## Ownership

**Pooja Kiran** — sole author and primary contributor. Designed and implemented the full pipeline: data preprocessing, LSTM training, anomaly detection, FastAPI backend, security layer, Docker deployment, and test suite.

External fork contributors are tracked via GitHub's fork network. Any external PRs are reviewed against the contribution guidelines below.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. Key expectations:

- **Branch naming:** `feat/<short-description>`, `fix/<issue-id>`, `docs/<topic>`
- **PR expectations:** Fill in the PR template (description, test evidence, checklist); link to the relevant issue
- **Required checks before merge:** `pytest` passes, `black` + `ruff` lint clean, `pyright` type-check passes, no secrets detected by pre-commit hooks
- **Commit style:** Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `chore:`)

---

## Roadmap

- [ ] Live demo deployment (AWS ECS or Render) with public Swagger UI
- [ ] GitHub Actions CI pipeline: lint → type-check → test → build Docker image (in progress — `.github/workflows/` scaffolded)
- [ ] Expand to C-MAPSS FD002/FD003/FD004 for multi-condition generalization
- [ ] Grafana dashboard for real-time anomaly monitoring

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Author

**Pooja Kiran**

- GitHub: [@poojakira](https://github.com/poojakira)
- LinkedIn: [Pooja Kiran](https://www.linkedin.com/in/poojakiran/)
