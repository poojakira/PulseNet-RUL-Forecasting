# PulseNet — Predictive Maintenance Operations Platform

[![CI](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![License Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

PulseNet is an end-to-end **anomaly-detection service** for industrial rotating
equipment, packaged as a production microservice rather than a notebook.

It reads turbofan-engine sensor telemetry, scores it with an unsupervised
model trained only on healthy operating cycles, and exposes the results to
operators through:

* a hardened **REST API** (FastAPI, JWT + RBAC, rate-limited, Prometheus-instrumented)
* an **operations console** (Streamlit) styled like a NOC/SCADA panel
* a **tamper-evident audit ledger** (SHA-256 hash-chained, Merkle-rooted, multi-tenant)
* an **async streaming pipeline** (Python `asyncio` with backpressure)
* a full **observability stack** (Prometheus + Grafana, structured JSON logs)

Reference dataset: [NASA C-MAPSS FD001](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) — a public-domain turbofan run-to-failure benchmark.

---

## What you can actually verify, today

Everything below is auditable from the source tree. No marketing numbers.

| Claim | Where to verify |
|---|---|
| API is JWT-authenticated with RBAC | `src/pulsenet/api/auth.py`, `src/pulsenet/api/routes/` |
| Tests exist for auth, models, pipeline, security, streaming | `tests/` (run `make test`) |
| Audit log is tamper-evident | `src/pulsenet/security/blockchain.py`, `tests/test_security.py::test_tamper_detection` |
| API is Prometheus-instrumented | `src/pulsenet/api/_prometheus.py`, `/metrics` endpoint |
| CI enforces lint + format + tests + types + security | `.github/workflows/ci.yml` |
| Container runs as non-root | `Dockerfile` (USER pulsenet:pulsenet, UID 10001) |
| Grafana dashboards auto-load on stack-up | `deploy/grafana/provisioning/` |

---

## Quick start

```bash
# 1. Install deps
make install-dev

# 2. Generate fixture data so the pipeline runs end-to-end without
#    downloading the real C-MAPSS dataset (fixture is clearly labeled
#    in data/FIXTURE_README.txt — do NOT use it for real benchmarks)
make fixture

# 3. Train + evaluate + benchmark
make train
make benchmark

# 4. Bring up the operations console
make dashboard         # http://localhost:8501

# 5. Or bring up the full stack (API + dashboard + MLflow + Prometheus + Grafana)
export PULSENET_JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
export GRAFANA_ADMIN_PASSWORD=$(python -c "import secrets; print(secrets.token_urlsafe(16))")
make stack-up
# api      -> http://localhost:8000  (Swagger: /docs, metrics: /metrics)
# console  -> http://localhost:8501
# mlflow   -> http://localhost:5000
# prom     -> http://localhost:9090
# grafana  -> http://localhost:3000
```

For real benchmarks, replace the fixture with the actual C-MAPSS files:

```bash
# 1. Download from NASA
#    https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
# 2. Drop train_FD001.txt, test_FD001.txt, RUL_FD001.txt into ./data/
# 3. Re-run the pipeline
make train benchmark
```

---

## Architecture

```
                  ┌──────────────────────────────────────┐
                  │  Sensor telemetry (live or replay)   │
                  └─────────────────┬────────────────────┘
                                    ▼
                  ┌──────────────────────────────────────┐
                  │  Async Streaming  (asyncio + queue)  │
                  │  - bounded queue + backpressure      │
                  │  - dynamic-batch FastAPI handler     │
                  └─────────────────┬────────────────────┘
                                    ▼
                  ┌──────────────────────────────────────┐
                  │  Inference (Isolation Forest /       │
                  │  LSTM-AE / Transformer-AE — pluggable)│
                  └────────┬─────────────────┬───────────┘
                           ▼                 ▼
        ┌───────────────────────┐  ┌────────────────────────┐
        │ FastAPI service       │  │ Cryptographic ledger   │
        │ - JWT + RBAC          │  │ - SHA-256 hash chain   │
        │ - rate limit (sliding)│  │ - Merkle root          │
        │ - X-Request-ID + JSON │  │ - per-tenant isolation │
        │ - /metrics, /healthz  │  └────────────────────────┘
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐  ┌────────────────────────┐
        │ Streamlit Console     │  │ Prometheus + Grafana   │
        │ (Fleet/Engine/Alarms/ │  │ provisioned dashboards │
        │  Audit/System Health) │  └────────────────────────┘
        └───────────────────────┘
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for component-by-component design notes.

---

## Repository layout

```
.
├── src/pulsenet/
│   ├── api/              FastAPI app, auth, RBAC, rate limit, Prometheus
│   ├── benchmarks/       Latency/throughput/quality/robustness measurements
│   ├── core/             Custom exceptions, threshold utilities
│   ├── dashboard/        Industrial Streamlit ops console (5 views)
│   ├── evaluation/       Detection-metric helpers
│   ├── mlops/            MLflow tracking + drift detector
│   ├── models/           IF + LSTM + Transformer + Ensemble + Registry
│   ├── pipeline/         Ingestion → preprocess → train → infer
│   ├── security/         Audit log, AES-Fernet encryption, hash-chain ledger
│   └── streaming/        Async producer/consumer + backpressure queue
├── tests/                pytest suite (unit + integration)
├── scripts/
│   ├── generate_test_fixture.py    Synthetic-fixture generator
│   ├── verify_benchmarks.py        Benchmark CLI runner
│   └── robotics_telemetry_bridge.py  Edge IoT simulator
├── deploy/
│   ├── prometheus.yml              Scrape config for the API service
│   └── grafana/provisioning/       Auto-loaded datasource + dashboard
├── .github/workflows/ci.yml        Lint → test → typecheck → security → docker
├── Dockerfile                      Multi-stage, non-root, healthchecked
├── docker-compose.yml              5-service production stack
├── Makefile                        All operator commands
├── config.example.yaml             Template (committed)
├── pyproject.toml                  Lint rules, optional-deps, build config
└── requirements.txt                Pinned runtime deps (CI uses this)
```

---

## How to run real benchmarks (and what they actually measure)

Once you have the real C-MAPSS dataset in place:

```bash
make train              # writes models/isolation_forest.joblib + scaler + feature_registry
make benchmark          # writes outputs/benchmarks/benchmark_results.json
ls outputs/benchmarks/  # benchmark_plots.png, benchmark_results.json
```

The benchmark suite (`src/pulsenet/benchmarks/benchmark.py`) measures the
following on **your hardware against your data** — there are no pre-packaged
numbers in this repo:

| Group | What is measured |
|---|---|
| Inference latency | Single-sample `model.predict()` mean / median / p95 / p99 |
| Throughput | Batch sizes 1, 8, 32, 64, 128, 256 (samples/sec) |
| Detection quality | Precision / Recall / F1 / ROC-AUC / PR-AUC versus RUL labels |
| Lead time | Cycles between first failure prediction and actual end-of-life |
| Network resilience | Survival fraction under simulated 10/20/30% packet loss (seeded) |
| Encryption overhead | Fernet AES-256 encrypt/decrypt latency |
| Robustness | F1 degradation under Gaussian noise + feature dropout (seeded) |

> **Why no published numbers in this README?** Benchmark numbers without the
> hardware spec they were measured on are misleading. Run `make benchmark` on
> the box you care about — the suite emits a JSON file you can attach to
> a PR or paste in your own README.

---

## Production engineering decisions

| Decision | Implementation | Why |
|---|---|---|
| Schema-validated config | `pulsenet.config.cfg` (Pydantic) | Catches misconfig at startup, not first request |
| Env-var overrides | `_apply_env_overrides()` in `config.py` | 12-factor; survives container restarts |
| Feature registry | `FeatureRegistry.process_offline/online` | Same transforms in train + inference, no skew |
| Versioned artifacts | `models/<name>_v<ts>.joblib` + `<name>.joblib` alias | Reproducible rollbacks |
| Dynamic batching | `DynamicBatcher` in `routes/predict.py` | Concurrent reqs collapse into one GPU call |
| Graceful shutdown | SIGTERM handler + FastAPI lifespan | Drains in-flight requests in K8s/ECS |
| Structured logs | JSON in prod, ANSI in dev (`logger.py`) | Datadog/CloudWatch-parseable |
| Per-tenant audit | `X-Tenant-ID` header + per-tenant ledger files | SaaS-ready isolation |
| Hash-chained audit log | SHA-256 + Merkle root in `BlackBoxLedger` | Tamper-evident; verifiable in O(log n) |
| Encryption-key rotation | `EncryptionManager.rotate_key()` with backup | Limit blast radius of compromise |
| Non-root container | `Dockerfile` UID 10001 | CIS Docker Benchmark 4.1 |
| Multi-stage build | `Dockerfile` builder → runtime | No build toolchain in shipped image |
| Healthcheck | `HEALTHCHECK CMD /healthz` | Orchestrator-driven restarts |

---

## Security posture

| Control | Threat mitigated | Evidence |
|---|---|---|
| JWT (HS256, 60-min default expiry) | Unauthenticated API access | `src/pulsenet/api/auth.py` |
| bcrypt password hashing | Credential theft from DB | `tests/test_security.py` exercises real bcrypt |
| RBAC (admin / engineer / operator) | Privilege escalation | `ROLE_PERMISSIONS` in `auth.py` |
| AES-256 (Fernet) at rest | Data exposure on disk | `EncryptionManager.encrypt_dataframe` |
| Sliding-window rate limiter | Brute-force / DoS | `_RateLimiter` in `api/app.py` |
| `X-Request-ID` correlation | Forensic tracing | Middleware in `api/app.py` |
| Hash-chained audit ledger | Prediction-log tampering | `BlackBoxLedger.validate_integrity()` |
| Dependency CVE scan | Known-vulnerable libs | `pip-audit` in CI |
| SAST (bandit) | Dangerous patterns in src | `bandit` in CI |
| CORS allow-list | Cross-origin abuse | Refuses `*` when `PULSENET_ENV=production` |

---

## CI / CD

`.github/workflows/ci.yml` runs four mandatory jobs on every push and PR:

```
Lint  →  Tests (3.11 + 3.12)  →  Typecheck  →  Security  →  Docker build
```

| Job | What runs | Blocking? |
|---|---|---|
| Lint | `ruff check` + `ruff format --check` | Yes |
| Tests | `pytest` with coverage on Python 3.11 + 3.12 | Yes |
| Typecheck | `pyright src/` | Yes |
| Security | `bandit -r src/` + `pip-audit -r requirements.txt` | Yes |
| Docker | `docker build` | Yes |

> The previous version had `continue-on-error: true` on typecheck — that's gone.
> All jobs now block merges.

---

## Limitations (honest)

* **No live deployed instance.** Roadmap item.
* **Tested only on C-MAPSS FD001** in this repo. FD002–FD004 introduce
  multiple operating conditions and fault modes — would require condition-
  conditional models and a multi-mode evaluation script (not yet written).
* **Streaming uses in-memory asyncio queues** — suitable for single-node.
  Multi-node scaling needs Kafka/Redis Streams (interface is queue-shaped
  so the swap is mechanical, but it's not done).
* **The "blockchain ledger" is a local hash-chain**, not a distributed
  consensus blockchain. It's tamper-evident, not tamper-proof; an attacker
  with write access to the file can replace the entire chain.
* **Unsupervised detection is precision-limited by design.** Without labeled
  failure windows the model favors recall — false alarms trigger an
  inspection, while missed failures cause unplanned downtime.

---

## Dataset

| Property | Value |
|---|---|
| Source | [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data) |
| Subset used | FD001 |
| Train engines | 100 (run-to-failure) |
| Test engines | 100 |
| Sensors per cycle | 21 |
| Operating conditions | 1 (sea level) |
| Fault mode | 1 (HPC degradation) |
| License | Public domain (US Government work) |

---

## Maintainer

**Pooja Kiran** — full system author.

* M.S. Information Technology (Security focus), Arizona State University
* B.E. Computer Science & Engineering, M.S. Ramaiah University of Applied Sciences
* Certifications: AWS Cloud Architecting, AWS Cloud Security Foundations, Honeywell Aerospace × ASU Technology Innovation Lab
* IEEE INDICON 2023 (author & presenter)

GitHub: [@poojakira](https://github.com/poojakira) · LinkedIn: [Pooja Kiran](https://www.linkedin.com/in/poojakiran/)

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
