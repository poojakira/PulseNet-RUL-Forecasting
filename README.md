> **Predictive maintenance that actually forecasts.** The 1D-CNN model scores
> **RMSE 13.19 on NASA C-MAPSS FD001**. For reference, that beats a RandomForest
> baseline (18.25), the Babu 2016 CNN (18.45), and the Zheng 2017 LSTM (16.14),
> and sits within 0.6 of the Li 2018 DCNN (12.61). It trains on CPU in about a
> minute from the dataset committed in this repo, and every prediction maps to a
> maintenance decision (immediate / plan / monitor / healthy). Reproduce it:
> `python benchmark/deep_rul_benchmark.py`. Numbers: [`docs/evidence/deep_rul_fd001.json`](docs/evidence/deep_rul_fd001.json).

---
# PulseNet — Secure MLOps for Industrial Predictive Maintenance

[![CI](https://img.shields.io/github/actions/workflow/status/poojakira/PulseNet-RUL-Forecasting/ci.yml?branch=main&label=CI)](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![STRIDE Threat Modeled](https://img.shields.io/badge/STRIDE-threat--modeled-orange)](THREAT_MODEL.md)
[![SARIF](https://img.shields.io/badge/SARIF-CodeQL%20output-blue)](https://docs.github.com/en/code-security/code-scanning)
[![NIST AI RMF](https://img.shields.io/badge/NIST-AI%20RMF-purple)](https://airc.nist.gov/)

---

PulseNet predicts how many operating cycles a turbofan engine has left (its
Remaining Useful Life) and turns that into a maintenance action. It's built on
the NASA C-MAPSS dataset. The reason it exists: most RUL demos stop at a model,
but a real deployment has to answer "who can call this, what happens if the
input is adversarial, and can I prove what the model decided six months later?"
So the pipeline carries the security controls a maintenance team would actually
need in production — RBAC, encryption at rest, a tamper-evident audit log, an
adversarial input guard, and JWT revocation — around a model that hits a
competitive RMSE.

---

## Table of Contents

1. [Security-First Design](#security-first-design)
2. [STRIDE Threat Model](#stride-threat-model)
3. [Security Controls Architecture](#security-controls-architecture)
4. [Continuous Assurance](#continuous-assurance)
5. [Models and Benchmarks](#models-and-benchmarks)
6. [VPC Simulation](#vpc-simulation)
7. [mTLS Setup](#mtls-setup)
8. [Requirements](#requirements)
9. [Installation and Running](#installation-and-running)
10. [Project Structure](#project-structure)
11. [NIST AI RMF Alignment](#nist-ai-rmf-alignment)

---

## Security-First Design

Most ML systems treat security as a post-deployment afterthought. PulseNet inverts this: security
controls are designed in from the data ingestion layer up to the inference API, with each layer
independently auditable.

| Property | Mechanism | Location |
|----------|-----------|----------|
| Confidentiality | AES-256-GCM encryption at rest and in transit | `src/pulsenet/security/encryption.py` |
| Integrity | SHA-256 hash-chained audit log | `src/pulsenet/security/audit.py` |
| Non-repudiation | Immutable append-only event log per tenant | `src/pulsenet/security/audit.py` |
| Authorization | JWT-based RBAC, multi-tenant isolation | `src/pulsenet/security/rbac.py` |
| Availability | Rate limiting + adversarial input guard | `src/pulsenet/security/adversarial_guard.py` |
| Observability | Prometheus metrics + Grafana dashboards | `src/pulsenet/api/metrics.py` |
| Static analysis | CodeQL SARIF output gated in CI | `.github/workflows/ci.yml` |

---

## STRIDE Threat Model

Full threat model in [THREAT_MODEL.md](THREAT_MODEL.md). Summary of the 12 identified findings:

| Component | Threat (STRIDE) | Mitigation | Control Type |
|-----------|-----------------|------------|--------------|
| Telemetry Ingestion | **Spoofing** — attacker injects false sensor readings to manipulate RUL predictions | mTLS mutual authentication on ingestion endpoint; input schema validation | Preventive |
| Model Inference API | **Tampering** — adversarial perturbations crafted to flip anomaly classification | Adversarial guard (statistical deviation check) before model call | Preventive |
| Audit Log | **Repudiation** — tenant denies issuing a prediction request | SHA-256 hash-chained, append-only audit log; entries are non-deletable | Detective / Corrective |
| Multi-tenant data | **Information Disclosure** — tenant A reads tenant B's RUL history | JWT claims enforce tenant scope; DB queries filtered by `tenant_id` at ORM layer | Preventive |
| FastAPI endpoint | **Denial of Service** — flood of unauthenticated requests saturates inference workers | Rate limiter (slowapi) keyed on JWT sub + auth check before queue entry | Preventive |
| CI/CD pipeline | **Elevation of Privilege** — malicious PR injects code bypassing RBAC | Branch protection + required CodeQL SARIF pass; pinned action versions | Preventive |
| JWT Token | **Spoofing** — stolen token used from unauthorized client | Short-lived tokens (15 min); RS256 signed; token binding to `client_id` claim | Preventive |
| Encryption Keys | **Information Disclosure** — key material exposed in container env vars | Keys stored in Docker secrets (local) or env vars; AWS Secrets Manager integration planned but NOT yet implemented | Preventive |
| Model Artifact | **Tampering** — model weights replaced with backdoored version | SHA-256 manifest checked at load time; artifacts in S3 with object lock | Detective |
| Prometheus Metrics | **Information Disclosure** — `/metrics` leaks tenant count and request rates | `/metrics` restricted to internal Docker network; not on public port | Preventive |
| Docker network | **Lateral Movement** — compromised dashboard pivots to model service | Network segmentation: `internal` network for model+DB, `external` for API only | Preventive |
| Rate Limiter bypass | **Denial of Service** — attacker rotates IPs to circumvent per-IP limit | Rate limiting keyed on JWT `sub` claim (not IP); circuit breaker for anonymous | Preventive |


---

## Security Controls Architecture

Data flow from raw telemetry to prediction output, with security controls annotated at each layer:

```
  Raw Telemetry
  (sensor stream)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  INGESTION LAYER                                     │
│                                                      │
│  ┌─────────────┐    ┌──────────────┐                 │
│  │ mTLS check  │───▶│ Schema       │                 │
│  │ (client     │    │ Validation   │                 │
│  │  cert auth) │    │ (Pydantic)   │                 │
│  └─────────────┘    └──────┬───────┘                 │
│                            │                         │
│  Control: Spoofing prevention, input integrity       │
└────────────────────────────┼─────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────┐
│  PREPROCESSING LAYER                                 │
│                                                      │
│  ┌─────────────────┐    ┌──────────────────┐         │
│  │ RBAC Check      │    │ Adversarial Guard│         │
│  │ (JWT tenant     │───▶│ (statistical     │         │
│  │  scope verify)  │    │  deviation gate) │         │
│  └─────────────────┘    └────────┬─────────┘         │
│                    (blocked)◀────┤                   │
│  Control: AuthZ enforcement, adversarial input block │
└────────────────────────────────────┼─────────────────┘
                                     │ (clean input)
                                     ▼
┌──────────────────────────────────────────────────────┐
│  INFERENCE LAYER                                     │
│                                                      │
│  ┌──────────────┐    ┌──────────────┐                │
│  │ SARIF gate   │    │ Model        │                │
│  │ (CodeQL pass │───▶│ Inference    │                │
│  │  required)   │    │ (IF / LSTM)  │                │
│  └──────────────┘    └──────┬───────┘                │
│                             │                        │
│  Control: Code integrity, supply chain assurance     │
└─────────────────────────────┼────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────┐
│  OUTPUT / STORAGE LAYER                              │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────┐        │
│  │ AES-256-GCM  │    │ Audit Log            │        │
│  │ Encryption   │    │ (SHA-256 hash-chained│        │
│  │ (at rest)    │    │  append-only)        │        │
│  └──────────────┘    └──────────────────────┘        │
│                                                      │
│  ┌──────────────────────────────────────┐            │
│  │ Prometheus metrics emitted           │            │
│  │ (prediction_requests_total,          │            │
│  │  adversarial_detections_total,       │            │
│  │  rbac_violations_total)              │            │
│  └──────────────────────────────────────┘            │
│                                                      │
│  Control: Confidentiality, non-repudiation,          │
│           continuous observability                   │
└──────────────────────────────────────────────────────┘
```

---

## Continuous Assurance

Security posture is monitored continuously via Prometheus metrics and Grafana dashboards. Alerts
fire on: adversarial input detection, RBAC violation attempts, audit log tampering, model drift
above threshold.

The /metrics endpoint is implemented in src/pulsenet/api/metrics.py and served on the internal Docker network only (not exposed on public port 8000).

**Prometheus metrics exposed at `/metrics` (internal network only):**

| Metric | Type | Description |
|--------|------|-------------|
| `prediction_requests_total` | Counter | Total inference requests, labeled by tenant and model |
| `adversarial_detections_total` | Counter | Inputs blocked by adversarial guard |
| `rbac_violations_total` | Counter | JWT authz failures, labeled by violation type |
| `audit_log_events_total` | Counter | Events written to hash-chained audit log |
| `inference_latency_seconds` | Histogram | Per-request model inference time (buckets: 1ms–1s) |
| `request_duration_seconds` | Histogram | Total FastAPI request duration |
| `model_drift_score` | Gauge | Current drift score vs. training distribution |
| `active_tenants` | Gauge | Number of tenants with active JWT sessions |

**Alert rules (Prometheus alertmanager):**

```yaml
# Adversarial spike: >5 detections in 5 minutes
- alert: AdversarialInputSpike
  expr: rate(adversarial_detections_total[5m]) > 1
  severity: warning

# RBAC brute-force: >10 violations in 2 minutes
- alert: RBACViolationBurst
  expr: rate(rbac_violations_total[2m]) > 5
  severity: critical

# Model drift: drift score above 0.15
- alert: ModelDriftExceeded
  expr: model_drift_score > 0.15
  severity: warning
```


---

## Models and Benchmarks

| Model | Task | Notes |
|-------|------|-------|
| **1D-CNN (Li 2018 architecture)** | **RUL regression (primary)** | Sliding 30-cycle windows, 14 sensors. **RMSE 13.19 on FD001.** |
| GradientBoosting / RandomForest | RUL regression (classical baseline) | Last-cycle engineered features. RMSE ~18.3. |
| Isolation Forest | Anomaly screening (secondary) | Unsupervised guard for out-of-distribution telemetry |

### Performance Results

Benchmarked on the **NASA C-MAPSS FD001** official per-unit train/test split
(no random split, no temporal leakage). Reproduce with
`python benchmark/deep_rul_benchmark.py`.

| Model | RMSE ↓ | Source |
|-------|--------|--------|
| **PulseNet 1D-CNN** | **13.19** | this repo, measured on CPU |
| DCNN — Li et al. 2018 | 12.61 | published SOTA-class |
| LSTM — Zheng et al. 2017 | 16.14 | published |
| CNN — Babu et al. 2016 | 18.45 | published |
| RandomForest (classical) | 18.25 | this repo baseline |

The 1D-CNN **beats every classical baseline and the 2016–2017 deep-learning
results**, landing within 0.6 RMSE of the Li 2018 DCNN — on CPU, in ~60 seconds,
on the real committed dataset. This is production-viable for maintenance
scheduling: each prediction feeds the `MaintenanceScheduler`, which acts on the
*conservative* RUL (prediction minus model uncertainty) so the dangerous
direction — over-estimating remaining life — is guarded against.

```python
from pulsenet.models.rul_forecaster import RULForecaster, MaintenanceScheduler

forecaster = RULForecaster().fit(train_df)
scheduler = MaintenanceScheduler()
for fc in forecaster.predict_last_cycle(live_df):
    decision = scheduler.decide(fc)
    print(decision.unit_number, decision.action.value, decision.reason)
    # e.g. 42 immediate "Conservative RUL 11 cycles <= 15. Ground the asset..."
```

<details>
<summary>Legacy anomaly-detection numbers (secondary guard, not the product)</summary>

Isolation Forest F1=0.54 / Precision=0.71 / Recall=0.43 on FD001. This runs as
a secondary out-of-distribution guard, not the maintenance forecaster. The RUL
regressor above is the primary model.
</details>

Note on adversarial guard false-positive rate: The statistical guard (z-score threshold=4.0σ) is tuned for high recall over precision. Operators can set ADVERSARIAL_GUARD_THRESHOLD=6.0 to reduce false alerts at the cost of missing more attacks. This is a deliberate tradeoff for safety-critical anomaly detection.

| Metric | Value | Dataset | Evidence |
|--------|-------|---------|----------|
| Isolation Forest F1 | 0.54 | NASA C-MAPSS FD001 | `results/validation_results.json` |
| Isolation Forest Precision | 0.71 | NASA C-MAPSS FD001 | `results/validation_results.json` |
| Isolation Forest Recall | 0.43 | NASA C-MAPSS FD001 | `results/validation_results.json` |
| Mean inference latency | 2.7ms | FD001 test set, batch=32 | `benchmark/latency_results.json` |
| P99 inference latency | 4.3ms | FD001 test set, batch=32 | `benchmark/latency_results.json` |
| Throughput | ~13,400 samples/sec | batch=32, single process | `benchmark/latency_results.json` |

Hardware: AWS c5.4xlarge (16 vCPU, 32 GB RAM), single process, batch size 32, Python 3.11, scikit-learn 1.4.0.

---

## VPC Simulation

The Docker Compose network topology simulates AWS VPC public/private subnet isolation:

```
  Internet
     │
     ▼
┌────────────┐   (pulsenet_external only)
│ API :8000  │
│ Dashboard  │
│ Grafana    │
└─────┬──────┘
      │  (api joins both networks as the only bridge)
      ▼
┌────────────────────────────────────────┐
│  pulsenet_internal  (internal: true)   │  ← No host routing
│                                        │
│  ┌──────────────┐  ┌─────────────────┐ │
│  │ PostgreSQL   │  │   Prometheus    │ │
│  │ (timeseries) │  │  (no host port) │ │
│  └──────────────┘  └─────────────────┘ │
└────────────────────────────────────────┘
```

- `pulsenet_external`: API gateway, dashboard, Grafana — reachable from host
- `pulsenet_internal`: PostgreSQL, Prometheus — `internal: true` blocks all external routing
- The API is the only service bridging both networks, enforcing auth on every inbound call

This mirrors AWS VPC design where a public subnet holds the load balancer/NAT gateway and a
private subnet holds application servers and databases with no direct internet route.

---

## mTLS Setup

PulseNet supports mutual TLS on the telemetry ingestion endpoint. Certificates are volume-mounted
into the API container (see `docker-compose.yml`, `certs/` volume mounts).

### Generate development certificates

```bash
mkdir certs

# Certificate Authority
openssl genrsa -out certs/ca.key 4096
openssl req -new -x509 -days 365 -key certs/ca.key \
  -subj "/CN=PulseNet-Dev-CA" -out certs/ca.crt

# Server certificate
openssl genrsa -out certs/server.key 4096
openssl req -new -key certs/server.key \
  -subj "/CN=api.pulsenet.local" -out certs/server.csr
openssl x509 -req -days 365 -in certs/server.csr \
  -CA certs/ca.crt -CAkey certs/ca.key -CAcreateserial \
  -out certs/server.crt

# Client certificate (for telemetry producer)
openssl genrsa -out certs/client.key 4096
openssl req -new -key certs/client.key \
  -subj "/CN=telemetry-agent-01" -out certs/client.csr
openssl x509 -req -days 365 -in certs/client.csr \
  -CA certs/ca.crt -CAkey certs/ca.key -CAcreateserial \
  -out certs/client.crt
```

### Enable mTLS in uvicorn

```python
import ssl
import uvicorn

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain("certs/server.crt", "certs/server.key")
ssl_context.load_verify_locations("certs/ca.crt")
ssl_context.verify_mode = ssl.CERT_REQUIRED  # enforce client cert

uvicorn.run(app, host="0.0.0.0", port=8443, ssl=ssl_context)
```

### Test mTLS with curl

```bash
curl --cert certs/client.crt \
     --key certs/client.key \
     --cacert certs/ca.crt \
     https://localhost:8443/health
```


---

## Requirements

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.2.0
prometheus-client>=0.20.0
python-jose[cryptography]>=3.3.0
cryptography>=42.0.0
slowapi>=0.1.9
pydantic>=2.6.0
streamlit>=1.32.0
sqlalchemy>=2.0.0
pytest>=8.0.0
pytest-cov>=5.0.0
```

Python 3.10+ required.

---

## Installation and Running

### Prerequisites
- Python 3.10 or newer
- pip (comes with Python)
- Git
- Docker & Docker Compose (optional, for full-stack deployment)
- OpenSSL (optional, for mTLS certificate generation)
- Key dependencies (installed automatically): FastAPI, PyTorch, scikit-learn, pandas, numpy, prometheus-client, cryptography, python-jose

### Install from source

```powershell
# Windows PowerShell
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting.git
cd PulseNet-RUL-Forecasting
py -m venv .venv
.venv\Scripts\activate
py -m pip install -e ".[dev]"
```

```bash
# Linux / Mac
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting.git
cd PulseNet-RUL-Forecasting
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Verify installation

```powershell
# Windows PowerShell
py -c "from pulsenet.security.rbac import *; from pulsenet.security.encryption import *; from pulsenet.security.audit import *; print('OK')"
```

```bash
# Linux / Mac
python -c "from pulsenet.security.rbac import *; from pulsenet.security.encryption import *; from pulsenet.security.audit import *; print('OK')"
```

### Run the API server (local development)

```powershell
# Windows PowerShell
py -m uvicorn src.pulsenet.api.main:app --host 0.0.0.0 --port 8000
```

```bash
# Linux / Mac
uvicorn src.pulsenet.api.main:app --host 0.0.0.0 --port 8000
```

### Run the dashboard

```powershell
# Windows PowerShell
py -m streamlit run src/pulsenet/dashboard/app.py
```

```bash
# Linux / Mac
streamlit run src/pulsenet/dashboard/app.py
```

### Run tests

```powershell
# Windows PowerShell
py -m pytest tests/ -v --cov=src/pulsenet --cov-fail-under=80
# Expected: all tests passed, coverage >= 80%
```

```bash
# Linux / Mac
pytest tests/ -v --cov=src/pulsenet --cov-fail-under=80
# Expected: all tests passed, coverage >= 80%
```

### Docker Compose (full stack with VPC simulation)

```powershell
# Windows PowerShell — generate secrets first
mkdir secrets
"supersecretdbpassword" | Out-File -Encoding ascii secrets/db_password.txt
"supersecretgrafana" | Out-File -Encoding ascii secrets/grafana_password.txt
openssl rand -base64 32 | Out-File -Encoding ascii secrets/aes_key.bin
openssl genrsa -out secrets/jwt_private_key.pem 2048
openssl rsa -in secrets/jwt_private_key.pem -pubout -out secrets/jwt_public_key.pem

docker compose up --build
```

```bash
# Linux / Mac
mkdir -p secrets
echo "supersecretdbpassword" > secrets/db_password.txt
echo "supersecretgrafana" > secrets/grafana_password.txt
openssl rand -base64 32 > secrets/aes_key.bin
openssl genrsa -out secrets/jwt_private_key.pem 2048
openssl rsa -in secrets/jwt_private_key.pem -pubout -out secrets/jwt_public_key.pem

docker compose up --build
```

Services after Docker start:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Grafana: http://localhost:3000
- Prometheus: internal only (`docker exec pulsenet_prometheus wget -qO- localhost:9090/metrics`)

### Common issues

| Problem | Fix |
|---------|-----|
| `py` not recognized (Windows) | Use `python` instead, or install Python from python.org and ensure it's on PATH |
| PyTorch install fails / takes forever | Install PyTorch separately first: `py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` |
| `ModuleNotFoundError: No module named 'pulsenet'` | Ensure you ran `pip install -e ".[dev]"` from the repo root |
| Permission denied on install | Use a virtual environment (see steps above) |
| Docker build fails on secrets | Ensure the `secrets/` directory exists with all required files before running `docker compose up` |
| `cryptography` fails to build on Windows | Install Visual C++ Build Tools or use `py -m pip install --only-binary :all: cryptography` |
| Port 8000 already in use | Change port: `py -m uvicorn src.pulsenet.api.main:app --port 8001` |
| `openssl` not found on Windows | Install OpenSSL via `winget install ShiningLight.OpenSSL` or use Git Bash which includes OpenSSL |

---

## Project Structure

```
PulseNet-RUL-Forecasting/
├── src/pulsenet/
│   ├── api/
│   │   ├── main.py               # FastAPI app entry point
│   │   ├── routes.py             # Prediction and health endpoints
│   │   └── metrics.py            # Prometheus metrics definitions + /metrics route
│   ├── security/
│   │   ├── rbac.py               # JWT-based RBAC, tenant isolation
│   │   ├── encryption.py         # AES-256-GCM encrypt/decrypt
│   │   ├── audit.py              # SHA-256 hash-chained append-only audit log
│   │   └── adversarial_guard.py  # Statistical adversarial input detection
│   ├── models/
│   │   ├── isolation_forest.py
│   │   ├── lstm_autoencoder.py
│   │   └── registry.py           # Artifact hash verification at load time
│   └── dashboard/
│       └── app.py                # Streamlit UI
├── tests/
│   ├── test_security.py          # Security control unit tests (5 tests)
│   ├── test_models.py            # Model accuracy regression tests
│   └── test_api.py               # FastAPI integration tests
├── benchmark/
│   └── latency_results.json      # Committed latency benchmark artifacts
├── results/
│   └── validation_results.json   # Committed model benchmark artifacts
├── monitoring/
│   ├── prometheus.yml
│   ├── alert_rules.yml
│   └── grafana/
├── certs/                        # mTLS certs (gitignored — see mTLS Setup)
├── secrets/                      # Docker secrets (gitignored)
├── docker-compose.yml
├── THREAT_MODEL.md
├── SECURITY.md
├── CONTRIBUTING.md
└── pyproject.toml
```

---

## NIST AI RMF Alignment

| NIST AI RMF Function | PulseNet Control |
|----------------------|-----------------|
| **GOVERN** — policies and accountability | SECURITY.md, CONTRIBUTING.md, branch protection, CODEOWNERS |
| **MAP** — context and risk identification | THREAT_MODEL.md; 12 STRIDE findings with ATT&CK mappings |
| **MEASURE** — quantify and monitor | Prometheus counters/histograms; committed benchmark artifacts in `results/` and `benchmark/` |
| **MANAGE** — respond and recover | Alertmanager rules; RBAC violation response playbook in SECURITY.md |

---

## ATT&CK v19 / ICS Techniques Defended

| Technique | Description | PulseNet Defense |
|-----------|-------------|-----------------|
| T1691 | ICS Lateral Movement | Internal Docker network isolation (`internal: true`) |
| T1692 | ICS Impair Process Control | Adversarial guard on sensor inputs before model call |
| T0843 | ICS Program Download | SHA-256 artifact hash verification at model load |
| T0873 | ICS Project File Infection | CodeQL SARIF gate required in CI before merge |
| T0846 | ICS Remote System Discovery | No unnecessary network exposure; internal-only Prometheus |

---

## License

Apache-2.0. See [LICENSE](LICENSE).
