> ℹ️ **REFERENCE ARCHITECTURE — The value here is the security architecture (STRIDE threat model, hash-chained audit, adversarial guards), not the ML performance. The Isolation Forest achieves F1=0.54 on NASA C-MAPSS FD001, which is not production-viable for maintenance scheduling. This is a secure MLOps reference implementation, not a predictive maintenance product.**

---
# PulseNet — Secure MLOps for Industrial Predictive Maintenance

[![CI](https://img.shields.io/github/actions/workflow/status/poojakira/PulseNet-RUL-Forecasting/ci.yml?branch=main&label=CI)](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)
[![STRIDE Threat Modeled](https://img.shields.io/badge/STRIDE-threat--modeled-orange)](THREAT_MODEL.md)
[![SARIF](https://img.shields.io/badge/SARIF-CodeQL%20output-blue)](https://docs.github.com/en/code-security/code-scanning)
[![NIST AI RMF](https://img.shields.io/badge/NIST-AI%20RMF-purple)](https://airc.nist.gov/)

---

PulseNet is a secure MLOps pipeline for industrial predictive maintenance (NASA C-MAPSS dataset)
demonstrating how security controls — threat modeling, RBAC, encryption, audit logging, and
adversarial guards — can be integrated across the full ML lifecycle without sacrificing performance.

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
| Isolation Forest | Anomaly detection | Unsupervised; trained on healthy operating cycles |
| LSTM Autoencoder | Sequence anomaly | Reconstruction error threshold tuned on FD001 |
| Ridge Regression | RUL point estimate | Baseline; MAE reported on test split |
| XGBoost | RUL + classification | Gradient boosted; feature importance via SHAP |

### Performance Results

Performance benchmarked on NASA C-MAPSS FD001 dataset. Results: **2.7ms mean inference latency,
P99=4.3ms, ~13,400 samples/sec at batch size 32. Isolation Forest F1=0.54, Precision=0.71,
Recall=0.43.**

> **Honest assessment:** The Isolation Forest F1=0.54 is below what you'd need for production
> deployment without human-in-the-loop review. Precision=0.71 is acceptable for alerting (low
> false-positive cost), but Recall=0.43 means ~57% of true anomalies are missed. The benchmark
> value here is demonstrating secure pipeline architecture around a realistic (imperfect) model,
> not claiming state-of-the-art detection.

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

### Local development

```bash
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting
cd PulseNet-RUL-Forecasting
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Run API server
uvicorn src.pulsenet.api.main:app --host 0.0.0.0 --port 8000

# Run dashboard
streamlit run src/pulsenet/dashboard/app.py
```

### Docker Compose (full stack with VPC simulation)

```bash
# Generate dev certs first (see mTLS Setup above)
# Create secrets directory
mkdir secrets
echo "supersecretdbpassword" > secrets/db_password.txt
echo "supersecretgrafana" > secrets/grafana_password.txt
openssl rand -base64 32 > secrets/aes_key.bin
# Generate JWT keypair into secrets/
openssl genrsa -out secrets/jwt_private_key.pem 2048
openssl rsa -in secrets/jwt_private_key.pem -pubout -out secrets/jwt_public_key.pem

docker compose up --build

# Services:
#   API:        http://localhost:8000
#   Dashboard:  http://localhost:8501
#   Grafana:    http://localhost:3000
#   Prometheus: internal only (docker exec pulsenet_prometheus wget -qO- localhost:9090/metrics)
```

### Run tests

```bash
pytest tests/ -v --cov=src/pulsenet --cov-fail-under=80
```

### Run security scan (CodeQL)

```bash
codeql database create codeql-db --language=python --source-root=src/
codeql database analyze codeql-db python-security-and-quality.qls \
  --format=sarif-latest --output=results/security.sarif
```

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
