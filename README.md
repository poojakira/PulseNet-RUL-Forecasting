# PulseNet-RUL-Forecasting

Remaining Useful Life forecasting on NASA C-MAPSS FD001 — built as a secure MLOps reference
implementation with a STRIDE threat model, RBAC, tenant audit trail, and SARIF CI security gates.

This is the ops side of ML security: what does a properly hardened ML inference pipeline
look like from data ingestion to prediction API?

![CI](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions/workflows/ci.yml/badge.svg)

---

## Threat model (STRIDE)

Full STRIDE analysis: `STRIDE_THREAT_MODEL.md`

| STRIDE category | Threat | Mitigation |
|----------------|--------|-----------|
| **Spoofing** | Sensor spoofing — malicious actors inject false telemetry readings | Mutual TLS + API key auth; data origin validation at ingestion gate |
| **Tampering** | Training data or model weights modified at rest | SHA-256 hash chain on pipeline artifacts; Ed25519-signed model checkpoints |
| **Repudiation** | Actor denies performing a critical action (model update, data deletion) | Append-only structured audit log — tenant, user, timestamp, prediction, input hash |
| **Information disclosure** | Proprietary sensor data or model IP exfiltrated | RBAC per tenant; encryption in transit and at rest; 0.019 ms mean encrypt overhead |
| **Denial of service** | Prediction endpoint flooded | Rate limiting; load balancing config in `docker-compose.yml` |
| **Elevation of privilege** | Low-privilege user gains admin access | Least-privilege RBAC enforced at API layer; scoped JWT tokens |

---

## What this implements

| Component | Implementation | Location |
|-----------|---------------|----------|
| RUL model | LSTM + Transformer ensemble on C-MAPSS FD001 | `src/pulsenet/models/` |
| Data lineage | SHA-256 hash chain from raw sensor data to training set | `src/pulsenet/pipeline/` |
| Input validation | Schema enforcement + anomaly gate before inference | `src/pulsenet/security/adversarial_telemetry_guard.py` |
| RBAC | Role-scoped API endpoints; tenant isolation at middleware layer | `src/pulsenet/api/auth.py`, `src/pulsenet/api/middleware/tenant.py` |
| Audit trail | Append-only JSONL log — prediction, input hash, tenant, user, timestamp | `src/pulsenet/security/audit.py` |
| Encryption | AES-GCM at rest; TLS in transit | `src/pulsenet/security/encryption.py` |
| CI gates | GitHub Actions: dependency scan + artifact hash check + SARIF output | `.github/workflows/ci.yml` |
| NIST AI RMF controls | Mapped controls in `docs/nist_ai_rmf_controls.yaml` | `docs/` |

---

## Benchmark results

Dataset: NASA C-MAPSS FD001 (official `.zip` — `data/official/CMAPSSData.zip`)

### Inference latency

| Metric | Value |
|--------|-------|
| Mean | 2.7 ms |
| Median | 2.5 ms |
| P95 | 3.9 ms |
| P99 | 4.3 ms |
| Target met | ✓ |

### Throughput (samples/sec)

| Batch size | Throughput |
|-----------|-----------|
| 1 | 329 |
| 32 | 13,429 |
| 128 | 31,424 |
| 256 | 52,368 |

### Anomaly detection (adversarial telemetry guard)

| Metric | Value |
|--------|-------|
| Recall | 1.0 (all 10 degrading engines detected) |
| Precision | 0.23 |
| F1 | 0.37 |
| Avg lead time | 195 cycles before failure |
| Detection rate | 10/10 engines |

High recall / lower precision is a deliberate design choice for a safety-critical path —
false negatives (missing a failing engine) are more costly than false positives (unnecessary inspection).
See `FAILURE_MODES.md` for the full trade-off rationale.

### Encryption overhead

| Operation | Mean | P95 |
|-----------|------|-----|
| Encrypt | 0.019 ms | 0.027 ms |
| Decrypt | 0.018 ms | 0.024 ms |

Full JSON: `reports/benchmark_results.json`

---

## Known limits

- Threat model covers data ingestion and API surface; training infrastructure compromise is not in scope
- RBAC is application-layer only — no hardware isolation, TEE, or confidential compute
- Audit log is append-only with hash-chain integrity; not blockchain-sealed
- Anomaly gate is tuned for FD001; other C-MAPSS subsets (FD002–FD004) need recalibration
- Network resilience drops below target under 10%+ packet loss — documented in benchmark (target_met: false)
- LSTM/Transformer ensemble not yet evaluated against adversarial sensor injection beyond Gaussian noise

---

## Setup

```bash
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting
cd PulseNet-RUL-Forecasting
pip install -r requirements.txt

# Run pipeline
python main_pipeline.py

# Start API
uvicorn src.pulsenet.api.app:app --reload

# POST /predict with sensor payload
# Audit log written to logs/audit.jsonl

# Run security tests
pytest tests/test_security.py -v

# Reproduce benchmarks
python scripts/run_validation.py
```

Docker:

```bash
docker-compose up
```

---

## References

- [NASA C-MAPSS Jet Engine Simulated Dataset](https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6)
- [NIST AI Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/system/files/documents/2023/01/26/AI%20RMF%201.0.pdf)
- [NIST SP 800-204D: DevSecOps for Microservices](https://csrc.nist.gov/pubs/sp/800/204/d/final)
- [STRIDE Threat Modelling — Microsoft](https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats)
- [MITRE ATLAS — Adversarial Threat Landscape for AI Systems](https://atlas.mitre.org/)


---

## SSE AI Infrastructure Relevance

**Role target:** System Software Engineer, AI Infrastructure

This project is a **production ML inference pipeline** — not just a model notebook. It addresses the systems engineering concerns that matter in AI infrastructure:

| Concern | Implementation |
|---|---|
| **Inference pipeline** | LSTM+Transformer served via FastAPI/uvicorn at 2.7 ms mean latency / 52K samples/sec throughput |
| **Model serving** | FastAPI with RBAC middleware, scoped JWT tokens, per-tenant rate limiting |
| **CI/CD for ML** | GitHub Actions: ruff lint, bandit SAST, pip-audit CVE scan, pytest with coverage, SBOM generation, SARIF output — all gates block on failure |
| **Artifact integrity** | SHA-256 hash chain from raw sensor data → training set → model checkpoint; Ed25519-signed artifacts |
| **Distributed systems hardening** | Multi-tenant isolation at API layer; AES-GCM encryption at rest (0.019 ms overhead measured); append-only JSONL audit log |
| **Observability & reliability** | Prometheus/Grafana in architecture; IR_PLAYBOOK.md; FAILURE_MODES.md; smoke_test.sh; structured logging |
| **Container** | Docker + docker-compose; non-root runtime; multi-stage build |
| **Performance benchmarking** | scripts/run_validation.py; measured latency and throughput in benchmark scripts |
| **NIST AI RMF / SRE** | NIST AI RMF 1.0 compliance mapping; STRIDE threat model with per-category mitigations; NIST SP 800-204D reference |
| **Scalable architecture** | Rate limiting + load balancing config; designed for horizontal scaling behind reverse proxy |

**Tech stack:** Python · PyTorch · FastAPI · Docker · Kubernetes (k8s manifests) · Prometheus · GitHub Actions · SARIF · Ed25519 · AES-GCM · JWT/RBAC · NIST AI RMF

**Additional keywords (honest):** deep learning � LSTM+Transformer � SLSA provenance � policy-as-code (CI gates enforce artifact signing policy) � scalable inference � high-performance (sub-3ms latency) � distributed systems (multi-tenant API + hash-chained audit) � SRE principles (error budgets, SLOs, incident playbook) � model lifecycle management (promotion gates via SARIF)
