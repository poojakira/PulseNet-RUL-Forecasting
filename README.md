# PulseNet RUL Forecasting

PulseNet is a research and engineering reference implementation for remaining-useful-life forecasting, anomaly detection, and security monitoring in industrial ML pipelines.

> **Current maturity:** pre-production. The repository contains useful components and tests, but it has not been independently validated on production traffic, certified for safety-critical use, or proven to meet a published service-level objective.

## What is implemented

- C-MAPSS data ingestion and preprocessing
- Isolation Forest, LSTM, Transformer, and ensemble model components
- FastAPI inference and training endpoints
- Authentication and role checks
- Audit and integrity-monitoring components
- Streaming, benchmarking, and dashboard modules
- MITRE ATT&CK mapping metadata for security findings

ATT&CK mappings are taxonomy metadata. They do **not** prove detection coverage, attack prevention, or operational effectiveness.

## Install

Python 3.10–3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install optional Torch models with:

```bash
pip install -e ".[ml,dev]"
```

## Required production configuration

The API intentionally refuses to start without a sufficiently long JWT secret outside test mode.

```bash
export PULSENET_ENV=production
export PULSENET_JWT_SECRET="replace-with-at-least-32-random-characters"
export PULSENET_USERS='{"admin":{"hashed_password":"<bcrypt-hash>","role":"admin"}}'
export PULSENET_ENCRYPTION_KEY="<fernet-key>"
```

Do not use wildcard CORS origins, locally generated encryption keys, filesystem audit logs, or the in-memory rate limiter as final production controls.

## Validation

Run the checks that are reproducible from the repository:

```bash
ruff check src tests attack_mapping scripts
ruff format --check src tests attack_mapping scripts
pytest tests -ra --cov=pulsenet --cov-report=term-missing
python -m build
python -m twine check dist/*
```

The official C-MAPSS integration test runs only when the NASA archive exists at:

```text
data/official/CMAPSSData.zip
```

```bash
pytest tests/test_rul_regression.py -ra
```

## Benchmark claim policy

A numeric result belongs in this README only when all of the following are committed:

1. The exact benchmark script and configuration
2. Dataset identity, checksum, and split definition
3. Hardware and software environment
4. Random seeds and repetition count
5. Machine-readable output artifact
6. CI or release workflow that reproduces or verifies the result

Earlier exact claims such as a fixed FD001 error, exact test count, exact coverage percentage, or fixed P99 security latency were removed because their cited evidence files were absent or did not establish those values.

## Production blockers

Before exposing PulseNet to real industrial traffic, address at least these items:

- Bind tenant identity to authenticated token claims instead of trusting `X-Tenant-ID`.
- Replace process-local rate limiting and batching state with bounded, observable infrastructure.
- Move blocking model inference off the event loop and enforce request deadlines.
- Use a secret manager/KMS and a versioned key ring; never auto-generate production keys.
- Replace local hash-chain files with durable append-only storage and external integrity anchoring.
- Apply strict numeric bounds and finite-value validation to all telemetry.
- Authenticate or isolate operational metrics and prevent unbounded metric labels.
- Run load, fault-injection, model-drift, rollback, disaster-recovery, and tenant-isolation tests.
- Establish model governance: lineage, approvals, signed artifacts, canary deployment, and rollback.

See `docs/SECURITY_AUDIT.md` for the initial engineering audit.

## Safety boundary

This software must not be the sole decision-maker for aviation, industrial-control, medical, or other safety-critical maintenance actions. Human review, independent sensors, fail-safe controls, and domain-specific certification remain necessary.
