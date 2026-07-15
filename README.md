# PulseNet RUL Forecasting

[![CI](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/PulseNet-RUL-Forecasting/actions)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

## What it is
PulseNet is an evidence-driven predictive maintenance reference service designed to forecast the Remaining Useful Life (RUL) of aerospace turbofan engines using the NASA C-MAPSS FD001 dataset. Unlike standard notebook-based ML projects, PulseNet is wrapped in a FastAPI service with dynamic batching, RBAC, audit logging, and adversarial telemetry-filtering controls.

## Motivation
Traditional ML deployments for predictive maintenance operate under the assumption that sensor telemetry is benign. In reality, industrial control systems are prime targets for sensor spoofing and False Data Injection Attacks (FDIA). An adversary manipulating temperature or vibration telemetry could force the ML model to predict an imminent failure, shutting down critical infrastructure.

## Objective
The objective is to provide a defensible ML architecture that explicitly filters suspicious telemetry *before* inference. PulseNet demonstrates controls that may support stricter trust-boundary deployment models, but production safety still depends on environment-specific validation, monitoring, and threat modeling.

## Data Flow
1. **Secure Ingestion**: `scripts/download_data.py` securely downloads the NASA dataset and cryptographically verifies its SHA-256 hash in memory.
2. **Anomaly Filtering**: Incoming telemetry streams pass through an Isolation Forest anomaly detector. Out-of-distribution or spoofed payloads are dropped and logged.
3. **Inference**: Clean telemetry is routed to the RUL Regressor to calculate the Remaining Useful Life.
4. **API Layer**: Prediction and training routes are protected with JWT/RBAC controls; token and health/probe endpoints remain public by design. `X-Tenant-ID` is propagated into response headers.
5. **Audit**: All actions (predictions, access events) are recorded in a hash-chained audit ledger to detect tampering.

## Technology Used
- **Core ML**: Scikit-Learn (Isolation Forest), Pandas, NumPy.
- **API & Security**: Python 3.12, FastAPI, JWT (JSON Web Tokens), Role-Based Access Control (RBAC).
- **Supply Chain**: `requirements.lock` records pinned dependencies; selected GitHub Actions are SHA-pinned where practical.
- **Testing**: Pytest for unit and adversarial testing.

## Benchmarks
*(Honest & Skeptic)*
- **Performance**: Includes local benchmark scripts for C-MAPSS FD001 RUL experiments. Treat reported RMSE as run-specific evidence, not a universal baseline or state-of-the-art claim.
- **Security**: The `FDIADetector` is evaluated against synthetic spoofed payloads in the included tests/benchmarks. Treat detection rates as benchmark-scoped, not guaranteed production TPR.
- **Resilience**: The API follows a fail-fast pattern; if required model weights are missing or corrupted, startup raises `RuntimeError` instead of serving predictions with an invalid model.

## Market Comparison
Compared to standard MLflow or AWS SageMaker endpoints, PulseNet fundamentally rejects the "trusted network" fallacy. It forces strict anomaly boundary checks at inference time. While commercial tools focus heavily on drift, PulseNet focuses on active adversarial sensor injection.

## How to Run

1. **Install dependencies:**
   ```bash
   uv sync
   ```

2. **Download and verify official data:**
   ```bash
   uv run scripts/download_data.py
   ```

3. **Generate API artifacts and signed manifest:**
   ```bash
   export PULSENET_ARTIFACT_MANIFEST_KEY="replace-with-a-random-secret"
   uv run python scripts/generate_api_artifacts.py
   ```
   This creates `models/isolation_forest.joblib`, `models/scaler.joblib`, `models/feature_registry.joblib`, and `models/api_artifacts.sha256.json`. The API refuses to load joblib artifacts when the manifest is missing, mismatched, unsigned, or signed with a different `PULSENET_ARTIFACT_MANIFEST_KEY`.

4. **Run local verification and tests:**
   ```bash
   uv run verify.py
   uv run pytest tests/
   ```

5. **Start the API Server:**
   ```bash
   export PULSENET_ARTIFACT_MANIFEST_KEY="same-secret-used-for-artifact-generation"
   uv run uvicorn src.pulsenet.api.app:app --host 0.0.0.0 --port 8000
   ```
