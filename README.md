# PulseNet – Secure RUL Forecasting for Safety‑Critical Telemetry

## Problem

Modern safety‑critical systems (engines, turbines, satellites, industrial assets) rely on telemetry streams to estimate Remaining Useful Life (RUL) and schedule maintenance.  
If telemetry is corrupted, delayed, replayed, or silently drifted, RUL forecasts become untrustworthy—and failures can be catastrophic.  

**PulseNet** is a security‑aware RUL forecasting pipeline that treats telemetry as a high‑value asset: it combines RUL modeling with data lineage, drift detection, and audit logging.

---

## Threat model

**Assets**

- Telemetry streams (time‑series sensor data)
- Preprocessed feature sets
- Trained RUL models and predictions
- Audit logs and lineage metadata

**Adversaries**

- Network‑adjacent attacker able to inject, drop, or replay telemetry
- Insider with access to intermediate data or preprocessing scripts
- Misconfigured or compromised upstream service corrupting data

**Attack surfaces**

- Telemetry ingestion endpoints
- Intermediate files / feature stores
- Model input pipeline
- Logging / monitoring infrastructure

**Defended (in scope)**

- Detection of tampered or replayed telemetry via lineage and hash checks  
- Detection of distribution drift in telemetry  
- Detection of missing or truncated data segments  

**Not defended (out of scope)**

- Physical sensor spoofing  
- Model extraction / IP theft  
- Compromise of the entire host or orchestrator  

See `docs/threat_model.md` for a detailed threat model.

---

## Approach

PulseNet implements a secure RUL forecasting pipeline with:

- **Data lineage & integrity**
  - Hashing of raw and preprocessed data
  - Lineage metadata linking telemetry → features → model inputs
- **Drift & anomaly detection**
  - Statistical drift checks on key features
  - Alerts when telemetry distribution shifts beyond configured thresholds
- **RUL modeling**
  - Time‑series RUL model (e.g., sequence model or regression on engineered features)
  - Evaluation on held‑out telemetry
- **Audit logging**
  - Append‑only audit log of data and model events
  - Hash‑chained entries to detect tampering

---

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion
        A[Telemetry Source] --> B[Ingestion Service]
    end

    B --> C[Raw Storage]
    C --> D[Preprocessing & Feature Engineering]
    D --> E[RUL Model]
    D --> F[Drift Detector]

    C --> G[Lineage Tracker]
    D --> G
    E --> H[Audit Logger]
    F --> H

    H --> I[Audit Log Store]

    classDef untrusted fill:#0a0a0a,stroke:#ff5555,color:#ffffff;
    classDef trusted fill:#111111,stroke:#4da6ff,color:#ffffff;

    class A,B untrusted;
    class C,D,E,F,G,H,I trusted;
