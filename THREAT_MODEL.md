# Threat Model: PulseNet RUL Forecasting

## 🎯 Assets

| Asset | Description | Impact of Compromise |
| --- | --- | --- |
| **Telemetry Data** | Turbofan sensor readings from NASA C-MAPSS. | False RUL predictions leading to safety incidents. |
| **ML Models** | Trained weights and architecture. | Model theft or backdoored inference. |
| **Audit Logs** | Hash-chained records of system activity. | Lack of accountability after a breach. |
| **Signing Keys** | JWT secrets used for authentication. | Unauthorized system-wide access. |
| **Tenant Data** | Isolated records for specific aircraft fleets. | Privacy breach and industrial espionage. |

## 👤 Adversaries

1. **Malicious External User**: Attempts to gain unauthorized access to prediction APIs or training endpoints.
2. **Insider (Disgruntled Employee)**: Has authorized access but attempts to tamper with models or logs to hide failures.
3. **Supply Chain Attacker**: Attempts to poison the NASA dataset or compromise a Python dependency.
4. **Data Poisoner**: Submits subtly altered telemetry to slowly drift the model's accuracy.

## ⚔️ Attacks & Mitigations

### 1. Telemetry Tampering (Input Attack)
- **Threat**: An adversary modifies sensor data to delay a "Critical Maintenance" alert.
- **Mitigation**: Input validation schemas (Pydantic), range checks, and out-of-distribution (OOD) detection.

### 2. Model Rollback / Replacement
- **Threat**: An attacker replaces a high-accuracy model with a version that has a hidden backdoor.
- **Mitigation**: Model artifacts are hashed and signed. The registry verifies hashes before loading.

### 3. Cross-Tenant Data Leakage
- **Threat**: Tenant A is able to view the prediction history of Tenant B.
- **Mitigation**: Middleware-enforced `X-Tenant-ID` filtering on all database queries and log retrievals.

### 4. Audit Log Tampering
- **Threat**: An attacker deletes evidence of a malicious action from the logs.
- **Mitigation**: SHA-256 hash-chaining. Deleting one record breaks the chain, which is detected by the daily verification job.

### 5. Dependency Poisoning
- **Threat**: A compromised version of `pandas` or `scikit-learn` is used in the pipeline.
- **Mitigation**: Pinned dependencies, `pip-audit` in CI, and SBOM generation for traceability.

## 🛡️ Risk Summary

PulseNet assumes a **Zero Trust** posture. We do not trust the network, the users, or even the underlying data until it has been verified against known-good hashes and authenticated against the RBAC policy.
