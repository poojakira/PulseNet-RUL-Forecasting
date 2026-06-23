# PulseNet Threat Model

## Data Flow Diagram

```mermaid
flowchart TB
    subgraph Users[Users & Clients]
        U[API Client]
        D[Dashboard User]
    end

    subgraph External[External Systems]
        NASA[(NASA C-MAPSS Archive)]
        MLFLOW[(MLflow Tracking)]
    end

    subgraph API[PulseNet API Layer]
        LB[Load Balancer / Ingress]
        FW[FastAPI App]
        AUTH[Auth / JWT]
        RL[Rate Limiter]
        TENANT[Tenant Middleware]
    end

    subgraph Security[Security Layer]
        ENC[Encryption Manager]
        AUDIT[Audit Logger<br/>Hash Chain]
        BLOCK[Blockchain Ledger]
    end

    subgraph ML[ML Pipeline]
        IF[Isolation Forest]
        LSTM[LSTM Shadow]
        FEAT[Feature Registry]
        SCALER[MinMax Scaler]
    end

    subgraph Storage[Data Stores]
        OTEL[OpenTelemetry]
        PROM[(Prometheus Metrics)]
        LEDGER[(Audit Ledger File)]
        KEY[(Encryption Key File)]
    end

    U -->|HTTPS| LB
    D -->|HTTPS| LB
    LB --> FW
    FW --> AUTH
    FW --> RL
    FW --> TENANT
    FW --> ENC
    FW --> AUDIT
    FW --> IF
    FW --> LSTM
    IF --> FEAT
    IF --> SCALER
    AUDIT --> LEDGER
    AUDIT --> BLOCK
    ENC --> KEY
    FW --> PROM
    FW -->|POST| OTEL
    NASA -->|SHA-256 Verify| FW

    style NASA fill:#f96,stroke:#333,stroke-width:2px
    style ENC fill:#f9f,stroke:#333,stroke-width:2px
    style AUDIT fill:#f9f,stroke:#333,stroke-width:2px
    style BLOCK fill:#f9f,stroke:#333,stroke-width:2px
```

## Assets

| Asset | Criticality | Description |
|-------|-------------|-------------|
| NASA FD001 archive | High | Official C-MAPSS telemetry — source of truth |
| Model artifacts | High | Trained IsolationForest, scaler, feature registry |
| JWT signing secret | Critical | Token integrity for all API auth |
| User password hashes | Critical | bcrypt hashes stored in env |
| Encryption key | Critical | AES-256 Fernet key for data-at-rest |
| Audit log ledger | High | Tamper-evident hash-chain of all access |
| Prediction endpoint | High | Availability and correctness of /api/v1/predict |

## Trust Boundaries

1. **Official data boundary** — Only `data.nasa.gov` archive content trusted after SHA-256 verification
2. **API boundary** — FastAPI receives untrusted client payloads and tenant headers
3. **Auth boundary** — JWT claims trusted only after signature verification
4. **Runtime boundary** — Model, scaler, key files are local runtime artifacts (not source-controlled)
5. **Container boundary** — Container images run as non-root with read-only filesystem

---

## STRIDE Threat Analysis

### Spoofing

```mermaid
graph TD
    S[Spoofing] --> S1[JWT token forgery]
    S --> S2[User credential theft]
    S --> S3[Tenant ID spoofing]
    S1 --> S1M1[Use strong HS256 + secret rotation]
    S1 --> S1M2[Short token expiry: 60min]
    S2 --> S2M1[bcrypt password hashing]
    S2 --> S2M2[Environment-based user store]
    S3 --> S3M1[Tenant header validation regex]
```

| Threat | DREAD Score | Mitigation | Verification |
|--------|-------------|------------|--------------|
| JWT token forgery | 8 (9,9,8,7,7) | HS256 with env-based secret, 60min expiry | test_api.py validates token verify |
| Credential brute force | 6 (7,7,6,5,5) | bcrypt hashing, env-based user store | test_auth invalid login returns 401 |
| Tenant ID spoofing | 5 (6,5,5,4,5) | Regex validation `[A-Za-z0-9_.-]{1,64}` | test_tenant_header_rejected returns 400 |

### Tampering

```mermaid
graph TD
    T[Tampering] --> T1[Audit log modification]
    T --> T2[Model artifact replacement]
    T --> T3[Encrypted data manipulation]
    T --> T4[Config file tampering]
    T1 --> T1M1[SHA-256 hash chain per entry]
    T1 --> T1M2[Integrity verification endpoint]
    T2 --> T2M1[Runtime-only models, not in git]
    T2 --> T2M2[Future: artifact signing]
    T3 --> T3M1[Fernet AEAD authentication tag]
    T4 --> T4M1[Read-only filesystem in container]
```

| Threat | DREAD Score | Mitigation | Verification |
|--------|-------------|------------|--------------|
| Audit log tampering | 9 (9,9,9,9,9) | Hash-chain ledger with SHA-256 | test_audit_logger::test_verify_integrity |
| Model replacement | 7 (8,7,7,6,7) | Models not in git, loaded at runtime | Runtime load from models/ directory |
| Encrypted data tampering | 8 (9,8,8,7,8) | Fernet AEAD authentication | test_encrypt_decrypt_roundtrip |
| Config tampering | 5 (6,5,5,4,5) | ReadOnlyRootFilesystem in K8s | K8s securityContext enforcement |

### Repudiation

```mermaid
graph TD
    R[Repudiation] --> R1[User denies API action]
    R --> R2[Operator denies model retrain]
    R1 --> R1M1[Audit logger records all access]
    R1 --> R1M2[Hash-chain ensures non-repudiation]
    R2 --> R2M1[MLflow tracking for all experiments]
    R2 --> R2M2[Audit log for /train endpoint]
```

| Threat | DREAD Score | Mitigation | Verification |
|--------|-------------|------------|--------------|
| User denies API action | 7 (8,7,7,6,7) | AuditLogger with hash-chain integrity | test_audit_logger::test_log_access |
| Operator denies retrain | 6 (7,6,6,5,6) | MLflow experiment tracking + audit | MLflow run_id in audit metadata |

### Information Disclosure

```mermaid
graph TD
    I[Information Disclosure] --> I1[Model prediction leakage]
    I --> I2[Encryption key exposure]
    I --> I3[JWT secret exposure]
    I --> I4[Audit log exposure]
    I1 --> I1M1[Optional payload encryption]
    I1 --> I1M2[Auth/RBAC on predict endpoint]
    I2 --> I2M1[Key in env var or file with 0o600]
    I2 --> I2M2[ExternalSecret for K8s]
    I3 --> I3M1[Only in env, never in code/git]
    I4 --> I4M1[RBAC: admin-only audit access]
```

| Threat | DREAD Score | Mitigation | Verification |
|--------|-------------|------------|--------------|
| Prediction leakage | 6 (7,6,6,5,6) | Encryptable payloads, RBAC enforced | test_predict_no_auth returns 401 |
| Key exposure | 9 (10,9,9,8,9) | K8s ExternalSecrets, .gitignore | .dockerignore excludes .runtime/ |
| JWT secret exposure | 9 (10,9,9,8,9) | Env-based, not in source control | .env.example has placeholder only |
| Audit log exposure | 5 (6,5,5,4,5) | Admin-only RBAC on /audit | test_audit_requires_auth |

### Denial of Service

```mermaid
graph TD
    D[Denial of Service] --> D1[Predict endpoint flood]
    D --> D2[Model loading resource exhaustion]
    D --> D3[Audit log disk fill]
    D1 --> D1M1[In-memory rate limiter: 100 req/60s]
    D1 --> D1M2[HPA autoscaling up to 10 pods]
    D2 --> D2M1[Resource limits in K8s: 2Gi memory]
    D3 --> D3M1[Local file rotation, future: append-only storage]
```

| Threat | DREAD Score | Mitigation | Verification |
|--------|-------------|------------|--------------|
| Predict flood | 7 (8,7,7,6,7) | Rate limiter + HPA | test_rate_limit returns 429 |
| Resource exhaustion | 6 (7,6,6,5,6) | K8s resource limits (2Gi, 2 CPU) | K8s deployment manifest |
| Disk fill (audit) | 5 (6,5,5,4,5) | Future: append-only object storage | N/A (planned) |

### Elevation of Privilege

```mermaid
graph TD
    E[Elevation of Privilege] --> E1[Operator escalates to admin]
    E --> E2[Tenant cross-contamination]
    E --> E3[Direct pod access via exec]
    E1 --> E1M1[Role-based JWT claims]
    E1 --> E1M2[Role enforced on audit endpoint]
    E2 --> E2M1[Tenant ID scoped to user]
    E2 --> E2M2[Header validation + traceability]
    E3 --> E3M1[PodSecurityPolicy / SecurityContext]
    E3 --> E3M2[NetworkPolicy denies exec traffic]
```

| Threat | DREAD Score | Mitigation | Verification |
|--------|-------------|------------|--------------|
| Operator to admin | 8 (9,8,8,7,8) | JWT role claim, RBAC on admin endpoints | test_audit_rbac returns 403 |
| Tenant cross-contamination | 6 (7,6,6,5,6) | Tenant header validation + reflection | test_tenant_header_is_reflected |
| Pod exec escalation | 7 (8,7,7,6,7) | SecurityContext: drop all caps, non-root | K8s securityContext in deployment |

---

## DREAD Scoring Guide

Each threat is rated 1-10 on five dimensions:

| Letter | Dimension | Description |
|--------|-----------|-------------|
| **D** | Damage Potential | How severe is the damage? |
| **R** | Reproducibility | How easy is it to reproduce? |
| **E** | Exploitability | How easy is it to exploit? |
| **A** | Affected Users | How many users are affected? |
| **D** | Discoverability | How easy is it to discover? |

Overall score = (D + R + E + A + D) / 5

---

## Mitigation Verification Matrix

| Threat | Mitigation | Tested | Test Name | K8s Enforced |
|--------|------------|--------|-----------|--------------|
| JWT forgery | HS256 + secret rotation | ✓ | test_auth | N/A |
| Invalid login | bcrypt + 401 response | ✓ | test_invalid_login | N/A |
| Tenant path traversal | Regex validation | ✓ | test_invalid_tenant_header_rejected | N/A |
| No auth predict | RBAC enforcement | ✓ | test_predict_no_auth | N/A |
| Operator audit access | RBAC admin-only | ✓ | test_audit_rbac | N/A |
| Audit tampering | Hash-chain ledger | ✓ | test_verify_integrity | N/A |
| Blockchain tampering | SHA-256 chain | ✓ | test_tamper_detection | N/A |
| Encryption roundtrip | Fernet AEAD | ✓ | test_encrypt_decrypt_roundtrip | N/A |
| Key rotation | Backup + new key | ✓ | test_key_rotation | N/A |
| Container escape | SecurityContext | N/A (integration) | N/A | ✓ |
| Network intrusion | NetworkPolicy | N/A (infra) | N/A | ✓ |
| Secret in git | .gitignore + .dockerignore | Manual | pre-commit check | ✓ ExternalSecret |

---

## Residual Risk

| Risk | Impact | Notes |
|------|--------|-------|
| Rate limiting is in-memory per-process | DoS across multiple pods | Move to Redis or Envoy for distributed rate limiting |
| Audit storage is local file | Disk fill, single point of failure | Migrate to append-only object store (S3) with retention policy |
| Model artifacts unsigned | Supply chain tampering | Add cosign/sigstore artifact signing in CI/CD |
| FD001 is public benchmark | Not representative of production | Replace with live telemetry for production deployment |
| No WAF in front of API | Web application attacks | Deploy AWS WAF or CloudFront with AWS WAF |
