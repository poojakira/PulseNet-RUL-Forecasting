# PulseNet Threat Model

**Methodology:** STRIDE  
**Last reviewed:** 2026-08-05  
**Reviewer:** Security Engineering  
**Version:** 1.1  
**Scope:** PulseNet MLOps pipeline — telemetry ingestion through inference API and result storage.  
**ATT&CK mapping:** MITRE ATT&CK v19, ICS sub-techniques T0843/T0873/T0846, T1691–T1695.

---

## System Description

PulseNet processes raw telemetry from industrial machinery (NASA C-MAPSS dataset format), runs
anomaly detection (Isolation Forest) and RUL forecasting (LSTM, XGBoost, Ridge), and exposes
predictions via a FastAPI service with multi-tenant JWT auth. Results are stored in PostgreSQL
and surfaced via a Streamlit dashboard and Prometheus/Grafana observability stack.

**Trust boundaries:**
1. External telemetry producers → Ingestion endpoint (mTLS boundary)
2. Authenticated tenants → API gateway (JWT boundary)
3. API gateway → Internal model service (Docker internal network boundary)
4. Internal services → PostgreSQL (ORM/credential boundary)
5. CI/CD pipeline → Production artifacts (SARIF/branch-protection boundary)

---

## Data Flow Diagram

```
[Telemetry Source] ──mTLS──▶ [Ingestion Endpoint] ──▶ [Schema Validation]
                                                                │
                                                                ▼
[JWT Client] ──HTTPS──▶ [API Gateway :8000] ──▶ [RBAC Check] ──▶ [Adversarial Guard]
                                                                          │
                                                       (blocked)◀─────────┤
                                                                          │ (clean)
                                                                          ▼
                                                               [Model Inference]
                                                                          │
                                                     ┌────────────────────┤
                                                     ▼                    ▼
                                              [AES-GCM store]     [Audit Log]
                                              [PostgreSQL]         [Hash chain]
                                                     │
                                                     ▼
                                          [Prometheus /metrics]
                                          (internal network only)
```

---

## STRIDE Findings

### Finding 1 — Telemetry Ingestion: Spoofing

| Field | Detail |
|-------|--------|
| **Component** | Telemetry Ingestion endpoint |
| **Threat category** | Spoofing |
| **ATT&CK** | T1190 (Exploit Public-Facing Application), T0843 (Program Download — ICS) |
| **Description** | An attacker without a valid client certificate impersonates a legitimate sensor node and injects fabricated telemetry. Crafted readings (e.g. all-nominal values during an active fault) suppress anomaly alerts, delaying maintenance intervention and potentially causing equipment failure. |
| **Impact** | High — suppressed alerts lead to undetected RUL degradation; safety-critical in OT/ICS context |
| **Likelihood** | Medium — requires network access to ingestion endpoint |
| **Mitigation** | mTLS mutual authentication: server presents cert; client must present a cert signed by the PulseNet CA. Unsigned connections are rejected at TLS handshake before any application code runs. |
| **Residual risk** | Compromise of a valid client key. Mitigated by short-lived client certs (90-day rotation) and cert revocation via CRL. |
| **Control type** | Preventive |
| **Verification** | `tests/test_security.py::test_mtls_rejects_unsigned_client` |

---

### Finding 2 — Model Inference API: Tampering

| Field | Detail |
|-------|--------|
| **Component** | Model Inference API |
| **Threat category** | Tampering |
| **ATT&CK** | T1685 (Adversarial ML — Evasion), T1692 (ICS Impair Process Control) |
| **Description** | An authenticated tenant crafts adversarial feature vectors — inputs that appear statistically plausible but are designed to cross the Isolation Forest decision boundary, forcing a "healthy" classification during an actual fault. This is a white-box evasion attack if the tenant can observe prediction outputs. |
| **Impact** | High — a missed anomaly means no maintenance alert; can lead to equipment damage |
| **Likelihood** | Low-medium — requires authenticated access and knowledge of the feature space |
| **Mitigation** | Adversarial guard (`src/pulsenet/security/adversarial_guard.py`) computes z-scores across all features against the training distribution. Inputs with any feature outside ±4σ are rejected with HTTP 422 before reaching the model. Detection is logged and counted by `adversarial_detections_total`. |
| **Residual risk** | Adversarial inputs crafted within ±4σ bounds. Statistical guard does not catch all evasion attacks; this is acknowledged in the model's honest benchmark (Recall=0.43). |
| **Control type** | Preventive |
| **Verification** | `tests/test_security.py::test_adversarial_input_flagged_with_recall_1` |

---

### Finding 3 — Audit Log: Repudiation

| Field | Detail |
|-------|--------|
| **Component** | Audit Log |
| **Threat category** | Repudiation |
| **ATT&CK** | T1070 (Indicator Removal), T1565 (Data Manipulation) |
| **Description** | A tenant (or a compromised internal service) disputes having issued a prediction request, or a malicious insider deletes/modifies audit log entries to hide unauthorized access. Without log integrity controls, there is no forensic basis to prove the event occurred. |
| **Impact** | Medium — regulatory and forensic consequence; doesn't affect real-time safety |
| **Likelihood** | Low — requires write access to audit log storage |
| **Mitigation** | SHA-256 hash-chained, append-only audit log (`src/pulsenet/security/audit.py`). Each entry's `previous_hash` field contains the digest of its predecessor. `verify_audit_log()` replays the full chain and reports any broken links, deleted entries, or modifications. The audit volume is mounted read-only from the application's perspective after initial write (append-only file descriptor). |
| **Residual risk** | An attacker with full disk access can rewrite the entire chain. Mitigate by shipping logs to an external WORM store (e.g. AWS CloudWatch Logs with log group retention policy). |
| **Control type** | Detective / Corrective |
| **Verification** | `tests/test_security.py::test_audit_log_hash_chain_integrity` |

---

### Finding 4 — Multi-tenant Data: Information Disclosure

| Field | Detail |
|-------|--------|
| **Component** | Multi-tenant data store (PostgreSQL) |
| **Threat category** | Information Disclosure |
| **ATT&CK** | T1530 (Data from Cloud Storage), T1213 (Data from Information Repositories) |
| **Description** | A JWT token belonging to Tenant A is used to query prediction history belonging to Tenant B. This is a horizontal privilege escalation ("confused deputy") attack enabled by missing or bypassable tenant scoping in database queries. |
| **Impact** | High — exposes proprietary RUL telemetry, operating schedules, equipment health data |
| **Likelihood** | Low — requires a valid JWT (own or stolen); exploitable only if query filtering is missing |
| **Mitigation** | Every database query is filtered by `tenant_id` extracted from the verified JWT `sub` claim at the ORM layer (`WHERE tenant_id = :jwt_tenant_id`). The JWT is RS256-signed; `tenant_id` cannot be forged without the private key. RBAC layer validates the claim before the query executes. |
| **Residual risk** | JWT theft. Mitigated by 15-minute token expiry and token binding to `client_id` claim. |
| **Control type** | Preventive |
| **Verification** | `tests/test_security.py::test_rbac_blocks_cross_tenant_access` |

---

### Finding 5 — FastAPI Endpoint: Denial of Service

| Field | Detail |
|-------|--------|
| **Component** | FastAPI inference endpoint |
| **Threat category** | Denial of Service |
| **ATT&CK** | T1499 (Endpoint Denial of Service), T1498 (Network Denial of Service) |
| **Description** | An unauthenticated (or authenticated) attacker floods the `/predict` endpoint with high-volume requests. Each request triggers schema validation and possibly inference, saturating CPU and blocking legitimate tenants. |
| **Impact** | Medium — service unavailability; no data loss |
| **Likelihood** | Medium — unauthenticated flood is straightforward |
| **Mitigation** | Two-layer rate limiting via `slowapi`: (1) unauthenticated requests limited by IP at 10 req/min; (2) authenticated requests limited by JWT `sub` claim at 100 req/min per tenant, preventing per-IP bypass via IP rotation. Auth check occurs before queue entry — unauthenticated requests are rejected immediately. Prometheus alert fires on request spike. |
| **Residual risk** | Distributed flood from many IPs. Recommend upstream WAF (AWS WAF / CloudFront) for production. |
| **Control type** | Preventive |
| **Verification** | Integration test in `tests/test_api.py::test_rate_limiter_enforced` |

---

### Finding 6 — CI/CD Pipeline: Elevation of Privilege

| Field | Detail |
|-------|--------|
| **Component** | GitHub Actions CI/CD pipeline |
| **Threat category** | Elevation of Privilege |
| **ATT&CK** | T1195.001 (Compromise Software Dependencies — Supply Chain), T0873 (ICS Project File Infection) |
| **Description** | A malicious pull request injects code that bypasses RBAC checks or weakens encryption, then self-approves or exploits a permissive branch protection rule to merge to main, shipping a backdoored artifact. |
| **Impact** | Critical — a backdoored model or auth bypass affects all tenants |
| **Likelihood** | Low — requires repository write access or a compromised contributor account |
| **Mitigation** | Branch protection on `main`: required reviews (min 2), no self-approval. CodeQL SARIF must pass as a required status check — PRs introducing security-flagged patterns are blocked. GitHub Actions versions are pinned by SHA (no floating tags). Dependency hashes pinned in `requirements.txt`. |
| **Residual risk** | Compromise of a reviewer's account. Mitigate with CODEOWNERS file and org-level SSO enforcement. |
| **Control type** | Preventive |
| **Verification** | `results/security.sarif` committed; CI badge on README |

---

### Finding 7 — Training Pipeline: Data Poisoning (Tampering)

| Field | Detail |
|-------|--------|
| **Component** | Model training pipeline (offline) |
| **Threat category** | Tampering |
| **ATT&CK** | T1685, T0868 (ICS Modify Control Logic) |
| **Description** | Attacker injects mislabeled or adversarial samples into the C-MAPSS training dataset, causing the trained Isolation Forest to misclassify fault conditions as healthy. Poisoning 5% of training data can shift the decision boundary enough to suppress anomaly alerts. |
| **Impact** | High |
| **Likelihood** | Low |
| **Mitigation** | Training data provenance tracked in data/official/CMAPSSData.zip SHA-256 manifest. Training runs in isolated CI environment with no external network access. Dataset version pinned in pyproject.toml. |
| **Control type** | Preventive / Detective |

---

### Finding 8 — Model Checkpoint Storage: Tampering

| Field | Detail |
|-------|--------|
| **Component** | Serialized model artifacts (.joblib files) |
| **Threat category** | Tampering |
| **ATT&CK** | T1683.001 (ML Supply Chain — Model Tampering), T0843 (ICS Program Download) |
| **Description** | A backdoored model checkpoint is substituted in the artifact store between training and serving. The backdoored model produces healthy predictions for specific sensor signatures chosen by the attacker. |
| **Impact** | Critical |
| **Likelihood** | Low |
| **Mitigation** | SHA-256 digest of each artifact committed to models/manifest.json at train time. Registry verifies digest on load, raises ModelIntegrityError on mismatch. S3 Object Lock (WORM) in production. |
| **Control type** | Detective |

---

### Finding 9 — Feature Engineering Pipeline: Tampering

| Field | Detail |
|-------|--------|
| **Component** | Feature extraction and normalization pipeline |
| **Threat category** | Tampering |
| **ATT&CK** | T1565 (Data Manipulation) |
| **Description** | If a compromised upstream service can inject values into the feature pipeline (e.g. via a poisoned Kafka topic), normalised features could shift the Isolation Forest score silently without triggering input validation. |
| **Impact** | Medium |
| **Likelihood** | Low |
| **Mitigation** | Feature pipeline inputs validated against training-distribution statistics (mean ± 5σ). Pipeline runs in isolated container with no external write access post-deployment. |
| **Control type** | Preventive |

---

### Finding 10 — CI/CD Pipeline Integrity: Elevation of Privilege

| Field | Detail |
|-------|--------|
| **Component** | GitHub Actions CI/CD pipeline |
| **Threat category** | Elevation of Privilege |
| **ATT&CK** | T1195.001 (Supply Chain Compromise), T0873 (ICS Project File Infection) |
| **Description** | A malicious PR injects code that weakens the adversarial guard threshold or disables RBAC checks, then is merged via a compromised reviewer account or bypassed branch protection. |
| **Impact** | Critical |
| **Likelihood** | Low |
| **Mitigation** | Branch protection: min 2 reviewers, no self-approval. CodeQL SARIF required status check. GitHub Actions pinned by SHA. CODEOWNERS enforced. |
| **Control type** | Preventive |

---

### Finding 11 — Data Lineage Store: Information Disclosure

| Field | Detail |
|-------|--------|
| **Component** | Data lineage metadata store (docs/DATA_LINEAGE.md + provenance.json) |
| **Threat category** | Information Disclosure |
| **ATT&CK** | T1083 (File and Directory Discovery), T1213 (Data from Information Repositories) |
| **Description** | Lineage metadata reveals training dataset versions, pipeline configurations, and model genealogy. If exposed externally, this helps an attacker fingerprint the model architecture and craft more targeted adversarial inputs. |
| **Impact** | Low-Medium |
| **Likelihood** | Low |
| **Mitigation** | Lineage metadata stored in internal-only paths, not served via the public API. Prometheus /metrics endpoint (which could expose model version) is internal-network only. |
| **Control type** | Preventive |

---

### Finding 12 — Admin Interface: Elevation of Privilege

| Field | Detail |
|-------|--------|
| **Component** | Administrative interface (model retraining trigger, tenant management) |
| **Threat category** | Elevation of Privilege |
| **ATT&CK** | T1078 (Valid Accounts), T1548 (Abuse Elevation Control Mechanism) |
| **Description** | The admin API endpoints (train trigger, tenant CRUD) require elevated JWT role. If an analyst-role JWT is accepted by an admin endpoint due to missing role check, a tenant could trigger arbitrary model retraining. |
| **Impact** | High |
| **Likelihood** | Low-Medium |
| **Mitigation** | Admin endpoints require role=admin claim in JWT. Role hierarchy enforced in RBAC middleware: viewer < analyst < operator < admin. Unit tests assert analyst JWT rejected by admin routes. |
| **Control type** | Preventive |

---

## Risk Summary

| Finding | Category | Impact | Likelihood | Residual Risk | Status |
|---------|----------|--------|------------|---------------|--------|
| 1 — Telemetry Spoofing | Spoofing | High | Medium | Low (cert rotation) | Mitigated |
| 2 — Adversarial Tampering | Tampering | High | Low-Med | Medium (within-σ evasion) | Partially mitigated |
| 3 — Audit Repudiation | Repudiation | Medium | Low | Low (external WORM recommended) | Mitigated |
| 4 — Tenant Data Disclosure | Info Disclosure | High | Low | Low (JWT theft window) | Mitigated |
| 5 — API DoS | DoS | Medium | Medium | Low-Med (no WAF) | Partially mitigated |
| 6 — CI/CD Privilege Escalation | Elevation | Critical | Low | Low (account compromise) | Mitigated |
| 7 — Training Data Poisoning | Tampering | High | Low | Low (pinned dataset, isolated CI) | Mitigated |
| 8 — Model Checkpoint Tampering | Tampering | Critical | Low | Low (manifest + WORM) | Mitigated |
| 9 — Feature Pipeline Tampering | Tampering | Medium | Low | Low (distribution validation) | Mitigated |
| 10 — CI/CD Pipeline Integrity | Elevation | Critical | Low | Low (branch protection + SHA pins) | Mitigated |
| 11 — Data Lineage Disclosure | Info Disclosure | Low-Med | Low | Low (internal-only paths) | Mitigated |
| 12 — Admin Interface EoP | Elevation | High | Low-Med | Low (RBAC role hierarchy) | Mitigated |

---

## Accepted Residual Risks

The following residual risks are accepted for the current prototype scope and should be addressed
before production deployment:

1. **Adversarial within-σ evasion** (Finding 2): The statistical guard does not catch all evasion
   attacks. Model Recall=0.43 means ~57% of true anomalies are missed. A human-in-the-loop review
   step is required before using PulseNet in a safety-critical environment.

2. **No external WORM log shipping** (Finding 3): The hash-chained log is rewritable by an attacker
   with full disk access. Ship logs to AWS CloudWatch Logs or an S3 WORM bucket for production.

3. **No JWT revocation list** (Finding 7): Stolen tokens are valid for up to 15 minutes. Implement
   a Redis-backed blocklist for privileged operations (admin, bulk export).

4. **No WAF for unauthenticated DoS** (Findings 5, 12): Rate limiting on the application layer
   does not protect against volumetric attacks. Add AWS WAF with Bot Control managed rules.

---

## Review Cadence

This threat model should be re-reviewed:
- On any change to the authentication or authorization flow
- On any new external-facing endpoint
- On any new data store or encryption boundary
- At least annually, or after any security incident

---

*Threat model follows STRIDE methodology. ATT&CK mappings reference MITRE ATT&CK v19 and ICS.*
