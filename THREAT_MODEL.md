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

### Finding 7 — JWT Token: Spoofing

| Field | Detail |
|-------|--------|
| **Component** | JWT authentication tokens |
| **Threat category** | Spoofing |
| **ATT&CK** | T1078 (Valid Accounts), T1539 (Steal Web Session Cookie) |
| **Description** | An attacker obtains a valid JWT (via network interception, credential stuffing, or session hijacking) and uses it from an unauthorized client to impersonate a legitimate tenant and issue prediction requests or read results. |
| **Impact** | High — full tenant account takeover for the token's lifetime |
| **Likelihood** | Low-medium — depends on token exfiltration opportunity |
| **Mitigation** | Tokens are RS256-signed (private key in Docker secret, never in env). Short-lived: 15-minute expiry enforced server-side. `client_id` claim binds the token to a registered client identifier checked on every request. HTTPS enforced; tokens never logged. |
| **Residual risk** | 15-minute window after theft. Mitigate with token revocation list (Redis-backed blocklist) for high-privilege operations. |
| **Control type** | Preventive |
| **Verification** | `tests/test_security.py::test_rbac_blocks_cross_tenant_access` (expired/wrong-tenant token cases) |

---

### Finding 8 — Encryption Keys: Information Disclosure

| Field | Detail |
|-------|--------|
| **Component** | AES-256-GCM encryption key material |
| **Threat category** | Information Disclosure |
| **ATT&CK** | T1552 (Unsecured Credentials), T1552.001 (Credentials in Files) |
| **Description** | Encryption keys stored as environment variables or hardcoded in source are exposed via `docker inspect`, container logs, or source code scanning. Any attacker with read access to the container environment or the repository can decrypt stored prediction data. |
| **Impact** | Critical — decrypts all at-rest tenant data |
| **Likelihood** | Medium — environment variable leakage is a common misconfiguration |
| **Mitigation** | Keys are stored exclusively in Docker secrets (mounted at `/run/secrets/`; tmpfs, never written to disk) and loaded at startup. The application reads key material from the secrets path; no key values appear in environment variables, config files, or application logs. In production, secrets are sourced from AWS Secrets Manager via the ECS secrets injection mechanism. |
| **Residual risk** | Docker secrets are readable by processes in the same container. Mitigate with read-once key loading and in-memory zeroing after use. |
| **Control type** | Preventive |
| **Verification** | `tests/test_security.py::test_aes_gcm_encryption_decryption`; secret scanning CI step |

---

### Finding 9 — Model Artifact: Tampering

| Field | Detail |
|-------|--------|
| **Component** | Serialized model artifact (.joblib / .pt files) |
| **Threat category** | Tampering |
| **ATT&CK** | T1683.001 (ML Supply Chain — Model Tampering), T0843 (ICS Program Download) |
| **Description** | A backdoored model artifact is substituted for the legitimate trained model, either in the artifact store (S3 bucket) or during the container build. The backdoored model produces "healthy" predictions for specific sensor patterns chosen by the attacker. |
| **Impact** | Critical — silent prediction manipulation affecting all tenants using the model |
| **Likelihood** | Low — requires write access to artifact store or CI pipeline |
| **Mitigation** | SHA-256 digest of each model artifact is recorded in `models/manifest.json` at train time. `src/pulsenet/models/registry.py` recomputes the digest on load and raises `ModelIntegrityError` if it does not match. S3 bucket has Object Lock (WORM) enabled; no direct write access from the inference container. |
| **Residual risk** | Compromise of the manifest itself. Mitigate by signing the manifest with a hardware key (AWS KMS asymmetric key). |
| **Control type** | Detective |
| **Verification** | `tests/test_models.py::test_model_artifact_hash_verified_on_load` |

---

### Finding 10 — Prometheus Metrics: Information Disclosure

| Field | Detail |
|-------|--------|
| **Component** | Prometheus `/metrics` endpoint |
| **Threat category** | Information Disclosure |
| **ATT&CK** | T1083 (File and Directory Discovery), T1046 (Network Service Discovery) |
| **Description** | The `/metrics` endpoint exposes tenant count, per-tenant request rates, adversarial detection counts, and RBAC violation rates. If reachable externally, an attacker can enumerate active tenants, infer usage patterns, and time attacks around low-activity periods. |
| **Impact** | Low-medium — metadata leakage; no direct data access |
| **Likelihood** | Medium — misconfigured port exposure is common |
| **Mitigation** | The `/metrics` endpoint is served on the internal Docker network (`pulsenet_internal`) only. The docker-compose.yml Prometheus service has no `ports:` mapping to the host. The FastAPI route sets `include_in_schema=False` to prevent OpenAPI exposure. Network policy enforced at the Docker driver level (`internal: true` on the network). |
| **Residual risk** | Misconfiguration in future deployments. Mitigate with an automated network exposure test in CI. |
| **Control type** | Preventive |
| **Verification** | Docker Compose network config; `tests/test_api.py::test_metrics_not_on_external_port` |

---

### Finding 11 — Docker Network: Lateral Movement

| Field | Detail |
|-------|--------|
| **Component** | Docker network topology |
| **Threat category** | Lateral Movement (Defense Evasion / TA0005) |
| **ATT&CK** | T1691 (ICS Lateral Movement), T1021 (Remote Services) |
| **Description** | A compromised Streamlit dashboard container (e.g. via a malicious Python dependency) uses its network access to pivot directly to the PostgreSQL database or model inference service, bypassing the API gateway and its auth controls entirely. |
| **Impact** | High — direct DB access allows reading or modifying all tenant data |
| **Likelihood** | Low — requires dashboard container compromise |
| **Mitigation** | Network segmentation: `pulsenet_internal` network has `internal: true` (Docker blocks any external routing). The dashboard is on `pulsenet_external` and communicates with the API on `pulsenet_internal` only via the API service's internal hostname. PostgreSQL and the model service have no interface on `pulsenet_external`. The API is the only bridge between networks and enforces auth on every call. |
| **Residual risk** | Container escape to host. Out of scope for Docker Compose; mitigate with Kubernetes Network Policies and seccomp profiles in production. |
| **Control type** | Preventive |
| **Verification** | `docker-compose.yml` network definitions; `internal: true` on `pulsenet_internal` |

---

### Finding 12 — Rate Limiter Bypass: Denial of Service

| Field | Detail |
|-------|--------|
| **Component** | API rate limiter (slowapi) |
| **Threat category** | Denial of Service |
| **ATT&CK** | T1499.004 (Application or System Exploitation) |
| **Description** | A per-IP rate limiter is trivially bypassed by rotating source IPs (e.g. from a botnet or cloud provider IP range). The attacker sustains a high request volume against the inference endpoint, exhausting worker capacity without triggering the per-IP threshold. |
| **Impact** | Medium — service degradation for legitimate tenants |
| **Likelihood** | Medium — IP rotation is straightforward with cloud infrastructure |
| **Mitigation** | Rate limiting is keyed on the JWT `sub` claim (tenant identity) when the request is authenticated, making IP rotation ineffective for authenticated DoS. For unauthenticated requests, a circuit breaker pattern drops connections once the unauthenticated request queue depth exceeds a threshold. Prometheus alert fires at >1 adversarial detection per second (proxy for abnormal request patterns). |
| **Residual risk** | Anonymous request flood before auth rejection. Recommend AWS WAF managed rule group for bot control at the load balancer layer. |
| **Control type** | Preventive |
| **Verification** | `tests/test_api.py::test_rate_limiter_keyed_on_jwt_sub` |

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
| 7 — JWT Spoofing | Spoofing | High | Low-Med | Low (no revocation list) | Mitigated |
| 8 — Key Disclosure | Info Disclosure | Critical | Medium | Low (Docker secrets) | Mitigated |
| 9 — Model Tampering | Tampering | Critical | Low | Low (manifest signing recommended) | Mitigated |
| 10 — Metrics Disclosure | Info Disclosure | Low-Med | Medium | Low | Mitigated |
| 11 — Lateral Movement | Lateral Movement | High | Low | Low (no K8s policies) | Mitigated |
| 12 — Rate Limit Bypass DoS | DoS | Medium | Medium | Low-Med (no WAF) | Partially mitigated |

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
