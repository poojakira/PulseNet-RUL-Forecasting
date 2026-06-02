# Threat Model — PulseNet-RUL-Forecasting

## System Description

Predictive maintenance system trained on NASA C-MAPSS FD001 data. Exposes a
FastAPI inference endpoint with JWT/RBAC auth, a model registry, a tenant-scoped
audit log, and KS-test drift detection.

---

## Data Flow Diagram (Trust Boundaries)

```
[NASA data.nasa.gov] ─── (SHA-256 verify) ──► [Data pipeline]
                                                     │
                                              [Feature store]
                                                     │
                                              [Training job]
                                                     │
                                           [Model Registry] ──► (artifact NOT signed yet — residual risk)
                                                     │
[Client] ──► (TLS) ──► [API Gateway] ──► [FastAPI /predict]
                            │                      │
                     (JWT verify)            [Model inference]
                            │                      │
                      [RBAC check]          [Audit log (hash-chain)]
                            │
                   [Tenant isolation]

Trust boundary 1: external network → API (TLS + JWT)
Trust boundary 2: raw NASA archive → trusted data (SHA-256 gate)
Trust boundary 3: model store → inference (artifact loaded from local path, not signed)
Trust boundary 4: tenant A data ↔ tenant B data (header-validated tenant ID)
```

---

## STRIDE Threat Table

| STRIDE | Threat | Attacker | Severity | Control | Status |
|--------|--------|----------|----------|---------|--------|
| **S** Spoofing | Forge JWT to impersonate admin | External | Critical | RS256 signature verification; short expiry | ✅ |
| **S** Spoofing | Forge `X-Tenant-ID` header to access other tenant's data | External | Critical | Tenant ID regex + length validation; RBAC | ✅ |
| **T** Tampering | Replace NASA archive with poisoned dataset | External / Insider | High | Hardcoded SHA-256 gate before extraction | ✅ |
| **T** Tampering | Modify trained model artifact on disk | Insider | High | Audit log covers deployment; artifact NOT signed | ⚠️ residual |
| **T** Tampering | Alter audit log entries retroactively | Insider | High | Per-entry hash chain | ✅ |
| **R** Repudiation | Admin denies deploying model version | Insider | Medium | Immutable audit trail with actor field | ✅ |
| **I** Info Disclosure | Model weights stolen via file-system access | Insider | High | Encryption at rest (KMS in production) | ⚠️ residual |
| **I** Info Disclosure | Training data membership inference via prediction API | External | Medium | Confidence rounding; rate limiting | ⚠️ partial |
| **D** DoS | Flood /predict endpoint | External | Medium | Process-local rate limiter (not distributed) | ⚠️ residual |
| **E** Elevation | Exploit misconfigured RBAC role to access admin endpoints | External | High | Principle of least privilege; role enum | ✅ |

---

## ML-Specific Threats

| ML Threat | Description | Likelihood | Impact | Mitigation |
|-----------|-------------|-----------|--------|-----------|
| Data poisoning | Attacker substitutes modified FD001 CSV before SHA-256 check | Low (check is pre-extraction) | Critical | SHA-256 gate; archive sourced from `data.nasa.gov` only |
| Adversarial inputs | Crafted sensor readings shift RUL prediction toward unsafe "healthy" outputs | Medium | High | Input range validation; drift detection on prediction distribution |
| Membership inference | Attacker queries API to determine whether a specific engine unit was in training set | Medium | Medium | Rounded confidence outputs; rate limiting; DP-SGD not yet applied |
| Model inversion | Reconstruct training sensor patterns from repeated API queries | Low | Medium | Rate limiting; output rounding; no raw logit exposure |
| Model theft via extraction | Attacker clones model behavior with enough queries | Low | Medium | Rate limiting; no model file endpoint exposed |
| Supply chain (dependencies) | Malicious package injected into requirements | Medium | High | `pip-audit` in CI; pinned versions |

---

## Residual Risks (Deployment Blockers)

| Risk | Notes |
|------|-------|
| Model artifacts not signed | Ed25519 signing not yet implemented. Production deployment must add artifact signatures before serving. |
| Rate limiting not distributed | Current rate limiter is process-local. Move to Redis or API gateway before multi-replica deploy. |
| Audit storage is local files | Move to append-only object storage (S3 + Object Lock) in production. |
| No DP-SGD on training | Membership inference advantage is non-zero. Add DP training or output perturbation for sensitive deployments. |
| JWT key rotation not automated | Manual rotation. Add automated rotation schedule and revocation endpoint. |

---

## Assets

| Asset | Sensitivity | Location |
|-------|------------|---------|
| NASA FD001 archive | Public (but integrity critical) | `data/official/` |
| Trained model + scaler | High | `models/` (not committed) |
| JWT signing key | Critical | Environment variable only |
| Tenant audit logs | High | `logs/` (not committed) |
| Prediction endpoint availability | Medium | FastAPI process |

---

## Not In Scope

- Physical access to inference hardware
- TLS certificate management (assumed handled by load balancer)
- Availability beyond single-process rate limiting
