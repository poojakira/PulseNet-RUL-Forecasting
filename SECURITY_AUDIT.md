# PulseNet Security Audit Report

**Date:** 2026-08-05  
**Auditor:** agent/security-hardening-v1 (strictness 10/10)  
**Scope:** `src/pulsenet/security/`, `src/pulsenet/api/auth.py`, `src/pulsenet/api/app.py`, `tests/test_security.py`, `.github/workflows/ci.yml`

---

## 1. Purpose

PulseNet is a secure MLOps pipeline for ICS (Industrial Control Systems) predictive
maintenance on NASA C-MAPSS FD001 turbofan engine sensor data.

Core security goals:
- **Confidentiality:** AES-256-GCM encryption of telemetry and model outputs
- **Integrity:** SHA-256 hash-chained append-only audit log with fail-closed verification
- **Authentication:** JWT (HS256) + bcrypt with `iss`/`aud`/`nbf`/`jti` claim validation
- **Adversarial defense:** Statistical z-score guard before inference
- **Supply-chain:** SHA-pinned CI actions, pip-audit dependency scan, CodeQL SARIF gate

---

## 2. Implemented Capabilities (from source)

### 2.1 `src/pulsenet/security/encryption.py`

| Capability | Implementation | Status |
|---|---|---|
| Module-level `encrypt(key, plaintext)` | AES-256-GCM via `AESGCM` | ✅ Implemented |
| Module-level `decrypt(key, token)` | Raises `InvalidTag` on any tamper — never returns default | ✅ Fail-closed |
| `EncryptionManager` class | Fernet (AES-128-CBC + HMAC-SHA256) wrapper | ✅ Implemented |
| `decrypt_cell()` | Re-raises on failure; does NOT return `0.0` silently | ✅ Fixed |
| Key rotation | `rotate_key()` with `.key.bak` backup | ✅ Implemented |
| AWS Secrets Manager | Not implemented | ❌ Documented gap |

Token format: `nonce (12 bytes) || ciphertext || GCM tag` — self-contained, verified on every decrypt.

### 2.2 `src/pulsenet/security/audit.py`

| Capability | Implementation | Status |
|---|---|---|
| Append-only NDJSON with SHA-256 hash chain | `AuditLogger.log_event()` | ✅ Implemented |
| Thread-safe writes | Module-level `threading.Lock` | ✅ Implemented |
| `verify_audit_log()` raises on violations | `AuditIntegrityError` raised; never returns `(False, [...])` | ✅ Fail-closed |
| Missing field detection | Checks `event_id`, `timestamp`, `event_type`, `tenant_id`, `details`, `previous_hash` | ✅ Implemented |
| Timestamp monotonicity check | Enforced | ✅ Implemented |

### 2.3 `src/pulsenet/api/auth.py`

| Capability | Implementation | Status |
|---|---|---|
| JWT `exp` validation | Enforced via python-jose | ✅ |
| JWT `iss` / `aud` validation | Enforced, configurable via env vars | ✅ |
| JWT `nbf` (not-before) | Set on creation, validated on decode | ✅ |
| JWT `jti` (anti-replay UUID) | Included in every token | ✅ |
| bcrypt password hashing | `bcrypt.hashpw` / `checkpw` | ✅ |
| RBAC via `ROLE_PERMISSIONS` | admin / engineer / operator roles | ✅ |
| Bare `except Exception` in `verify_token` | None — only `JWTError` caught | ✅ |

### 2.4 `src/pulsenet/api/app.py`

| Capability | Status |
|---|---|
| CORS wildcard blocked in production | ✅ |
| Per-IP rate limiter (100 req/min) | ✅ |
| Request correlation IDs | ✅ |
| Global exception handler (masks details in prod) | ✅ |
| Fail-loud model load (raises `RuntimeError` if missing) | ✅ |

### 2.5 `.github/workflows/ci.yml`

| Action | SHA-pinned | Comment |
|---|---|---|
| `actions/checkout` (main) | `11bd71901bbe5b1630ceea73d27597364c9af683` | ✅ |
| `actions/checkout` (attack-v19-core) | `11d5960a326750d5838078e36cf38b85af677262` | ✅ Fixed in this PR |
| `actions/setup-python` | `ece7cb06caefa5fff74198d8649806c4678c61a1` | ✅ |
| `actions/upload-artifact` | `ea165f8d65b6e75b540449e92b4886f43607fa02` | ✅ |
| `codecov/codecov-action` | `b9fd7d16f6d7d1b5d2bec1a2887e65ceed900238` | ✅ Fixed in this PR |
| `docker/setup-buildx-action` | `8d2750c68a42422c14e847fe6c8ac0403b4cbd6f` | ✅ |
| `docker/build-push-action` | `10e90e3645eae34f1e60eeb005ba3a3d33f178e8` | ✅ |
| `github/codeql-action/init` | `192325c86100d080feab897ff886c34abd4c83a3` | ✅ |
| `github/codeql-action/analyze` | `192325c86100d080feab897ff886c34abd4c83a3` | ✅ |
| `pip-audit` dependency scan | Added in `test` job | ✅ Added in this PR |
| `permissions: contents: read` at workflow level | Present | ✅ |

---

## 3. Critical Findings (pre-patch state documented for record)

### CRITICAL-1 (FIXED): `decrypt_cell()` was silently returning `0.0` on tamper

**File:** `src/pulsenet/security/encryption.py`  
**Severity:** Critical — now resolved  
**Root cause:** `except (ValueError, TypeError, Exception)` caught all errors and returned `0.0`, allowing attacker-controlled data to reach the ICS model as a "valid" sensor reading.  
**Fix applied:** Re-raises immediately. `decrypt_cell()` docstring now explicitly documents `InvalidToken` propagation.

### CRITICAL-2 (FIXED): Module-level AES-256-GCM functions were absent

**File:** `src/pulsenet/security/encryption.py`  
**Severity:** Critical — now resolved  
**Root cause:** Only `EncryptionManager` (Fernet/AES-128-CBC) existed. Tests fell back to inline implementations, meaning the real module was never exercised for tamper detection.  
**Fix applied:** Module-level `encrypt(key: bytes, plaintext: bytes) -> bytes` and `decrypt(key: bytes, token: bytes) -> bytes` added using `AESGCM`. `decrypt` re-raises `InvalidTag` and never returns a default.

### CRITICAL-3 (FIXED): JWT missing `iss`, `aud`, `nbf`, `jti` claims

**File:** `src/pulsenet/api/auth.py`  
**Severity:** High — now resolved  
**Root cause:** Without issuer/audience validation, JWTs from other services were accepted. Without `jti`, stolen tokens could not be tracked.  
**Fix applied:** `create_token()` and `verify_token()` now include and validate `iss`, `aud`, `nbf`, `jti`.

### CRITICAL-4 (FIXED): Floating `@v4` action tags in CI

**File:** `.github/workflows/ci.yml`  
**Severity:** High — now resolved  
**Root cause:** `actions/checkout@v4` (inner checkout for attack-v19-core) and `codecov/codecov-action@v4` were tag references that could be silently updated by the action author to inject malicious code.  
**Fix applied:** Both pinned to immutable commit SHAs.

---

## 4. High Findings

### HIGH-1 (FIXED): `verify_audit_log()` returned `(False, list)` instead of raising

**File:** `src/pulsenet/security/audit.py`  
**Severity:** High — now resolved  
**Root cause:** Callers could write `verify_audit_log(path)` and discard the return value, silently ignoring integrity failures.  
**Fix applied:** `AuditIntegrityError` exception defined and raised on any violation. The method now documents `(True, [])` as the only non-raising return value.

---

## 5. Medium Findings

### MEDIUM-1 (FIXED): No `pip-audit` dependency vulnerability scanning

**File:** `.github/workflows/ci.yml`  
**Severity:** Medium — now resolved  
**Fix applied:** `pip install pip-audit && pip-audit --desc` step added to `test` job.

### MEDIUM-2 (OPEN): AWS Secrets Manager documented but not implemented

**File:** `src/pulsenet/security/encryption.py`  
**Severity:** Medium — documentation gap  
**Status:** Encryption key is loaded from env var or local file. `boto3`/Secrets Manager is not in the dependency tree. README claim should be corrected or implementation added.

### MEDIUM-3 (OPEN): Rate limiter keyed on IP, not JWT `sub`

**File:** `src/pulsenet/api/app.py`  
**Severity:** Low-medium  
**Status:** README implies per-user rate limiting; implementation uses source IP. Proxies/load balancers may share IPs, reducing effectiveness. Not exploitable on its own.

---

## 6. Unsupported README Claims

| README Claim | Reality |
|---|---|
| "Keys stored in Docker secrets / AWS Secrets Manager" | Keys loaded from `PULSENET_ENCRYPTION_KEY` env var or `.runtime/pulsenet-fernet.key` local file |
| "Rate limiting per JWT sub" | Rate limiting is per source IP |
| "AES-256-GCM encryption at rest and in transit" | `EncryptionManager` uses Fernet (AES-128-CBC + HMAC-SHA256); module-level functions use AES-256-GCM |

---

## 7. Remediation Plan (prioritized)

| Priority | Finding | Action | File | Status |
|---|---|---|---|---|
| P0 | CRITICAL-1 | `decrypt_cell()` re-raises on failure | `encryption.py` | ✅ Done |
| P0 | CRITICAL-2 | Add module-level AES-256-GCM functions | `encryption.py` | ✅ Done |
| P0 | CRITICAL-3 | Add `iss`, `aud`, `nbf`, `jti` to JWT | `auth.py` | ✅ Done |
| P0 | CRITICAL-4 | Pin `codecov` and inner `checkout` actions | `ci.yml` | ✅ Done |
| P1 | HIGH-1 | `verify_audit_log()` raises `AuditIntegrityError` | `audit.py` | ✅ Done |
| P1 | MEDIUM-1 | Add `pip-audit` to CI | `ci.yml` | ✅ Done |
| P2 | MEDIUM-2 | Implement Secrets Manager or correct README | `encryption.py` / `README.md` | Open |
| P3 | MEDIUM-3 | Rate limiter per JWT sub (requires Redis) | `app.py` | Open |

---

## 8. Evidence Policy

See `evidence_policy.json` at repo root for machine-readable experiment provenance.

Committed metrics (from `docs/evidence/validation_results.json`):
- Isolation Forest F1: **0.54** (Precision: 0.71, Recall: 0.43)
- Mean inference latency: **2.7 ms**
- P99 inference latency: **4.3 ms**
- Dataset: NASA C-MAPSS FD001

---

*This audit was performed by an automated security hardening agent. All findings in §3-§5 that are marked "Fixed" have corresponding code changes and tests in this PR.*
