# PulseNet Production Security Audit

Date: 2026-07-23

## Executive assessment

PulseNet is a capable research prototype, not a production-grade industrial service. The highest-risk problems are identity-boundary failures, unsafe key lifecycle behavior, blocking and unbounded runtime components, weak audit durability, and benchmark claims that were not reproducible from committed evidence.

This document records the initial audit. It is not a certification or penetration-test report.

## Severity summary

| Severity | Count | Meaning |
|---|---:|---|
| Critical | 2 | Cross-tenant or cryptographic control can fail by design |
| High | 7 | Material confidentiality, integrity, availability, or operational risk |
| Medium | 8 | Reliability, observability, governance, or hardening deficiency |

## Critical findings

### C-01: Tenant identity is controlled by an untrusted header

`TenantMiddleware` accepts `X-Tenant-ID` and places it into request state without binding it to the authenticated JWT subject or authorization policy. A valid user can therefore request another tenant identifier unless every downstream route independently prevents it.

**Required remediation**

- Add an immutable tenant claim to issued JWTs.
- Resolve tenant context only after authentication.
- Reject a header that conflicts with the token claim.
- Apply tenant scoping at the persistence/repository layer, not only middleware.
- Add negative integration tests proving cross-tenant reads and writes fail.

### C-02: Production encryption can silently create a local key

`EncryptionManager` generates and stores a local Fernet key when configuration is missing. In a horizontally scaled or ephemeral deployment, instances can use different keys or lose the key after replacement. Current rotation also replaces the active key without retaining a decrypt-capable key ring or re-encrypting existing data.

**Required remediation**

- Fail closed in production when no KMS/secret-manager key is configured.
- Introduce key identifiers and a versioned decrypt key ring.
- Re-encrypt data or retain old decrypt-only keys during rotation.
- Record rotation events and test backup/restore.

## High findings

### H-01: Tenant context is omitted from batch audit logging

The batch prediction path writes audit records without passing the request tenant, causing records to fall into the default `public` tenant.

### H-02: Model inference blocks the async event loop

The dynamic batch worker invokes synchronous model methods directly. Slow CPU inference blocks unrelated requests and health endpoints. Use a bounded worker pool or separate model-serving process with deadlines and backpressure.

### H-03: Inference queue and request waits are unbounded

The queue has no maximum size, admission policy, timeout, cancellation cleanup, or overload response. Sustained traffic can exhaust memory and leave unresolved futures.

### H-04: Audit logs are rewriteable

Each audit record contains an ordinary SHA-256 digest of its own content. Anyone who can modify the file can recalculate hashes. Use append-only durable storage, chained/HMAC-authenticated records, external anchoring, restricted writer identities, and alerting.

### H-05: Ledger durability is not guaranteed

The local hash-chain ledger flushes only periodically or for critical events, catches persistence errors, and writes files non-atomically. A crash can lose acknowledged events. Hash chaining alone does not make a locally rewriteable file tamper-proof.

### H-06: Model/scaler trust policy is unsafe

The scaler is loaded with `trusted=True`. Model artifacts need allowlisted types, cryptographic signatures, provenance metadata, immutable storage, and pre-deployment scanning.

### H-07: Input models allow non-finite and unrealistic telemetry

Sensor fields lack finite-value checks and domain bounds. NaN, infinity, extreme values, and physically impossible readings can reach preprocessing and models.

## Medium findings

### M-01: Rate limiting is process-local

The in-memory limiter is inconsistent across workers, resets on restart, grows by client key, and does not define trusted proxy handling. Use an external atomic limiter and authenticated principal/tenant quotas.

### M-02: Metrics can leak operational data

The metrics endpoint is unauthenticated and uses request paths as labels. Protect or isolate it and use normalized route templates to control cardinality.

### M-03: JWT error details are returned to clients

Token verification includes exception text in the response. Return a generic authentication error and log details internally.

### M-04: User storage is process-local configuration

Users loaded from environment variables do not provide revocation, rotation, lockout, MFA, audit history, or centralized policy. Integrate an identity provider for production.

### M-05: Configuration errors fail open to defaults

Malformed configuration can fall back to defaults after printing a warning. Production configuration should fail startup with structured errors.

### M-06: Environment override parsing is incomplete

List and structured fields such as CORS origins cannot be safely represented by the current scalar conversion logic.

### M-07: Runtime state is module-global

Model cache, batcher, users, and rate limiter are module-level globals, complicating multi-worker consistency, testing, reloads, and lifecycle management.

### M-08: Model governance is incomplete

There is no enforced approval chain for training data, artifacts, promotion, rollback, drift response, or emergency disablement.

## Benchmark and research integrity

Exact metrics must not be manually copied into documentation. A claim pipeline should emit signed JSON containing:

- Git commit and dirty-state flag
- Dataset name, source, checksum, and split
- Model/configuration hash
- Python and dependency lock hash
- Hardware and operating system
- Seeds, repetitions, confidence intervals
- Raw and aggregate metrics
- Start/end timestamps

CI should validate schema and provenance. Expensive official benchmarks can run through a controlled release workflow and attach immutable artifacts.

## Required production acceptance gates

1. Tenant-isolation integration suite passes against the real data store.
2. Authentication supports revocation and key rotation.
3. All secrets come from an approved secret manager.
4. Signed model artifacts are verified before load.
5. Load tests establish P50/P95/P99 latency and saturation behavior.
6. Fault injection covers model timeout, storage loss, queue overload, corrupt artifacts, and dependency outage.
7. Audit events survive crash/restart and are externally integrity-anchored.
8. Backup and disaster-recovery exercises meet defined RPO/RTO.
9. Model benchmark artifacts are reproducible and reviewed.
10. A human safety owner approves the deployment boundary and rollback plan.

## Portfolio-wide findings

The related public repositories require the same correction pattern:

- Remove unsupported exact metrics until reproducible evidence exists.
- Make each package declare its real runtime dependencies.
- Test the actual source package rather than only copied ATT&CK mapping code.
- Replace copy-pasted CI with a centrally versioned reusable workflow.
- Separate experimental attack code from deployable defensive services.
- Treat MITRE ATT&CK mappings as classification metadata, not proof of coverage.
- Add threat models, support policies, release signing, SBOMs, and vulnerability disclosure.
