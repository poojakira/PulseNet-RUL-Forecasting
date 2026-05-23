# Threat Model

## Assets

- NASA FD001 archive and extracted telemetry files.
- Trained model artifacts and scaler/feature registry files.
- JWT signing secret and user password hashes.
- Tenant-scoped audit logs and hash-chain ledger entries.
- Prediction endpoint availability and integrity.

## Trust Boundaries

1. Official data boundary: only `data.nasa.gov` archive content is trusted after
   SHA-256 verification.
2. API boundary: FastAPI receives untrusted client payloads and tenant headers.
3. Auth boundary: JWT claims are trusted only after signature verification.
4. Runtime boundary: model, scaler, and key files are local runtime artifacts,
   not source-controlled evidence.

## Threats And Controls

| Threat | Control |
| --- | --- |
| Dataset replacement or poisoned archive | Hardcoded NASA archive SHA-256 check before extraction |
| Zip path traversal | Extraction rejects members resolving outside destination |
| Tenant path traversal through `X-Tenant-ID` | Tenant IDs restricted to `[A-Za-z0-9_.-]` and 64 chars |
| Hardcoded API credentials | Users and JWT secret are loaded from environment |
| Predict endpoint abuse | Auth/RBAC, request validation, process-local rate limiter |
| Audit log tampering | Per-entry hash verification |
| Secret leakage through git or container build | `.gitignore` and `.dockerignore` exclude keys, ledgers, CSVs, models |

## Residual Risk

- Rate limiting is not distributed. A production deployment should move this to
  Redis, Envoy, or an API gateway.
- Audit storage is local file based. Production should use append-only object
  storage or a managed log backend with retention policy.
- Model artifacts are not signed yet. A production rollout should add artifact
  signatures and SBOM/provenance gates.
- FD001 is a public benchmark, not live company telemetry. This repository uses
  it because it is official, legally redistributable, and reproducible.
