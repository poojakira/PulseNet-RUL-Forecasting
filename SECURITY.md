# Security Policy

## Reporting a Vulnerability
Report vulnerabilities to: poojakiranbhardwaj@gmail.com

## Response Timeline
- **48 hours**: Acknowledgment of receipt
- **5 days**: Initial assessment and severity classification
- **30 days**: Fix deployed or mitigation plan communicated

## Dependency Vulnerability Management

The CI `Security Audit` job runs `pip-audit` against `requirements.txt` on every
push. The following advisories were **remediated by upgrading** the affected
dependencies to patched releases:

- `python-multipart` 0.0.20 → 0.0.32
- `requests` 2.32.5 → 2.33.1
- `python-jose` 3.4.0 → 3.5.0 (unlocks `pyasn1` 0.6.3, fixing PYSEC-2026-2263)
- `cryptography` 49.0.0 → 48.0.1 (compatible with the mlflow 3.x bound)
- `mlflow` 2.20.1 → 3.14.0 (clears ~37 advisories)
- `pyarrow` 18.1.0 → 23.0.1
- `pytest` 8.3.5 → 9.0.3
- `starlette` 0.41.3 → 0.49.3 (highest release the pinned FastAPI supports)

### Accepted residual advisories (no compatible fix available)

These are explicitly ignored in the `pip-audit` CI step (with justification) and
tracked here. Each has **no patched release that is compatible** with the project:

| ID | Package | Reason accepted |
|----|---------|-----------------|
| CVE-2025-3000 | torch 2.12.0 | No patched release published upstream. |
| PYSEC-2026-1325 | ecdsa 0.19.2 | No patched release; pure transitive dependency of `python-jose[cryptography]`. |
| PYSEC-2026-161, -248, -249, -2280, -2281 | starlette 0.49.3 | Fixed only in starlette 1.x, which no released FastAPI version supports (FastAPI pins `starlette <0.50`). Upgrading requires a FastAPI major migration and is tracked separately. |

These will be re-evaluated whenever a compatible upstream fix becomes available.

