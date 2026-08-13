# Archive Decision Record

**Decision date:** 2026-08-11  
**Decision:** ARCHIVE  
**Readiness:** Experimental; not production-ready

## Decision

PulseNet is not retained as an active product. The model and test code remain as
a historical experimental reference, but deployment and portfolio claims are
withdrawn.

## Evidence ledger

| Type | Statement | Evidence | Publisher/date | Retrieved | Confidence |
|---|---|---|---|---|---|
| FACT | The referenced C-MAPSS data is simulated engine-degradation data, not telemetry collected from a deployed fleet. | [NASA dataset page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data): NASA describes multivariate engine time series, sensor noise, and fault growth. | NASA; precise publication date unable to verify from the page | 2026-08-11 | High |
| FACT | ATT&CK mapping code was outside the packaged `src` tree and was exercised only by its own test module. | Repository search before removal found imports only in `attack_mapping/` and `tests/test_attack_mapping.py`. | Repository commit `34ee4fd`; 2026-08-11 audit | 2026-08-11 | High |
| FACT | The dependency used a floating Git branch while the lock referenced an older commit, causing `uv sync --locked` to fail after the dependency branch changed. | Local command output from `uv sync --locked --extra dev`, 2026-08-11. | Reproducible local audit | 2026-08-11 | High |
| FACT | The former README cited files and modules that did not exist, including a latency artifact, `rbac.py`, `adversarial_guard.py`, and `api/main.py`. | Repository file inventory and former README at commit `34ee4fd`. | Reproducible repository audit | 2026-08-11 | High |
| FACT | The former deployment manifests were not accompanied by recorded load, recovery, penetration, or operational acceptance results. | Repository file inventory at commit `34ee4fd`; no such result artifacts found. | Reproducible repository audit | 2026-08-11 | High |
| INFERENCE | A public simulator benchmark does not establish safe maintenance utility on physical equipment. | The dataset provenance above and absence of field-validation evidence. | Portfolio decision board; 2026-08-11 | 2026-08-11 | High |
| INFERENCE | Maintaining this repository as a product would dilute the portfolio's ML-security thesis and impose disproportionate dependency and safety-review burden. | Scope and production-gate findings in this record. | Portfolio decision board; 2026-08-11 | 2026-08-11 | Medium |

## Production gate result

| Area | Score (0-5) | Rationale |
|---|---:|---|
| Problem validation | 1 | Public benchmark relevance is established; an evidenced operator workflow is not. |
| Differentiation | 1 | No verified advantage under an equivalent external evaluation protocol. |
| Functional completeness | 2 | Model and API code exist, but the operational path is incomplete and internally inconsistent. |
| Security | 1 | Security helpers exist; tenancy, identity, secrets, immutable audit storage, and abuse controls are not production-validated. |
| Reliability | 0 | No accepted load, failure, recovery, backup, or SLO evidence. |
| Test quality | 2 | Unit coverage exists; collection and API execution failed during the fresh audit and async tests can skip without the plugin. |
| Observability | 1 | Metrics code exists; no validated operational monitoring system remains. |
| Documentation | 1 | Prior documentation materially exceeded implemented reality. |
| Deployment reproducibility | 0 | Deployment assets were unverified and have been removed. |
| Integration readiness | 0 | No validated maintenance-system or fleet-telemetry integration. |
| Maintainability | 1 | Large dependency surface and mixed model/API/security concerns. |
| Evidence quality | 1 | Historical JSON exists, but current audit did not reproduce the headline benchmark and field evidence is absent. |

Scores are review judgments, not measured product metrics.

## Removed surfaces

- Floating ATT&CK Git dependency and disconnected mapping helper.
- Static dashboard and generated dashboard data.
- Kubernetes, Terraform, Docker Compose, monitoring, and container manifests.
- Stale generated coverage, SARIF, and SBOM files.
- Unsupported production, security-control, latency, and field-use claims.

## Reopening criteria

Reopening requires a new owner and all of the following before implementation:

1. A documented, lawful dataset and operator workflow representative of the
   intended equipment and failure modes.
2. Independent domain-expert review of labels, target definition, failure costs,
   and maintenance-action boundaries.
3. A preregistered evaluation protocol, leakage analysis, baselines under the
   same preprocessing/split/scoring method, calibration, and uncertainty plan.
4. A new architecture and threat model tied to a real deployment boundary.
5. Reproducible tests for load, failure, rollback, data drift, and recovery.
6. Human approval for every maintenance-impacting action.

Until all six are evidenced, this repository remains archived.
