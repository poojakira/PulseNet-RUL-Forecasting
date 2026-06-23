# PulseNet RUL Forecasting

PulseNet is an evidence-driven predictive maintenance service for NASA
C-MAPSS FD001 turbofan telemetry. The repository is scoped to a defensible
ML-security portfolio artifact: official data lineage, reproducible anomaly
detection, API auth/RBAC, tenant traceability, audit logging, and CI smoke
verification.

## Current Evidence

- Official dataset: NASA C-MAPSS FD001.
- Archive URL: https://data.nasa.gov/docs/legacy/CMAPSSData.zip
- Archive SHA-256:
  `74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f`
- Loader: `src/pulsenet/pipeline/official_cmapss.py`
- Verification: `python verify.py`
- Data lineage: `docs/DATA_LINEAGE.md`
- Threat model: `docs/THREAT_MODEL.md`
- Measured evidence: `docs/evidence/validation_results.json`
- Metrics chart: `docs/evidence/validation_metrics.svg`

No generated fixture data is used in tests, CI, or smoke verification.

## Security Scope

- JWT authentication requires configured users and a configured signing secret.
- RBAC protects prediction, training, audit, and verification endpoints.
- `X-Tenant-ID` is propagated into response headers and audit metadata for
  traceability.
- The audit ledger hash-chains API access events and verifies tampering.
- Container runtime runs as a non-root user and does not bake secrets.

This is not presented as a deployed production fleet. It is a reproducible
reference implementation with measured local verification.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python verify.py
```

To regenerate full official-data validation metrics:

```powershell
python scripts/run_validation.py
```

## CI Gates

GitHub Actions run:

- `ruff check`
- `ruff format --check`
- `pytest`
- `python verify.py`
- Docker build

CI must fail on lint, test, or official-data verification failure.
## License
MIT

## Security
See SECURITY.md for vulnerability reporting.

## Contributing
See CONTRIBUTING.md for guidelines.
