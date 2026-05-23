# PulseNet Design Notes

## Scope

PulseNet is a single-service reference implementation for official NASA
C-MAPSS FD001 predictive maintenance verification. It is not documented as a
deployed production fleet.

## Architecture

1. `official_cmapss.py` verifies the NASA archive hash and extracts FD001.
2. `ingestion.py` parses the 26-column C-MAPSS schema.
3. `preprocessing.py` adds rolling features and fits scalers on train data only.
4. `IsolationForestModel` trains on early-cycle healthy windows.
5. FastAPI exposes authenticated prediction, training, health, and audit routes.
6. Audit logs include hash integrity checks and tenant-scoped files.

## Security Boundaries

- Data provenance is enforced through archive SHA-256 verification.
- JWT signing secret and users must come from environment configuration.
- Tenant identifiers are constrained to a safe character set before audit paths
  are built.
- Runtime artifacts, keys, ledgers, model binaries, and extracted raw data are
  ignored by git.

## Limitations

- GPU acceleration is optional and only used when compatible libraries are
  installed.
- API rate limiting is process-local, not distributed.
- The checked-in validation is local evidence, not a cloud SLO or fleet metric.
