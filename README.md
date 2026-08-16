> **ARCHIVED - Experimental Research Prototype**
>
> This repository is archived, not production-ready, not an active project, and not a security product. It is retained only as a historical NASA C-MAPSS RUL forecasting reference.

# PulseNet RUL Forecasting

PulseNet contains experimental code for Remaining Useful Life forecasting on the NASA C-MAPSS simulated turbofan dataset. Previous production, deployment, security-product, and benchmark-readiness claims are withdrawn.

## Current Status

- Status: archived reference.
- Dataset: NASA C-MAPSS simulated engine-degradation data.
- Runtime support: none promised.
- Deployment support: withdrawn.
- Metrics: do not cite README numbers; reproduce any local result from code and data in a fresh environment.

## What Remains

- Model and pipeline source retained for historical review.
- Tests retained as regression and behavior examples.
- `ARCHIVE.md` records the archive decision, evidence, limitations, and reopening criteria.

## Verification

```bash
python scripts/verify_archive.py
python -m compileall -q src benchmark scripts
```

Broader tests may require installing the pinned project dependencies and are not evidence of production readiness.

## Reopening

Do not restore deployment files, generated evidence artifacts, dashboards, SBOM/SARIF outputs, or production claims unless every reopening criterion in `ARCHIVE.md` is satisfied and independently reviewed.