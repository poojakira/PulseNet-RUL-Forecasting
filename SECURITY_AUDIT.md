# Security and Readiness Audit

**Audit date:** 2026-08-11
**Result:** ARCHIVE; not production-ready

## Critical findings

1. **No secure tenant boundary.** The prototype accepts `X-Tenant-ID` from the
   caller, while JWT authorization is not bound to that tenant. Header validation
   prevents malformed IDs but does not authorize tenant access.
2. **No validated inference service.** A fresh test run hung on the first health
   endpoint test. Model/scaler startup also depends on local artifacts that are
   not provisioned by a supported release process.
3. **Safety claims exceeded evidence.** C-MAPSS is simulator data; no field
   validation, maintenance outcome study, or domain safety case was found.
4. **Deployment claims exceeded evidence.** Removed manifests had no recorded
   load, recovery, rollback, penetration, or operator acceptance results.
5. **Audit-log language exceeded the boundary.** A local hash chain can expose
   some modifications when the trusted head and key assumptions hold. It does
   not create immutable storage or legal non-repudiation.

## High findings

- The former rate limiter was process-local and source-IP based, despite stronger
  documentation claims. It was not suitable for distributed enforcement.
- The `/metrics` route was mounted on the same application and had no route-level
  authentication; network placement alone was not enforced in code.
- Training was launched as an in-process background task with no durable job
  state, approval workflow, cancellation, or rollback transaction.
- Model and scaler discovery used relative filesystem paths and had no signed
  release manifest or complete provenance verification path.
- Broad exception handlers in inference could conceal model/data contract
  failures and return behavior inconsistent with an operator runbook.

## Test evidence

Fresh audit commands used the repository at commit `34ee4fd` before archive edits:

- Full collection failed because the disconnected ATT&CK helper required an
  incompatible external package.
- Core suite excluding that test did not complete within the bounded audit
  interval and was terminated.
- `tests/test_security.py`: 9 passed in the available system environment.
- `tests/test_models.py`: 12 passed.
- `tests/test_rul_forecaster.py`: 12 passed.
- `tests/test_pipeline.py`: 7 passed and 4 async tests skipped because the async
  plugin was unavailable in that environment.
- `tests/test_api.py` hung on the first health endpoint test and was terminated.

These are local audit observations, not release metrics or proof of security.

## Resolution

The repository is archived instead of receiving incremental production
hardening. See [ARCHIVE.md](ARCHIVE.md). Historical code remains unsupported and
must not be deployed.

The final Python 3.12 `pip-audit 2.10.1` run reported known advisories in ten
packages: `ecdsa`, `starlette`, `mlflow`, `protobuf`, `pyarrow`, `python-dotenv`,
`python-jose`, `python-multipart`, `torch`, and `transformers`. These findings
remain unresolved; none is accepted as safe. Incrementally upgrading this mixed
service and ML stack was rejected because the repository has no approved product
use case. Execution is limited to disposable historical reproduction.

## Post-archive verification

After removing the floating ATT&CK dependency and resolving `uv.lock` with
Python 3.12, the following bounded check completed on 2026-08-11:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  -p pytest_asyncio.plugin -q -o addopts='' --ignore=tests/test_api.py
```

Result: 113 passed, one skipped, and 17 warnings in 72.74 seconds. The excluded
API module remains unsupported because its first test hangs. This result applies
only to the audited commit and environment and does not establish production,
field, safety, or security performance.
