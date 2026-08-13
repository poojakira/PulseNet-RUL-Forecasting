# Archived Repository Runbook

## Supported operation

There is no supported service operation or deployment. The only supported use is
local, isolated reproduction for source review. Do not expose the API to a
network or connect the code to operational equipment.

## Environment

Use Python 3.11 or 3.12 with the committed lockfile:

```bash
uv sync --locked --extra dev
```

Do not bypass a stale lock with an unconstrained install. The 2026-08-11 audit
found known vulnerabilities in the historical dependency graph; use a disposable
unprivileged environment with no secrets, production data, equipment access, or
network exposure after installation. Review [SECURITY.md](SECURITY.md).

## Verification

```bash
uv run python scripts/verify_archive.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest \
  -p pytest_asyncio.plugin -q -o addopts='' --ignore=tests/test_api.py
uv run ruff check src tests benchmark scripts
```

Record failures and skips exactly. Async tests that skip because their plugin is
missing are not passing tests. API requests that hang are release blockers, not
performance results.

## Benchmark reproduction

```bash
uv run python benchmark/deep_rul_benchmark.py \
  --output /tmp/pulsenet-fd001.json --overwrite
```

Keep generated results outside the repository until an independent reviewer has
checked data hashes, code commit, package lock, hardware, seed, train/test
construction, target construction, and scoring. A result on C-MAPSS remains a
simulator-dataset result and must not be relabeled as field performance.

## Failure handling

- Stop on dataset hash or schema mismatch.
- Stop if a model artifact cannot be loaded with the declared safe loader.
- Stop on test collection errors, hangs, security-audit findings, or unexplained
  metric drift.
- Do not remediate equipment or maintenance systems from this code.

## Security contact

Report vulnerabilities through the process in [SECURITY.md](SECURITY.md). There
is no on-call rotation, SLO, disaster-recovery process, or production incident
response for this archived repository.
