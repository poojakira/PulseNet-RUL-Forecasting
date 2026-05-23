# Contributing

PulseNet contributions must preserve the official-data evidence boundary.

## Required Checks

```powershell
python -m ruff check src tests scripts verify.py
python -m ruff format --check src tests scripts verify.py
python -m pytest tests -q
python verify.py
```

## Data Rules

- Do not add generated telemetry fixtures.
- Tests must use `data/official/CMAPSSData.zip` or a verified download from
  `https://data.nasa.gov/docs/legacy/CMAPSSData.zip`.
- Any new metric must be produced by a checked-in command and documented with
  dataset source, row counts, and hash.

## Security Rules

- Do not commit secrets, local key files, ledgers, model binaries, or runtime
  CSV outputs.
- Do not add permissive production defaults such as wildcard CORS, hardcoded
  JWT secrets, or plaintext passwords.
- Do not make lint, tests, verification, or dependency checks advisory.
