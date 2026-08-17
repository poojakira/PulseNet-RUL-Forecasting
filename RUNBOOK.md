# Archived Repository Runbook

> **⚠️ ARCHIVED PROJECT** — This repository is not under active development.
> No maintenance, updates, or support should be expected. Use only for local
> source review and offline reproduction.

## Prerequisites

- Python 3.12+ (`py --version` on Windows, `python3 --version` on Linux)
- Git
- ~200MB disk (for dependencies including torch)

## Clone

**Windows (PowerShell):**
```powershell
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting.git
cd PulseNet-RUL-Forecasting
```

**Linux/macOS:**
```bash
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting.git
cd PulseNet-RUL-Forecasting
```

## Supported operation

There is no supported service operation or deployment. The only supported use is
local, isolated reproduction for source review. Do not expose the API to a
network or connect the code to operational equipment.

## Environment setup

Requires **Python 3.12+**. Use the committed `requirements.txt` for
installation (the `pyproject.toml` does not declare runtime dependencies).

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

### Security note

The 2026-08-11 audit found known vulnerabilities in the historical dependency
graph. Use a disposable, unprivileged environment with no secrets, production
data, equipment access, or network exposure after installation. Review
[SECURITY.md](SECURITY.md).

## Verification

### Running tests

The full test command, skipping adversarial tests that require additional setup:

**Linux / macOS:**

```bash
pytest tests/ -v --ignore=tests/adversarial
```

**Windows (PowerShell):**

```powershell
pytest tests/ -v --ignore=tests/adversarial
```

To run only fast unit tests (no torch dependency required):

**Linux / macOS:**

```bash
pytest tests/ -v --ignore=tests/adversarial --ignore=tests/test_api.py -k "not transformer"
```

**Windows (PowerShell):**

```powershell
pytest tests/ -v --ignore=tests/adversarial --ignore=tests/test_api.py -k "not transformer"
```

### Linting

**Linux / macOS:**

```bash
ruff check src tests benchmark scripts
```

**Windows (PowerShell):**

```powershell
ruff check src tests benchmark scripts
```

### Archive verification script

**Linux / macOS:**

```bash
python scripts/verify_archive.py
```

**Windows (PowerShell):**

```powershell
python scripts/verify_archive.py
```

## Benchmark reproduction

**Linux / macOS:**

```bash
python benchmark/deep_rul_benchmark.py \
  --output /tmp/pulsenet-fd001.json --overwrite
```

**Windows (PowerShell):**

```powershell
python benchmark/deep_rul_benchmark.py `
  --output $env:TEMP\pulsenet-fd001.json --overwrite
```

Keep generated results outside the repository until an independent reviewer has
checked data hashes, code commit, package lock, hardware, seed, train/test
construction, target construction, and scoring. A result on C-MAPSS remains a
simulator-dataset result and must not be relabeled as field performance.

## Known issues

1. **`transformer_model.py` torch import guard** — The `_PositionalEncoding` and
   `_TransformerAutoencoder` classes require PyTorch. They are guarded behind an
   `if TORCH_AVAILABLE:` block. If torch is not installed, importing the module
   will not fail, but instantiating `TransformerModel` will raise `ImportError`.

2. **Async test configuration** — Tests use `pytest-asyncio` with
   `asyncio_mode = "auto"`. If `pytest-asyncio` is not installed, async tests
   will be collected but skip or error. This is expected in minimal environments.

3. **`pyproject.toml` has no extras** — The `pyproject.toml` declares no
   `[project.optional-dependencies]`. Commands like `uv sync --extra dev` will
   fail. Use `pip install -r requirements.txt` instead.

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
