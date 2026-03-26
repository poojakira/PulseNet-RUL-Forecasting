# Contributing to PulseNet

Thank you for your interest in contributing to PulseNet! This guide will help you get started.

---

## Development Setup

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional, for containerized runs)
- NVIDIA GPU + CUDA toolkit (optional, for GPU-accelerated training)

### Local Environment

```bash
# Clone the repository
git clone https://github.com/poojakira/PulseNet.git && cd PulseNet

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your configuration

# Run tests to verify setup
PYTHONPATH=src pytest tests/ -v
```

### Docker Environment

```bash
docker-compose up --build
```

---

## Project Structure

All source code lives under `src/pulsenet/`. Key modules:

| Module | Purpose |
|--------|---------|
| `api/` | FastAPI server, JWT auth, RBAC routes |
| `models/` | ML models (IF, LSTM, Transformer, Ensemble) |
| `pipeline/` | Data ingestion, preprocessing, orchestration |
| `security/` | AES-256 encryption, blockchain audit |
| `streaming/` | Async producer/consumer pipeline |
| `dashboard/` | Streamlit real-time dashboard |
| `mlops/` | MLflow tracking, drift detection |
| `benchmarks/` | Performance benchmarking suite |

---

## Coding Standards

- **Style**: Follow PEP 8. We use [Ruff](https://docs.astral.sh/ruff/) for linting.
- **Types**: Add type hints to all function signatures. We use Pyright for type checking.
- **Docstrings**: Google-style docstrings for all public classes and functions.
- **Logging**: Use `from pulsenet.logger import get_logger` — never use `print()` for diagnostics.
- **Config**: All tunable parameters go in `config.yaml`, never hardcoded.

### Linting

```bash
# Run linter
ruff check src/ tests/

# Auto-fix
ruff check --fix src/ tests/
```

---

## Testing

We use `pytest` with 52+ test cases across four suites:

```bash
# Full suite with coverage
PYTHONPATH=src pytest tests/ -v --cov=src/pulsenet --cov-report=term-missing

# Individual suites
PYTHONPATH=src pytest tests/test_models.py -v       # ML models
PYTHONPATH=src pytest tests/test_api.py -v          # API endpoints
PYTHONPATH=src pytest tests/test_security.py -v     # Security & Blockchain
PYTHONPATH=src pytest tests/test_pipeline.py -v     # Data Pipeline
```

### Writing Tests

- Place tests in `tests/` with the naming convention `test_<module>.py`
- Use fixtures from `tests/conftest.py` for shared data
- Aim for 100% test passing locally before opening a PR

---

## Pull Request Workflow

1. **Fork** the repository and create a feature branch from `main`.
2. **Implement** your changes following the coding standards above.
3. **Test** — ensure all 52+ tests pass and add new ones as needed.
4. **Lint & Typecheck** — run `make lint` and `make typecheck`.
5. **Commit** with clear, descriptive messages:
   ```
   feat: add ensemble model with majority voting
   fix: correct threshold optimization edge case
   docs: update API endpoint documentation
   ```
6. **Push** and open a Pull Request against `main`. Ensure you fill out the required `PULL_REQUEST_TEMPLATE.md`.
7. **CI** will automatically run tests, pyright, and ruff. All checks must pass.

---

## Branch Naming

| Prefix | Purpose |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation |
| `refactor/` | Code refactoring |
| `test/` | Test additions/fixes |
| `ci/` | CI/CD changes |

---

## Reporting Issues

- Use the GitHub templates (`bug_report.md` or `feature_request.md`) to report tracking items.
- Include reproduction steps, expected vs. actual behavior
- Attach relevant logs (sanitize any secrets)

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
