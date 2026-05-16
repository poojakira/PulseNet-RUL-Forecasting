# =============================================================================
# PulseNet — Developer & Operator Makefile
# =============================================================================
# Run `make help` to list all targets.
# =============================================================================

.PHONY: help install install-dev lock lint lint-fix format test test-fast \
	typecheck security audit fixture train serve dashboard benchmark plot \
	docker docker-up docker-down stack-up stack-down clean

PYTHON  ?= python3
PIP     ?= pip
PYTEST  ?= pytest
RUFF    ?= ruff
SRC     := src/pulsenet

# ----------- Dependency management -----------

## install         — Install runtime dependencies (production)
install:
	$(PIP) install -r requirements.txt

## install-dev     — Install runtime + dev dependencies
install-dev: install
	$(PIP) install -e ".[dev]"

## lock            — Regenerate requirements.lock (deterministic deps)
lock:
	$(PIP) install pip-tools
	pip-compile --resolver=backtracking --output-file=requirements.lock requirements.txt

# ----------- Code quality -----------

## lint            — Run ruff linter (must pass for CI)
lint:
	$(RUFF) check $(SRC) tests/

## lint-fix        — Auto-fix lint issues
lint-fix:
	$(RUFF) check --fix $(SRC) tests/

## format          — Run ruff formatter
format:
	$(RUFF) format $(SRC) tests/

## typecheck       — Run pyright type checker (must pass for CI)
typecheck:
	PYTHONPATH=src pyright $(SRC)

## security        — Run bandit + pip-audit
security:
	bandit -r $(SRC) -ll -ii --skip B101,B104,B311,B301
	pip-audit -r requirements.txt --strict

## audit           — Full local CI: lint + typecheck + tests + security
audit: lint format-check typecheck test security
	@echo "All checks passed."

format-check:
	$(RUFF) format --check $(SRC) tests/

# ----------- Testing -----------

## test            — Run pytest with coverage (CI uses this)
test:
	PYTHONPATH=src $(PYTEST) tests/ -v --cov=$(SRC) --cov-report=term-missing

## test-fast       — Run tests without coverage (faster)
test-fast:
	PYTHONPATH=src $(PYTEST) tests/ -v

# ----------- ML pipeline -----------

## fixture         — Generate small fixture data so the pipeline runs locally
##                   (NOT real C-MAPSS — clearly labeled in data/FIXTURE_README.txt)
fixture:
	$(PYTHON) scripts/generate_test_fixture.py --out data/

## train           — Run full pipeline (ingest → preprocess → train → eval → infer)
train:
	PYTHONPATH=src $(PYTHON) main_pipeline.py --mode full

## benchmark       — Run benchmark suite (writes outputs/benchmarks/*.json + plots)
benchmark:
	PYTHONPATH=src $(PYTHON) main_pipeline.py --mode benchmark

## plot            — Generate benchmark plots from the latest results
plot:
	PYTHONPATH=src $(PYTHON) -m pulsenet.benchmarks.benchmark --plots-only

## demo            — End-to-end local demo: fixture → train → benchmark
##                   Use this when you don't yet have the real C-MAPSS data.
demo: fixture train benchmark
	@echo ""
	@echo "Demo complete. Open the dashboard: make dashboard"
	@echo "Or view raw results in outputs/benchmarks/"

# ----------- Services -----------

## serve           — Start FastAPI server (dev mode, --reload)
serve:
	PYTHONPATH=src uvicorn pulsenet.api.app:app --reload --host 0.0.0.0 --port 8000

## dashboard       — Launch Streamlit operations console
dashboard:
	PYTHONPATH=src streamlit run $(SRC)/dashboard/app.py

# ----------- Docker -----------

## docker          — Build Docker image
docker:
	docker build -t pulsenet:latest .

## docker-up       — Start full stack (api + dashboard + mlflow + prom + grafana)
docker-up:
	docker compose up -d --build

## docker-down     — Stop full stack
docker-down:
	docker compose down

## stack-up        — Alias for docker-up
stack-up: docker-up

## stack-down      — Alias for docker-down
stack-down: docker-down

# ----------- Cleanup -----------

## clean           — Remove caches and benchmark artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	rm -rf outputs/benchmarks/*.json outputs/benchmarks/*.png

## help            — Show this help
help:
	@grep -E '^##' Makefile | sed 's/^## /  /'
