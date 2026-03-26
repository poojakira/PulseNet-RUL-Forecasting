.PHONY: install lint test typecheck docker train serve dashboard benchmark clean

# ---------------------------------------------------------------------------
# 🛠  PulseNet Makefile — common development targets
# ---------------------------------------------------------------------------

PYTHON   ?= python3
PIP      ?= pip
PYTEST   ?= pytest
RUFF     ?= ruff
SRC      := src/pulsenet

## install        — Install all dependencies
install:
	$(PIP) install -r requirements.txt

## lint           — Run ruff linter + formatter check
lint:
	$(RUFF) check $(SRC) tests/
	$(RUFF) format --check $(SRC) tests/

## lint-fix       — Auto-fix lint issues
lint-fix:
	$(RUFF) check --fix $(SRC) tests/
	$(RUFF) format $(SRC) tests/

## test           — Run full test suite with coverage
test:
	PYTHONPATH=src $(PYTEST) tests/ -v --cov=$(SRC) --cov-report=term-missing

## test-fast      — Run tests without coverage (faster)
test-fast:
	PYTHONPATH=src $(PYTEST) tests/ -v

## typecheck      — Run pyright type checker
typecheck:
	PYTHONPATH=src pyright $(SRC)

## train          — Run full training pipeline
train:
	PYTHONPATH=src $(PYTHON) main_pipeline.py --mode full

## train-ddp      — Multi-GPU DDP training (requires torchrun)
train-ddp:
	torchrun --nproc_per_node=$${NUM_GPUS:-2} $(SRC)/benchmarks/ddp_benchmark.py

## serve          — Start FastAPI server (dev)
serve:
	PYTHONPATH=src uvicorn pulsenet.api.app:app --reload --host 0.0.0.0 --port 8000

## dashboard      — Launch Streamlit monitoring dashboard
dashboard:
	PYTHONPATH=src streamlit run $(SRC)/dashboard/app.py

## benchmark      — Run performance benchmarks
benchmark:
	PYTHONPATH=src $(PYTHON) main_pipeline.py --mode benchmark

## docker         — Build Docker image
docker:
	docker build -t pulsenet:latest .

## docker-up      — Start full stack via docker-compose
docker-up:
	docker-compose up -d

## clean          — Remove caches and artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	rm -rf outputs/benchmarks/*.json outputs/benchmarks/*.png

## help           — Show this help
help:
	@grep -E '^##' Makefile | sed 's/## /  /'
