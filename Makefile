UV ?= uv

.PHONY: archive-check install test-retained lint benchmark

archive-check:
	python3 scripts/verify_archive.py

install:
	$(UV) sync --locked --extra dev

test-retained:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(UV) run pytest -p pytest_asyncio.plugin -q -o addopts='' --ignore=tests/test_api.py

lint:
	$(UV) run ruff check src tests benchmark scripts
	$(UV) run ruff format --check src tests benchmark scripts

benchmark:
	$(UV) run python benchmark/deep_rul_benchmark.py --output /tmp/pulsenet-fd001.json --overwrite
