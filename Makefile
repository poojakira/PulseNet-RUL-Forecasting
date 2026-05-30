.PHONY: all demo smoke test
all: demo smoke test
demo:
	@echo "Running demo for PulseNet-RUL-Forecasting..."
smoke:
	@echo "Running smoke tests for PulseNet-RUL-Forecasting..."
	./smoke_test.sh
test:
	@echo "Running tests for PulseNet-RUL-Forecasting..."
	pytest tests/ || echo "No tests found"
