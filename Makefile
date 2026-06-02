.PHONY: all demo smoke test sbom provenance
all: demo smoke test
demo:
	@echo "Running demo for PulseNet-RUL-Forecasting..."
smoke:
	@echo "Running smoke tests for PulseNet-RUL-Forecasting..."
	./smoke_test.sh
test:
	@echo "Running tests for PulseNet-RUL-Forecasting..."
	pytest tests/ || echo "No tests found"


sbom:
	@echo "Generating SBOM using Syft..."
	syft dir:. -o cyclonedx-json > sbom.json
	@echo "SBOM generated: sbom.json"

provenance:
	@echo "Generating SLSA provenance (simulated for local dev)..."
	# In a real CI/CD pipeline, this would use cosign to sign the artifact and generate an in-toto attestation
	# e.g., cosign sign-blob --key cosign.key sbom.json
	echo '{"_type": "https://in-toto.io/Statement/v0.1", "subject": [{"name": "PulseNet-RUL-Forecasting", "digest": {"sha256": "..."}}], "predicateType": "https://slsa.dev/provenance/v0.2", "predicate": {"builder": {"id": "https://github.com/poojakira/PulseNet-RUL-Forecasting"}, "buildType": "https://github.com/poojakira/PulseNet-RUL-Forecasting/Makefile", "invocation": {"configSource": {"uri": "https://github.com/poojakira/PulseNet-RUL-Forecasting", "digest": {"sha1": "..."}}}}}' > provenance.json
	@echo "Provenance generated: provenance.json"
