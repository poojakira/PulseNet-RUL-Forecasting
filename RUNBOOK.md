# Runbook

## Engineering Update - 2026-07-27

Repository: PulseNet-RUL-Forecasting
Purpose: RUL forecasting and fleet operations dashboard

## Build

- Install: make install
- Lint: make lint
- Format: make format
- Test: make test
- Package build: make build
- Security scan: make security
- Full local gate: make verify

## Dashboard

Streamlit/Plotly 3D fleet dashboard in src/pulsenet/dashboard/app.py; launch with make dashboard.

## Dependencies And Data

Single-test run with repo coverage gate needs --no-cov or full-suite coverage context.

## Validation Snapshot

Validated: dashboard py_compile/Ruff passed; tests/test_dashboard.py passed with --no-cov.

## Operating Limits

- Re-check Linux and GitHub Actions after pushing to main.
- Treat local dashboard scores as evidence indicators, not certifications.
- Do not cite production readiness until clean CI, dependency audit, license status, and runtime smoke tests are current.