# Runbook — PulseNet RUL Forecasting

Step-by-step instructions to get PulseNet running from a fresh clone.

## Prerequisites

- Python 3.10 or newer
- pip (comes with Python)
- ~500 MB disk space (dependencies + data)
- Git

## 1. Clone and Set Up

```bash
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting
cd PulseNet-RUL-Forecasting
```

Create a virtual environment:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

Install:

```bash
pip install -e ".[dev]"
```

## 2. Download the NASA C-MAPSS Data

```bash
python scripts/download_data.py
```

This downloads `CMAPSSData.zip` (~12 MB) to `data/official/` and checks the SHA-256 hash. If you're behind a proxy or the NASA URL is down, download manually from https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data and place the zip at `data/official/CMAPSSData.zip`.

## 3. Configure Environment Variables

```bash
copy .env.example .env
```

Edit `.env` and set at minimum:

- `PULSENET_JWT_SECRET` — any random string (used to sign API tokens)
- `PULSENET_ENCRYPTION_KEY` — generate with:
  ```bash
  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  ```

## 4. Run the Training Pipeline

```bash
python main_pipeline.py --mode full
```

This will:
1. Load and parse `train_FD001.txt` and `test_FD001.txt`
2. Engineer features (rolling means, normalization)
3. Train the Isolation Forest model
4. Print evaluation metrics

Expected output: model metrics for each configured model, training completes in under 30 seconds on a laptop.

## 5. Run Tests

```bash
pytest tests/ -v
```

Or with coverage enforcement:

```bash
pytest tests/ -v --cov=pulsenet --cov-report=term-missing --cov-fail-under=80
```

## 6. Start the API Server

```bash
uvicorn pulsenet.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Test it:

```bash
curl http://localhost:8000/health
```

API documentation: http://localhost:8000/docs

## 7. Start the Dashboard (Optional)

```bash
pip install streamlit plotly
streamlit run src/pulsenet/dashboard/app.py
```

Opens in browser at http://localhost:8501.

## 8. Run Benchmarks

```bash
python scripts/run_benchmark.py
```

Output goes to `reports/benchmark_results.json` and `reports/benchmark_report.md`.

## 9. Docker (Alternative to Steps 1-6)

```bash
docker build -t pulsenet:latest .
docker-compose up -d
```

Services:
- API: http://localhost:8000
- Dashboard: http://localhost:8501

## Common Issues

**"NASA C-MAPSS archive not present" in tests:**
Run `python scripts/download_data.py` first. Some tests require the real data.

**Import errors after install:**
Make sure you installed in editable mode (`pip install -e ".[dev]"`). The source lives under `src/`, so `PYTHONPATH=src` is needed if you don't use editable installs.

**"PULSENET_ENCRYPTION_KEY not set":**
Copy `.env.example` to `.env` and generate a key (see step 3).

**Tests fail with coverage below 80%:**
This is intentional. The CI gate requires 80% coverage. Run `pytest --no-cov` to skip the coverage check during development.

## Makefile Shortcuts

If you have `make` available:

| Command | What it does |
|---------|-------------|
| `make install` | Install dependencies from requirements.txt |
| `make test` | Run pytest with coverage |
| `make test-fast` | Run pytest without coverage |
| `make lint` | Check code style with ruff |
| `make lint-fix` | Auto-fix lint issues |
| `make serve` | Start the API server |
| `make dashboard` | Launch the Streamlit dashboard |
| `make benchmark` | Run performance benchmarks |
| `make verify` | Lint + typecheck + test (local CI gate) |
| `make clean` | Remove caches and build artifacts |
