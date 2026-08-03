# PulseNet — Remaining Useful Life Prediction for Turbofan Engines

PulseNet predicts how many operating cycles a turbofan engine has left before failure, using NASA's C-MAPSS dataset (FD001 subset). It includes a training pipeline, a FastAPI inference server, and a Streamlit dashboard.

## What It Does

- Trains models on NASA C-MAPSS FD001 run-to-failure data (100 training engines, 100 test engines)
- Predicts Remaining Useful Life (RUL) at any point in an engine's operating history
- Serves predictions via a REST API with JWT authentication
- Provides a monitoring dashboard (Streamlit)

## What It Cannot Do

- It only works with C-MAPSS-format sensor data (21 sensors, 3 operational settings). It won't generalize to arbitrary industrial equipment without retraining on new data.
- The anomaly detection model (Isolation Forest) has poor precision (F1 = 0.37 in benchmarks). It catches all failures but generates many false alarms.
- The RUL regression model (Random Forest) achieves RMSE of 15–25 cycles on FD001, which is typical for a classical baseline but not state-of-the-art.
- This is a reference implementation, not a production-certified system. It has not been validated on real-world fleet data.

## Models

| Model | Type | What It Predicts |
|-------|------|-----------------|
| Isolation Forest | Anomaly detection | Binary: "degrading" vs "healthy" |
| Random Forest | Regression | RUL in cycles (0–125, capped) |
| LSTM | Deep learning (sequence) | RUL in cycles |
| Transformer | Deep learning (attention) | RUL in cycles |

The default model is Isolation Forest for anomaly detection. The RUL regression module uses Random Forest with rolling-mean features.

## Benchmark Results (FD001)

From `results/validation_results.json` (ran on official NASA C-MAPSS data):

**Isolation Forest (anomaly detection):**
- F1: 0.54, Precision: 0.71, Recall: 0.43, ROC-AUC: 0.70
- Training time: 0.26 seconds on 45 features, 20,631 training rows

**RUL Regression (Random Forest, official per-unit split):**
- RMSE: 15–25 cycles (test asserts this range; exact value depends on run)
- Uses the C-MAPSS asymmetric scoring function (late predictions penalised more than early ones)
- No random splitting — train and test engines are disjoint, as intended by NASA

**Inference latency (Isolation Forest):**
- Mean: 2.7ms, P99: 4.3ms (500 samples)
- Throughput: ~13,400 samples/sec at batch size 32

## Requirements

- Python 3.10+
- ~250 MB RAM per instance
- GPU optional (only used for LSTM/Transformer training)

## Install

```bash
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting
cd PulseNet-RUL-Forecasting

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -e ".[dev]"
```

## Download the Data

```bash
python scripts/download_data.py
```

This downloads the NASA C-MAPSS dataset (~12 MB zip) to `data/official/` and verifies its SHA-256 hash.

## Run the Pipeline

```bash
# Full pipeline: ingest → preprocess → train → evaluate
python main_pipeline.py --mode full

# Just train
python main_pipeline.py --mode train

# Run predictions
python main_pipeline.py --mode predict

# Run benchmarks
python main_pipeline.py --mode benchmark
```

## Start the API Server

```bash
# Copy and edit environment config
copy .env.example .env
# Fill in PULSENET_JWT_SECRET and PULSENET_ENCRYPTION_KEY

# Start the server
uvicorn pulsenet.api.app:app --host 0.0.0.0 --port 8000
```

API docs at http://localhost:8000/docs once running.

## Start the Dashboard

```bash
streamlit run src/pulsenet/dashboard/app.py
```

Opens at http://localhost:8501.

## Run Tests

```bash
pytest tests/ -v --cov=pulsenet --cov-report=term-missing
```

Tests require 80% coverage to pass (configured in pyproject.toml).

## Docker

```bash
docker build -t pulsenet:latest .
docker-compose up -d
```

## Project Structure

```
src/pulsenet/
  api/          - FastAPI server with JWT auth, prediction and health routes
  dashboard/    - Streamlit monitoring dashboard
  evaluation/   - RUL regression metrics (RMSE, C-MAPSS score)
  models/       - Isolation Forest, LSTM, Transformer, Ensemble
  pipeline/     - Data ingestion, preprocessing, orchestration
  streaming/    - Async producer/consumer for real-time inference
  security/     - Encryption, audit logging
  mlops/        - MLflow experiment tracking
```

## Dataset

NASA C-MAPSS FD001: Simulated turbofan engine degradation data.
- 100 training engines (run to failure)
- 100 test engines (truncated before failure)
- 21 sensor channels + 3 operational settings
- Source: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

## License

Apache-2.0
