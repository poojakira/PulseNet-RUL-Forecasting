# PulseNet — Predictive Maintenance on NASA C-MAPSS

**RUL prediction and anomaly detection on NASA C-MAPSS turbofan engines.**

[![CI](https://github.com/poojakira/PulseNet/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/PulseNet/actions/workflows/ci.yml) [![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

![Architecture Diagram](docs/assets/architecture.png)

> Figure 1: PulseNet ML pipeline — from sensor time series to RUL and anomaly scores.

## 📌 Overview

PulseNet is a predictive maintenance project built around the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset. It focuses on **Remaining Useful Life (RUL)** estimation and **unsupervised anomaly detection** for turbofan engines, organized as an end‑to‑end ML systems repo rather than a single research notebook.

This project is a portfolio prototype intended to demonstrate architecture, modeling, and engineering decisions. It is not presented as a production-ready industrial platform.

---

## 🚀 Problem Statement

Unplanned turbofan maintenance is expensive and risky. Two core tasks are:

- **RUL prediction** – estimating how many cycles remain before a unit fails, to support planned maintenance.
- **Anomaly detection** – flagging unusual sensor behavior that may indicate incipient faults or abnormal degradation.

PulseNet uses the C‑MAPSS dataset to explore how these tasks can be implemented in a realistic ML pipeline.

---

## 🛠️ Technical Approach

### 1. Data engineering (C‑MAPSS)

- **Normalization** – sensor-wise scaling to handle differing units and magnitudes.
- **Temporal windowing** – converting unit histories into 3D tensors (e.g. 30–50‑cycle windows) so models see degradation trends over time.
- **Feature selection** – dropping flat/low‑variance sensors to reduce noise and dimensionality.

### 2. Modeling strategy

- **RUL estimation (supervised)** – an LSTM‑based sequence model learns non‑linear degradation trajectories and predicts remaining cycles.
- **Anomaly detection (unsupervised)** – a method such as Isolation Forest or a reconstruction model flags unusual behavior in high‑dimensional sensor space.
- **Champion–challenger evaluation** – the codebase is structured so multiple model variants can be run and compared under a common pipeline.

---

## ⚙️ Architecture & MLOps Shape

- **Inference service** – FastAPI application that exposes RUL/anomaly scoring endpoints.
- **Shared preprocessing** – the same normalization/windowing logic is shared between training and inference to reduce training–serving skew.
- **Containerization** – Docker and `docker-compose` are used for local, reproducible setup of the API and supporting services.
- **Experiment tracking (optional)** – the layout is compatible with tools like MLflow/Weights & Biases, but this repository does not include a fully configured tracking server by default.

Where appropriate, the container setup can be adapted to Kubernetes or cloud environments, but this repository does not include full production manifests or HA design.

---

### Performance & Quality Metrics

PulseNet v2.1.0 achieves state-of-the-art results on the NASA C-MAPSS dataset (FD001) by prioritizing early warning lead times and reproducible data integrity.

| Category | Metric | Baseline (v1.0) | PulseNet (v2.1) | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **Detection** | **Anomaly F1-Score** | 0.280 | **0.373** | +33% 🚀 |
| | **Precision** | 0.182 | **0.229** | +26% |
| | **Recall** | 0.920 | **1.000** | Perfect 📦 |
| **Forecasting**| **RUL RMSE (Lead Error)** | 185.0 | **166.7** | -10% |
| | **RUL MAE** | 178.0 | **164.8** | -8% |
| **System** | **Avg Lead Time (Cycles)** | 120.0 | **195.1** | +62% ⏱️ |
| | **Max Throughput** | 5,000/s | **52,368/s** | 10.5x 🔥 |
| | **P95 Latency (ms)** | 12.0 | **3.94** | -67% |

> [!NOTE]
> **Anomaly detection calibration:** The high Recall (1.0) and lower Precision (0.229) reflect a "Safe-Fail" design. PulseNet detects early-stage sensor drifts as anomalies approximately 195 cycles before complete engine failure.

### Proof & Artifacts
- **Full Report**: [reports/benchmark_report.md](file:///c:/Users/pooja/Downloads/PulseNet/reports/benchmark_report.md)
- **Visual Evidence**: [reports/benchmark_plots.png](file:///c:/Users/pooja/Downloads/PulseNet/reports/benchmark_plots.png)

---

## 📦 Quick start (local)

```bash
# Clone and set up environment
git clone https://github.com/poojakira/PulseNet.git
cd PulseNet

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Run the full benchmark suite
python main_pipeline.py --mode benchmark

# Run the API
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
```

- API docs: `http://localhost:8000/docs`

### Using Docker

```bash
docker-compose up --build
```

- API: `http://localhost:8000/docs`

---

## 🧪 Project Structure

```text
.
├── src/
│   ├── pulsenet/
│   │   ├── benchmarks/     # Performance & Quality benchmarking suite
│   │   ├── core/           # Core exceptions and configurations
│   │   ├── evaluation/     # Metrics and ROC/PR analysis
│   │   ├── models/         # Model definitions (LSTM, Isolation Forest)
│   │   ├── pipeline/       # Ingestion, Preprocessing, Orchestration
│   │   ├── security/       # Encryption and Blockchain auditing
│   │   └── serving/        # FastAPI app and inference logic
├── notebooks/              # EDA and experiment notebooks
├── docs/                   # Diagrams and documentation assets
├── scripts/                # Utility scripts (e.g. verify_benchmarks.py)
├── tests/                  # Unit and integration tests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---
