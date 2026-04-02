# PulseNet — Industrial Predictive Maintenance
**RUL prediction and anomaly detection on NASA C-MAPSS for turbofan engines.**

![Architecture Diagram](docs/assets/architecture.png)

## 📌 Overview
PulseNet is an enterprise-ready predictive maintenance platform designed for high-availability aerospace monitoring. It specializes in **Remaining Useful Life (RUL)** estimation and **unsupervised anomaly detection** using the NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset.

## 🚀 Problem Statement
In the aerospace industry, unplanned engine maintenance can cost millions in grounded flights and logistical delays. More critically, undetected engine degradation poses severe safety risks. 
- **RUL Prediction**: Knowing *exactly* when a component will fail allows for just-in-time maintenance, optimizing spare part inventory and reducing downtime.
- **Anomaly Detection**: Identifying "never-seen-before" sensor drifts early can prevent catastrophic failures even before the RUL threshold is reached.

PulseNet bridges the gap between academic research and industrial deployment by providing a secure, scalable, and auditable pipeline.

## 🛠️ Technical Approach

### 1. Data Engineering (C-MAPSS)
- **Normalization**: Sensor-specific Min-Max scaling to handle varying units (K, psia, rpm).
- **Temporal Windowing**: Raw sensor snapshots are transformed into 3D temporal tensors (Window Size: 30-50 cycles) to provide the model with degradation trends.
- **Feature Selection**: Automated removal of flat-line sensors and low-variance features to reduce noise.

### 2. Modeling Strategy
- **RUL Estimation (Supervised)**: A Deep **LSTM (Long Short-Term Memory)** network captures the non-linear degradation path as the engine approaches its EOL (End of Life).
- **Anomaly Detection (Unsupervised)**: **Isolation Forest** identifies statistical outliers in high-dimensional sensor space, flagging potential "infant mortality" or sudden hardware shocks.
- **Ensemble Inference**: The system can run multiple models in parallel (Champion-Challenger) to validate new models against trusted baselines without affecting production uptime.

## ⚙️ MLOps & Architecture
- **Inference Engine**: FastAPI-based server with **Dynamic Batching** to maximize GPU/CPU utilization under heavy load.
- **Unified Feature Registry**: Ensures zero training-serving skew by using the exact same normalization/windowing logic in both training and real-time inference.
- **Security**: 
  - **AES-256 (Fernet)**: All sensitive flight telemetry is encrypted at rest and in transit.
  - **Blockchain-Secured Audit Trail**: Every inference and model transition is recorded in a cryptographically signed SHA-256 hash-chain for compliance and safety audits.
- **Deployment**: Containerized via Docker and orchestrated with Docker Compose (Kubernetes-ready).

## 📊 Experimental Results
*Results based on FD001 dataset benchmarks (Experimental v2.1.0)*

| Metric | Baseline (v1.0) | PulseNet (v2.1.0) |
| :--- | :--- | :--- |
| **RUL RMSE** | 18.5 | **14.2** |
| **RUL MAE** | 15.2 | **11.8** |
| **Anomaly F1** | 0.82 | **0.91** |
| **Inference Latency** | 12ms | **1.70ms** |

*Note: Metrics are preliminary and vary based on operational settings.*

## 📦 Quick Start with Docker
The entire stack (API, Dashboard, MLflow) can be started with a single command:

```bash
docker-compose up --build
```

- **API**: `http://localhost:8000/docs`
- **Dashboard**: `http://localhost:8501`
- **Experiment Tracking**: `http://localhost:5000`

---

## 📈 Project Health & Roadmap
We are actively maintaining PulseNet to reach Tier-1 industrial standard. Check out our **[ROADMAP.md](ROADMAP.md)** for:
- 🟢 **Good First Issues**: UI Polish, API Documentation, Unit Tests.
- 🟡 **Enhancements**: Prometheus exporters, Attention-based models.
- 🔴 **Future Goals**: Edge deployment on NVIDIA Orin/Jetson.
