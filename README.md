# PulseNet — Industrial Engine Telemetry Anomaly Detection Pipeline

* MLOps project · Python · PyTorch · scikit-learn · MLflow · FastAPI · Streamlit**

---

## Overview

PulseNet is a predictive maintenance platform built to monitor aerospace engine health and identify early signs of hardware failure. Developed by Pooja Kiran and Rhutvik Pachghare, the system processes turbofan telemetry (NASA C-MAPSS) through a multi-model ML ensemble. It provides automated health scoring, unified feature management, shadow deployment for model validation, and a blockchain-secured audit trail for incident reporting.

---

## The Problem

Jet engines and industrial machinery generate continuous streams of sensor data. When hardware starts to degrade, initial signals are often subtle and non-linear. Standard threshold-based alerts frequently fire too late, leading to increased maintenance costs or hardware failure.

This project focuses on resolving several practical engineering challenges:
1. **Training-Serving Skew**: Ensuring feature engineering (rolling averages, scaling) is consistent between offline training and online inference.
2. **Safe Model Deployment**: Validating "challenger" (shadow) models against production favorites before switching traffic.
3. **Data Integrity**: Securing sensitive telemetry at rest and providing an immutable log of engine status changes.
4. **Inference Throughput**: Handling high-frequency sensor streams with efficient batching for real-time monitoring.

---

## What We Built

### 3‑Model Anomaly Ensemble (`src/pulsenet/models/`)

PulseNet uses three distinct model types to capture different anomaly signatures. Their outputs are combined into a single health index (0–100%):

| Model                    | Architecture                                         | Purpose                                             |
|-------------------------|-----------------------------------------------------|-----------------------------------------------------|
| **Isolation Forest**    | scikit-learn Ensemble                               | Global spatial outlier detection over sensor space  |
| **LSTM Autoencoder**    | PyTorch Recurrent AE                                | Temporal pattern anomalies in sequence windows      |
| **Transformer AE**     | PyTorch Attention AE                                | Long-range context and complex sensor interactions  |

- **Unified Feature Registry**: A centralized component ensures that scaling and rolling statistics are calculated identically during training and live inference.
- **Shadow Deployment**: The API supports a "Shadow Mode" where a second model mirrors production traffic to compare predictions without affecting the live status.

### Security & Integrity (`src/pulsenet/security/`)

- **AES-256 Encryption**: Every feature batch is encrypted using Fernet (AES-128/256) before being written to disk.
- **Blockchain Audit Trail**: Anomaly alerts and engine status transitions are recorded in a SHA-256 hash-chain (ledger).
- **Merkle Tree Validation**: High-performance integrity verification via Merkle Roots to detect tampering in historical data.
- **Multi-Tenant Isolation**: Audit logs and ledger entries support tenant-specific filtering and isolation.

### MLOps Orchestrator (`src/pulsenet/pipeline/`)

The `PipelineOrchestrator` manages the data lifecycle:
1. **Ingestion**: Automated loading of NASA C-MAPSS datasets with variance-based feature selection.
2. **Preprocessing**: Centralized normalization (MinMax) and 3D temporal tensor generation for deep learning models.
3. **Training**: MLflow-tracked experiments for recording hyperparameters, metrics, and model artifacts.
4. **Dynamic Batching**: A FastAPI-based inference runner that groups concurrent requests to increase GPU/CPU throughput.

---

## Tech Stack

| Layer          | Tools / Libraries                     |
|----------------|----------------------------------------|
| **ML Backend** | PyTorch, scikit‑learn, NumPy, pandas  |
| **API Layer**  | FastAPI (Async), Uvicorn, Pydantic    |
| **Security**   | Cryptography (Fernet), SHA-256 Ledger  |
| **Tracking**   | MLflow, Feature Registry              |
| **UI**         | Streamlit, Plotly                     |
| **Tooling**    | Docker, pytest, Ruff, Pyright         |

---

## Results & Validation

### System Performance (Benchmarked on CPU/GPU)

- **Inference Latency**: Median latency of **1.70ms** per sample (P95: 3.56ms, P99: 4.08ms).
- **Throughput**:
    - **624 samples/sec** (Single-request mode)
    - **15,717 samples/sec** (Batch size 32)
    - **78,638 samples/sec** (Batch size 256)
- **Encryption Overhead**: Negligible mean overhead of **0.01ms** per encryption/decryption cycle.
- **Resource Usage**: Core inference engine runs in ~174MB RAM, with minimal GPU VRAM footprint (~838MB).

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/poojakira/PulseNet.git
cd PulseNet

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure (Optional)
cp .env.example .env

# 4. Run the Full Suite (Initialize → Serve → Monitor)
python main_pipeline.py --mode full    # Initialize models and process data
python main.py                         # Start API Server (at http://localhost:8000)
streamlit run src/pulsenet/dashboard/app.py  # Launch Dashboard (at http://localhost:8501)
```

---

## CLI Reference

```bash
python main_pipeline.py --mode full        # End-to-end: ingest → train → evaluate → log
python main_pipeline.py --mode benchmark   # Measure throughput and latency metrics
python main_pipeline.py --mode stream      # Simulation mode for async producer/consumer
python main_pipeline.py --mode train       # (Re)train the active model
python main.py                             # Start Production-grade FastAPI server
```

---

## Team Contributions

### Pooja Kiran
- **Model Engineering**: Designed the 3-model anomaly detection ensemble (Isolation Forest, LSTM, Transformer).
- **Feature Registry**: Implemented the unified feature store to resolve training-serving skew.
- **Security**: Designed the SHA-256 blockchain ledger and Merkle tree verification protocol.

### Rhutvik Pachghare
- **Systems Engineering**: Built the FastAPI backend with dynamic batching and multi-tenant audit logs.
- **MLOps**: Developed the shadow deployment logic and inference orchestration engine.
- **Validation**: Engineered the automated test suite and Streamlit monitoring dashboard.

---

**Version:** v2.1.0 · **License:** Apache 2.0
