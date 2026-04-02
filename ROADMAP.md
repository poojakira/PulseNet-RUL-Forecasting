# PulseNet Roadmap 🚀

This document outlines the planned enhancements and community contribution opportunities for the PulseNet Predictive Maintenance platform.

## 🟢 Good First Issues
*These are perfect for new contributors to get familiar with the codebase.*

1.  **Unit Tests for Blockchain Ledger**: `[good-first-issue]`
    - Add comprehensive tests for `src/pulsenet/pipeline/blockchain_ledger.py` to ensure hash chain integrity.
2.  **Dashboard UI Polish**: `[good-first-issue]` `[ui/ux]`
    - Improve the Streamlit dashboard layout, specifically the sensor heatmap sidebar and metric formatting.
3.  **Documentation: API Examples**: `[good-first-issue]` `[docs]`
    - Add a `notebooks/api_tutorial.ipynb` demonstrating how to call the FastAPI endpoints from Python.

## 🟡 Enhancements
*Feature requests and architectural improvements.*

4.  **Prometheus/Grafana Integration**: `[enhancement]` `[ops]`
    - Export real-time model latency and drift metrics to a Prometheus endpoint.
5.  **Attention-based RUL Model**: `[enhancement]` `[ml]`
    - Implement an attention mechanism in the LSTM pipeline to improve the "Approach" section's technical depth.
6.  **Automated Model Retraining**: `[enhancement]` `[mlops]`
    - Create a GitHub Action or a separate worker service that triggers model retraining when data drift is detected.

## 🔴 Future Goals
7.  **Multi-Engine Signal Fusion**: Support combining signals from multiple engine types (not just C-MAPSS FD001).
8.  **Edge Deployment**: Optimize the inference engine for NVIDIA Orin/Jetson devices using TensorRT.
