# PulseNet NVIDIA HPC System Design

This document outlines the core architecture and hardware optimizations applied during the transition to an NVIDIA containerized distributed environment.

## Overview
PulseNet is an enterprise-ready predictive maintenance platform. With the move to NVIDIA HPC, the stack relies strictly on:
- **PyTorch DDP (Distributed Data Parallel)** for multi-node training
- **AMP (Automatic Mixed Precision)** utilizing NVIDIA Tensor Cores
- **RAPIDS cuML** for accelerated anomaly detection on GPU
- **NVIDIA NGC Base Container (`nvcr.io/nvidia/pytorch:23.10-py3`)** for the runtime environment

## Pipeline Strategy
Our `main_pipeline.py` orchestrates the complete lifecycle:
1. **DDP Initialization**: Bootstraps NCCL backend via `torchrun`.
2. **Dynamic Batching**: Groups concurrent inference requests to fully saturate the GPU during FastAPI loads.
3. **Hardware Telemetry**: Incorporates `pynvml` context gathering for GPU health, power usage, and temperature mapping.

*For complete implementation details on the test plans and deployment, please refer to the primary codebase configuration files.*
