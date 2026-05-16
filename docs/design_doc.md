# PulseNet System Design Document

## Overview

PulseNet is a production-grade predictive maintenance platform built on NASA C-MAPSS turbofan engine data. The system performs unsupervised anomaly detection to identify engines approaching failure, deployed as a secure REST API with real-time streaming capabilities.

## Hardware Acceleration Strategy

The codebase supports **optional GPU acceleration** at multiple levels:

| Component | CPU Path | GPU Path |
|-----------|----------|----------|
| Isolation Forest | scikit-learn | RAPIDS cuML (if available) |
| LSTM Autoencoder | PyTorch CPU | PyTorch CUDA + AMP |
| Transformer Autoencoder | PyTorch CPU | PyTorch CUDA + AMP |
| Multi-GPU Training | N/A | PyTorch DDP via `torchrun` |

### DDP (Distributed Data Parallel) Support

For multi-GPU training, the system supports PyTorch DDP with NCCL backend:

```bash
torchrun --nproc_per_node=NUM_GPUS src/pulsenet/benchmarks/ddp_benchmark.py
```

Implementation details:
- NCCL backend on Linux, Gloo fallback on Windows
- `DistributedSampler` for data sharding across GPUs
- `SyncBatchNorm` conversion for Transformer model
- Only Rank 0 saves model artifacts (prevents write collisions)
- AMP (Automatic Mixed Precision) via `torch.amp.GradScaler` for Tensor Core utilization

### Container Strategy

The Dockerfile uses `nvcr.io/nvidia/pytorch:23.10-py3` as the base image, which includes:
- CUDA toolkit and cuDNN
- PyTorch with CUDA support pre-built
- NCCL for multi-GPU communication

This enables GPU inference without any additional setup if deployed on GPU-equipped instances (AWS `p3`/`g4dn`, GCP `a2`).

## Pipeline Strategy

The `main_pipeline.py` orchestrates five stages:

1. **Ingestion** — Load C-MAPSS .txt files, validate schema, handle NaN/Inf
2. **Preprocessing** — Rolling features, MinMax normalization, encrypted feature storage
3. **Training** — Model-specific formatting (flat array vs 3D sequences), versioned artifact saving
4. **Evaluation** — Multi-model comparison on test set with ground truth labels
5. **Inference** — Predict on test data, log results to blockchain ledger

## Key Design Patterns

- **Feature Registry** — Single source of truth for feature transformations (prevents training-serving skew)
- **Model Registry** — Lazy-loading, multi-model comparison, best-model selection
- **Dynamic Batching** — Groups concurrent API requests for GPU throughput optimization
- **Blockchain Audit** — Tamper-evident prediction logging for regulated environments
- **Shadow Mode** — Parallel inference with primary + shadow model for safe rollouts
