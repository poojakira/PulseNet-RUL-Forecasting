# PulseNet System Architecture

This document describes the design decisions, component interactions, and production patterns implemented in PulseNet.

---

## High-Level Design

PulseNet is structured as a **modular ML microservice** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway Layer                          │
│  FastAPI + CORS + Rate Limiting + JWT Auth + Request IDs         │
├─────────────────────────────────────────────────────────────────┤
│                        Business Logic                             │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ /predict │  │ /train       │  │ /audit, /verify-chain    │  │
│  │ (batch)  │  │ (background) │  │ (blockchain queries)     │  │
│  └────┬─────┘  └──────┬───────┘  └────────────┬─────────────┘  │
├───────┼────────────────┼───────────────────────┼────────────────┤
│       v                v                       v                 │
│  ┌─────────┐    ┌───────────┐         ┌──────────────┐         │
│  │ Dynamic │    │ Training  │         │ BlackBox     │         │
│  │ Batcher │    │ Pipeline  │         │ Ledger       │         │
│  └────┬────┘    └─────┬─────┘         └──────┬───────┘         │
├───────┼────────────────┼──────────────────────┼─────────────────┤
│       v                v                      v                  │
│  ┌────────────────────────────────┐   ┌────────────────┐       │
│  │       Model Registry           │   │ Encryption     │       │
│  │  IF | LSTM | Transformer | Ens │   │ Manager        │       │
│  └────────────────────────────────┘   └────────────────┘       │
├─────────────────────────────────────────────────────────────────┤
│                     Infrastructure                                │
│  Docker | Prometheus | Structured Logging | GPU Telemetry        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

### 1. Model Architecture: Unsupervised Anomaly Detection

**Decision:** Use unsupervised models (Isolation Forest, LSTM/Transformer autoencoders) rather than supervised RUL regression.

**Rationale:**
- In real-world deployment, labeled failure data is scarce and expensive to obtain
- Unsupervised approaches only need "normal" operating data — which is abundant
- The system can detect **novel** failure modes not seen during training
- Aligns with how predictive maintenance is actually deployed in industry

**Trade-off:** Lower precision compared to supervised methods, but higher generalization to unseen failure patterns.

### 2. Feature Registry Pattern

**Decision:** Centralized `FeatureRegistry` class that owns all feature transformations.

```python
# Training (offline)
registry.process_offline(df)  # Rolling stats, groupby transforms

# Inference (online)  
registry.process_online(data, history)  # Same transforms, single event
```

**Rationale:** Eliminates training-serving skew — the #1 cause of silent ML system failures in production. The same code path produces features whether you're training a model or serving a prediction.

### 3. Dynamic Request Batching

**Decision:** The `/predict` endpoint uses a `DynamicBatcher` that accumulates concurrent requests and processes them as a single batch.

```
Request 1 ──┐
Request 2 ──┤──→ [Batcher waits 50ms or fills batch_size=32] ──→ model.predict(batch)
Request 3 ──┘
```

**Rationale:** GPU inference is throughput-bound, not latency-bound. Batching concurrent requests amortizes the GPU kernel launch overhead across multiple predictions, achieving near-linear throughput scaling.

### 4. Blockchain Audit Ledger

**Decision:** SHA-256 hash-chained blocks with Merkle tree verification for prediction logging.

**Rationale:** In regulated industries (aviation, energy), maintenance decisions must be auditable and tamper-evident. If someone modifies a historical prediction, the hash chain breaks and `validate_integrity()` detects it. The Merkle root provides O(log n) verification of any single entry.

**Not a real blockchain** — this is a local append-only ledger with cryptographic integrity guarantees. No consensus mechanism, no distributed nodes. This is the correct pattern for audit trails (similar to Certificate Transparency logs).

### 5. Multi-Tenancy via Header Injection

**Decision:** `X-Tenant-ID` header → per-tenant data isolation in audit trails, ledger, and access logs.

**Rationale:** Enables SaaS deployment where multiple airline operators share the same service but have isolated prediction histories. Middleware pattern keeps tenant logic out of business logic.

### 6. Graceful Shutdown

**Decision:** SIGTERM/SIGINT handler + FastAPI lifespan context manager.

```python
signal.signal(signal.SIGTERM, _signal_handler)

@asynccontextmanager
async def lifespan(app):
    # startup: load model, start batcher
    yield
    # shutdown: stop batcher, flush ledger
```

**Rationale:** Container orchestrators (K8s, ECS) send SIGTERM before killing a pod. Without graceful shutdown, in-flight predictions are lost and the audit ledger may be corrupted.

### 7. Configuration Architecture

**Decision:** Pydantic schema validation + YAML file + environment variable overrides.

Priority order: `ENV_VAR > config.yaml > hardcoded defaults`

**Rationale:**
- Pydantic catches misconfigurations at startup (fail fast)
- YAML file is human-readable and version-controlled
- Env vars enable per-environment customization without rebuilding the image
- Matches 12-factor app methodology

---

## Data Flow

### Training Pipeline

```
Raw C-MAPSS .txt files
    → load_raw() validates schema (26 columns, no NaN)
    → drop_noisy_columns() removes constant/redundant sensors
    → compute_rolling_features() adds temporal context
    → normalize() via MinMaxScaler (fit on train, transform test)
    → Isolation Forest trained on healthy cycles only (time_in_cycles ≤ 50)
    → Model + Scaler + FeatureRegistry config saved as versioned artifacts
```

### Inference Path

```
HTTP POST /predict (JSON sensor readings)
    → JWT validation
    → DynamicBatcher accumulates requests
    → FeatureRegistry.process_online() (same transforms as training)
    → model.predict() + model.score() + model.health_index()
    → Audit entry logged to blockchain ledger
    → PredictionResponse returned with health score + status
```

### Streaming Path

```
SensorProducer reads CSV row-by-row (simulates live telemetry)
    → AsyncStreamQueue (bounded, with backpressure at 80% capacity)
    → InferenceConsumer drains in batches of 32
    → model.predict() on batch
    → Critical events logged to blockchain
    → Metrics updated (processed count, latency, anomaly rate)
```

---

## Security Architecture

```
                    ┌───────────────────┐
                    │   Client Request  │
                    └────────┬──────────┘
                             │
                    ┌────────v──────────┐
                    │  Rate Limiter     │  ← 100 req/min per IP
                    └────────┬──────────┘
                             │
                    ┌────────v──────────┐
                    │  JWT Validation   │  ← HS256, 60min expiry
                    └────────┬──────────┘
                             │
                    ┌────────v──────────┐
                    │  RBAC Check       │  ← role → permissions mapping
                    └────────┬──────────┘
                             │
                    ┌────────v──────────┐
                    │  Business Logic   │
                    └────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              v              v              v
    ┌─────────────┐  ┌────────────┐  ┌──────────────┐
    │ Encrypted   │  │ Audit Log  │  │ Blockchain   │
    │ Data at Rest│  │ (hashed)   │  │ Ledger       │
    └─────────────┘  └────────────┘  └──────────────┘
```

Key management:
- Production: `PULSENET_ENCRYPTION_KEY` env var (injected from secrets manager)
- Development: Auto-generated key saved to `secret.key` (0600 permissions)
- Rotation: `rotate_key()` backs up old key, generates new one, logs the event

---

## Deployment Topology

### Docker Compose (Development/Demo)

```yaml
services:
  api:        # FastAPI + uvicorn (port 8000)
  dashboard:  # Streamlit monitoring (port 8501)
  mlflow:     # Experiment tracking (port 5000)
```

### Production (AWS ECS / Kubernetes)

```
ALB/Ingress → API Service (2+ replicas, auto-scaling on CPU)
                   │
                   ├── Model artifacts from S3/ECR
                   ├── Secrets from AWS Secrets Manager
                   ├── Logs to CloudWatch / ELK
                   └── Metrics to Prometheus/Grafana
```

Kubernetes probes:
- `/healthz` — liveness (process alive)
- `/readyz` — readiness (model loaded and serving)

---

## Model Registry & Versioning

```
models/
├── isolation_forest.joblib           # Latest (symlink-like)
├── isolation_forest_v20260115_143022.joblib  # Versioned artifact
├── isolation_forest_model_card.yaml  # Metadata (author, date, samples, arch)
├── scaler.joblib                     # Feature scaler (must match model version)
└── feature_registry.joblib           # Feature column ordering
```

The `ModelRegistry` class supports:
- Lazy-loading (import model class only when requested)
- Multi-model comparison (`compare_all()`)
- Best-model selection by metric (`best_model(X, y, metric="f1")`)

---

## Observability

| Signal | Implementation | Where |
|--------|---------------|-------|
| Logs | Structured JSON (production) / colored text (dev) | stdout → CloudWatch |
| Metrics | Prometheus counters + histograms | `/metrics` endpoint |
| Traces | X-Request-ID correlation headers | Every request/response |
| GPU | pynvml telemetry (utilization, VRAM, temp, power) | `/health` response |
| Drift | KL divergence vs reference distribution | `MLOpsTracker.detect_drift()` |

---

## Key Files

| File | Purpose |
|------|---------|
| `main_pipeline.py` | CLI orchestrator (full/train/predict/benchmark/stream modes) |
| `src/pulsenet/config.py` | Pydantic config with YAML + env override |
| `src/pulsenet/api/app.py` | FastAPI factory with all middleware |
| `src/pulsenet/models/registry.py` | Multi-model management |
| `src/pulsenet/pipeline/orchestrator.py` | 5-stage pipeline controller |
| `src/pulsenet/streaming/queue.py` | Async bounded queue with backpressure |
| `src/pulsenet/security/blockchain.py` | Hash-chained audit ledger |
| `src/pulsenet/benchmarks/benchmark.py` | Performance measurement suite |
