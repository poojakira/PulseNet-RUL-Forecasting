# PulseNet-RUL-Forecasting

> **ARCHIVED** — Experimental project, not maintained.

Remaining useful life prediction on NASA C-MAPSS turbofan data with adversarial input validation for sensor integrity.

## Key Metrics

| Metric | Value |
|--------|-------|
| Dataset | NASA C-MAPSS FD001–FD004 |
| Input features | 21 sensor channels + 3 operational settings |
| Output | RUL prediction (cycles remaining) |
| Evaluation | RMSE + NASA scoring function |
| Adversarial check | Sensor-input perturbation bounds |
| Training samples | ~20,000 degradation trajectories |
| Status | Archived / Experimental |

## Architecture

```
┌──────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Sensor Streams  │────▶│  Input Validator  │────▶│  RUL Predictor  │
│  21 channels     │     │  Adversarial check│     │  Neural network │
└──────────────────┘     └──────────────────┘     └─────────────────┘
        │                         │                        │
        ▼                         ▼                        ▼
  C-MAPSS time series      Reject out-of-bound       RMSE + score
  (multivariate)           sensor readings           benchmarked
```

**Data Pipeline:**
1. Load C-MAPSS degradation trajectories (run-to-failure sequences)
2. Normalize sensor channels per operational condition
3. Apply sliding window to create fixed-length input sequences
4. Validate inputs against learned sensor operational envelopes
5. Predict RUL via neural network
6. Evaluate against published benchmark metrics (RMSE, scoring function)

**Datasets:**

| Subset | Engines (train) | Engines (test) | Fault modes | Operating conditions |
|--------|----------------|---------------|-------------|---------------------|
| FD001  | 100            | 100           | 1 (HPC)     | 1                   |
| FD002  | 260            | 259           | 1 (HPC)     | 6                   |
| FD003  | 100            | 100           | 2           | 1                   |
| FD004  | 249            | 248           | 2           | 6                   |

## Adversarial Input Validation

The input validator checks sensor readings against physical plausibility bounds before inference. This addresses a practical deployment concern in safety-critical ML systems:

- **Sensor spoofing** — adversarial manipulation of input channels to trigger incorrect predictions
- **Faulty sensors** — hardware degradation producing out-of-range values
- **Distribution shift** — operational conditions outside the training envelope

The validator learns per-channel operational envelopes from training data and rejects inputs that violate these bounds, flagging potential adversarial manipulation or sensor failure before the prediction reaches downstream maintenance scheduling.

## Quick Start

```bash
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting.git && cd PulseNet-RUL-Forecasting
pip install -r requirements.txt

# Download NASA C-MAPSS data
python scripts/download_data.py

# Train and evaluate on FD001
python main.py --dataset FD001

# Run with adversarial input validation enabled
python main.py --dataset FD001 --validate-inputs
```

## Relevance to AI Security

Safety-critical ML systems (predictive maintenance, autonomous vehicles, medical devices) face a threat model where adversarial inputs have physical consequences. A manipulated RUL prediction could defer maintenance on failing equipment or trigger unnecessary shutdowns.

Input validation against operational envelopes is a minimal-cost defense that catches both adversarial perturbations and sensor faults. This project applies AI security thinking — input validation, bounds checking, anomaly rejection — to a domain where model failure has real-world safety implications beyond information security.

The approach demonstrates that AI security extends beyond LLMs and classifiers into any ML system where predictions drive physical-world decisions.

## License

Apache License 2.0
