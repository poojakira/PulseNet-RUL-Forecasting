# PulseNet-RUL-Forecasting

A predictive maintenance ML experiment using the NASA C-MAPSS turbofan engine degradation dataset. Forecasts Remaining Useful Life (RUL) from multivariate sensor time series.

**Status:** Archived. Experimental. Not maintained, not production-ready.

## What It Does

- Trains neural network models on NASA C-MAPSS FD001–FD004 datasets
- Predicts remaining useful life from 21 sensor channels + 3 operational settings
- Includes adversarial sensor-input robustness checks
- Outputs benchmark metrics (RMSE, score function) for comparison with published results

## What It Is Not

This is a single-person ML experiment on a standard benchmark dataset. It is not a production system. There is no live deployment, no real-time inference serving, and no operational monitoring beyond what's needed for experiment tracking.

## Quick Start

```bash
git clone https://github.com/poojakira/PulseNet-RUL-Forecasting.git && cd PulseNet-RUL-Forecasting
pip install -r requirements.txt
python main.py
```

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation). Download via `scripts/download_data.py`.

## License

MIT.
