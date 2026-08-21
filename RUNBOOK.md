# RUNBOOK  --  PulseNet-RUL-Forecasting

## Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (CPU works but slow)
- ~2 GB disk for NASA C-MAPSS dataset

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify PyTorch sees GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Download Data

```bash
python scripts/download_cmapss.py --output data/
```

Expected structure: `data/train_FD001.txt`, `data/test_FD001.txt`, `data/RUL_FD001.txt` (FD001–FD004).

## Train Model

```bash
python train.py --config configs/default.yaml --dataset FD001
```

Checkpoints saved to `checkpoints/`. Training logs to `logs/`. Typical training: ~50 epochs, ~20 min on GPU.

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --dataset FD001
```

Output: RMSE and score on test set printed to stdout and saved to `results/eval_FD001.json`.

## Generate Reports

```bash
python scripts/generate_report.py --results-dir results/ --output reports/summary.pdf
```

Produces per-engine RUL curves and aggregate metrics as PDF.

## Test

```bash
pytest tests/ -v
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA out of memory | Reduce `batch_size` in config YAML |
| Download script 403 | Manually download from NASA Prognostics repo, place in `data/` |
| RMSE unexpectedly high | Check `max_rul` clipping value in config (default: 125) |
| Missing `RUL_FD00x.txt` | Required for evaluation  --  re-download dataset |
| Report generation fails | Install `pip install matplotlib reportlab` |
