"""
Deep 1D-CNN RUL forecaster for NASA C-MAPSS (PyTorch, CPU-trainable).

Classical last-cycle regressors plateau near RMSE 18 on FD001 because a single
cycle carries little degradation trajectory. State-of-the-art results come from
feeding a *sliding window* of the most recent cycles into a temporal model.

This implements the deep-CNN approach of Li, Ding & Sun (2018), "Remaining
useful life estimation in prognostics using deep convolution neural networks"
(Reliability Engineering & System Safety), which established RMSE ~12.6 on
FD001 - a result classical ML cannot reach. The network is small enough to
train on CPU in a few minutes because FD001 has only 100 engines.

Methodology (no leakage, official protocol):
- 14 informative sensors (constant channels dropped), min-max scaled on TRAIN.
- Sliding windows of length L=30 over each engine's trajectory.
- Piecewise-linear RUL target capped at 125.
- Evaluated on the LAST window of each test engine vs RUL_FD001 ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The 14 sensors that carry degradation signal on FD001 (the standard subset
# used across the C-MAPSS literature; the other 7 are constant or monotone-flat).
FD001_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

WINDOW = 30
RUL_CAP = 125


def _piecewise_rul(n_cycles: int, cap: int = RUL_CAP) -> np.ndarray:
    rul = np.arange(n_cycles)[::-1].astype(float)  # T-1, T-2, ..., 0
    return np.minimum(rul, cap)


def _make_windows(
    df: pd.DataFrame, sensors: list[str], window: int, with_targets: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Build (N, window, n_sensors) sequences per engine.

    For training we emit every window with its RUL target. For test we emit
    only the final window of each engine (the online inference point).
    """
    seqs: list[np.ndarray] = []
    targets: list[float] = []
    for _, unit in df.groupby("unit_number"):
        vals = unit[sensors].to_numpy(dtype=np.float32)
        n = len(vals)
        if with_targets:
            rul = _piecewise_rul(n)
            if n < window:
                pad = np.repeat(vals[:1], window - n, axis=0)
                seqs.append(np.vstack([pad, vals]))
                targets.append(float(rul[-1]))
            else:
                for end in range(window, n + 1):
                    seqs.append(vals[end - window : end])
                    targets.append(float(rul[end - 1]))
        else:
            if n < window:
                pad = np.repeat(vals[:1], window - n, axis=0)
                seqs.append(np.vstack([pad, vals]))
            else:
                seqs.append(vals[n - window : n])
            targets.append(0.0)
    return np.asarray(seqs, dtype=np.float32), np.asarray(targets, dtype=np.float32)


@dataclass
class DeepRULResult:
    rmse: float
    cmapss_score: float
    n_test_engines: int
    epochs: int
    window: int
    model: str = "1D-CNN (Li et al. 2018 architecture)"


def _cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    return float(
        np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0))
    )


def train_and_evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rul_truth: np.ndarray,
    *,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
) -> DeepRULResult:
    """Train the 1D-CNN and evaluate on the official FD001 test protocol."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    sensors = FD001_SENSORS
    # Min-max scale on training data only.
    mins = train[sensors].min()
    maxs = train[sensors].max()
    rng = (maxs - mins).replace(0, 1.0)
    tr = train.copy()
    te = test.copy()
    tr[sensors] = (tr[sensors] - mins) / rng
    te[sensors] = (te[sensors] - mins) / rng

    x_train, y_train = _make_windows(tr, sensors, WINDOW, with_targets=True)
    x_test, _ = _make_windows(te, sensors, WINDOW, with_targets=False)

    # Normalize the RUL target to [0,1] (divide by cap). Regressing a bounded
    # target with tanh/sigmoid-friendly scale converges far better than raw
    # 0-125 values, then we scale predictions back.
    y_train_norm = y_train / float(RUL_CAP)

    # (N, window, C) -> (N, C, window) for Conv1d
    x_train_t = torch.tensor(x_train).permute(0, 2, 1)
    y_train_t = torch.tensor(y_train_norm).unsqueeze(1)
    x_test_t = torch.tensor(x_test).permute(0, 2, 1)

    n_sensors = len(sensors)

    class DCNN(nn.Module):
        """Li, Ding & Sun (2018) deep-CNN: five conv layers (tanh), FC head.

        Four conv layers with 10 filters and kernel 10 (same padding) extract
        temporal degradation features; a 5th 1-filter conv fuses channels; the
        FC head regresses the normalized RUL. tanh is used throughout per the
        paper - ReLU collapses this small-signal regression.
        """

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_sensors, 10, kernel_size=10, padding="same"),
                nn.Tanh(),
                nn.Conv1d(10, 10, kernel_size=10, padding="same"),
                nn.Tanh(),
                nn.Conv1d(10, 10, kernel_size=10, padding="same"),
                nn.Tanh(),
                nn.Conv1d(10, 10, kernel_size=10, padding="same"),
                nn.Tanh(),
                nn.Conv1d(10, 1, kernel_size=3, padding="same"),
                nn.Tanh(),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(WINDOW, 100),
                nn.Tanh(),
                nn.Linear(100, 1),
                nn.Sigmoid(),  # bounded [0,1] to match normalized target
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.features(x))

    model = DCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(x_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        y_pred_norm = model(x_test_t).squeeze(1).numpy()
    # Scale normalized predictions back to cycles.
    y_pred = np.clip(y_pred_norm * float(RUL_CAP), 0.0, RUL_CAP)

    y_true = np.minimum(rul_truth.astype(float), RUL_CAP)
    rmse_val = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return DeepRULResult(
        rmse=round(rmse_val, 3),
        cmapss_score=round(_cmapss_score(y_true, y_pred), 1),
        n_test_engines=len(y_true),
        epochs=epochs,
        window=WINDOW,
    )
