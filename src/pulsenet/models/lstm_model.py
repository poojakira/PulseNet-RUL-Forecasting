# pyright: reportGeneralTypeIssues=false
"""
LSTM / GRU time-series anomaly detection via reconstruction error.

Architecture: Encoder-Decoder LSTM autoencoder trained on healthy sequences.
Anomaly = high reconstruction error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from pulsenet.logger import get_logger
from pulsenet.models.base import BaseAnomalyModel

log = get_logger(__name__)

if not TORCH_AVAILABLE:
    log.warning("PyTorch not installed — LSTM model unavailable")


class _LSTMAutoencoder(nn.Module):
    """LSTM Encoder-Decoder for sequence reconstruction."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = nn.LSTM(
            n_features,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        _, (hidden, cell) = self.encoder(x)
        # Decode — repeat last hidden for each time step
        seq_len = x.size(1)
        # hidden shape: (num_layers, batch, hidden_size)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder(decoder_input, (hidden, cell))
        return self.output_layer(decoded)


class LSTMModel(BaseAnomalyModel):
    """LSTM autoencoder anomaly detector."""

    name = "lstm"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        sequence_length: int = 30,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 64,
        threshold: Optional[float] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM model")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = sequence_length
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.model: Optional[
            Union[_LSTMAutoencoder, nn.parallel.DistributedDataParallel]
        ] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._n_features: int = 0

    def _ensure_model(self) -> None:
        """Ensure model is trained or loaded."""
        if self.model is None:
            raise RuntimeError(f"Model {self.name} is not trained or loaded.")

    def train(self, X: np.ndarray, **kwargs: Any) -> None:
        """Train on sequences shaped (N, seq_len, features)."""
        if X.ndim != 3:
            raise ValueError(
                f"LSTM expects 3D sequence tensors (N, seq, features), got {X.ndim}D"
            )

        self._n_features = X.shape[2]
        raw_model = _LSTMAutoencoder(
            self._n_features, self.hidden_size, self.num_layers, self.dropout
        )

        dataset = TensorDataset(torch.FloatTensor(X))

        if dist.is_initialized():
            local_rank = dist.get_rank()
            self.device = torch.device(f"cuda:{local_rank}")
            raw_model = raw_model.to(self.device)
            self.model = nn.parallel.DistributedDataParallel(
                raw_model, device_ids=[local_rank]
            )
            sampler = torch.utils.data.DistributedSampler(dataset)
            loader: Any = DataLoader(
                dataset, batch_size=self.batch_size, sampler=sampler
            )
        else:
            self.model = raw_model.to(self.device)
            sampler = None
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Use explicit device type for GradScaler
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        device = next(self.model.parameters()).device
        scaler = torch.amp.GradScaler("cuda" if device.type == "cuda" else "cpu")  # type: ignore

        self.model.train()
        for epoch in range(self.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                with torch.autocast(device_type=("cuda" if device.type == "cuda" else "cpu")):
                    output = self.model(batch)
                    loss = criterion(output, batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item())

            if (epoch + 1) % 10 == 0 and (
                not dist.is_initialized() or dist.get_rank() == 0
            ):
                log.info(
                    f"LSTM Epoch {epoch + 1}/{self.epochs}",
                    extra={"loss": f"{total_loss / len(loader):.6f}"},
                )

        # Set threshold from training reconstruction error
        train_errors = self._compute_errors(X)
        self.threshold = float(np.percentile(train_errors, 95))

        # GPU memory reporting
        if self.device.type == "cuda":
            peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
            reserved_mb = torch.cuda.max_memory_reserved(self.device) / 1024**2
            log.info(
                "GPU memory usage after training",
                extra={
                    "peak_allocated_mb": f"{peak_mb:.1f}",
                    "peak_reserved_mb": f"{reserved_mb:.1f}",
                    "device": str(self.device),
                },
            )
            torch.cuda.reset_peak_memory_stats(self.device)

        log.info("LSTM trained", extra={"threshold": f"{self.threshold:.6f}"})

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions from reconstruction error."""
        if self.threshold is None:
            raise ValueError("Threshold must be set before prediction.")
        errors = self._compute_errors(X)
        return (errors > self.threshold).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Reconstruction error as anomaly score."""
        return self._compute_errors(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw reconstruction error (higher = more anomalous)."""
        return self._compute_errors(X)

    def health_index(self, X: np.ndarray) -> np.ndarray:
        """Convert reconstruction error to 0-100 health index."""
        errors = self._compute_errors(X)
        if self.threshold is None or self.threshold == 0:
            return np.full_like(errors, 100.0)
        
        # Mapping: 0 error -> 100%, threshold -> 50%, 2*threshold -> 0%
        health = np.clip(100 * (1 - (errors / (self.threshold * 2))), 0, 100)
        return health

    def _compute_errors(self, X: np.ndarray) -> np.ndarray:
        self._ensure_model()
        if X.ndim != 3:
            raise ValueError(
                f"LSTM expects 3D sequence tensors (N, seq, features), got {X.ndim}D"
            )

        if self.model is None:
            raise RuntimeError("Model is missing after validation.")
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            output = self.model(tensor)
            errors = torch.mean((tensor - output) ** 2, dim=(1, 2)).cpu().numpy()
        return errors

    @staticmethod
    def _window_flat(X: np.ndarray, seq_len: int) -> np.ndarray:
        """Create sliding windows from flat (N, features) array."""
        seqs = []
        for i in range(len(X) - seq_len + 1):
            seqs.append(X[i : i + seq_len])
        return np.array(seqs)

    def save(self, path: Union[Path, str]) -> None:
        """Persist model and state to disk."""
        self._ensure_model()
        if self.model is None:
            raise RuntimeError("Model is missing after validation.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Handle DDP wrapping
        state_dict = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.parallel.DistributedDataParallel)
            else self.model.state_dict()
        )

        torch.save(
            {
                "state_dict": state_dict,
                "threshold": self.threshold,
                "n_features": self._n_features,
                "config": {
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                },
            },
            path,
        )
        log.info("LSTM model saved", extra={"path": str(path)})

    def load(self, path: Union[Path, str]) -> None:
        """Load model and state from disk."""
        data = torch.load(path, map_location=self.device, weights_only=True)
        self._n_features = data["n_features"]
        model_cfg = data["config"]

        raw_model = _LSTMAutoencoder(
            self._n_features,
            model_cfg["hidden_size"],
            model_cfg["num_layers"],
            model_cfg["dropout"],
        ).to(self.device)

        raw_model.load_state_dict(data["state_dict"])
        self.model = raw_model
        self.threshold = data["threshold"]
        log.info("LSTM model loaded", extra={"path": str(path)})
