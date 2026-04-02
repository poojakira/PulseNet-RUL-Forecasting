"""
Transformer autoencoder for sequence anomaly detection.

Lightweight positional-encoded Transformer encoder with linear decoder,
trained to reconstruct normal sequences. Anomaly = high reconstruction error.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pulsenet.logger import get_logger
from pulsenet.models.base import BaseAnomalyModel

log = get_logger(__name__)

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]  # type: ignore


class _TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.decoder(h)


class TransformerModel(BaseAnomalyModel):
    """Transformer autoencoder anomaly detector."""

    name = "transformer"

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        sequence_length: int = 30,
        learning_rate: float = 0.0001,
        epochs: int = 50,
        batch_size: int = 64,
        threshold: Optional[float] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer model")
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = sequence_length
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold = threshold
        self.model: Any = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._n_features: int = 0

    def train(self, X: np.ndarray | Any, **kwargs) -> None:
        if X.ndim != 3:
            raise ValueError(
                f"Transformer expects 3D sequence tensors (N, seq, features), got {X.ndim}D"
            )
        self._n_features = X.shape[2]
        self.model = _TransformerAutoencoder(
            self._n_features,
            self.d_model,
            self.nhead,
            self.num_layers,
            self.dropout,
        )

        dataset = TensorDataset(torch.FloatTensor(X))

        if dist.is_initialized():
            local_rank = dist.get_rank()
            self.device = torch.device(f"cuda:{local_rank}")
            self.model = self.model.to(self.device)
            # Convert any potential BatchNorms to SyncBatchNorm for DDP
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank]
            )
            sampler = torch.utils.data.DistributedSampler(dataset)
            loader = DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
        else:
            self.model = self.model.to(self.device)
            sampler = None
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()  # type: ignore

        self.model.train()
        for epoch in range(self.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():  # type: ignore
                    output = self.model(batch)
                    loss = criterion(output, batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 and (
                not dist.is_initialized() or dist.get_rank() == 0
            ):
                log.info(
                    f"Transformer Epoch {epoch + 1}/{self.epochs}",
                    extra={"loss": f"{total_loss / len(loader):.6f}"},
                )

        train_errors = self._compute_errors(X)
        self.threshold = float(np.percentile(train_errors, 95))
        log.info("Transformer trained", extra={"threshold": f"{self.threshold:.6f}"})

    def predict(self, X: np.ndarray | Any) -> np.ndarray:
        errors = self._compute_errors(X)
        return (errors > self.threshold).astype(int)

    def score(self, X: np.ndarray | Any) -> np.ndarray:
        return self._compute_errors(X)

    def _compute_errors(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError(
                f"Transformer expects 3D sequence tensors (N, seq, features), got {X.ndim}D"
            )
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X).to(self.device)
            output = self.model(tensor)
            errors = torch.mean((tensor - output) ** 2, dim=(1, 2)).cpu().numpy()
        return errors

    @staticmethod
    def _window_flat(X: np.ndarray, seq_len: int) -> np.ndarray:
        seqs = []
        for i in range(len(X) - seq_len + 1):
            seqs.append(X[i : i + seq_len])
        return np.array(seqs)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "threshold": self.threshold,
                "n_features": self._n_features,
                "config": {
                    "d_model": self.d_model,
                    "nhead": self.nhead,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                },
            },
            path,
        )

    def load(self, path: Path | str) -> None:
        data = torch.load(path, map_location=self.device, weights_only=True)
        self._n_features = data["n_features"]
        cfg = data["config"]
        self.model = _TransformerAutoencoder(
            self._n_features,
            cfg["d_model"],
            cfg["nhead"],
            cfg["num_layers"],
            cfg["dropout"],
        ).to(self.device)
        self.model.load_state_dict(data["state_dict"])
        self.threshold = data["threshold"]
