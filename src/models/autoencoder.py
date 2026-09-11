"""P2: normal-only reconstruction autoencoder over the flattened window.

Fitted on eligible-negative windows of clean (``fault_family == none``)
development episodes. The raw score is the per-window mean squared reconstruction
error mapped to [0, 1] by a monotone, clipped min-max scaling whose bounds are
quantiles of the error on development negatives (``ErrorScaler``). The scaler is
pure numpy so it can be stored in the checkpoint and used without torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ErrorScaler:
    lower: float
    upper: float
    fit_split: str = "development"
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99

    @classmethod
    def fit(cls, errors: np.ndarray, *, lower_quantile: float = 0.01, upper_quantile: float = 0.99,
            split: str = "development") -> "ErrorScaler":
        if split != "development":
            raise ValueError("the error scaling is fitted on development negatives only")
        values = np.asarray(errors, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("errors must be a non-empty finite 1-D array")
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
        lower = float(np.quantile(values, lower_quantile))
        upper = float(np.quantile(values, upper_quantile))
        if upper <= lower:
            upper = lower + 1.0
        return cls(lower, upper, split, lower_quantile, upper_quantile)

    def transform(self, errors: np.ndarray) -> np.ndarray:
        values = np.asarray(errors, dtype=np.float64)
        return np.clip((values - self.lower) / (self.upper - self.lower), 0.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {"lower": self.lower, "upper": self.upper, "fit_split": self.fit_split,
                "lower_quantile": self.lower_quantile, "upper_quantile": self.upper_quantile,
                "scaling": "minmax_clipped"}

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "ErrorScaler":
        return cls(float(document["lower"]), float(document["upper"]), str(document["fit_split"]),
                   float(document["lower_quantile"]), float(document["upper_quantile"]))


def _build():
    import torch
    from torch import nn

    class ReconstructionAutoencoder(nn.Module):
        model_id = "p2_reconstruction_ae"

        def __init__(self, input_size: int, hidden_sizes: Sequence[int] = (256, 64),
                     bottleneck: int = 16, dropout: float = 0.1):
            super().__init__()
            sizes = [int(input_size), *[int(size) for size in hidden_sizes]]
            encoder = []
            for a, b in zip(sizes[:-1], sizes[1:]):
                encoder += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
            encoder.append(nn.Linear(sizes[-1], int(bottleneck)))
            decoder_sizes = [int(bottleneck), *reversed(sizes[1:])]
            decoder = []
            for a, b in zip(decoder_sizes[:-1], decoder_sizes[1:]):
                decoder += [nn.Linear(a, b), nn.GELU(), nn.Dropout(dropout)]
            decoder.append(nn.Linear(decoder_sizes[-1], int(input_size)))
            self.encoder = nn.Sequential(*encoder)
            self.decoder = nn.Sequential(*decoder)
            self.input_size = int(input_size)

        def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
            flat = x.reshape(x.shape[0], -1)
            if flat.shape[1] != self.input_size:
                raise ValueError(f"expected {self.input_size} inputs, got {flat.shape[1]}")
            return self.decoder(self.encoder(flat))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Mean squared reconstruction error per window: [n, T, C] -> [n]."""
            flat = x.reshape(x.shape[0], -1)
            return ((self.reconstruct(x) - flat) ** 2).mean(dim=1)

    return ReconstructionAutoencoder


def __getattr__(name: str):
    if name == "ReconstructionAutoencoder":
        return _build()
    raise AttributeError(name)
