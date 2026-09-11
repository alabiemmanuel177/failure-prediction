"""P5: compact causal transformer encoder (optional capacity baseline).

A square subsequent (upper-triangular ``-inf``) attention mask stops every position
from attending to later positions, and the learned positional embedding is indexed
by absolute step so the logit at step ``t`` is a function of steps ``<= t`` only.
"""

from __future__ import annotations

import torch
from torch import nn


class CompactCausalTransformer(nn.Module):
    model_id = "p5_compact_transformer"

    def __init__(self, in_channels: int, time_steps: int = 10, d_model: int = 64, heads: int = 4,
                 layers: int = 2, feedforward: int = 128, dropout: float = 0.1):
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.time_steps = int(time_steps)
        self.input = nn.Linear(in_channels, d_model)
        self.position = nn.Parameter(torch.zeros(self.time_steps, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=feedforward, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    @staticmethod
    def causal_mask(steps: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((steps, steps), float("-inf"), device=device), diagonal=1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        steps = x.shape[1]
        if steps > self.time_steps:
            raise ValueError(f"window has {steps} steps; model built for <= {self.time_steps}")
        hidden = self.input(x) + self.position[:steps]
        mask = self.causal_mask(steps, x.device)
        hidden = self.encoder(hidden, mask=mask, is_causal=True)
        return self.head(self.norm(hidden)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1]
