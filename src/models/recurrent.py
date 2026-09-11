"""P4: GRU baseline with a parameter budget within +-25 % of P3 (asserted in tests)."""

from __future__ import annotations

import torch
from torch import nn


class GRUPredictor(nn.Module):
    model_id = "p4_gru"

    def __init__(self, in_channels: int, hidden_size: int = 128, layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            in_channels, hidden_size, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Logits at every time step (a unidirectional GRU is causal by construction)."""
        hidden, _ = self.gru(x)                              # [n, T, H]
        return self.head(self.dropout(hidden)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x)[:, -1]
