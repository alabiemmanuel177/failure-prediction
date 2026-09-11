"""P3: causal dilated temporal convolutional network (primary learned predictor).

Every convolution is left-padded by ``(kernel_size - 1) * dilation`` and never
right-padded, so the logit at time step ``t`` depends on steps ``<= t`` only. The
unit test ``tests/test_models_causality.py`` perturbs step ``t + 1`` and asserts the
output at step ``t`` is unchanged.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


_ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU}


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: [n, C, T]
        return self.conv(nn.functional.pad(x, (self.left_pad, 0)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel_size: int, dilation: int,
                 dropout: float, activation: str):
        super().__init__()
        act = _ACTIVATIONS[activation]
        self.body = nn.Sequential(
            CausalConv1d(in_channels, channels, kernel_size, dilation), act(), nn.Dropout(dropout),
            CausalConv1d(channels, channels, kernel_size, dilation), act(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_channels, channels, 1) if in_channels != channels else nn.Identity()
        self.out_act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.body(x) + self.skip(x))


class CausalTCN(nn.Module):
    model_id = "p3_causal_tcn"

    def __init__(self, in_channels: int, hidden_channels: int = 64, kernel_size: int = 3,
                 dilations: Sequence[int] = (1, 2, 4), dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        if kernel_size < 1 or not dilations:
            raise ValueError("kernel_size must be >= 1 and dilations non-empty")
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        blocks = []
        channels = in_channels
        for dilation in dilations:
            blocks.append(ResidualBlock(channels, hidden_channels, kernel_size, int(dilation),
                                        dropout, activation))
            channels = hidden_channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_channels, 1)
        self.receptive_field = 1 + sum(2 * (kernel_size - 1) * int(d) for d in dilations)

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Logits at every time step: x [n, T, C] -> [n, T]."""
        hidden = self.blocks(x.transpose(1, 2))            # [n, H, T]
        return self.head(hidden.transpose(1, 2)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logit for the decision at the last time step: x [n, T, C] -> [n]."""
        return self.forward_sequence(x)[:, -1]
