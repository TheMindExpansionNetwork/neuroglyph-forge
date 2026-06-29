"""Subject-specific adapter layers for fine-tuning."""

from __future__ import annotations

import torch
import torch.nn as nn


class SubjectAdapter(nn.Module):
    """Lightweight subject/session linear adapter on encoder features."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)