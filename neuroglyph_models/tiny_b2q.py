"""Compact Brain2Qwerty-inspired conv classifier for 14-channel EPOC X."""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyB2Q(nn.Module):
    def __init__(self, n_channels: int = 14, n_classes: int = 8, hidden: int = 256):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, time) e.g. (B, 14, 25) for 0.5s @ 50 Hz.
        """
        x = self.encoder(x)
        x = self.pool(x)
        return self.classifier(x)