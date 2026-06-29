"""EEGNet baseline (Lawhern et al.) sized for 14 channels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 14,
        n_classes: int = 2,
        n_time: int = 25,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_time)
            out = self._features(dummy)
            flat = out.view(1, -1).shape[1]
        self.fc = nn.Linear(flat, n_classes)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sep(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input (B, C, T) -> (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)