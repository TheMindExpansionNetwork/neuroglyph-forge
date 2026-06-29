"""Live inference: sliding EEG buffer → prediction events for Hermes/Unreal."""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neuroglyph_agent.policy import PredictionEvent
from neuroglyph_core import MODEL_SAMPLE_RATE, N_CHANNELS, N_TIME_SAMPLES
from neuroglyph_models.heads import TASK_HEADS
from neuroglyph_train.train import build_model


@dataclass
class LiveDecoderConfig:
    checkpoint: Path
    confidence_threshold: float = 0.75
    mode: str = "unreal_control"


class LiveDecoder:
    def __init__(self, config: LiveDecoderConfig):
        blob = torch.load(config.checkpoint, weights_only=True)
        self.task = blob["task"]
        self.classes = TASK_HEADS[self.task]["classes"]
        self.model = build_model(blob["model"], blob["n_channels"], blob["n_classes"])
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        self.threshold = config.confidence_threshold
        self.mode = config.mode
        self._buffer: deque[np.ndarray] = deque(maxlen=N_TIME_SAMPLES)

    def push_sample(self, sample_14ch: np.ndarray) -> PredictionEvent | None:
        """Push one multichannel sample at MODEL_SAMPLE_RATE (50 Hz)."""
        self._buffer.append(sample_14ch.astype(np.float32))
        if len(self._buffer) < N_TIME_SAMPLES:
            return None
        window = np.stack(list(self._buffer), axis=1)  # 14 x T
        x = torch.tensor(window).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)[0]
        conf, idx = probs.max(dim=0)
        pred = self.classes[int(idx)]
        if float(conf) < self.threshold:
            return None
        return PredictionEvent(
            source="neuroglyph_decoder",
            prediction=pred,
            confidence=float(conf),
            mode=self.mode,
            timestamp_ms=int(time.time() * 1000),
        )

    def decode_file(self, npy_path: Path) -> list[dict]:
        """Offline stream from array shape (n_samples, 14) at 50 Hz."""
        arr = np.load(npy_path)
        events = []
        self._buffer.clear()
        for row in arr:
            ev = self.push_sample(row)
            if ev:
                events.append(json.loads(__import__("json").dumps(ev, default=lambda o: o.__dict__)))
        return events