"""Confidence gates and rate limits for BCI → agent actions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PredictionEvent:
    source: str
    prediction: str
    confidence: float
    mode: str
    timestamp_ms: int


def should_act(event: PredictionEvent, threshold: float = 0.75) -> bool:
    return event.confidence >= threshold and event.prediction not in ("", "unknown")


@dataclass
class ActionRateLimiter:
    """Max one Unreal/agent action per min_interval_sec."""

    min_interval_sec: float = 0.5
    _last_action_mono: float = field(default=0.0, repr=False)

    def allow(self) -> bool:
        now = time.monotonic()
        if now - self._last_action_mono < self.min_interval_sec:
            return False
        self._last_action_mono = now
        return True

    def should_emit(self, event: PredictionEvent, threshold: float = 0.75) -> bool:
        return should_act(event, threshold) and self.allow()