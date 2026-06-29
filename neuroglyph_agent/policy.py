"""Confidence gates for BCI → agent actions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictionEvent:
    source: str
    prediction: str
    confidence: float
    mode: str
    timestamp_ms: int


def should_act(event: PredictionEvent, threshold: float = 0.75) -> bool:
    return event.confidence >= threshold and event.prediction not in ("", "unknown")