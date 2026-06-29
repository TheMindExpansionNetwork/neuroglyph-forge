"""Bridge decoded predictions into Hermes-friendly events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuroglyph_agent.policy import PredictionEvent, should_act


def format_hermes_message(event: PredictionEvent) -> str:
    return json.dumps(
        {
            "source": event.source,
            "prediction": event.prediction,
            "confidence": event.confidence,
            "mode": event.mode,
            "timestamp": event.timestamp_ms,
            "actionable": should_act(event),
        },
        indent=2,
    )


def load_checkpoint_meta(checkpoint: Path) -> dict[str, Any]:
    import torch

    blob = torch.load(checkpoint, weights_only=True)
    return {
        "task": blob.get("task"),
        "model": blob.get("model"),
        "n_classes": blob.get("n_classes"),
        "val_acc": blob.get("val_acc"),
    }