"""Append high-signal steps to MindBot JSONL bus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuroglyph_agent.policy import PredictionEvent

DEFAULT_BUS = Path(__file__).resolve().parents[1] / "data" / "mindbot_export" / "live_steps.jsonl"


def append_step(
    event: PredictionEvent,
    *,
    situation: str,
    unreal_action: dict[str, Any] | None = None,
    path: Path = DEFAULT_BUS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "id": f"step-{event.timestamp_ms}",
        "session_id": "live-demo",
        "timestamp_ms": event.timestamp_ms,
        "situation": situation,
        "bci": {
            "prediction": event.prediction,
            "confidence": event.confidence,
            "mode": event.mode,
            "task": event.task,
        },
        "hermes": {"tools_called": [], "summary": "live_bci_demo gated emit"},
        "unreal": unreal_action or {},
        "mindbot": {"cot_snippet": "", "dream_tag": None},
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")