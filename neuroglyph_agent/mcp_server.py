"""NeuroGlyph MCP tool surface for Hermes / MCP clients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuroglyph_agent.hermes_bridge import load_checkpoint_meta
from neuroglyph_agent.policy import PredictionEvent, should_act
from neuroglyph_data.synthetic import write_synthetic_processed
from neuroglyph_train.train import main as train_main


TOOL_SCHEMAS = [
    {
        "name": "record_typing_session",
        "description": "Record a mock or live EPOC typing session with key timestamps.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject_id": {"type": "string"},
                "session_id": {"type": "string"},
                "duration_sec": {"type": "number", "default": 30},
                "out_dir": {"type": "string", "default": "data/raw"},
            },
        },
    },
    {
        "name": "preprocess_session",
        "description": "Build synthetic/processed epoch tensors for training smoke tests.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "enum": ["hand", "zone", "char29", "intent"]},
                "out_dir": {"type": "string", "default": "data/processed"},
            },
            "required": ["task"],
        },
    },
    {
        "name": "train_decoder",
        "description": "Train TinyB2Q/EEGNet on processed epochs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "data_dir": {"type": "string"},
                "model": {"type": "string"},
                "epochs": {"type": "integer"},
            },
            "required": ["task"],
        },
    },
    {
        "name": "evaluate_decoder",
        "description": "Return checkpoint metadata (val acc, task, classes).",
        "input_schema": {
            "type": "object",
            "properties": {"checkpoint_path": {"type": "string"}},
            "required": ["checkpoint_path"],
        },
    },
    {
        "name": "send_prediction_to_unreal",
        "description": "Map a high-confidence prediction to a suggested Unreal MCP action.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prediction": {"type": "string"},
                "confidence": {"type": "number"},
                "threshold": {"type": "number", "default": 0.75},
            },
            "required": ["prediction", "confidence"],
        },
    },
]


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    args = arguments or {}
    if name == "record_typing_session":
        from neuroglyph_recorder.session_runner import run_session
        import asyncio

        path = asyncio.run(
            run_session(
                args.get("subject_id", "sub-001"),
                args.get("session_id", "ses-001"),
                float(args.get("duration_sec", 30)),
                Path(args.get("out_dir", "data/raw")),
                mock=True,
            )
        )
        return {"status": "ok", "meta_path": str(path)}
    if name == "preprocess_session":
        path = write_synthetic_processed(Path(args.get("out_dir", "data/processed")), task=args["task"])
        return {"status": "ok", "tensor_path": str(path)}
    if name == "train_decoder":
        argv = [
            "--task",
            args.get("task", "hand"),
            "--data",
            args.get("data_dir", "data/processed"),
            "--model",
            args.get("model", "tiny_b2q"),
            "--epochs",
            str(args.get("epochs", 5)),
        ]
        train_main(argv)
        return {"status": "ok", "argv": argv}
    if name == "evaluate_decoder":
        meta = load_checkpoint_meta(Path(args["checkpoint_path"]))
        return {"status": "ok", "checkpoint": meta}
    if name == "send_prediction_to_unreal":
        from neuroglyph_unreal.ue_actions import prediction_to_unreal_action

        ev = PredictionEvent(
            source="neuroglyph_decoder",
            prediction=args["prediction"],
            confidence=float(args["confidence"]),
            mode="unreal_control",
            timestamp_ms=0,
        )
        if not should_act(ev, threshold=float(args.get("threshold", 0.75))):
            return {"status": "ignored", "reason": "low confidence"}
        return {"status": "ok", "action": prediction_to_unreal_action(ev.prediction)}
    raise ValueError(f"unknown tool: {name}")


def list_tools() -> list[dict]:
    return TOOL_SCHEMAS


if __name__ == "__main__":
    print(json.dumps(list_tools(), indent=2))