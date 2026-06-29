"""Map BCI predictions to Unreal blueprint-oriented actions."""

from __future__ import annotations

from typing import Any, Union

from neuroglyph_agent.policy import PredictionEvent


ACTION_MAP = {
    "left": {"toolset": "editor_toolset.toolsets.scene.SceneTools", "tool": "trigger_blueprint_event", "event": "MoveLeft"},
    "right": {"toolset": "editor_toolset.toolsets.scene.SceneTools", "tool": "trigger_blueprint_event", "event": "MoveRight"},
    "select": {"event": "ConfirmSelection", "actor": "BP_NeuroPawn"},
    "space": {"event": "ConfirmSelection", "actor": "BP_NeuroPawn"},
    "spawn": {"event": "SpawnDebugActor", "actor": "BP_NeuroDirector"},
    "delete": {"event": "DeleteSelection", "actor": "BP_NeuroDirector"},
}


def prediction_to_unreal_action(prediction: Union[str, PredictionEvent]) -> dict[str, Any]:
    if isinstance(prediction, PredictionEvent):
        key = prediction.prediction.lower().strip()
        conf = prediction.confidence
    else:
        key = prediction.lower().strip()
        conf = None
    base = ACTION_MAP.get(key, {"event": "NeuroGlyphIntent", "payload": key})
    args = {
        "actor": base.get("actor", "BP_NeuroPawn"),
        "event": base.get("event", "NeuroGlyphIntent"),
        "payload": base.get("payload", key),
    }
    if conf is not None:
        args["confidence"] = conf
    return {
        "mcp_server": "unreal-mcp",
        "call_tool": {
            "toolset_name": base.get("toolset", "EditorToolset.EditorAppToolset"),
            "tool_name": base.get("tool", "trigger_blueprint_event"),
            "arguments": args,
        },
        "note": "Use Hermes mcp_unreal-mcp_call_tool or HTTP probe per unreal-engine-mcp skill.",
    }