# MindBot export (JSONL)

High-signal trajectories for synergetic cognition training — **not** raw EEG.

## Schema (`mindbot_step_v1`)

```json
{
  "id": "step-00001",
  "session_id": "ses-001",
  "timestamp_ms": 1782769600123,
  "bci": {
    "prediction": "spawn",
    "confidence": 0.88,
    "mode": "unreal_control",
    "task": "intent"
  },
  "hermes": {
    "tools_called": ["mcp_unreal_mcp_call_tool"],
    "plan_summary": "High-confidence spawn → SpawnDebugActor"
  },
  "unreal": {
    "event": "SpawnDebugActor",
    "success": true
  },
  "user_correction": null,
  "narrative_hook": "Operator intended spawn at 0.88 — suggest next scene beat"
}
```

## Collecting

1. Run live demo / real sessions with logging enabled (future: `scripts/log_mindbot_step.py`).
2. Append one JSON object per line under `data/mindbot_export/`.
3. Target **10k–50k** curated steps (quality over volume).

## Example seed

See `example.jsonl` (synthetic smoke rows for schema validation).