# MindBot / synergetic cognition hook

NeuroGlyph Forge is the **sensory-motor slice** of MindBot — not the whole swarm.

## Where it sits

```
EPOC X → NeuroGlyph (decode) → Hermes (plan) → Unreal (act)
                              ↘ MindBot bus (narrative / memory / stream)
```

## Event shape for MindBot

Reuse `PredictionEvent` JSON:

```json
{
  "source": "neuroglyph_decoder",
  "prediction": "spawn",
  "confidence": 0.88,
  "mode": "unreal_control",
  "timestamp_ms": 1782769600123
}
```

Hermes can:
1. Call Unreal MCP when `mode=unreal_control` and confidence ≥ threshold.
2. Append to session narrative for live stream (“operator intended spawn”).
3. Queue dreaming/CoT prompts: *“Given BCI intent spawn at 0.88 confidence, suggest next scene beat.”*

## Dataset for MindBot training

Export **high-signal** trajectories (not raw EEG dumps):

| Field | Content |
|-------|---------|
| `context` | Last N predictions + confidence |
| `hermes_plan` | Tool chain chosen |
| `unreal_outcome` | Blueprint event result |
| `user_correction` | If user overrode |

Target volume: 10k–50k curated steps (per your preference), stored as JSONL under `data/mindbot_export/`.

## Not in scope yet

- End-to-end fine-tune of MindBot on EEG (decode stays in TinyB2Q).
- Replacing Hermes; Forge **feeds** Hermes/MindBot with structured intents.