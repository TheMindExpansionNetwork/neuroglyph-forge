# Unreal Engine — NeuroGlyph MVP checklist

## Editor plugin

1. Install **Model ContextProtocol** plugin (Epic / community UE 5.x MCP).
2. Enable **EditorToolset** in plugin settings.
3. Start MCP HTTP server (default often `http://127.0.0.1:8000/mcp`).

## Hermes config

Merge `configs/hermes-mcp-snippet.yaml` → restart Hermes.

## Blueprints (minimal)

### `BP_NeuroDirector` (Actor)

- Event **NeuroSpawn** → `Spawn Actor from Class` (debug mesh).
- Event **NeuroDelete** → destroy selected / last spawned.

### `BP_NeuroPawn` (or existing pawn)

Custom events matching `neuroglyph_unreal/ue_actions.py`:

| Prediction | Blueprint event |
|------------|-----------------|
| `left` | `MoveLeft` |
| `right` | `MoveRight` |
| `select` | `Select` |
| `cancel` | `Cancel` |
| `spawn` | `NeuroSpawn` on director |
| `delete` | `NeuroDelete` |

Wire **confidence** as float parameter if you want in-editor gating (Hermes already gates at 0.75).

## Test from Python (no agent)

```bash
python -c "
from neuroglyph_agent.mcp_server import call_tool
print(call_tool('send_prediction_to_unreal', {'prediction':'spawn','confidence':0.9}))
"
```

## Test from Hermes

Ask: *“Send neuroglyph prediction spawn at 0.9 confidence to Unreal.”*

Requires `mcp_unreal_mcp_*` tools live and editor running.

## Stream / live performance

For reality-live streams: overlay widget showing last `PredictionEvent` + confidence (UMG reads replicated var on pawn).