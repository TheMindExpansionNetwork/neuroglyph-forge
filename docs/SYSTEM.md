# NeuroGlyph Forge — system architecture

Five layers from scalp to Unreal. Each layer has a **contract** (data shape + failure mode).

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ 1. SENSE    │───▶│ 2. SYNC      │───▶│ 3. LEARN    │───▶│ 4. REASON    │───▶│ 5. ACT      │
│ EPOC X      │    │ keys+markers │    │ TinyB2Q     │    │ Hermes       │    │ Unreal MCP  │
│ Cortex API  │    │ epochs       │    │ train/eval  │    │ confidence   │    │ BP events   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
```

## 1. Sense (hardware)

| Item | Spec |
|------|------|
| Device | EMOTIV EPOC X, 14 EEG + references |
| Raw EEG | Cortex **Premium** WebSocket `subscribe` |
| Rate | 256 Hz capture → 50 Hz model grid |
| Band | 0.1–20 Hz (+ 60 Hz notch), avg reference |

**Failure modes:** poor contact quality, jaw blink artifacts, Bluetooth lag. Mitigation: QC plots (future), short calibration block, reject bad epochs.

## 2. Sync (data contract)

| Artifact | Format |
|----------|--------|
| `*_meta.json` | Pydantic `SessionMeta` + `KeyEvent` list (unix ms) |
| EEG export | Cortex CSV, columns = channel names |
| Alignment | `clock_offset_ms` tuned per session |
| Epoch | **[-0.2s, +0.3s]** around keydown → `(14, 25)` tensor |

Pipeline: `make_epochs_cli` → `data/processed/processed_{task}.pt`

**Relation to Brain2Qwerty:** same *keystroke-aligned 500 ms window* idea; we skip their 64ch Conv+Transformer+LM until EPOC signal justifies it.

## 3. Learn (ML)

| Stage | Task | Classes | Success bar |
|-------|------|---------|-------------|
| A | `hand` | 2 | val acc > 60% |
| B | `zone` | 8 | beats chance ~12.5% |
| C | `intent` | 8 Unreal verbs | stable live demo |
| D | `char29` | B2Q-style | CER ↓ vs naive |

Commands:
- `python -m neuroglyph_train.train --task hand`
- `python -m neuroglyph_train.evaluate --checkpoint checkpoints/tiny_b2q_hand.pt`
- `python -m neuroglyph_train.finetune_subject --base-checkpoint ...`
- `python -m neuroglyph_train.export_model --checkpoint ...`

**Datasets:** SpanishBCBL = *reference only*. Real weights = **NeuroGlyph-EPOC-Typing-v1** (you record).

## 4. Reason (Hermes / MindBot)

Hermes **never** ingests raw EEG.

```
LiveDecoder → PredictionEvent {prediction, confidence, mode}
     → policy.should_act (threshold 0.75)
     → Hermes: plan / clarify / chain tools
     → optional: synergetic cognition layer (MindBot swarm) for lore/stream UX
```

Wire tools:
- In-process: `neuroglyph_agent.mcp_server.call_tool`
- Native MCP: `neuroglyph_agent.hermes_stdio_mcp` (see `configs/hermes-mcp-snippet.yaml`)

## 5. Act (Unreal)

Downstream only. `prediction_to_unreal_action` emits `call_tool` hints for UE ModelContextProtocol plugin.

Prerequisites: UE editor running, `EditorToolset` enabled, Hermes `mcp_servers.unreal-mcp` URL.

**MVP demo:** zone/intent → `MoveLeft` / `SpawnDebugActor` on `BP_NeuroPawn` / `BP_NeuroDirector`.

## Safety & ops

- Low-confidence predictions are **dropped** (no Unreal spam).
- Cron/live loops should rate-limit actions (e.g. max 2/s).
- Store checkpoints + session meta under `data/` and `checkpoints/`; never commit Cortex secrets.

## End-to-end operator script

```bash
cd D:/neuroglyph-forge
source .venv/Scripts/activate
bash scripts/pipeline.sh hand
```

## What “real” means next

1. **One real EPOC session** exported to `data/raw/` → preprocess → train hand.
2. **Hermes MCP** registered + restart → agent can `train_decoder` / `send_prediction_to_unreal`.
3. **UE blueprint** `BP_NeuroPawn` listening for mapped events.
4. Optional: live loop `LiveDecoder` + Cortex stream (Premium).

That closes the loop from biology to game engine.

**Visual brain map:** `docs/BRAINMAP.md` · `docs/brainmap.html` · `assets/brand/brainmap-gpt-image-2.png`