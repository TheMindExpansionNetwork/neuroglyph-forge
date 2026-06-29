# Roadmap

## Phase 0 — Done (scaffold + smoke)
- [x] TinyB2Q / EEGNet, synthetic train, pytest
- [x] Agent tool surface + Unreal action map
- [x] Epoch builder + evaluate/finetune/export CLI

## Phase 1 — First real signal (you + headset)
- [ ] Cortex Premium raw EEG subscription in `cortex_client.py`
- [ ] 3× 30 min collection via `scripts/collect_session.py --live`
- [ ] Clock offset calibration utility
- [ ] Hand classifier > 60% val on **real** data

## Phase 2 — Controllable demo
- [ ] `intent` head trained on prompted Unreal verbs
- [ ] `LiveDecoder` wired to Cortex 50 Hz stream
- [ ] Hermes stdio MCP in config + UE `BP_NeuroPawn` events
- [ ] Stream overlay (confidence + last prediction)

## Phase 3 — Mind typing moonshot
- [ ] Zone + char29 heads, session adapter
- [ ] N-gram / small LM rescoring (Brain2Qwerty-style post-process)
- [ ] Compare against Brain2Qwerty v1 **event timing** from SpanishBCBL (not weights)

## Phase 4 — MindBot synergetic layer
- [ ] Decoder events → MindBot message bus (`neuroglyph_agent/mindbot_bus.py`)
- [ ] **Situation-aware adaptation** — `docs/ADAPTIVE_ORCHESTRATION.md`, `AGENT_GOAL.md`, `data/adaptive_queue.json`
- [ ] Dreaming/CoT logs decoded intent for live stream narrative
- [ ] Dataset export for MindBot fine-tune (high-signal BCI trajectories)

## Parallel references
- Clone `brain2qwerty` for lab pipeline sanity: `python -m brain2qwerty_v1.main debug`
- HF dataset: `bcbl190626/SpanishBCBL` (262 GB — use event-only neuralset path first)