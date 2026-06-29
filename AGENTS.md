# NeuroGlyph Forge — agent context

## Project intent

Build a MindBot-compatible BCI stack: EMOTIV EPOC X typing EEG → trainable decoder → Hermes agent → Unreal MCP.

## Conventions

- Epoch window: **-0.2s to +0.3s** around keydown (0.5s total), resampled to **50 Hz → 25 samples**, **14 channels**.
- Training order: `hand` → `zone` → `char29` / `intent`.
- Synthetic data in `data/processed` is valid for CI/smoke; real EPOC data goes under `data/raw`.

## Commands

- Tests: `pytest tests/ -q`
- Train: `python -m neuroglyph_train.train --task hand --epochs 10`
- Mock record: `python -m neuroglyph_recorder.session_runner`
- Agent tools: `python -c "from neuroglyph_agent.mcp_server import call_tool; print(call_tool('preprocess_session', {'task':'hand'}))"`

## Hermes / Unreal

- Use `unreal-engine-mcp` skill for editor HTTP MCP (`Mcp-Session-Id`, `list_toolsets`, `call_tool`).
- Map predictions via `neuroglyph_unreal.ue_actions.prediction_to_unreal_action`.