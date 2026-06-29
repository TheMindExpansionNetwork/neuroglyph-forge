# Adaptive orchestration — situational MindBot / NeuroGlyph

**Standing agent goal:** keep the stack **useful under changing context** without re-asking the user for the same steering. Detect situation → adjust thresholds, tools, and queued work → log trajectories for MindBot training.

## Situations (modes)

| Situation | When | Actuation | Hermes focus |
|-----------|------|-----------|--------------|
| `dev_pipeline` | Tests, refactor, docs | None | `terminal`, `file`, pytest |
| `collect_epoc` | Headset / Cortex setup | Record only | Cortex skill path, recorder |
| `live_bci` | Streaming EEG + decoder | Gated Unreal/typing | `live_bci_demo`, rate limit |
| `train_finetune` | Modal/local training | None | `engine`, datasets, Modal |
| `integration` | MCP, UE, Comfy | Tool calls | `hermes mcp`, unreal skill |
| `creative_promo` | Brand, brainmap, assets | Image gen | Comfy / OpenAI images |
| `stream_narrative` | Live / MindBot show | Soft actions + narration | JSONL bus, high-signal logs |
| `observe_only` | Low confidence / debug | Log, no UE | Raise threshold, no spam |

Situations **stack**: e.g. `live_bci` + `observe_only` when val_acc &lt; 0.6 on real data.

## Adaptation rules (implemented)

Code: `neuroglyph_agent/situations.py`

- **Confidence floor** per situation (live 0.55, observe 0.85, stream 0.50).
- **Min action interval** per situation (live 0.5s, stream 0.25s).
- **Skill hints** injected into Hermes context (which doc to load first).
- **Queue** (`data/adaptive_queue.json`): deferred goals with priority + trigger.

## Queue format

```json
{
  "version": 1,
  "active_situation": "dev_pipeline",
  "items": [
    {
      "id": "goal-real-hand-60",
      "priority": 10,
      "situation": "collect_epoc",
      "goal": "Hand classifier >60% val on real EPOC sessions",
      "trigger": "cortex_credentials_set",
      "status": "pending"
    }
  ]
}
```

## Agent playbook (Hermes / MindBot collaborator)

1. **On session start** — read `data/adaptive_queue.json` + project `ROADMAP.md` phase; infer situation from workspace (tests failing → `dev_pipeline`, user mentions headset → `collect_epoc`).
2. **On blocker** — enqueue adaptation (e.g. “HF SpanishBCBL failed → document brain2qwerty clone path”) instead of stopping cold.
3. **On pivot** — user `//queue` or new topic: push previous in-flight work to queue with `status: paused`, switch `active_situation`.
4. **On success** — mark queue item `done`, append one line to `data/mindbot_export/` with `situation` + outcome (training bus).
5. **Never** fabricate train metrics or MCP connectivity — situation `observe_only` for honest partial state.

## CLI

```bash
python -m neuroglyph_agent.situations --show
python -m neuroglyph_agent.situations --set live_bci
python scripts/live_bci_demo.py --situation live_bci
```

## Roadmap link

Phase 4 MindBot layer in `docs/ROADMAP.md` — this doc is the **operational** spec for “how we adapt.”

## Related

- `docs/MINDBOT_INTEGRATION.md` — JSONL bus
- `docs/BRAINMAP.md` — Sense → Act layers
- `neuroglyph_agent/policy.py` — rate limits