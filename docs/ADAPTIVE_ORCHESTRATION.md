# Adaptive orchestration — situational MindBot / NeuroGlyph

**Standing agent goal:** keep the stack **useful under changing context** without re-asking the user for the same steering. **Perceive → infer → reconcile → act → narrate.**

## Synergetic cognition loop (implemented)

```
WorldSignals (filesystem, env, checkpoints)
        ↓
propose_transitions() + reconcile_queue()
        ↓
SituationRouter (thresholds, Unreal gate, rate limit)
        ↓
BCI / train / integrate / creative actions
        ↓
mindbot JSONL + adaptive_history.jsonl
```

**Deep engine:** `neuroglyph_agent/adaptive_engine.py`  
**Profiles + queue file:** `neuroglyph_agent/situations.py`

### Hermes session start (agent)

```bash
python -m neuroglyph_agent.adaptive_engine --brief --reconcile
# read data/adaptive_brief.md — suggested commands + narrative
```

Optional autopilot (≥80% confidence transitions only):

```bash
python -m neuroglyph_agent.adaptive_engine --autopilot --brief
```

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

Situations **stack**: e.g. `live_bci` + `observe_only` when val_acc &lt; 0.6 on real data, or synthetic-perfect ckpt during stream.

## Triggers (machine-checkable)

| Trigger | Meaning |
|---------|---------|
| `cortex_credentials_set` | `EMOTIV_CLIENT_ID` + `SECRET` in `.env` or Hermes `.env` |
| `real_processed_hand_exists` | `processed_hand.pt` from non-synthetic session |
| `live_bci_demo_stable` | ≥1 line in `mindbot_export/live_steps.jsonl` |
| `always_pending` | Manual / integration goals |

Completions auto-mark goals `done` (see `completion_satisfied()` in adaptive_engine).

## Transition examples

| Signals | Proposal |
|---------|----------|
| Cortex creds, no raw sessions | → `collect_epoc` |
| Raw sessions, no processed real | → `train_finetune` (run epochs) |
| Real processed + creds | → `live_bci` |
| Real val_acc &lt; 0.6 | overlay `observe_only` |
| Synthetic val_acc = 1.0 + stream mode | overlay `observe_only` (honest smoke) |

Logged to `data/adaptive_history.jsonl` when `--autopilot` applies a change.

## Queue format

```json
{
  "version": 1,
  "active_situation": "dev_pipeline",
  "overlay": null,
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

Playbook commands per goal id: `GOAL_PLAYBOOK` in `adaptive_engine.py`.

## Agent playbook (Hermes / MindBot collaborator)

1. **On session start** — `--brief --reconcile`; read `AGENT_GOAL.md` + `adaptive_brief.md`.
2. **On blocker** — enqueue new `QueuedGoal` with trigger + situation; do not stop cold.
3. **On pivot** — user `//queue`: pause active goals, `--set <situation>`, regenerate brief.
4. **On success** — reconcile marks `done`; append MindBot JSONL with `situation` + outcome.
5. **Never** fabricate metrics — `WorldSignals` is the source of truth for transitions.

## CLI

```bash
python -m neuroglyph_agent.situations --show
python -m neuroglyph_agent.situations --set live_bci
python -m neuroglyph_agent.adaptive_engine --brief --reconcile
python scripts/live_bci_demo.py --situation live_bci
```

## Roadmap link

Phase 4 MindBot layer in `docs/ROADMAP.md`.

## Related

- `docs/MINDBOT_INTEGRATION.md` — JSONL bus
- `docs/BRAINMAP.md` — Sense → Act layers
- `neuroglyph_agent/policy.py` — rate limits