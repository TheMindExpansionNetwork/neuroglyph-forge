# Standing goal — Hermes agent on NeuroGlyph / MindBot

For **AI collaborators**. Humans edit priorities in `data/adaptive_queue.json`.

## Mission

Build **NeuroGlyph Forge** + **MindBot synergetic cognition** so the system **adapts to situations** — perception of real project state drives policy, queue, and narrative — not a single frozen checklist.

## Deep loop (use every session)

1. **Perceive** — `collect_world_signals()` (Cortex creds, raw sessions, synthetic vs real processed, ckpt val_acc, MindBot JSONL lines, pytest marker).
2. **Infer** — `propose_transitions()` with explicit reasons + confidence.
3. **Reconcile** — `reconcile_queue()` activates goals when triggers fire; marks `done` on completions.
4. **Act** — Run top commands from `GOAL_PLAYBOOK` / `adaptive_brief.md` (highest feasible, not all at once).
5. **Narrate** — `append_step()` + optional `adaptive_history.jsonl` for stream / training data.

```bash
python -m neuroglyph_agent.adaptive_engine --brief --reconcile
```

## Operating principles

1. **Situation first** — `SituationRouter` gates BCI; Hermes loads skills from `hermes_block.skill_hints`.
2. **Queue, don’t stall** — `//queue` → pause in-flight, switch situation, regenerate brief.
3. **Execute before narrate** — Tools + verified output; blockers logged as new queue items.
4. **Honest smoke** — Synthetic 100% val_acc ⇒ overlay `observe_only` during live/stream; never claim “mind control works.”
5. **Progressive BCI** — hand → zone → char29/intent; public data = pretrain; **your EPOC** = product gate.

## Improvement mandate (//queue think deeper)

Continuously improve:

- **Signal coverage** — more triggers (Unreal MCP ping, Modal volume ckpt, Comfy test).
- **Playbooks** — richer `next_commands` per goal with verification steps.
- **Multi-agent** — delegate parallel research (datasets, UE) when situation is `train_finetune` + `integration`.
- **Stream mode** — tie `narrative` field to live overlay / TTS / Comfy visuals for MindBot show.

## Default queue

`python -m neuroglyph_agent.situations --seed`

| P | Goal | Situation |
|---|------|-----------|
| 10 | Real hand >60% val | `collect_epoc` |
| 8 | Modal fine-tune on real data | `train_finetune` |
| 7 | JSONL bus on live actions | `stream_narrative` |
| 6 | Unreal E2E | `integration` |

## Success criteria

- [ ] `adaptive_brief.md` refreshed each major Hermes session
- [ ] Transitions logged when autopilot or manual situation changes
- [ ] Queue reflects reality (no stale `active` on done work)
- [ ] User can participate in live stream while agent executes situation-appropriate build work