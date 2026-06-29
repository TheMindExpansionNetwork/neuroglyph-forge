# Standing goal — Hermes agent on NeuroGlyph / MindBot

This file is for **AI collaborators** (Hermes, subagents, future MindBot swarm). Humans can edit priorities.

## Mission

Help build **NeuroGlyph Forge** and the **MindBot synergetic cognition** layer so the system **adapts to situations** instead of one rigid script.

## Operating principles

1. **Situation first** — Load `data/adaptive_queue.json` and `docs/ADAPTIVE_ORCHESTRATION.md`; set behavior via `SituationRouter` (confidence, Unreal on/off, which skill to load).
2. **Queue, don’t stall** — On `//queue` or topic pivot: pause in-flight work in the queue, switch situation, continue with what fits *now*.
3. **Execute before narrate** — Tools and real output beat plans; honest blockers beat fabricated success.
4. **High-signal memory** — MindBot JSONL + compact Hermes memory for *preferences*; skills for *procedures*; queue for *goals*.
5. **Progressive BCI** — hand → zone → char29/intent; public datasets = pretrain only; real EPOC = product gate.

## Default queue (seed)

Run once: `python -m neuroglyph_agent.situations --seed`

| Priority | Goal | Situation |
|----------|------|-----------|
| 10 | Real hand >60% val | `collect_epoc` |
| 8 | Modal fine-tune on real data | `train_finetune` |
| 7 | JSONL bus on live actions | `stream_narrative` |

## When user says “adapt to situations”

- Implement or tune **`neuroglyph_agent/situations.py`**
- Wire **`live_bci_demo --situation`**
- Extend **`mindbot_step_v1`** with `situation` field
- Update **skill `neuroglyph-forge`** with situation table
- Do **not** duplicate long task logs in MEMORY.md

## Success criteria (agent)

- [ ] Queue file exists and reflects current phase
- [ ] Live demo respects situation gates
- [ ] Each major session leaves ≥1 JSONL step or queue state update
- [ ] User can stream/participate while swarm handles build + content (Phase 4 ROADMAP)