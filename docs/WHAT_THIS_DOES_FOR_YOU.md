# What NeuroGlyph Forge does for you

You have an **EMOTIV EPOC X** and a **4070 12GB** PC. This stack turns that into:

## Near-term (weeks)

| Capability | What you get |
|------------|----------------|
| **Hand intent** | Model guesses **left vs right hand** (later: keyboard zone) from 14-channel EEG around each key press. |
| **Hermes agent** | Predictions become **tool calls** — scripts, Comfy, narration for stream. |
| **Unreal** | **BP_NeuroPawn** moves/emotes from gated predictions. |
| **MindBot bus** | Gated actions → **JSONL** for your synergetic training data. |

## Medium-term (English typing)

| Stage | Task |
|-------|------|
| 1 | **hand** L/R (language-agnostic) |
| 2 | **zone** QWERTY rows — **your** English typing |
| 3 | **char29** / intent — short English commands you define |

**You do not need Spanish data.** SpanishBCBL is optional 64ch lab research — not your EPOC English path.

## Realtime (after first real checkpoint)

```
EPOC X → Cortex localhost → LiveDecoder → gate → Hermes / Unreal / keys
```

Your headset is enough for v1. Bottleneck is **labeled sessions**, not GPU.

## Honest “brain communication”

- Not full sentence mind-reading on day one.
- **Yes**: learn patterns around **keys you press** while EEG records.
- **Yes**: streaming loop with rate limits.

First win: one real session processed, val acc **> 55%** on hand. Second: **`live_bci_demo --live`**.

## Data layout

```
data/catalog.json     manifest
data/raw/             Cortex sessions (gold)
data/processed/       training epochs
data/external/        optional Kaggle etc.
data/uploads/         HF staging
```

HF upload is **optional** until you have real sessions worth publishing.

See: `FIRST_SESSION_PROTOCOL.md`, `ENGLISH_DATA_PATH.md`, `LOCAL_GPU.md`.