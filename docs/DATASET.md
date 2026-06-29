# Dataset plan — NeuroGlyph-EPOC-Typing-v1

## Public references (not EPOC-compatible for weights)

| Dataset | URL | Use |
|---------|-----|-----|
| Brain2Qwerty v1 SpanishBCBL | https://huggingface.co/datasets/bcbl190626/SpanishBCBL | Labeling, CER, sentence splits |
| Brain2Qwerty repo | https://github.com/facebookresearch/brain2qwerty | Architecture inspiration |

## Required custom collection

Minimum pilot:

| Field | Target |
|-------|--------|
| subjects | 1–3 |
| sessions / subject | 3 |
| duration / session | 30–45 min |
| sample rate | 256 Hz |
| sync | Cortex markers + local key log |

## Recording checklist

1. EMOTIV Cortex running; app authorized; **Raw EEG** subscription (Premium).
2. `python -m neuroglyph_recorder.session_runner --live --subject sub-001 --session ses-001`
3. Type prompted sentences; export EEG CSV/EDF from Cortex after session.
4. Preprocess (MNE — install `pip install -e '.[preprocess]'`) → `data/processed/*.pt`
5. Train: `python -m neuroglyph_train.train --task hand`

## Optional reference clone

```bash
git clone https://github.com/facebookresearch/brain2qwerty ../brain2qwerty
```

Run their v1 pipeline on SpanishBCBL to validate your GPU/env before collecting EPOC data.