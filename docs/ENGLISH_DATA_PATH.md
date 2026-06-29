# English-first data path (no Spanish required)

## You need

1. **Your EPOC recordings** — English QWERTY (`docs/DATASET.md`).
2. **Optional**: Kaggle Emotiv **hand movement** (motor pretrain, any language).

## You do **not** need

- **SpanishBCBL** — Spanish *sentence* typing on 64ch lab EEG; reference for Brain2Qwerty only.
- **ArEEG Arabic** — 14ch word task; later optional, not step one.

## Priority list

| P | Dataset | Status |
|---|---------|--------|
| **P0** | Your Cortex sessions | Record first |
| **P1** | Kaggle Emotiv hands | `download_public_datasets.py kaggle-hands` |
| **P2** | EID-M 14ch EPOC+ | Manual download |
| Skip | SpanishBCBL, 64ch PhysioNet | Wrong task or channels |

## Hugging Face (later)

```bash
python scripts/push_dataset_hf.py --repo-id TheMindExpansionNetwork/neuroglyph-epoc-typing-en-v1
python scripts/push_dataset_hf.py --push   # after huggingface-cli login
```

## Train order

`hand` → `zone` → `char29` / English intent tokens