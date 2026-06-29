# Trainable datasets for NeuroGlyph Forge (14ch EPOC X)

## Can train **now** (with caveats)

| Dataset | Channels | Task | How | Fit for typing? |
|---------|----------|------|-----|-----------------|
| **Synthetic** (`scripts/download_public_datasets.py synthetic`) | 14 | hand/zone/intent | Built-in | Smoke only |
| **Kaggle Emotiv hands** | 5→14 mapped | hand L/R | [Kaggle](https://www.kaggle.com/datasets/fabriciotorquato/brain-wave-data-from-hands-movement-of-eeg) → `--csv` | Motor pretrain, not keystroke |
| **EID-M / EID-S** (OpenBCI list) | 14 EPOC+ | rest / identity | [Google Drive](https://drive.google.com/drive/folders/1t6tL434ZOESb06ZvA4Bw1p9chzxzbRbj) | Pretrain spatial filters |
| **EEG-eye state** (UCI) | Emotiv | eyes open/closed | [UCI](https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State) | Artifact QC, not typing |

```bash
python scripts/download_public_datasets.py synthetic --task hand --n 800
python scripts/download_public_datasets.py kaggle-hands --csv path/to/eeg.csv
python scripts/download_public_datasets.py hf-spanishbcbl
```

## Reference / pipeline validation (not 14ch EPOC weights)

| Dataset | URL | Notes |
|---------|-----|-------|
| **SpanishBCBL** | https://huggingface.co/datasets/bcbl190626/SpanishBCBL | Brain2Qwerty typing sentences; **64ch lab EEG/MEG** — use to validate B2Q pipeline, not to deploy on EPOC |
| **Brain2Qwerty repo** | https://github.com/facebookresearch/brain2qwerty | Clone + run v1 on SpanishBCBL before collecting EPOC |
| **ArEEG_Words** | https://data.mendeley.com/datasets/7m472ykkx7 | 14ch Emotiv **word** tasks — manual download; good future importer |
| **PhysioNet MMIDB** | https://physionet.org/content/eegmmidb/1.0.0/ | Left/right **imagery** 64ch — channel map differs from EPOC |

## Gold standard (you record)

**NeuroGlyph-EPOC-Typing-v1** — keystroke-aligned 256 Hz → epochs per `docs/DATASET.md`.

Public sets = **pretrain / motor hand**; real typing accuracy needs your Cortex sessions.

## Train after prepare

```bash
# Local CPU/GPU
python -m neuroglyph_train.train --task hand --epochs 40 --data data/processed

# Fine-tune on new subject
python -m neuroglyph_train.finetune_subject --base-checkpoint checkpoints/tiny_b2q_hand.pt --epochs 30

# Modal GPU (see docs/MODAL_TRAINING.md)
modal run neuroglyph_cloud/modal_train.py --task hand --epochs 40
```