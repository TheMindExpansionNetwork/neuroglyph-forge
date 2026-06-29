# Modal GPU training & fine-tuning

Based on [modal-examples](https://github.com/modal-labs/modal-examples) (`06_gpu_and_ml`, `long-training`).

## 1. Install Modal

```bash
pip install modal
modal setup
```

Create a free account at [modal.com](https://modal.com/) if prompted.

## 2. Prepare data locally

```bash
cd D:/neuroglyph-forge
source .venv/Scripts/activate
python scripts/download_public_datasets.py synthetic --task hand --n 800
# optional: real public CSV
# python scripts/download_public_datasets.py kaggle-hands --csv ~/Downloads/eeg.csv
```

## 3. Cloud train (T4 GPU)

Uploads `data/processed/*.pt` to Modal volume `neuroglyph-data`, trains, saves to `neuroglyph-checkpoints`:

```bash
modal run neuroglyph_cloud/modal_train.py --task hand --epochs 40 --model tiny_b2q
```

## 4. Cloud fine-tune

Uploads existing `checkpoints/*.pt`, runs low-LR fine-tune:

```bash
modal run neuroglyph_cloud/modal_train.py --finetune --task hand --epochs 30 --freeze-encoder
```

## 5. Pull checkpoints back

```bash
modal volume get neuroglyph-checkpoints tiny_b2q_hand.pt checkpoints/
```

## Architecture

| Piece | Path |
|-------|------|
| Shared train loop | `neuroglyph_train/engine.py` |
| Modal app | `neuroglyph_cloud/modal_train.py` |
| Local CLI | `python -m neuroglyph_train.train` |

Volumes persist between runs — use the same volume for iterative fine-tuning on new sessions you upload.

## When to use Modal vs local

| Scenario | Recommendation |
|----------|----------------|
| Laptop, small synthetic | Local `train.py` |
| 40+ epochs, hyperparameter sweeps | Modal T4/A10G |
| Subject fine-tune after new EPOC sessions | Modal `--finetune` |

## Related Modal examples

- [GPU quickstart](https://modal.com/docs/guide/gpu)
- [Long resumable training](https://modal.com/docs/examples/long-training)
- [Torch profiling](https://modal.com/docs/examples/torch_profiling)