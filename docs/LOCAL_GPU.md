# Local training — RTX 4070 12GB

TinyB2Q / EEGNet use **under 1 GB VRAM** at batch 32. **12 GB is plenty** (try batch 128 for speed).

## Enable CUDA (venv may be CPU torch today)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Train

```bash
python -m neuroglyph_train.train --task hand --epochs 40 --data data/processed
```

Engine auto-picks `cuda` when available.

**Realtime decode** is tiny — CPU OK. GPU = training only.

Modal is optional when the 4070 is at your desk.