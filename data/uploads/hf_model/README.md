---
license: mit
tags:
- eeg
- emotiv
- bci
- typing
library_name: pytorch
---

# NeuroGlyph EPOC Typing v1

English keystroke-aligned **hand** decoder (left/right) for EMOTIV EPOC X (14ch, 50 Hz, 25 samples).

## Load

```python
import torch
from neuroglyph_train.train import build_model

blob = torch.load("tiny_b2q_hand.pt", map_location="cpu", weights_only=True)
model = build_model(blob["model"], blob["n_channels"], blob["n_classes"])
model.load_state_dict(blob["state_dict"])
```

## Train your own

https://github.com/TheMindExpansionNetwork/neuroglyph-forge
