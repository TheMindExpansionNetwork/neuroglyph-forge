#!/usr/bin/env python3
"""Upload NeuroGlyph checkpoint + model card to Hugging Face."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

README = """---
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
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--repo-id", default="TheMindExpansionNetwork/NeuroGlyph-EPOC-Typing-v1")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "uploads" / "hf_model")
    p.add_argument("--push", action="store_true")
    args = p.parse_args(argv)

    if not args.checkpoint.exists():
        print(f"missing {args.checkpoint}")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    dest_ckpt = args.out / args.checkpoint.name
    shutil.copy2(args.checkpoint, dest_ckpt)
    (args.out / "README.md").write_text(README, encoding="utf-8")
    meta = torch_load_meta(args.checkpoint)
    meta["verified_at"] = __import__("datetime").datetime.utcnow().isoformat() + "Z"
    meta["repo"] = args.repo_id
    meta["language"] = "en"
    (args.out / "model_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"staged {args.out}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=str(args.out), repo_id=args.repo_id, repo_type="model")
        print(f"https://huggingface.co/{args.repo_id}")
    return 0


def torch_load_meta(path: Path) -> dict:
    import torch

    b = torch.load(path, weights_only=True)
    return {k: b[k] for k in ("model", "task", "n_channels", "n_classes", "val_acc", "finetuned") if k in b}


if __name__ == "__main__":
    raise SystemExit(main())