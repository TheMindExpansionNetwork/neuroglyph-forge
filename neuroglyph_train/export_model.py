"""Export trained decoder for live inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neuroglyph_models.heads import TASK_HEADS
from neuroglyph_train.train import build_model


def export_torchscript(checkpoint: Path, out_path: Path) -> Path:
    blob = torch.load(checkpoint, weights_only=True)
    model = build_model(blob["model"], blob["n_channels"], blob["n_classes"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    example = torch.randn(1, blob["n_channels"], 25)
    traced = torch.jit.trace(model, example)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    meta = {
        "task": blob["task"],
        "classes": TASK_HEADS[blob["task"]]["classes"],
        "n_channels": blob["n_channels"],
    }
    out_path.with_suffix(".meta.json").write_text(
        __import__("json").dumps(meta, indent=2), encoding="utf-8"
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("checkpoints/exported.ts"))
    args = p.parse_args(argv)
    path = export_torchscript(args.checkpoint, args.output)
    print(f"exported {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())