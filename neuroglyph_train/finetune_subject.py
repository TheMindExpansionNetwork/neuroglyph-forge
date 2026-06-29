"""Subject-specific fine-tune (freeze encoder, train classifier + adapter)."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroglyph_train.engine import finetune_decoder


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--task", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--output", type=Path, default=Path("checkpoints"))
    args = p.parse_args(argv)

    finetune_decoder(
        base_checkpoint=args.base_checkpoint,
        data_dir=args.data,
        task=args.task,
        epochs=args.epochs,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())