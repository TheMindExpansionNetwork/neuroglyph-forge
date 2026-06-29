"""Subject-specific fine-tune (freeze encoder, train classifier + adapter)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from neuroglyph_data.splits import train_val_loaders
from neuroglyph_models.adapters import SubjectAdapter
from neuroglyph_models.heads import TASK_HEADS
from neuroglyph_train.train import build_model, eval_epoch, run_epoch


class FinetuneWrapper(nn.Module):
    def __init__(self, base: nn.Module, hidden: int = 256):
        super().__init__()
        self.base = base
        self.adapter = SubjectAdapter(hidden)

    def forward(self, x):
        logits = self.base(x)
        # Adapter hooks on pooled features would need architecture change;
        # for MVP we train full base with low LR when freeze_encoder=false.
        return logits


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

    blob = torch.load(args.base_checkpoint, weights_only=True)
    task = args.task or blob["task"]
    head = TASK_HEADS[task]
    model = build_model(blob["model"], blob["n_channels"], head["n_classes"])
    model.load_state_dict(blob["state_dict"])

    if args.freeze_encoder and hasattr(model, "encoder"):
        for p_enc in model.encoder.parameters():
            p_enc.requires_grad = False

    train_loader, val_loader = train_val_loaders(args.data, task=task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / f"finetune_{task}.pt"
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, loss_fn, device)
        va_loss, va_acc = eval_epoch(model, val_loader, loss_fn, device)
        print(f"finetune {epoch}/{args.epochs} val_acc={va_acc:.3f}")
        if va_acc > best:
            best = va_acc
            torch.save(
                {
                    "model": blob["model"],
                    "task": task,
                    "state_dict": model.state_dict(),
                    "n_channels": blob["n_channels"],
                    "n_classes": head["n_classes"],
                    "val_acc": va_acc,
                    "finetuned": True,
                },
                out,
            )
    print(f"saved {out} best_val_acc={best:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())