"""Training loop for NeuroGlyph decoders."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from neuroglyph_data.splits import train_val_loaders
from neuroglyph_models.eegnet import EEGNet
from neuroglyph_models.heads import TASK_HEADS
from neuroglyph_models.tiny_b2q import TinyB2Q


def build_model(name: str, n_channels: int, n_classes: int, n_time: int = 25) -> nn.Module:
    if name == "eegnet":
        return EEGNet(n_channels=n_channels, n_classes=n_classes, n_time=n_time)
    return TinyB2Q(n_channels=n_channels, n_classes=n_classes)


def run_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += X.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train NeuroGlyph decoder")
    p.add_argument("--task", default="hand", choices=list(TASK_HEADS.keys()))
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--model", default="tiny_b2q", choices=["tiny_b2q", "eegnet"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", type=Path, default=Path("checkpoints"))
    args = p.parse_args(argv)

    head = TASK_HEADS[args.task]
    train_loader, val_loader = train_val_loaders(args.data, task=args.task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, n_channels=14, n_classes=head["n_classes"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.output / f"{args.model}_{args.task}.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, loss_fn, device)
        va_loss, va_acc = eval_epoch(model, val_loader, loss_fn, device)
        print(
            f"epoch {epoch}/{args.epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.3f}"
        )
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "model": args.model,
                    "task": args.task,
                    "state_dict": model.state_dict(),
                    "n_channels": 14,
                    "n_classes": head["n_classes"],
                    "val_acc": va_acc,
                },
                ckpt_path,
            )

    print(f"saved {ckpt_path} best_val_acc={best_acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())