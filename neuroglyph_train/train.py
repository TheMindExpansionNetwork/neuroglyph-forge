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

    from neuroglyph_train.engine import train_decoder

    train_decoder(
        task=args.task,
        data_dir=args.data,
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        output_dir=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())