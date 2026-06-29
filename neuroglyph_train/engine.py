"""Shared train/eval loops (local + Modal)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from neuroglyph_data.splits import train_val_loaders
from neuroglyph_models.heads import TASK_HEADS
from neuroglyph_train.train import build_model, eval_epoch, run_epoch


def train_decoder(
    *,
    task: str = "hand",
    data_dir: Path | str = Path("data/processed"),
    model_name: str = "tiny_b2q",
    epochs: int = 10,
    lr: float = 1e-3,
    output_dir: Path | str = Path("checkpoints"),
    device: torch.device | None = None,
    batch_size: int = 32,
) -> Path:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    head = TASK_HEADS[task]
    train_loader, val_loader = train_val_loaders(data_dir, task=task, batch_size=batch_size)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, n_channels=14, n_classes=head["n_classes"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    ckpt_path = output_dir / f"{model_name}_{task}.pt"

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, loss_fn, device)
        va_loss, va_acc = eval_epoch(model, val_loader, loss_fn, device)
        print(
            f"epoch {epoch}/{epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.3f} device={device}",
            flush=True,
        )
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "model": model_name,
                    "task": task,
                    "state_dict": model.state_dict(),
                    "n_channels": 14,
                    "n_classes": head["n_classes"],
                    "val_acc": va_acc,
                },
                ckpt_path,
            )

    print(f"saved {ckpt_path} best_val_acc={best_acc:.3f}", flush=True)
    return ckpt_path


def finetune_decoder(
    *,
    base_checkpoint: Path | str,
    data_dir: Path | str = Path("data/processed"),
    task: str | None = None,
    epochs: int = 20,
    lr: float = 1e-4,
    freeze_encoder: bool = False,
    output_dir: Path | str = Path("checkpoints"),
    device: torch.device | None = None,
) -> Path:
    base_checkpoint = Path(base_checkpoint)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blob = torch.load(base_checkpoint, weights_only=True)
    task = task or blob["task"]
    head = TASK_HEADS[task]
    model = build_model(blob["model"], blob["n_channels"], head["n_classes"])
    model.load_state_dict(blob["state_dict"])

    if freeze_encoder and hasattr(model, "encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False

    train_loader, val_loader = train_val_loaders(data_dir, task=task)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    out = output_dir / f"finetune_{task}.pt"
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, loss_fn, device)
        va_loss, va_acc = eval_epoch(model, val_loader, loss_fn, device)
        print(f"finetune {epoch}/{epochs} val_acc={va_acc:.3f}", flush=True)
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
                    "base_checkpoint": str(base_checkpoint),
                },
                out,
            )
    print(f"saved {out} best_val_acc={best:.3f}", flush=True)
    return out