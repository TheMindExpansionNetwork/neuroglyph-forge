"""Evaluation metrics: accuracy, confusion matrix, char CER proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from neuroglyph_data.splits import load_processed_dir
from neuroglyph_models.heads import TASK_HEADS
from neuroglyph_train.train import build_model


@torch.no_grad()
def evaluate_checkpoint(checkpoint: Path, data_dir: Path) -> dict:
    blob = torch.load(checkpoint, weights_only=True)
    task = blob["task"]
    head = TASK_HEADS[task]
    model = build_model(blob["model"], blob["n_channels"], blob["n_classes"])
    model.load_state_dict(blob["state_dict"])
    model.eval()

    X, y, _ = load_processed_dir(data_dir, task=task)
    logits = model(X)
    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean().item()
    cm = confusion_matrix(y.numpy(), pred.numpy(), labels=list(range(head["n_classes"])))
    report = classification_report(
        y.numpy(),
        pred.numpy(),
        target_names=head["classes"],
        output_dict=True,
        zero_division=0,
    )
    return {
        "task": task,
        "accuracy": acc,
        "n_samples": int(y.shape[0]),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "classes": head["classes"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)
    metrics = evaluate_checkpoint(args.checkpoint, args.data)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())