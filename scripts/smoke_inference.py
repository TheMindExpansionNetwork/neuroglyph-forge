#!/usr/bin/env python3
"""Smoke test: load checkpoint, run on sample epochs from processed_hand.pt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


def run_smoke_inference(
    checkpoint: Path,
    data: Path,
    *,
    n_samples: int = 8,
    min_accuracy: float = 0.45,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Forward pass on up to ``n_samples`` epochs; return report dict."""
    from neuroglyph_train.train import build_model

    blob = torch.load(checkpoint, weights_only=True)
    model = build_model(blob["model"], blob["n_channels"], blob["n_classes"])
    model.load_state_dict(blob["state_dict"])
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    pt = torch.load(data, weights_only=True)
    X, y = pt["X"], pt["y"]
    n = min(n_samples, X.shape[0])
    Xs, ys = X[:n].to(device), y[:n].to(device)
    with torch.no_grad():
        pred = model(Xs).argmax(1)
    acc = (pred == ys).float().mean().item()
    preds = pred.cpu().tolist()
    labels = ys.cpu().tolist()

    return {
        "checkpoint": str(checkpoint.resolve()),
        "data": str(data.resolve()),
        "device": str(device),
        "n_samples": n,
        "accuracy": acc,
        "val_acc_in_ckpt": blob.get("val_acc"),
        "predictions_sample": preds[: min(16, n)],
        "labels_sample": labels[: min(16, n)],
        "ok": acc > min_accuracy,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "processed_hand.pt")
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--min-accuracy", type=float, default=0.45, help="Exit 1 if accuracy below this (use 0 for verify-only)")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "smoke_report.json")
    args = p.parse_args(argv)

    if not args.checkpoint.is_file():
        print(f"checkpoint not found: {args.checkpoint}")
        return 1
    if not args.data.is_file():
        candidates = sorted((ROOT / "data" / "processed").rglob("processed_hand.pt"))
        if candidates:
            args.data = candidates[0]
        else:
            print(f"data not found: {args.data}")
            return 1

    report = run_smoke_inference(
        args.checkpoint, args.data, n_samples=args.n_samples, min_accuracy=args.min_accuracy
    )
    print(f"accuracy={report['accuracy']:.4f} (n={report['n_samples']})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())