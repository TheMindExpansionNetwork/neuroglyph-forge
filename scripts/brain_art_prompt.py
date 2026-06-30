#!/usr/bin/env python3
"""
Build a gpt-image-2 prompt from real smoke / eval / epoch tensors (NeuroGlyph).

Art = visualization of what the fine-tuned decoder *saw* on sample EEG epochs,
not generic product marketing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_checkpoint_meta(ckpt: Path) -> dict[str, Any]:
    b = torch.load(ckpt, weights_only=True)
    return {
        "task": b.get("task", "hand"),
        "model": b.get("model", "tiny_b2q"),
        "n_channels": int(b.get("n_channels", 14)),
        "n_classes": int(b.get("n_classes", 2)),
        "val_acc": float(b.get("val_acc", 0.0)),
    }


def epoch_visual_brief(X: torch.Tensor, y: int, pred: int, classes: list[str]) -> str:
    """Turn one epoch (C,T) into words an image model can paint."""
    x = X.float()
    c, t = x.shape
    # per-channel energy for composition hints
    energy = (x.pow(2).mean(dim=1)).tolist()
    top_ch = sorted(range(c), key=lambda i: energy[i], reverse=True)[:4]
    label_name = classes[y] if y < len(classes) else str(y)
    pred_name = classes[pred] if pred < len(classes) else str(pred)
    match = y == pred
    return (
        f"One keystroke-aligned EEG epoch: {c} parallel micro-wave ribbons over {t} time steps at 50 Hz. "
        f"Ground truth motor intent: **{label_name}** hand; decoder prediction: **{pred_name}** "
        f"({'aligned glow' if match else 'fractured split — prediction wrong, tension between teal truth and amber error'}). "
        f"Loudest cortical bands on sensor indices {top_ch} rendered as brighter filament thickness. "
    )


def run_smoke_preds(ckpt: Path, data: Path, n: int = 32) -> dict[str, Any]:
    from neuroglyph_train.train import build_model

    blob = torch.load(ckpt, weights_only=True)
    model = build_model(blob["model"], blob["n_channels"], blob["n_classes"])
    model.load_state_dict(blob["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    pt = torch.load(data, weights_only=True)
    X, y = pt["X"], pt["y"]
    n = min(n, X.shape[0])
    with torch.no_grad():
        pred = model(X[:n].to(device)).argmax(1).cpu()
    ys, ps = y[:n].tolist(), pred.tolist()
    acc = sum(a == b for a, b in zip(ys, ps)) / n
    # pick one interesting epoch: first mismatch or index 0
    idx = next((i for i in range(n) if ys[i] != ps[i]), 0)
    from neuroglyph_models.heads import TASK_HEADS

    classes = TASK_HEADS[blob["task"]]["classes"]
    brief = epoch_visual_brief(X[idx], ys[idx], ps[idx], classes)
    bitstream = "".join("L" if p == 0 else "R" for p in ps[: min(24, n)])
    return {
        "n_smoke": n,
        "smoke_accuracy": acc,
        "labels_bits": bitstream,
        "highlight_epoch_index": idx,
        "epoch_brief": brief,
        "classes": classes,
    }


def build_image_prompt(meta: dict, smoke: dict, *, eval_metrics: dict | None) -> str:
    cm_line = ""
    if eval_metrics and eval_metrics.get("confusion_matrix"):
        cm = eval_metrics["confusion_matrix"]
        cm_line = (
            f"Confusion matrix as abstract dual-key sculpture: left-correct mass {cm[0][0]}, "
            f"left-as-right errors {cm[0][1]}, right-as-left {cm[1][0]}, right-correct {cm[1][1]}. "
        )

    return (
        "Generative **data painting** from a real BCI training smoke test (synthetic EPOC-style 14-channel EEG, "
        "English QWERTY keystroke epochs, fine-tuned TinyB2Q decoder on RTX 4070). "
        f"Task: {meta['task']} · val_acc after fine-tune: {meta['val_acc']:.2f} · "
        f"smoke batch accuracy on {smoke['n_smoke']} epochs: {smoke['smoke_accuracy']:.2f}. "
        f"{cm_line}"
        f"{smoke['epoch_brief']}"
        f"Repeat a subtle motif of alternating keys along the bottom edge inspired by decoder bitstream: {smoke['labels_bits']} "
        "(L=left R=right, as tiny luminous key caps, not readable text). "
        "Visual style: dark void #070b14, neural filaments teal #2dd4bf, gold #fbbf24 for correct alignment, "
        "magenta fracture lines only where prediction disagrees with label. "
        "Include a faint ring of 14 EEG electrodes. "
        "Looks like scientific art + synesthetic brain-to-keyboard energy — **not** a software screenshot, **not** logos, **no** typography. "
        "Cinematic 16:9, museum-quality abstract neuroscience."
    )


def partner_payload(prompt: str) -> dict[str, Any]:
    return {
        "type": "image",
        "model": "openai/images-generations",
        "prompt": prompt,
        "aspect_ratio": "16:9",
        "client_os": "windows",
        "params": {"model": "gpt-image-2", "quality": "medium", "size": "1536x1024"},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "finetune_hand.pt")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "processed_hand.pt")
    p.add_argument("--eval-json", type=Path, default=None)
    p.add_argument("--out-prompt", type=Path, default=ROOT / "data" / "brain_art_prompt.txt")
    p.add_argument("--out-json", type=Path, default=ROOT / "assets" / "brand" / "comfy-partner_generate-brain-smoke.json")
    p.add_argument("--write-eval", type=Path, default=ROOT / "data" / "eval_hand.json")
    args = p.parse_args(argv)

    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    if not args.data.is_file():
        raise SystemExit(f"missing processed data: {args.data}")

    meta = load_checkpoint_meta(args.checkpoint)
    smoke = run_smoke_preds(args.checkpoint, args.data)

    eval_metrics = None
    if args.eval_json and args.eval_json.is_file():
        eval_metrics = json.loads(args.eval_json.read_text(encoding="utf-8"))
    else:
        from neuroglyph_train.evaluate import evaluate_checkpoint

        eval_metrics = evaluate_checkpoint(args.checkpoint, args.data.parent)
        args.write_eval.parent.mkdir(parents=True, exist_ok=True)
        args.write_eval.write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")

    prompt = build_image_prompt(meta, smoke, eval_metrics=eval_metrics)
    payload = partner_payload(prompt)

    args.out_prompt.write_text(prompt, encoding="utf-8")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    bundle = {
        "meta": meta,
        "smoke": {k: v for k, v in smoke.items() if k != "epoch_brief"},
        "eval_accuracy": eval_metrics.get("accuracy"),
        "prompt_chars": len(prompt),
        "partner_generate": args.out_json.as_posix(),
    }
    (ROOT / "data" / "brain_art_bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print("BRAIN_ART_PROMPT\n", prompt[:2000], ("…" if len(prompt) > 2000 else ""))
    print("\nWrote", args.out_prompt, args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())