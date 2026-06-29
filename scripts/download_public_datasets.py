#!/usr/bin/env python3
"""Download / prepare public datasets for NeuroGlyph training."""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuroglyph_data.public_datasets import load_kaggle_hands_csv, write_processed_pt
from neuroglyph_data.synthetic import write_synthetic_processed


def cmd_synthetic(out: Path, task: str, n: int) -> None:
    write_synthetic_processed(out, task=task, n_epochs=n)
    print(f"synthetic {task} -> {out}")


def cmd_kaggle(csv: Path, out: Path) -> None:
    X, y = load_kaggle_hands_csv(csv)
    write_processed_pt(out, X, y, task="hand", source="kaggle_emotiv_hands")
    print(f"kaggle hands n={X.shape[0]} -> {out}")


def cmd_hf_spanish(out: Path) -> None:
    """SpanishBCBL metadata — full epochs require brain2qwerty pipeline."""
    meta_dir = out.parent / "external" / "spanishbcbl"
    meta_dir.mkdir(parents=True, exist_ok=True)
    import json

    note = {
        "dataset": "bcbl190626/SpanishBCBL",
        "hf_url": "https://huggingface.co/datasets/bcbl190626/SpanishBCBL",
        "usage": "Clone brain2qwerty and run v1 training; not 14ch EPOC-native.",
        "clone": "git clone https://github.com/facebookresearch/brain2qwerty ../brain2qwerty",
    }
    try:
        from datasets import load_dataset

        ds = load_dataset("bcbl190626/SpanishBCBL", split="train", streaming=True)
        rows = []
        for i, row in enumerate(ds):
            rows.append({k: row[k] for k in list(row.keys())[:8]})
            if i >= 49:
                break
        note["sample_rows"] = rows
    except Exception as e:
        note["hf_error"] = str(e)
        note["hint"] = "Install brain2qwerty deps or download HF snapshot manually"

    (meta_dir / "README.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(f"Wrote {meta_dir / 'README.json'}")
    write_synthetic_processed(out, task="hand", n_epochs=600)
    print(f"Wrote synthetic hand for immediate train -> {out}")


def main() -> int:
    p = argparse.ArgumentParser(description="Prepare trainable datasets")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synthetic", help="CI / smoke epochs")
    s.add_argument("--task", default="hand")
    s.add_argument("--n", type=int, default=500)

    k = sub.add_parser("kaggle-hands", help="Emotiv Insight hand movement CSV")
    k.add_argument("--csv", type=Path, required=True)

    h = sub.add_parser("hf-spanishbcbl", help="HF B2Q dataset sample + synthetic fallback")

    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.cmd == "synthetic":
        cmd_synthetic(args.out, args.task, args.n)
    elif args.cmd == "kaggle-hands":
        cmd_kaggle(args.csv, args.out)
    elif args.cmd == "hf-spanishbcbl":
        cmd_hf_spanish(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())