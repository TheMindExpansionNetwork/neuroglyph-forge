#!/usr/bin/env python3
"""Pre-download HF model + dataset snapshots for local fine-tune."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_REPO = "TheMindExpansionNetwork/NeuroGlyph-EPOC-Typing-v1"
DATASET_REPO = "TheMindExpansionNetwork/neuroglyph-epoc-typing-en-v1"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, default=ROOT / "checkpoints" / "hf" / "NeuroGlyph-EPOC-Typing-v1")
    p.add_argument("--dataset-dir", type=Path, default=ROOT / "data" / "hf" / "neuroglyph-epoc-typing-en-v1")
    p.add_argument("--model-only", action="store_true")
    p.add_argument("--dataset-only", action="store_true")
    args = p.parse_args(argv)

    from huggingface_hub import snapshot_download

    if not args.dataset_only:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(repo_id=MODEL_REPO, repo_type="model", local_dir=str(args.model_dir))
        print(f"model -> {path}")

    if not args.model_only:
        args.dataset_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", local_dir=str(args.dataset_dir))
        print(f"dataset -> {path}")
        # Copy raw sessions into data/raw/hf_import if present
        raw_src = args.dataset_dir / "raw"
        if raw_src.is_dir():
            import shutil

            dest = ROOT / "data" / "raw" / "hf_import"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(raw_src, dest)
            print(f"imported raw -> {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())