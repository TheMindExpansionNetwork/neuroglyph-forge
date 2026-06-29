#!/usr/bin/env python3
"""Stage raw sessions for Hugging Face (English EPOC typing)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "uploads" / "hf_dataset")
    p.add_argument("--repo-id", default="TheMindExpansionNetwork/neuroglyph-epoc-typing-en-v1")
    p.add_argument("--push", action="store_true")
    args = p.parse_args(argv)

    if not args.raw.exists() or not list(args.raw.iterdir()):
        print("No raw data — see docs/FIRST_SESSION_PROTOCOL.md")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "raw"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(args.raw, dest)
    (args.out / "dataset_meta.json").write_text(
        json.dumps({"language": "en", "repo": args.repo_id}, indent=2),
        encoding="utf-8",
    )
    print(f"staged {args.out}")
    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(folder_path=str(args.out), repo_id=args.repo_id, repo_type="dataset")
        print(f"https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())