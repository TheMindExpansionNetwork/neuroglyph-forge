"""Interactive collection session (CLI) for real data gathering."""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from neuroglyph_recorder.prompts import next_prompt
from neuroglyph_recorder.session_runner import run_session


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Guided EPOC typing collection")
    p.add_argument("--subject", default="sub-001")
    p.add_argument("--session", default=None)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--seconds-per-block", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--live", action="store_true")
    args = p.parse_args(argv)

    session_base = args.session or time.strftime("ses-%Y%m%d-%H%M%S")
    for b in range(args.blocks):
        sid = f"{session_base}-b{b}"
        prompt = next_prompt(b, b)
        print(f"\n=== Block {b + 1}/{args.blocks} ===")
        print(f"Type this sentence repeatedly for {args.seconds_per_block:.0f}s:")
        print(f"  >> {prompt}")
        print("Press Enter when ready...")
        input()
        asyncio.run(
            run_session(
                args.subject,
                sid,
                args.seconds_per_block,
                args.out,
                mock=not args.live,
            )
        )
        print(f"Saved metadata: {args.out / (sid + '_meta.json')}")
        print("Export EEG CSV from Cortex into the same folder before preprocessing.")

    print("\nNext: place EEG CSV in data/raw/ then:")
    print("  python -m neuroglyph_data.make_epochs_cli --raw data/raw --out data/processed --task hand")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())