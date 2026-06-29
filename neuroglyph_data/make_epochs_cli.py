"""CLI entry for epoch building from raw sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

from neuroglyph_data.make_epochs import build_processed_from_raw
from neuroglyph_data.synthetic import write_synthetic_processed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/processed"))
    p.add_argument("--task", default="hand")
    p.add_argument("--synthetic", action="store_true", help="Skip raw; write synthetic tensors")
    p.add_argument("--clock-offset-ms", type=float, default=0.0)
    args = p.parse_args(argv)
    if args.synthetic:
        path = write_synthetic_processed(args.out, task=args.task)
    else:
        path = build_processed_from_raw(args.raw, args.out, task=args.task, clock_offset_ms=args.clock_offset_ms)
    print(f"processed: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())