"""Estimate clock_offset_ms between keyboard and EEG CSV using marker keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroglyph_recorder.marker_sync import estimate_offset_from_markers


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tune clock_offset_ms for a session")
    p.add_argument("--meta", type=Path, required=True, help="Session *_meta.json")
    p.add_argument("--cortex-markers", type=Path, help="JSON list of cortex marker times (ms)")
    p.add_argument("--offset-ms", type=float, default=None, help="Apply this offset directly")
    args = p.parse_args(argv)

    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    if args.offset_ms is not None:
        meta["clock_offset_ms"] = args.offset_ms
    elif args.cortex_markers:
        host = [e["timestamp_unix_ms"] for e in meta.get("events", [])]
        cortex = json.loads(args.cortex_markers.read_text(encoding="utf-8"))
        sync = estimate_offset_from_markers(host, cortex)
        meta["clock_offset_ms"] = sync.offset_ms
        print(f"estimated offset_ms={sync.offset_ms:.2f}")
    else:
        print("Provide --cortex-markers or --offset-ms")
        return 1

    args.meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"updated {args.meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())