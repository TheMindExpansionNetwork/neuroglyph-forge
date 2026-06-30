#!/usr/bin/env python3
"""Check Cortex + env before a live EPOC session."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


async def probe(live: bool) -> int:
    load_env()
    from neuroglyph_recorder.cortex_client import CortexClient, CortexConfig

    cid = os.environ.get("EMOTIV_CLIENT_ID", "")
    sec = os.environ.get("EMOTIV_CLIENT_SECRET", "")
    print(f"cortex_creds={'yes' if cid and sec else 'NO — copy .env.example to .env'}")
    print(f"mode={'live' if live else 'mock'}")

    client = CortexClient(CortexConfig(mock=not live))
    try:
        await client.connect()
        if live:
            session = await client.bootstrap()
            print(f"headset={session.headset_id} session={session.session_id}")
        n = 0
        async for _sample, _ in client.stream_eeg():
            n += 1
            if n >= 5:
                break
        print(f"eeg_samples={n} OK")
        if live:
            await client.close()
        return 0
    except Exception as e:
        print(f"FAIL: {e}")
        print("Fix: EMOTIV Launcher running, headset on, app approved, Premium EEG.")
        return 1


def main() -> int:
    live = "--live" in sys.argv
    return asyncio.run(probe(live))


if __name__ == "__main__":
    raise SystemExit(main())