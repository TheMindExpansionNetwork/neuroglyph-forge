"""Run a typing recording session (mock or live Cortex)."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from neuroglyph_data.schema import KeyEvent, SessionMeta
from neuroglyph_recorder.cortex_client import CortexClient, CortexConfig
from neuroglyph_recorder.eeg_writer import EegCsvWriter
from neuroglyph_recorder.keylogger import KeyLogger


async def run_session(
    subject_id: str,
    session_id: str,
    duration_sec: float,
    out_dir: Path,
    mock: bool = True,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cortex = CortexClient(CortexConfig(mock=mock))
    await cortex.connect()

    meta = SessionMeta(subject_id=subject_id, session_id=session_id)
    pending_markers: list[str] = []
    t0 = time.time()

    def on_key(ev):
        meta.events.append(
            KeyEvent(key=ev.key, timestamp_unix_ms=ev.timestamp_unix_ms, trial_id=f"trial-{len(meta.events)}")
        )
        pending_markers.append(ev.key)

    logger = KeyLogger(on_key=on_key)
    if not mock:
        logger.start()
        session = await cortex.bootstrap()
        meta.cortex_session_id = session.session_id
        meta.headset_id = session.headset_id

    eeg_path = out_dir / f"{session_id}_eeg.csv"
    ticks = 0
    with EegCsvWriter(eeg_path, sample_rate=meta.sample_rate) as writer:
        deadline = time.time() + duration_sec
        async for sample, counter in cortex.stream_eeg():
            if time.time() >= deadline:
                break
            ts = counter if counter is not None else ticks
            writer.write_row(float(ts), sample)
            ticks += 1
            while pending_markers:
                label = pending_markers.pop(0)
                await cortex.inject_marker(label)

    if not mock:
        logger.stop()
        await cortex.close()

    meta_path = out_dir / f"{session_id}_meta.json"
    meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / f"{session_id}_eeg_stats.json").write_text(
        json.dumps({"ticks": ticks, "duration_sec": duration_sec, "csv": eeg_path.name}, indent=2),
        encoding="utf-8",
    )
    return meta_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="sub-001")
    p.add_argument("--session", default="ses-001")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--live", action="store_true", help="Use pynput + real Cortex (not mock)")
    args = p.parse_args(argv)
    path = asyncio.run(
        run_session(args.subject, args.session, args.duration, args.out, mock=not args.live)
    )
    print(f"session meta written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())