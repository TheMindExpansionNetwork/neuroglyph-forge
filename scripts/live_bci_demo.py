"""
Live BCI demo: EEG stream → LiveDecoder → Hermes/Unreal action plan.

Mock (default):
  python scripts/live_bci_demo.py --checkpoint checkpoints/tiny_b2q_hand.pt

With Cortex (Premium, env vars set):
  python scripts/live_bci_demo.py --live --task hand
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from neuroglyph_agent.hermes_bridge import format_hermes_message
from neuroglyph_agent.live_decoder import LiveDecoder, LiveDecoderConfig
from neuroglyph_agent.policy import ActionRateLimiter, PredictionEvent, should_act
from neuroglyph_recorder.cortex_client import CortexClient, CortexConfig
from neuroglyph_unreal.ue_actions import prediction_to_unreal_action


async def run_demo(checkpoint: Path, live: bool, seconds: float, task: str) -> int:
    dec = LiveDecoder(LiveDecoderConfig(checkpoint=checkpoint, confidence_threshold=0.55))
    limiter = ActionRateLimiter(min_interval_sec=0.5)
    cortex = CortexClient(CortexConfig(mock=not live))
    await cortex.connect()
    if live:
        await cortex.bootstrap()

    # Downsample 256 Hz → 50 Hz for model
    step = max(1, 256 // 50)
    buf_i = 0
    n_pred = 0
    import time

    deadline = time.monotonic() + seconds

    async for sample, _ in cortex.stream_eeg():
        if time.monotonic() >= deadline:
            break
        buf_i += 1
        if buf_i % step != 0:
            continue
        ev = dec.push_sample(__import__("numpy").array(sample))
        if not ev or not limiter.should_emit(ev):
            continue
        n_pred += 1
        msg = format_hermes_message(ev)
        ue = prediction_to_unreal_action(ev)
        print(json.dumps({"hermes": msg, "unreal": ue}, indent=2))

    print(f"demo done predictions_emitted={n_pred}")
    await cortex.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tiny_b2q_hand.pt"))
    p.add_argument("--live", action="store_true")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--task", default="hand")
    args = p.parse_args(argv)
    if not args.checkpoint.exists():
        print(f"missing checkpoint {args.checkpoint} — run scripts/pipeline.sh hand first")
        return 1
    return asyncio.run(run_demo(args.checkpoint, args.live, args.seconds, args.task))


if __name__ == "__main__":
    raise SystemExit(main())