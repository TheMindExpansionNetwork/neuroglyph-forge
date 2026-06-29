"""
Live BCI demo: EEG stream → LiveDecoder → Hermes/Unreal action plan.

Mock (default):
  python scripts/live_bci_demo.py --checkpoint checkpoints/tiny_b2q_hand.pt

With Cortex (Premium, env vars set):
  python scripts/live_bci_demo.py --live --task hand --situation live_bci
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
from neuroglyph_agent.mindbot_bus import append_step
from neuroglyph_agent.situations import (
    AdaptiveQueue,
    Situation,
    SituationRouter,
    SituationState,
    infer_situation_from_env,
)
from neuroglyph_recorder.cortex_client import CortexClient, CortexConfig
from neuroglyph_unreal.ue_actions import prediction_to_unreal_action


async def run_demo(
    checkpoint: Path,
    live: bool,
    seconds: float,
    task: str,
    situation: Situation | None = None,
) -> int:
    import torch

    blob = torch.load(checkpoint, weights_only=True)
    val_acc = float(blob.get("val_acc", 1.0))

    if situation is None:
        state = infer_situation_from_env(cortex_mock=not live, checkpoint_val_acc=val_acc)
    else:
        state = SituationState(active=situation)

    router = SituationRouter(state)
    prof = state.profile()
    dec = LiveDecoder(
        LiveDecoderConfig(checkpoint=checkpoint, confidence_threshold=prof.confidence_min)
    )
    print(json.dumps({"situation": router.hermes_context_block()}, indent=2), flush=True)

    cortex = CortexClient(CortexConfig(mock=not live))
    await cortex.connect()
    if live:
        await cortex.bootstrap()

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
        if not ev or not router.should_emit(ev):
            continue
        n_pred += 1
        msg = format_hermes_message(ev)
        ue = prediction_to_unreal_action(ev)
        print(json.dumps({"hermes": msg, "unreal": ue}, indent=2))
        if state.effective() in (Situation.STREAM_NARRATIVE, Situation.LIVE_BCI):
            append_step(ev, situation=state.effective().value, unreal_action=ue)

    print(f"demo done predictions_emitted={n_pred}")
    await cortex.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tiny_b2q_hand.pt"))
    p.add_argument("--live", action="store_true")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--task", default="hand")
    p.add_argument("--situation", type=str, choices=[s.value for s in Situation], default=None)
    args = p.parse_args(argv)
    if not args.checkpoint.exists():
        print(f"missing checkpoint {args.checkpoint} — run scripts/pipeline.sh hand first")
        return 1
    sit = Situation(args.situation) if args.situation else None
    return asyncio.run(run_demo(args.checkpoint, args.live, args.seconds, args.task, sit))


if __name__ == "__main__":
    raise SystemExit(main())