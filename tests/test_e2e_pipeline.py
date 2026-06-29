"""Integration: synthetic CSV session → epochs → train → live decode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neuroglyph_data.make_epochs import build_processed_from_raw
from neuroglyph_data.schema import KeyEvent, SessionMeta
from neuroglyph_agent.live_decoder import LiveDecoder, LiveDecoderConfig
from neuroglyph_train.train import main as train_main


def _write_fake_raw(tmp_path: Path, n_samples: int = 5000, fs: int = 256) -> None:
    from neuroglyph_data.schema import EPOC_X_CHANNELS

    t = np.arange(n_samples) / fs
    data = {ch: np.sin(2 * np.pi * 10 * t + i) for i, ch in enumerate(EPOC_X_CHANNELS)}
    data["Timestamp"] = t
    df = pd.DataFrame(data)
    df.to_csv(tmp_path / "ses-test_eeg.csv", index=False)

    events = []
    for i in range(40):
        key = "a" if i % 2 == 0 else "j"  # left vs right hand
        events.append(
            KeyEvent(
                key=key,
                timestamp_unix_ms=int((2.0 + i * 0.5) * 1000),
                trial_id=f"t{i}",
            )
        )
    meta = SessionMeta(subject_id="sub-test", session_id="ses-test", sample_rate=fs, events=events)
    (tmp_path / "ses-test_meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")


def test_raw_to_live_decode(tmp_path):
    raw = tmp_path / "raw"
    proc = tmp_path / "processed"
    ckpt_dir = tmp_path / "ckpt"
    raw.mkdir()
    _write_fake_raw(raw)

    build_processed_from_raw(raw, proc, task="hand")
    train_main(["--task", "hand", "--data", str(proc), "--epochs", "4", "--output", str(ckpt_dir)])
    ckpt = ckpt_dir / "tiny_b2q_hand.pt"
    assert ckpt.exists()

    dec = LiveDecoder(LiveDecoderConfig(checkpoint=ckpt, confidence_threshold=0.0))
    sample = np.random.randn(14).astype(np.float32)
    for _ in range(30):
        ev = dec.push_sample(sample)
    # May or may not fire depending on random weights; ensure no crash
    assert dec.model is not None