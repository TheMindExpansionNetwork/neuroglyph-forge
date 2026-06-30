"""Smoke inference on synthetic processed data."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_inference_runs():
    ckpt = ROOT / "checkpoints" / "tiny_b2q_hand.pt"
    if not ckpt.exists():
        pytest.skip("no checkpoint")
    data = ROOT / "data" / "processed" / "processed_hand.pt"
    if not data.exists():
        pytest.skip("no processed data")
    import subprocess
    import sys

    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/smoke_inference.py"), "--checkpoint", str(ckpt)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    assert (ROOT / "data" / "smoke_report.json").exists()