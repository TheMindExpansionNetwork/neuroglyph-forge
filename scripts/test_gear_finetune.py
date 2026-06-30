#!/usr/bin/env python3
"""
End-to-end gear test: mock Cortex session → epochs → fine-tune → evaluate.

Proves your PC can fine-tune before/without live EPOC. For live headset, also run:
  scripts/cortex_probe.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=False)


def main() -> int:
    raw = ROOT / "data" / "raw" / "gear_test"
    proc = ROOT / "data" / "processed" / "gear_test"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    proc.mkdir(parents=True, exist_ok=True)

    # 1) Mock session (same code path as --live except Cortex mock EEG)
    from neuroglyph_recorder.session_runner import run_session

    asyncio.run(run_session("sub-gear-test", "ses-gear-001", 8.0, raw, mock=True))
    eeg = raw / "ses-gear-001_eeg.csv"
    meta = raw / "ses-gear-001_meta.json"
    if not eeg.exists() or not meta.exists():
        print("FAIL: session files missing")
        return 1

    # Mock mode does not capture keys — inject aligned events for epoch test
    import numpy as np
    import pandas as pd

    df = pd.read_csv(eeg)
    tcol = df.columns[0]
    ts = df[tcol].to_numpy(dtype=np.float64)
    if ts.max() < 1e6:
        ts_ms = ts * 1000.0
    else:
        ts_ms = ts
    meta_obj = json.loads(meta.read_text(encoding="utf-8"))
    keys = ["a", "j", "s", "k", "d", "l"] * 20
    events = []
    for i, k in enumerate(keys):
        idx = min(len(ts_ms) - 1, 50 + i * 8)
        events.append({"key": k, "timestamp_unix_ms": int(round(float(ts_ms[idx]))), "trial_id": f"t{i}", "target_sentence": None})
    meta_obj["events"] = events
    meta_obj["synthetic_keys_injected"] = True
    meta.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")

    rows = len(df) - 1
    print(f"mock session: eeg_rows={rows} events={len(events)}")

    # 2) Epochs
    r = run(
        [
            sys.executable,
            "-m",
            "neuroglyph_data.make_epochs_cli",
            "--raw",
            str(raw),
            "--out",
            str(proc),
            "--task",
            "hand",
        ]
    )
    if r.returncode != 0:
        print("FAIL: make_epochs")
        return 1

    # 3) Fine-tune from existing base (synthetic pretrain)
    base = ROOT / "checkpoints" / "tiny_b2q_hand.pt"
    if not base.exists():
        run([sys.executable, "-m", "neuroglyph_train.train", "--task", "hand", "--epochs", "5", "--data", str(ROOT / "data/processed")])

    r = run(
        [
            sys.executable,
            "-m",
            "neuroglyph_train.finetune_subject",
            "--base-checkpoint",
            str(base),
            "--data",
            str(proc),
            "--task",
            "hand",
            "--epochs",
            "8",
            "--freeze-encoder",
            "--output",
            str(ROOT / "checkpoints" / "gear_test"),
        ]
    )
    if r.returncode != 0:
        print("FAIL: finetune")
        return 1

    ft = ROOT / "checkpoints" / "gear_test" / "finetune_hand.pt"
    if not ft.exists():
        print("FAIL: no finetune checkpoint")
        return 1

    r = run([sys.executable, "-m", "neuroglyph_train.evaluate", "--checkpoint", str(ft)])
    if r.returncode != 0:
        print("FAIL: evaluate")
        return 1

    import torch

    blob = torch.load(ft, weights_only=True)
    report = {
        "device_train": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "cuda_available": torch.cuda.is_available(),
        "finetune_ckpt": str(ft),
        "val_acc": blob.get("val_acc"),
        "finetuned": blob.get("finetuned", True),
    }
    print("PASS gear fine-tune pipeline")
    print(json.dumps(report, indent=2))
    (ROOT / "data" / "gear_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())