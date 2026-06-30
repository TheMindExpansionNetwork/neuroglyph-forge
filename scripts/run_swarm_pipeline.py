#!/usr/bin/env python3
"""
Full local swarm: download sample data → train → fine-tune → test → HF upload (if logged in).

No live EPOC required; uses synthetic + gear_test raw when no sessions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], env: dict | None = None) -> int:
    print("+", " ".join(cmd), flush=True)
    e = os.environ.copy()
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=ROOT, env=e).returncode


def hf_logged_in() -> bool:
    try:
        from huggingface_hub import HfApi

        HfApi().whoami()
        return True
    except Exception:
        return False


def main() -> int:
    py = sys.executable
    report: dict = {"steps": []}

    def step(name: str, code: int):
        report["steps"].append({"name": name, "ok": code == 0})
        if code != 0:
            print(f"FAIL at {name}", flush=True)
            (ROOT / "data" / "swarm_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            return False
        return True

    # 1) Dependencies
    if run([py, "-m", "pip", "install", "-e", ".[cloud]", "-q"]) != 0:
        return 1

    # 2) Sample / training data
    py_dl = ROOT / "scripts" / "download_hf_artifacts.py"
    if py_dl.exists():
        run([py, str(py_dl)])

    if not step("synthetic_download", run([py, "scripts/download_public_datasets.py", "synthetic", "--task", "hand", "--n", "1200"])):
        return 1

    # 3) Gear-style raw epochs path (proves raw→processed pipeline)
    if run([py, "scripts/test_gear_finetune.py"]) != 0:
        report["steps"][-1]["note"] = "gear test failed but continuing if processed exists"
    else:
        report["steps"].append({"name": "gear_finetune_test", "ok": True})

    # 4) Train base model on synthetic (GPU if available)
    if not step("train_hand", run([py, "-m", "neuroglyph_train.train", "--task", "hand", "--epochs", "25", "--data", str(ROOT / "data/processed")])):
        return 1

    base = ROOT / "checkpoints" / "tiny_b2q_hand.pt"
    proc_gear = ROOT / "data" / "processed" / "gear_test"
    if proc_gear.exists() and (proc_gear / "processed_hand.pt").exists():
        if not step(
            "finetune_gear",
            run(
                [
                    py,
                    "-m",
                    "neuroglyph_train.finetune_subject",
                    "--base-checkpoint",
                    str(base),
                    "--data",
                    str(proc_gear),
                    "--task",
                    "hand",
                    "--epochs",
                    "12",
                    "--freeze-encoder",
                    "--output",
                    str(ROOT / "checkpoints"),
                ]
            ),
        ):
            return 1
        report["finetune_ckpt"] = str(ROOT / "checkpoints" / "finetune_hand.pt")
    else:
        report["finetune_ckpt"] = str(base)

    ckpt = Path(report["finetune_ckpt"])
    if not ckpt.exists():
        ckpt = base

    if not step("evaluate", run([py, "-m", "neuroglyph_train.evaluate", "--checkpoint", str(ckpt)])):
        return 1

    # Ensure main processed tensor exists for smoke
    main_pt = ROOT / "data" / "processed" / "processed_hand.pt"
    if not main_pt.exists():
        candidates = sorted((ROOT / "data" / "processed").rglob("processed_hand.pt"))
        if candidates and not main_pt.exists():
            import shutil

            shutil.copy2(candidates[0], main_pt)

    # 5) Smoke inference
    smoke = ROOT / "scripts" / "smoke_inference.py"
    if smoke.exists():
        step("smoke_inference", run([py, str(smoke), "--checkpoint", str(ckpt)]))
    else:
        report["steps"].append({"name": "smoke_inference", "ok": False, "note": "script pending"})

    # 6) Stage + push dataset (gear_test + any raw)
    raw_has = list((ROOT / "data/raw").rglob("*_meta.json"))
    if raw_has:
        run([py, "scripts/push_dataset_hf.py", "--raw", str(ROOT / "data/raw"), "--out", str(ROOT / "data/uploads/hf_dataset")])
        if hf_logged_in():
            step("push_dataset", run([py, "scripts/push_dataset_hf.py", "--raw", str(ROOT / "data/raw"), "--push"]))
        else:
            report["hf_dataset"] = "staged_only_no_token"

    push_model = ROOT / "scripts" / "push_model_hf.py"
    if push_model.exists() and hf_logged_in():
        step("push_model", run([py, str(push_model), "--checkpoint", str(ckpt), "--push"]))
    elif push_model.exists():
        run([py, str(push_model), "--checkpoint", str(ckpt)])
        report["hf_model"] = "staged_only_no_token"
    else:
        report["hf_model"] = "push_model_hf_pending"

    import torch

    report["cuda"] = torch.cuda.is_available()
    report["checkpoint"] = str(ckpt)
    if (ROOT / "data/smoke_report.json").exists():
        report["smoke"] = json.loads((ROOT / "data/smoke_report.json").read_text(encoding="utf-8"))

    report["ok"] = all(s.get("ok") for s in report["steps"] if s["name"] not in ("smoke_inference",))
    (ROOT / "data/swarm_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SWARM_REPORT", json.dumps(report, indent=2), flush=True)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())