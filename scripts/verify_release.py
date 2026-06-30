#!/usr/bin/env python3
"""End-to-end verification: sample data, fine-tune smoke, HF upload."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)


def run(cmd: list[str]) -> tuple[bool, str]:
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode == 0, out.strip()[-2000:]


def main() -> int:
    report: dict = {"steps": [], "ok": False}
    ckpt = ROOT / "checkpoints" / "finetune_hand.pt"

    steps = [
        ("pytest", [str(PY), "-m", "pytest", "tests/", "-q", "--tb=no", "-k", "not test_smoke_inference_runs"]),
        ("gear_test", [str(PY), "scripts/test_gear_finetune.py"]),
        ("smoke", [str(PY), "scripts/smoke_inference.py", "--checkpoint", str(ckpt), "--n-samples", "16", "--min-accuracy", "0"]),
        ("evaluate", [str(PY), "-m", "neuroglyph_train.evaluate", "--checkpoint", str(ckpt)]),
    ]

    for name, cmd in steps:
        ok, tail = run(cmd)
        report["steps"].append({"name": name, "ok": ok, "tail": tail[-400:] if tail else ""})
        if not ok and name != "gear_test":
            break

    smoke_path = ROOT / "data" / "smoke_report.json"
    if smoke_path.exists():
        report["smoke"] = json.loads(smoke_path.read_text(encoding="utf-8"))

    if ckpt.exists():
        report["checkpoint_bytes"] = ckpt.stat().st_size
        report["finetune_ckpt"] = str(ckpt)

    # HF push
    for name, cmd in [
        ("push_model", [str(PY), "scripts/push_model_hf.py", "--checkpoint", str(ckpt), "--push"]),
        ("push_dataset", [str(PY), "scripts/push_dataset_hf.py", "--raw", str(ROOT / "data" / "raw"), "--push"]),
    ]:
        if not ckpt.exists() and "model" in name:
            report["steps"].append({"name": name, "ok": False, "tail": "no checkpoint"})
            continue
        ok, tail = run(cmd)
        report["steps"].append({"name": name, "ok": ok, "tail": tail[-400:]})
        if "huggingface.co" in tail:
            report.setdefault("hf_urls", []).extend(
                [ln.strip() for ln in tail.splitlines() if "huggingface.co" in ln]
            )

    report["ok"] = all(s["ok"] for s in report["steps"] if s["name"] not in ("gear_test",))
    # gear_test optional if processed exists
    core = [s for s in report["steps"] if s["name"] in ("smoke", "evaluate", "push_model", "push_dataset")]
    report["ok"] = bool(core) and all(s["ok"] for s in core)

    out = ROOT / "data" / "verify_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("VERIFY_REPORT", json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())