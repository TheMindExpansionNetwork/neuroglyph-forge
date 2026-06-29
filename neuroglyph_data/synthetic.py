"""Synthetic epoch generation for smoke tests without hardware."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuroglyph_data.schema import KeyEvent, SessionMeta
from neuroglyph_models.heads import TASK_HEADS


def synthetic_session(
    n_events: int = 200,
    n_channels: int = 14,
    n_time: int = 25,
    task: str = "hand",
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    head = TASK_HEADS[task]
    classes = head["classes"]
    label_fn = head["label_fn"]
    keys = list("qwertyuiopasdfghjklzxcvbnm ")

    X = []
    y = []
    for i in range(n_events):
        key = keys[rng.integers(0, len(keys))]
        if task == "hand" and label_fn:
            label = label_fn(key)
            if label is None:
                continue
            cls_idx = classes.index(label)
        elif label_fn:
            label = label_fn(key)
            cls_idx = classes.index(label)
        else:
            cls_idx = rng.integers(0, len(classes))

        # Class-conditional bias so a tiny model can learn something
        base = rng.standard_normal((n_channels, n_time)).astype(np.float32) * 0.1
        base += (cls_idx + 1) * 0.05
        X.append(base)
        y.append(cls_idx)

    if not X:
        raise RuntimeError("no valid synthetic events")

    return torch.tensor(np.stack(X)), torch.tensor(y, dtype=torch.long)


def write_synthetic_processed(
    out_dir: Path,
    n_events: int = 256,
    task: str = "hand",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = synthetic_session(n_events=n_events, task=task)
    path = out_dir / f"synthetic_{task}.pt"
    torch.save({"X": X, "y": y, "task": task}, path)

    meta = SessionMeta(
        subject_id="sub-synth",
        session_id="ses-synth",
        events=[KeyEvent(key="a", timestamp_unix_ms=0)],
    )
    (out_dir / "session_meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps({"files": [path.name], "task": task}, indent=2),
        encoding="utf-8",
    )
    return path