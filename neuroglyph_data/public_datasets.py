"""Import public EEG datasets into NeuroGlyph epoch format (14×25 @ 50 Hz)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuroglyph_data.schema import EPOC_X_CHANNELS

# Insight 5ch → index in EPOC 14ch (pad zeros for missing sites)
INSIGHT_TO_EPOC = {
    "AF3": "AF3",
    "AF4": "AF4",
    "T7": "T7",
    "T8": "T8",
    "Pz": "P7",  # approximate map for motor studies
}


def map_channels_to_epoc(
    data: np.ndarray, source_names: list[str], n_time: int = 25
) -> np.ndarray:
    """data shape (n_trials, n_src_ch, n_time) or (n_src_ch, n_time) -> (..., 14, n_time)."""
    single = data.ndim == 2
    if single:
        data = data[np.newaxis, ...]
    n_trials, _, t = data.shape
    out = np.zeros((n_trials, 14, n_time), dtype=np.float32)
    name_to_idx = {c: i for i, c in enumerate(EPOC_X_CHANNELS)}
    src_idx = {n: i for i, n in enumerate(source_names)}
    for src_name, epoc_name in INSIGHT_TO_EPOC.items():
        if src_name not in src_idx or epoc_name not in name_to_idx:
            continue
        si = src_idx[src_name]
        ei = name_to_idx[epoc_name]
        trial = data[:, si, :]
        if t != n_time:
            # linear resample per trial
            x_old = np.linspace(0, 1, t)
            x_new = np.linspace(0, 1, n_time)
            resampled = np.stack(
                [np.interp(x_new, x_old, trial[i]) for i in range(n_trials)], axis=0
            )
            out[:, ei, :] = resampled
        else:
            out[:, ei, :] = trial
    if single:
        return out[0]
    return out


def write_processed_pt(
    out_dir: Path,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    source: str,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.int64))
    fname = f"public_{task}.pt"
    path = out_dir / fname
    torch.save({"X": X_t, "y": y_t, "task": task, "source": source}, path)
    manifest = {
        "task": task,
        "source": source,
        "files": [fname],
        "n_epochs": int(X_t.shape[0]),
        "shape": list(X_t.shape),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_kaggle_hands_csv(csv_path: Path, max_per_class: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Kaggle 'Brain wave data from hands movement of EEG' (Emotiv Insight, 5ch).
    Expects columns including channel values and a movement/label column.
    """
    import csv

    rows: list[dict] = []
    with csv_path.open(encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("empty CSV")

    # Heuristic: find label column
    label_keys = [k for k in rows[0] if "class" in k.lower() or "label" in k.lower() or "movement" in k.lower()]
    ch_keys = [k for k in rows[0] if k.upper() in ("AF3", "AF4", "T7", "T8", "PZ", "Pz")]
    if not ch_keys:
        ch_keys = [k for k in rows[0].keys() if k not in label_keys][:5]

    label_key = label_keys[0] if label_keys else None
    buckets: dict[int, list[np.ndarray]] = {0: [], 1: []}
    window = 25

    for row in rows:
        try:
            vals = [float(row[k]) for k in ch_keys[:5]]
        except (KeyError, ValueError):
            continue
        if label_key:
            raw = row[label_key].strip().lower()
            if raw in ("0", "left", "l"):
                lab = 0
            elif raw in ("1", "right", "r"):
                lab = 1
            else:
                try:
                    lab = int(float(raw)) % 2
                except ValueError:
                    continue
        else:
            lab = len(buckets[0]) % 2

        if len(buckets[lab]) >= max_per_class:
            continue
        # single-sample epoch: repeat window from scalar features (weak baseline)
        arr = np.tile(np.array(vals, dtype=np.float32)[:, None], (1, window))
        mapped = map_channels_to_epoc(arr, [c.upper() for c in ch_keys[:5]], n_time=window)
        buckets[lab].append(mapped)

    X_list, y_list = [], []
    for lab, arrs in buckets.items():
        for a in arrs:
            X_list.append(a)
            y_list.append(lab)
    if not X_list:
        raise ValueError("no samples parsed — check CSV columns")
    return np.stack(X_list), np.array(y_list, dtype=np.int64)