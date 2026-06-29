"""Keypress-aligned epoch extraction (numpy + optional MNE)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from neuroglyph_core import MODEL_SAMPLE_RATE, N_CHANNELS, N_TIME_SAMPLES
from neuroglyph_data.convert_emotiv import load_emotiv_csv
from neuroglyph_data.schema import SessionMeta
from neuroglyph_models.heads import TASK_HEADS


def _resample_window(window: np.ndarray, src_hz: float, dst_hz: float = MODEL_SAMPLE_RATE) -> np.ndarray:
    """Linear resample each channel along time axis."""
    n_src = window.shape[1]
    n_dst = max(1, int(round(n_src * dst_hz / src_hz)))
    x_src = np.linspace(0, 1, n_src)
    x_dst = np.linspace(0, 1, n_dst)
    out = np.zeros((window.shape[0], n_dst), dtype=np.float32)
    for c in range(window.shape[0]):
        out[c] = np.interp(x_dst, x_src, window[c])
    if n_dst != N_TIME_SAMPLES:
        # crop/pad to fixed model width
        if n_dst > N_TIME_SAMPLES:
            start = (n_dst - N_TIME_SAMPLES) // 2
            out = out[:, start : start + N_TIME_SAMPLES]
        else:
            pad = N_TIME_SAMPLES - n_dst
            out = np.pad(out, ((0, 0), (0, pad)), mode="edge")
    return out


def epochs_from_session(
    eeg: np.ndarray,
    eeg_ts_ms: np.ndarray,
    events: list,
    *,
    sample_rate_hz: float,
    task: str,
    t_start: float = -0.2,
    t_end: float = 0.3,
    clock_offset_ms: float = 0.0,
    label_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    head = TASK_HEADS[task]
    classes = head["classes"]
    label_fn = label_fn or head["label_fn"]

    X_list = []
    y_list = []
    for ev in events:
        key = ev.key if hasattr(ev, "key") else ev["key"]
        if label_fn is None:
            continue
        label = label_fn(key)
        if label is None:
            continue
        if label not in classes:
            continue
        t_ms = (ev.timestamp_unix_ms if hasattr(ev, "timestamp_unix_ms") else ev["timestamp_unix_ms"]) + clock_offset_ms
        t0 = t_ms + t_start * 1000.0
        t1 = t_ms + t_end * 1000.0
        i0 = int(np.searchsorted(eeg_ts_ms, t0))
        i1 = int(np.searchsorted(eeg_ts_ms, t1))
        if i1 - i0 < 4:
            continue
        window = eeg[i0:i1].T  # channels x time
        window = _resample_window(window, src_hz=sample_rate_hz)
        # robust scale per epoch
        med = np.median(window)
        mad = np.median(np.abs(window - med)) + 1e-6
        window = np.clip((window - med) / mad, -5.0, 5.0)
        X_list.append(window.astype(np.float32))
        y_list.append(classes.index(label))

    if not X_list:
        raise RuntimeError("no valid epochs from session")
    return torch.tensor(np.stack(X_list)), torch.tensor(y_list, dtype=torch.long)


def build_processed_from_raw(
    raw_dir: Path,
    out_dir: Path,
    task: str = "hand",
    eeg_glob: str = "*eeg*.csv",
    clock_offset_ms: float = 0.0,
) -> Path:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_files = sorted(raw_dir.glob("*_meta.json"))
    if not meta_files:
        raise FileNotFoundError(f"no *_meta.json in {raw_dir}")

    all_X = []
    all_y = []
    for meta_path in meta_files:
        meta = SessionMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
        eeg_candidates = sorted(raw_dir.glob(eeg_glob))
        if not eeg_candidates:
            eeg_candidates = sorted(raw_dir.glob(f"{meta.session_id}_eeg.csv"))
        if not eeg_candidates:
            raise FileNotFoundError(f"no EEG CSV matching {eeg_glob} in {raw_dir}")
        eeg_path = eeg_candidates[0]
        eeg, ts = load_emotiv_csv(eeg_path, channels=meta.channels)
        offset = clock_offset_ms if clock_offset_ms != 0.0 else float(meta.clock_offset_ms)
        X, y = epochs_from_session(
            eeg,
            ts,
            meta.events,
            sample_rate_hz=float(meta.sample_rate),
            task=task,
            clock_offset_ms=offset,
        )
        all_X.append(X)
        all_y.append(y)

    X_cat = torch.cat(all_X, dim=0)
    y_cat = torch.cat(all_y, dim=0)
    out_path = out_dir / f"processed_{task}.pt"
    torch.save({"X": X_cat, "y": y_cat, "task": task}, out_path)
    (out_dir / "manifest.json").write_text(
        json.dumps({"files": [out_path.name], "task": task, "n_epochs": int(X_cat.shape[0])}, indent=2),
        encoding="utf-8",
    )
    return out_path