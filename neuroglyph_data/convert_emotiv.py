"""Load EMOTIV Cortex CSV exports into aligned arrays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neuroglyph_data.schema import EPOC_X_CHANNELS


def load_emotiv_csv(path: Path, channels: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (data [n_samples, n_channels], timestamps_ms).
    Expects Cortex-style CSV with a time column (Counter, Timestamp, or first column ms).
    """
    channels = channels or EPOC_X_CHANNELS
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    time_col = None
    for candidate in ("timestamp", "counter", "time"):
        if candidate in cols:
            time_col = cols[candidate]
            break
    if time_col is None:
        time_col = df.columns[0]

    ch_cols = []
    for ch in channels:
        match = next((c for c in df.columns if c.upper() == ch.upper()), None)
        if match:
            ch_cols.append(match)
    if len(ch_cols) != len(channels):
        missing = set(channels) - {c.upper() for c in ch_cols}
        raise ValueError(f"CSV missing channels: {missing}")

    ts = df[time_col].to_numpy(dtype=np.float64)
    if ts.max() < 1e6:
        ts = ts * 1000.0  # seconds → ms heuristic
    data = df[ch_cols].to_numpy(dtype=np.float32)
    return data, ts