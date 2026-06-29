"""Optional MNE-based filtering (used when mne is installed)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mne


def filter_eeg_array(
    data: np.ndarray,
    sfreq: float,
    l_freq: float = 0.1,
    h_freq: float = 20.0,
    notch: float | None = 60.0,
) -> np.ndarray:
    try:
        import mne
    except ImportError as e:
        raise RuntimeError("Install preprocess extra: pip install -e '.[preprocess]'") from e

    info = mne.create_info(ch_names=[f"ch{i}" for i in range(data.shape[1])], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data.T, info, verbose=False)
    raw.filter(l_freq, h_freq, verbose=False)
    if notch:
        raw.notch_filter(notch, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    return raw.get_data().T.astype(np.float32)