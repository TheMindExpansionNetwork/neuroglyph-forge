"""Write Cortex EEG stream to CSV compatible with make_epochs."""

from __future__ import annotations

import csv
from pathlib import Path

from neuroglyph_data.schema import EPOC_X_CHANNELS


class EegCsvWriter:
    def __init__(self, path: Path, sample_rate: int = 256):
        self.path = path
        self.sample_rate = sample_rate
        self._file = None
        self._writer = None
        self.n_rows = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["Timestamp", *EPOC_X_CHANNELS])
        return self

    def __exit__(self, *args):
        if self._file:
            self._file.close()

    def write_row(self, counter_or_ms: float, channels: list[float]) -> None:
        assert self._writer is not None
        # Store seconds for loader heuristic (t < 1e6 → ×1000 to ms)
        t_sec = counter_or_ms / 1000.0 if counter_or_ms > 1e6 else counter_or_ms / float(self.sample_rate)
        if self.n_rows == 0 and counter_or_ms > 1e4:
            t_sec = counter_or_ms / 1000.0
        self._writer.writerow([t_sec, *channels])
        self.n_rows += 1