"""Clock sync helpers between host keyboard time and Cortex headset clock."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClockSync:
    """Linear map: host_unix_ms ≈ scale * cortex_counter_ms + offset_ms."""

    offset_ms: float = 0.0
    scale: float = 1.0

    def host_to_cortex(self, host_unix_ms: float) -> float:
        return self.scale * host_unix_ms + self.offset_ms

    def cortex_to_host(self, cortex_ms: float) -> float:
        return (cortex_ms - self.offset_ms) / self.scale


def estimate_offset_from_markers(
    host_markers_ms: list[float],
    cortex_markers_ms: list[float],
) -> ClockSync:
    """Least-squares offset when scales match (MVP)."""
    if not host_markers_ms or not cortex_markers_ms:
        return ClockSync()
    n = min(len(host_markers_ms), len(cortex_markers_ms))
    diffs = [cortex_markers_ms[i] - host_markers_ms[i] for i in range(n)]
    offset = sum(diffs) / n
    return ClockSync(offset_ms=offset, scale=1.0)