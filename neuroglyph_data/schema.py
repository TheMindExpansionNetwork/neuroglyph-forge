"""Session metadata and EPOC channel constants."""

from __future__ import annotations

from pydantic import BaseModel, Field

EPOC_X_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]


class KeyEvent(BaseModel):
    key: str
    timestamp_unix_ms: int
    trial_id: str | None = None
    target_sentence: str | None = None


class SessionMeta(BaseModel):
    subject_id: str
    session_id: str
    sample_rate: int = 256
    headset: str = "EMOTIV EPOC X"
    channels: list[str] = Field(default_factory=lambda: list(EPOC_X_CHANNELS))
    events: list[KeyEvent] = Field(default_factory=list)
    clock_offset_ms: float = 0.0
    cortex_session_id: str | None = None
    headset_id: str | None = None