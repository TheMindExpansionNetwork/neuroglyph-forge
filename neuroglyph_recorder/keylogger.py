"""Timestamped keyboard logger for typing sessions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable


@dataclass
class KeyPress:
    key: str
    timestamp_unix_ms: int


class KeyLogger:
    def __init__(self, on_key: Callable[[KeyPress], None] | None = None):
        self.on_key = on_key
        self.events: list[KeyPress] = []
        self._listener = None

    def _handle(self, key) -> None:
        try:
            char = key.char
        except AttributeError:
            name = getattr(key, "name", str(key))
            if name == "space":
                char = " "
            else:
                return
        if not char:
            return
        ev = KeyPress(key=char, timestamp_unix_ms=int(time.time() * 1000))
        self.events.append(ev)
        if self.on_key:
            self.on_key(ev)

    def start(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as e:
            raise RuntimeError("Install recorder extras: pip install -e '.[recorder]'") from e
        self._listener = keyboard.Listener(on_press=self._handle)
        self._listener.start()

    def stop(self) -> None:
        if self._listener:
            self._listener.stop()
            self._listener = None

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(e) for e in self.events], indent=2), encoding="utf-8")