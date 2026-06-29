"""Shared constants loaded from configs/epocx_256hz.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "epocx_256hz.yaml"

EPOCH_T_START = -0.2
EPOCH_T_END = 0.3
N_CHANNELS = 14
MODEL_SAMPLE_RATE = 50
N_TIME_SAMPLES = int((EPOCH_T_END - EPOCH_T_START) * MODEL_SAMPLE_RATE)  # 25


@lru_cache(maxsize=1)
def load_device_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)