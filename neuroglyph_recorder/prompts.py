"""Guided typing prompts for NeuroGlyph-EPOC-Typing-v1 collection."""

from __future__ import annotations

PROMPTS_SHORT = [
    "the quick brown fox",
    "mind bot types with eeg",
    "neuro glyph forge alpha",
    "left hand right hand test",
    "spawn actor in unreal",
]

PROMPTS_UNREAL_INTENT = [
    "left left select cancel",
    "right right next previous",
    "spawn delete select cancel",
]


def next_prompt(block: int, index: int) -> str:
    pool = PROMPTS_SHORT if block == 0 else PROMPTS_UNREAL_INTENT
    return pool[index % len(pool)]