"""Task heads and label vocabularies for progressive decoding."""

from __future__ import annotations

LEFT_HAND_KEYS = set("qwertasdfgzxcvb")
RIGHT_HAND_KEYS = set("yuiophjklnm")

ZONE_MAP = {
    "left_top": set("qwert"),
    "left_home": set("asdfg"),
    "left_bottom": set("zxcvb"),
    "right_top": set("yuiop"),
    "right_home": set("hjkl"),
    "right_bottom": set("nm"),
    "space": set(" "),
}

INTENT_COMMANDS = [
    "left",
    "right",
    "select",
    "cancel",
    "next",
    "previous",
    "spawn",
    "delete",
]

CHAR29_CLASSES = list("abcdefghijklmnopqrstuvwxyz") + ["space", "number", "special"]


def hand_label(key: str) -> str | None:
    k = key.lower()
    if len(k) != 1:
        return None
    if k in LEFT_HAND_KEYS:
        return "left"
    if k in RIGHT_HAND_KEYS:
        return "right"
    return None


def zone_label(key: str) -> str:
    k = key.lower()
    for zone, keys in ZONE_MAP.items():
        if k in keys:
            return zone
    if k.isdigit():
        return "other"
    return "other"


def char29_label(key: str) -> str:
    k = key.lower()
    if k == " ":
        return "space"
    if len(k) == 1 and k.isalpha():
        return k
    if k.isdigit():
        return "number"
    return "special"


TASK_HEADS: dict[str, dict] = {
    "hand": {"classes": ["left", "right"], "n_classes": 2, "label_fn": hand_label},
    "zone": {
        "classes": [
            "left_top",
            "left_home",
            "left_bottom",
            "right_top",
            "right_home",
            "right_bottom",
            "space",
            "other",
        ],
        "n_classes": 8,
        "label_fn": zone_label,
    },
    "char29": {"classes": CHAR29_CLASSES, "n_classes": len(CHAR29_CLASSES), "label_fn": char29_label},
    "intent": {"classes": INTENT_COMMANDS, "n_classes": len(INTENT_COMMANDS), "label_fn": None},
}