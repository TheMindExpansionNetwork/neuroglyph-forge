#!/usr/bin/env python3
"""Generate NeuroGlyph Forge brand + brainmap images (OpenAI gpt-image-2 / fallback)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "brand"
OUT.mkdir(parents=True, exist_ok=True)

EPOC_CHANNELS = "AF3 F7 F3 FC5 T7 P7 O1 O2 P8 T8 FC6 F4 F8 AF4"

LOGO_PROMPT = (
    "App icon logo for NeuroGlyph Forge: stylized human brain merged with "
    "golden ancient glyph symbol, thin cyan EEG waveform ring, dark navy background, "
    "minimal flat vector style, high contrast, no text, centered, professional tech brand"
)

BANNER_PROMPT = (
    "Wide promotional banner for NeuroGlyph Forge BCI platform: EMOTIV EEG headset, "
    "typing keyboard ghost keys, neural pathways flowing into Unreal Engine hologram city, "
    "teal and gold color grade, cinematic sci-fi, no readable small text"
)

BRAINMAP_PROMPT = (
    "Technical infographic brain map diagram, dark navy background, titled visually as "
    "NeuroGlyph Forge system architecture without long readable paragraphs. "
    "Five horizontal layers left to right with glowing arrows: "
    "1 SENSE: EMOTIV EPOC X headset 14 EEG channels "
    f"({EPOC_CHANNELS}) Cortex WebSocket 256Hz plus keyboard keypress timestamps. "
    "2 SYNC: CSV EEG and JSON metadata aligned to 500ms epochs tensor 14 by 25 at 50Hz. "
    "3 LEARN: TinyB2Q and EEGNet training pipeline hand zone intent char29 ladder. "
    "4 REASON: Hermes AI agent receives PredictionEvent confidence only never raw EEG, "
    "MindBot synergetic cognition bus branch. "
    "5 ACT: Unreal Engine MCP blueprint events MoveLeft SpawnDebugActor. "
    "Side branch: Comfy Cloud MCP for generative art. "
    "Include small top-down scalp diagram with 14 electrode dots in realistic EPOC layout. "
    "Style: clean sci-fi HUD, teal cyan and gold accents, isometric subtle depth, "
    "poster quality for GitHub documentation, legible short layer labels only"
)

SOCIAL_PREVIEW_PROMPT = (
    "GitHub repository social preview card 1200x630: NeuroGlyph Forge, "
    "EEG brainmap infographic left third, bold title area NeuroGlyph Forge, "
    "tagline EPOC X to Hermes to Unreal, dark gradient teal purple, "
    "minimal text large typography, professional open source BCI project"
)

IMAGE_MODELS = ("gpt-image-2", "gpt-image-1.5", "gpt-image-1")


def _openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"')
    if key:
        return key
    env_path = Path.home() / "AppData/Local/hermes/.env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def generate_openai(path: Path, prompt: str, size: str = "1024x1024") -> tuple[bool, str]:
    key = _openai_key()
    if not key:
        print("OPENAI_API_KEY not set", file=sys.stderr)
        return False, ""

    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    last_err = ""
    for model in IMAGE_MODELS:
        body = json.dumps(
            {"model": model, "prompt": prompt, "n": 1, "size": size}
        ).encode("utf-8")
        req = Request(
            "https://api.openai.com/v1/images/generations",
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=240) as resp:
                data = json.loads(resp.read().decode())
        except HTTPError as e:
            last_err = f"{model}: HTTP {e.code} {e.read().decode()[:400]}"
            print(last_err, file=sys.stderr)
            continue
        except Exception as e:
            last_err = f"{model}: {e}"
            print(last_err, file=sys.stderr)
            continue

        item = (data.get("data") or [{}])[0]
        if item.get("b64_json"):
            path.write_bytes(base64.b64decode(item["b64_json"]))
            print(f"wrote {path} (model={model})")
            return True, model
        if item.get("url"):
            import urllib.request

            urllib.request.urlretrieve(item["url"], path)
            print(f"wrote {path} from url (model={model})")
            return True, model
        last_err = f"{model}: unexpected {data}"
    return False, last_err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="logo + banner + brainmap + social")
    ap.add_argument("--brainmap", action="store_true")
    ap.add_argument("--social", action="store_true")
    args = ap.parse_args()
    do_all = args.all or not (args.brainmap or args.social)

    results: dict[str, str] = {}
    ok = 0

    jobs: list[tuple[str, str, str]] = []
    if do_all:
        jobs.extend(
            [
                ("logo-gpt-image-2.png", LOGO_PROMPT, "1024x1024"),
                ("banner-gpt-image-2.png", BANNER_PROMPT, "1536x1024"),
                ("brainmap-gpt-image-2.png", BRAINMAP_PROMPT, "1536x1024"),
                ("social-preview-gpt-image-2.png", SOCIAL_PREVIEW_PROMPT, "1536x1024"),
            ]
        )
    else:
        if args.brainmap:
            jobs.append(("brainmap-gpt-image-2.png", BRAINMAP_PROMPT, "1536x1024"))
        if args.social:
            jobs.append(("social-preview-gpt-image-2.png", SOCIAL_PREVIEW_PROMPT, "1536x1024"))

    for fname, prompt, size in jobs:
        success, model = generate_openai(OUT / fname, prompt, size)
        if success:
            ok += 1
            results[fname] = model

    meta = {
        "project": "NeuroGlyph Forge",
        "models_tried": list(IMAGE_MODELS),
        "generated": results,
        "prompts": {
            "logo": LOGO_PROMPT,
            "banner": BANNER_PROMPT,
            "brainmap": BRAINMAP_PROMPT,
            "social_preview": SOCIAL_PREVIEW_PROMPT,
        },
        "files": sorted(p.name for p in OUT.glob("*.png")),
    }
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())