#!/usr/bin/env python3
"""Generate NeuroGlyph Forge promo images (OpenAI gpt-image-1, optional Comfy)."""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "brand"
OUT.mkdir(parents=True, exist_ok=True)

LOGO_PROMPT = (
    "App icon logo for 'NeuroGlyph Forge': stylized human brain merged with "
    "golden ancient glyph symbol, thin cyan EEG waveform ring, dark navy background, "
    "minimal flat vector style, high contrast, no text, centered, professional tech brand"
)

BANNER_PROMPT = (
    "Wide promotional banner for NeuroGlyph Forge BCI platform: EMOTIV EEG headset, "
    "typing keyboard ghost keys, neural pathways flowing into Unreal Engine hologram city, "
    "teal and gold color grade, cinematic sci-fi, no readable small text"
)


def generate_openai(path: Path, prompt: str, size: str = "1024x1024") -> bool:
    key = os.environ.get("OPENAI_API_KEY", "").strip().strip('"')
    if not key:
        # Hermes .env
        env_path = Path.home() / "AppData/Local/hermes/.env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        print("OPENAI_API_KEY not set — skip OpenAI image gen", file=sys.stderr)
        return False

    try:
        from urllib.request import Request, urlopen
    except ImportError:
        return False

    body = json.dumps(
        {
            "model": "gpt-image-1",
            "prompt": prompt,
            "n": 1,
            "size": size,
        }
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
        with urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"OpenAI image API failed: {e}", file=sys.stderr)
        return False

    item = (data.get("data") or [{}])[0]
    if item.get("b64_json"):
        path.write_bytes(base64.b64decode(item["b64_json"]))
        print(f"wrote {path}")
        return True
    if item.get("url"):
        import urllib.request

        urllib.request.urlretrieve(item["url"], path)
        print(f"wrote {path} from url")
        return True
    print(f"unexpected response: {data}", file=sys.stderr)
    return False


def main() -> int:
    ok = 0
    if generate_openai(OUT / "logo-gpt-image-1.png", LOGO_PROMPT, "1024x1024"):
        ok += 1
    if generate_openai(OUT / "banner-gpt-image-1.png", BANNER_PROMPT, "1536x1024"):
        ok += 1
    meta = {
        "project": "NeuroGlyph Forge",
        "prompts": {"logo": LOGO_PROMPT, "banner": BANNER_PROMPT},
        "files": [p.name for p in OUT.glob("*.png")],
    }
    (OUT / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())