"""Optional: generate one promo image via Comfy Cloud MCP (requires Hermes-configured key)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "brand"


def _api_key() -> str:
    k = os.environ.get("COMFY_CLOUD_API_KEY", "").strip()
    if k:
        return k
    env = Path.home() / "AppData/Local/hermes/.env"
    if env.exists():
        for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("COMFY_CLOUD_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    cfg = Path.home() / "AppData/Local/hermes/config.yaml"
    if cfg.exists():
        import yaml

        data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        hdr = (data.get("mcp_servers") or {}).get("comfy-cloud", {}).get("headers") or {}
        return str(hdr.get("X-API-Key", ""))
    return ""


async def main() -> int:
    key = _api_key()
    if not key:
        print("No Comfy API key — set COMFY_CLOUD_API_KEY or Hermes mcp_servers.comfy-cloud", file=sys.stderr)
        return 1
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        print("pip install mcp", file=sys.stderr)
        return 1

    url = "https://cloud.comfy.org/mcp"
    headers = {"X-API-Key": key}
    prompt = (
        "Square logo NeuroGlyph Forge, brain + golden rune, cyan EEG ring, dark background, vector icon"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "partner_generate",
                {
                    "prompt": prompt,
                    "media_type": "image",
                },
            )
    text = ""
    for block in result.content:
        if hasattr(block, "text"):
            text += block.text
    out = OUT / "logo-comfy-partner.json"
    out.write_text(text or json.dumps({"raw": str(result)}), encoding="utf-8")
    print(f"Comfy partner_generate response saved to {out}")
    print(text[:2000] if text else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))