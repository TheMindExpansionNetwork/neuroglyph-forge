#!/usr/bin/env python3
"""One-shot Comfy Cloud partner_generate (gpt-image-2) via MCP SDK."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD = ROOT / "assets" / "brand" / "comfy-partner_generate-neuroglyph-hero.json"
MCP_URL = "https://cloud.comfy.org/mcp"


def load_api_key() -> str:
    key = os.environ.get("COMFY_CLOUD_API_KEY", "").strip()
    if key:
        return key
    hermes_env = Path(os.environ.get("USERPROFILE", "")) / "AppData" / "Local" / "hermes" / ".env"
    if hermes_env.is_file():
        for line in hermes_env.read_text(encoding="utf-8").splitlines():
            if line.startswith("COMFY_CLOUD_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("COMFY_CLOUD_API_KEY not set")


async def run(payload: dict) -> dict:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    headers = {"X-API-Key": load_api_key()}
    async with streamablehttp_client(MCP_URL, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("partner_generate", payload)
            out: dict = {"content": []}
            for block in result.content:
                if hasattr(block, "text"):
                    out["content"].append(block.text)
                else:
                    out["content"].append(str(block))
            return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "comfy_image2_result.json")
    args = p.parse_args()
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    print("partner_generate gpt-image-2 — may take 15–20+ min…")
    print("prompt:", payload.get("prompt", "")[:120], "…")
    result = asyncio.run(run(payload))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("saved", args.out)
    print(json.dumps(result, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())