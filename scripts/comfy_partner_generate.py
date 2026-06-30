#!/usr/bin/env python3
"""One-shot Comfy Cloud partner_generate (gpt-image-2) via MCP SDK."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# MCP + pywin32 are installed in the Hermes venv; neuroglyph's uv-based interpreter
# cannot load pywintypes from that site-packages. Re-exec with Hermes Python when needed.
_HERMES_PYTHON = (
    Path(os.environ.get("USERPROFILE", ""))
    / "AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe"
)


def _maybe_reexec_with_hermes_python() -> None:
    if os.environ.get("NEUROGLYPH_COMFY_USE_CURRENT_PYTHON") == "1":
        return
    if not _HERMES_PYTHON.is_file():
        return
    try:
        import pywintypes  # noqa: F401
    except ModuleNotFoundError:
        import subprocess

        env = {**os.environ, "NEUROGLYPH_COMFY_USE_CURRENT_PYTHON": "1"}
        print(f"Re-launching with Hermes Python: {_HERMES_PYTHON}", flush=True)
        raise SystemExit(
            subprocess.call([str(_HERMES_PYTHON), *sys.argv], env=env)
        )


_maybe_reexec_with_hermes_python()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD = ROOT / "assets" / "brand" / "comfy-partner_generate-brain-smoke.json"
FALLBACK_PAYLOAD = ROOT / "assets" / "brand" / "comfy-partner_generate-neuroglyph-hero.json"
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
    payload_path = args.payload
    if not payload_path.is_file() and FALLBACK_PAYLOAD.is_file():
        payload_path = FALLBACK_PAYLOAD
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
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