"""Minimal partner_generate test — short prompt, poll job."""
import asyncio
import json
import os
import time
from pathlib import Path

MCP_URL = "https://cloud.comfy.org/mcp"
ROOT = Path(__file__).resolve().parents[1]


def key():
    for line in (Path(os.environ["USERPROFILE"]) / "AppData/Local/hermes/.env").read_text().splitlines():
        if line.startswith("COMFY_CLOUD_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no key")


async def call(session, name, args):
    r = await session.call_tool(name, args)
    texts = [b.text for b in r.content if hasattr(b, "text")]
    return "\n".join(texts), getattr(r, "structuredContent", None)


async def main():
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    payload = {
        "type": "image",
        "model": "openai/images-generations",
        "prompt": "Abstract teal neural waves, 14 EEG channels, dark background, no text.",
        "aspect_ratio": "1:1",
        "client_os": "windows",
        "params": {"model": "gpt-image-2", "quality": "low", "size": "1024x1024"},
        "description": "neuroglyph-smoke-test",
    }
    headers = {"X-API-Key": key()}
    async with streamablehttp_client(MCP_URL, headers=headers, timeout=120) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            text, structured = await call(s, "partner_generate", payload)
            print("PARTNER_RESPONSE", text[:3000])
            if structured:
                print("STRUCTURED", json.dumps(structured, indent=2)[:2000])
            # try parse prompt_id
            blob = text
            if "prompt_id" in blob or "submitted" in blob:
                import re

                m = re.search(r'prompt_id["\s:]+([a-f0-9-]+)', blob, re.I)
                if m:
                    pid = m.group(1)
                    for i in range(40):
                        st, _ = await call(s, "get_job_status", {"prompt_id": pid})
                        print(f"poll {i}", st[:200])
                        if "completed" in st.lower() or "success" in st.lower():
                            out, _ = await call(s, "get_output", {"prompt_id": pid})
                            print("OUTPUT", out[:3000])
                            break
                        if "failed" in st.lower() or "error" in st.lower():
                            break
                        await asyncio.sleep(15)


asyncio.run(main())