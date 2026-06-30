"""List partner_generate input schema from Comfy Cloud MCP."""
import asyncio
import json
import os
from pathlib import Path

MCP_URL = "https://cloud.comfy.org/mcp"


def key():
    env = Path(os.environ["USERPROFILE"]) / "AppData/Local/hermes/.env"
    for line in env.read_text().splitlines():
        if line.startswith("COMFY_CLOUD_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no key")


async def main():
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(MCP_URL, headers={"X-API-Key": key()}) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = await s.list_tools()
            for t in tools.tools:
                if t.name == "partner_generate":
                    print(json.dumps(t.model_dump(), indent=2, default=str))
                    return
            print("not found", [x.name for x in tools.tools])


asyncio.run(main())