"""
Stdio MCP server for Hermes Agent.

Add to ~/AppData/Local/hermes/config.yaml:

  mcp_servers:
    neuroglyph:
      command: \"D:/neuroglyph-forge/.venv/Scripts/python.exe\"
      args: [\"-m\", \"neuroglyph_agent.hermes_stdio_mcp\"]
      env:
        PYTHONPATH: \"D:/neuroglyph-forge\"
"""

from __future__ import annotations

import json
import sys

from neuroglyph_agent.mcp_server import TOOL_SCHEMAS, call_tool


def main() -> int:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("pip install mcp", file=sys.stderr)
        return 1

    mcp = FastMCP("neuroglyph-forge")

    for spec in TOOL_SCHEMAS:
        name = spec["name"]

        def _make_handler(tool_name: str):
            def handler(**kwargs):
                return json.dumps(call_tool(tool_name, kwargs), indent=2)

            return handler

        mcp.tool(name=name)(_make_handler(name))

    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())