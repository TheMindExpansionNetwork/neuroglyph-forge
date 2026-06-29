"""HTTP JSON-RPC helper for Unreal Editor MCP (read-only / action planning)."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_UE_MCP_URL = os.environ.get("NEUROGLYPH_UE_MCP_URL", "http://127.0.0.1:8000/mcp")


class UnrealMCPBridge:
    def __init__(self, base_url: str = DEFAULT_UE_MCP_URL, session_id: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id

    def _post(self, payload: dict, extra_headers: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if extra_headers:
            headers.update(extra_headers)
        req = Request(self.base_url, data=json.dumps(payload).encode(), headers=headers, method="POST")
        with urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            if resp.headers.get("Mcp-Session-Id") and not self.session_id:
                self.session_id = resp.headers.get("Mcp-Session-Id")
            return json.loads(body) if body else {}

    def initialize(self) -> dict:
        return self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "neuroglyph-forge", "version": "0.1.0"},
                },
            }
        )

    async def send_prediction(self, prediction: str, confidence: float) -> dict[str, Any]:
        from neuroglyph_unreal.ue_actions import prediction_to_unreal_action

        if confidence < 0.75:
            return {"status": "ignored", "reason": "low confidence"}
        action = prediction_to_unreal_action(prediction)
        # Live call requires editor running + session; return planned action for Hermes to execute.
        try:
            self.initialize()
            reachable = True
        except Exception as exc:  # noqa: BLE001
            reachable = False
            action["probe_error"] = str(exc)
        return {"status": "planned", "unreal_reachable": reachable, "action": action}