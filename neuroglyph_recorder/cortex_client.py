"""EMOTIV Cortex JSON-RPC 2.0 client (live + mock)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from neuroglyph_data.schema import EPOC_X_CHANNELS

logger = logging.getLogger(__name__)

CORTEX_URL = os.environ.get("NEUROGLYPH_CORTEX_URL", "wss://localhost:6868")
EEG_STREAM_CHANNELS = list(EPOC_X_CHANNELS)


@dataclass
class CortexConfig:
    client_id: str = field(default_factory=lambda: os.environ.get("EMOTIV_CLIENT_ID", ""))
    client_secret: str = field(default_factory=lambda: os.environ.get("EMOTIV_CLIENT_SECRET", ""))
    headset_id: str = field(default_factory=lambda: os.environ.get("NEUROGLYPH_HEADSET_ID", ""))
    mock: bool = True
    license: str = ""
    debit: int = 1


@dataclass
class CortexSession:
    cortex_token: str
    session_id: str
    headset_id: str


class CortexClient:
    """Cortex WebSocket client with a single reader task (no recv races)."""

    def __init__(self, config: CortexConfig | None = None):
        self.config = config or CortexConfig()
        self._ws = None
        self._id = 0
        self._session: CortexSession | None = None
        self._recv_task: asyncio.Task | None = None
        self._eeg_queue: asyncio.Queue[list[float] | None] = asyncio.Queue(maxsize=512)
        self._pending: dict[int, asyncio.Future] = {}

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def connect(self) -> None:
        if self.config.mock:
            return
        import websockets

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        self._ws = await websockets.connect(CORTEX_URL, ssl=ctx, max_size=2**22)
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        if self._ws:
            await self._ws.close()
        self._ws = None
        try:
            self._eeg_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    def _dispatch(self, msg: dict) -> None:
        if "eeg" in msg:
            row = msg["eeg"]
            if isinstance(row, list) and len(row) >= 2 + len(EEG_STREAM_CHANNELS):
                values = [float(x) for x in row[2 : 2 + len(EEG_STREAM_CHANNELS)]]
                try:
                    self._eeg_queue.put_nowait(values)
                except asyncio.QueueFull:
                    pass
            return
        rid = msg.get("id")
        if rid is not None and rid in self._pending:
            fut = self._pending.pop(rid)
            if not fut.done():
                fut.set_result(msg)

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        while True:
            raw = await self._ws.recv()
            self._dispatch(json.loads(raw))

    async def request(self, method: str, params: dict | None = None, timeout: float = 30.0) -> dict:
        if self.config.mock:
            return self._mock_response(method, params or {})
        assert self._ws is not None
        rid = self._next_id()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[rid] = fut
        payload = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        await self._ws.send(json.dumps(payload))
        try:
            msg = await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            raise TimeoutError(f"Cortex {method} timed out") from None
        if "error" in msg:
            raise RuntimeError(f"Cortex {method}: {msg['error']}")
        return msg.get("result", {})

    def _mock_response(self, method: str, params: dict) -> dict:
        if method == "authorize":
            return {"cortexToken": "mock-token"}
        if method == "createSession":
            return {"id": "mock-session"}
        if method == "requestAccess":
            return {"accessGranted": True}
        if method == "queryHeadsets":
            return [{"id": "MOCK-EPOC", "status": "connected", "connectedBy": "mock"}]
        if method == "injectMarker":
            return {"marker": params}
        return {"mock": True, "method": method}

    async def request_access(self) -> bool:
        if not self.config.client_id or not self.config.client_secret:
            raise ValueError("Set EMOTIV_CLIENT_ID and EMOTIV_CLIENT_SECRET (see docs/CORTEX_SETUP.md)")
        res = await self.request(
            "requestAccess",
            {"clientId": self.config.client_id, "clientSecret": self.config.client_secret},
        )
        return bool(res.get("accessGranted"))

    async def authorize(self) -> str:
        params: dict[str, Any] = {
            "clientId": self.config.client_id,
            "clientSecret": self.config.client_secret,
        }
        if self.config.license:
            params["license"] = self.config.license
        if self.config.debit:
            params["debit"] = self.config.debit
        res = await self.request("authorize", params)
        token = res.get("cortexToken")
        if not token:
            raise RuntimeError(f"authorize failed: {res}")
        return token

    async def query_headsets(self) -> list[dict]:
        res = await self.request("queryHeadsets", {})
        return res if isinstance(res, list) else []

    async def create_session(self, cortex_token: str, headset_id: str) -> str:
        res = await self.request(
            "createSession",
            {"cortexToken": cortex_token, "headset": headset_id, "status": "active"},
        )
        sid = res.get("id")
        if not sid:
            raise RuntimeError(f"createSession failed: {res}")
        return sid

    async def subscribe_eeg(self, cortex_token: str, session_id: str) -> None:
        await self.request(
            "subscribe",
            {"cortexToken": cortex_token, "session": session_id, "streams": ["eeg"]},
        )

    async def inject_marker(self, label: str, port: str = "keyboard") -> dict:
        if self.config.mock:
            return await self.request("injectMarker", {"label": label, "value": label, "port": port})
        if not self._session:
            return {}
        return await self.request(
            "injectMarker",
            {
                "cortexToken": self._session.cortex_token,
                "session": self._session.session_id,
                "label": label,
                "value": label,
                "port": port,
                "isSync": True,
            },
        )

    async def bootstrap(self) -> CortexSession:
        if self.config.mock:
            self._session = CortexSession("mock-token", "mock-session", "MOCK-EPOC")
            return self._session

        await self.request_access()
        token = await self.authorize()
        headsets = await self.query_headsets()
        hid = self.config.headset_id
        if not hid:
            connected = [h for h in headsets if h.get("status") == "connected"]
            if not connected:
                raise RuntimeError(f"No connected headset. queryHeadsets={headsets}")
            hid = connected[0]["id"]
        session_id = await self.create_session(token, hid)
        await self.subscribe_eeg(token, session_id)
        self._session = CortexSession(token, session_id, hid)
        logger.info("Cortex session %s headset %s", session_id, hid)
        return self._session

    async def stream_eeg(self) -> AsyncIterator[tuple[list[float], float | None]]:
        if self.config.mock:
            i = 0
            while True:
                yield [0.01 * (i % 5)] * 14, float(i)
                i += 1
                await asyncio.sleep(1.0 / 256.0)
            return

        while True:
            sample = await self._eeg_queue.get()
            if sample is None:
                break
            yield sample, None

    async def stream_eeg_mock(self, n_samples: int = 10) -> AsyncIterator[list[float]]:
        count = 0
        async for sample, _ in self.stream_eeg():
            yield sample
            count += 1
            if count >= n_samples:
                break