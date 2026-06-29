"""Cortex client mock mode and bootstrap contract."""

from __future__ import annotations

import asyncio

from neuroglyph_recorder.cortex_client import CortexClient, CortexConfig


def test_cortex_mock_bootstrap():
    async def _run():
        c = CortexClient(CortexConfig(mock=True))
        await c.connect()
        sess = await c.bootstrap()
        assert sess.cortex_token == "mock-token"
        n = 0
        async for _ in c.stream_eeg_mock(n_samples=5):
            n += 1
        assert n == 5
        await c.close()

    asyncio.run(_run())


def test_cortex_mock_authorize_chain():
    async def _run():
        c = CortexClient(CortexConfig(mock=True, client_id="x", client_secret="y"))
        await c.connect()
        assert await c.request_access() is True
        token = await c.authorize()
        assert token == "mock-token"
        await c.close()

    asyncio.run(_run())