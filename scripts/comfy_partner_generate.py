#!/usr/bin/env python3
"""Comfy Cloud partner_generate (gpt-image-2) via MCP SDK — save image to disk."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

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
        raise SystemExit(subprocess.call([str(_HERMES_PYTHON), *sys.argv], env=env))


_maybe_reexec_with_hermes_python()

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAYLOAD = ROOT / "assets/brand/comfy-partner_generate-brain-smoke.json"
FALLBACK_PAYLOAD = ROOT / "assets/brand/comfy-partner_generate-neuroglyph-hero.json"
MCP_URL = "https://cloud.comfy.org/mcp"


def load_api_key() -> str:
    key = os.environ.get("COMFY_CLOUD_API_KEY", "").strip()
    if key:
        return key
    hermes_env = Path(os.environ.get("USERPROFILE", "")) / "AppData/Local/hermes/.env"
    if hermes_env.is_file():
        for line in hermes_env.read_text(encoding="utf-8").splitlines():
            if line.startswith("COMFY_CLOUD_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("COMFY_CLOUD_API_KEY not set")


def save_from_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("data:"):
        m = re.match(r"data:image/[^;]+;base64,(.+)", url, re.DOTALL)
        if not m:
            raise ValueError("unsupported data url")
        dest.write_bytes(base64.b64decode(m.group(1)))
        return dest
    raise ValueError("https download not implemented in script; use download_command from MCP text")


async def poll_job(session, prompt_id: str, timeout_s: int) -> dict:
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = await session.call_tool("get_job_status", {"prompt_id": prompt_id})
        text = " ".join(b.text for b in r.content if hasattr(b, "text"))
        if re.search(r"completed|success", text, re.I):
            out = await session.call_tool("get_output", {"prompt_id": prompt_id})
            return {"status": "completed", "status_text": text, "output": out}
        if re.search(r"failed|error|cancel", text, re.I):
            return {"status": "failed", "status_text": text}
        await asyncio.sleep(20)
    return {"status": "timeout", "status_text": text}


async def run_generate(payload: dict, timeout_s: int, status_path: Path | None = None) -> dict:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    def write_status(phase: str, **extra: object) -> None:
        if not status_path:
            return
        status_path.write_text(
            json.dumps({"phase": phase, "ts": time.time(), **extra}, indent=2),
            encoding="utf-8",
        )

    async def heartbeat(phase: str) -> None:
        while True:
            write_status(phase, heartbeat=True)
            await asyncio.sleep(45)

    headers = {"X-API-Key": load_api_key()}
    write_status("connecting")
    async with streamablehttp_client(MCP_URL, headers=headers, timeout=timeout_s) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            write_status("initialized")
            print("Comfy MCP OK — partner_generate…", flush=True)
            hb = asyncio.create_task(heartbeat("partner_generate_in_flight"))
            try:
                result = await asyncio.wait_for(
                    session.call_tool("partner_generate", payload),
                    timeout=timeout_s,
                )
            finally:
                hb.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await hb
            write_status("partner_generate_returned")
            texts = [b.text for b in result.content if hasattr(b, "text")]
            structured = getattr(result, "structuredContent", None)
            out: dict = {"content_text": "\n".join(texts), "structured": structured}

            if structured and isinstance(structured, dict):
                if structured.get("status") == "completed" and structured.get("results"):
                    out["results"] = structured["results"]
                    return out
                pid = structured.get("prompt_id") or structured.get("job_id")
                if pid and structured.get("status") == "submitted":
                    polled = await poll_job(session, str(pid), timeout_s)
                    out["poll"] = polled
            # prompt_id in text
            joined = out["content_text"]
            m = re.search(r"prompt_id['\"\\s:]+([a-f0-9-]{8,})", joined, re.I)
            if m and "results" not in out:
                polled = await poll_job(session, m.group(1), timeout_s)
                out["poll"] = polled
            return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--payload", type=Path, default=DEFAULT_PAYLOAD)
    p.add_argument("--out", type=Path, default=ROOT / "data/comfy_brain_art_result.json")
    p.add_argument("--image-out", type=Path, default=ROOT / "assets/brand/brain-art-gpt-image-2.png")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument(
        "--status",
        type=Path,
        default=ROOT / "data/comfy_partner_status.json",
        help="heartbeat JSON for detached runs",
    )
    args = p.parse_args()
    payload_path = args.payload
    if not payload_path.is_file():
        payload_path = FALLBACK_PAYLOAD if FALLBACK_PAYLOAD.is_file() else payload_path
    if not payload_path.is_file():
        raise SystemExit(f"payload not found: {args.payload}")
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    # strip markdown bold — image APIs choke on ** sometimes
    if "prompt" in payload:
        payload["prompt"] = payload["prompt"].replace("**", "")

    print("payload:", payload_path, flush=True)
    print("prompt:", payload.get("prompt", "")[:160], "…", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text(
        json.dumps({"phase": "starting", "ts": time.time(), "payload": str(payload_path)}),
        encoding="utf-8",
    )
    try:
        result = asyncio.run(run_generate(payload, args.timeout, args.status))
    except Exception as exc:
        err_path = args.out.with_suffix(".error.json")
        err_body = {"error": type(exc).__name__, "message": str(exc), "ts": time.time()}
        err_path.write_text(json.dumps(err_body, indent=2), encoding="utf-8")
        args.status.write_text(json.dumps({"phase": "failed", **err_body}), encoding="utf-8")
        print("ERROR", exc, flush=True)
        return 1

    # save image
    saved = None
    if result.get("results"):
        url = result["results"][0].get("url")
        if url:
            saved = save_from_url(url, args.image_out)
            result["saved_image"] = str(saved)

    # trim huge base64 from json on disk
    slim = json.loads(json.dumps(result, default=str))
    if slim.get("structured") and slim["structured"].get("results"):
        for r in slim["structured"]["results"]:
            if r.get("url", "").startswith("data:"):
                r["url"] = "<base64 omitted>"
    if slim.get("results"):
        for r in slim["results"]:
            if r.get("url", "").startswith("data:"):
                r["url"] = "<base64 omitted>"
    args.out.write_text(json.dumps(slim, indent=2), encoding="utf-8")
    args.status.write_text(
        json.dumps(
            {"phase": "saved", "ts": time.time(), "out": str(args.out), "image": str(saved) if saved else None},
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved", args.out, flush=True)
    if saved:
        print("image", saved, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())