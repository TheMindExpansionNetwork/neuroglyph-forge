# Hermes — Comfy Cloud MCP

Docs: https://docs.comfy.org/agent-tools/cloud

## Headless setup (this project)

API keys: [platform.comfy.org/profile/api-keys](https://platform.comfy.org/profile/api-keys) (`comfyui-…` prefix).

Add to `~/AppData/Local/hermes/.env`:

```bash
COMFY_CLOUD_API_KEY=comfyui-…
```

Config (`hermes config set` or `~/.hermes/config.yaml`):

```yaml
mcp_servers:
  comfy-cloud:
    url: https://cloud.comfy.org/mcp
    headers:
      X-API-Key: "<your comfyui- key>"
    timeout: 300
```

Verify:

```bash
hermes mcp test comfy-cloud
```

Restart Hermes or `/reload-mcp` in session. Tools appear as `mcp_comfy_cloud_*` (e.g. `partner_generate`, `run_template`).

## Promo images

From Hermes chat (with comfy-cloud connected):

> Generate a square logo for **NeuroGlyph Forge**: EEG brainwaves forming a golden glyph, teal/purple cyber aesthetic, dark background. Use Comfy Cloud partner_generate or image workflow.

Or run locally after MCP is configured:

```bash
python scripts/generate_brand_assets.py
```