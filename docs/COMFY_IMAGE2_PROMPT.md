# Comfy Cloud MCP — GPT Image 2 prompts (NeuroGlyph)

Use in Hermes when **comfy-cloud** MCP is loaded (`/reload-mcp` after config). Tool: **`partner_generate`**.

## MCP payload (copy JSON args)

```json
{
  "type": "image",
  "model": "openai/images-generations",
  "prompt": "Wide cinematic hero banner for a tech product called NeuroGlyph Forge. Dark navy background (#070b14), teal (#2dd4bf) and gold (#fbbf24) accents. Center: stylized human head silhouette with glowing neural pathways connecting to a QWERTY keyboard and a game engine viewport wireframe. Subtle EEG electrode ring around the scalp (14 nodes, consumer headset shape, not medical horror). Mood: synergetic cognition, live stream energy, premium sci-fi UI. No text, no logos, no watermarks. Ultra clean composition, soft volumetric light, 16:9.",
  "aspect_ratio": "16:9",
  "client_os": "windows",
  "params": {
    "model": "gpt-image-2",
    "quality": "medium",
    "size": "1536x1024"
  }
}
```

**Expect 15–20+ minutes** on one call. Poll with `get_job_status` / `get_output` if you get a `prompt_id` instead of a direct URL.

---

## Alternate prompts

### Brain map (square, docs / GitHub Pages)

```
Top-down infographic brain map on dark charcoal. Five labeled zones as soft glass panels (no readable tiny text): RECORD, EPOCHS, TRAIN, HERMES, UNREAL. Teal arrows flow left to right. Small icons: EEG headset, epoch waveform, GPU chip, agent node, Unreal logo-style geometric mark (generic, not trademark). Violet highlights. Minimal, Notion-meets-cyberpunk. Square 1:1, no watermark.
```

### Comfy + BCI stream overlay (9:16)

```
Vertical stream overlay frame, transparent-friendly edges. NeuroGlyph Forge aesthetic: dark panel, teal border glow. Central motif: brain waves morphing into musical notes and blueprint lines. EMOTIV-style 14-channel arc. Space left for OBS chat. No text. gpt-image-2 friendly, high contrast, readable on 1080p stream.
```

### MindBot / surreal accent (user taste)

```
Surreal tech dreamscape in Dalí-meets-cyberpunk: melting clock faces made of EEG waveforms, long-legged elephants carrying GPU towers, ants carrying keyboard keys along neural cables. Deep teal and gold palette, paranoiac double-image hint of a headset inside a galaxy. Wide 16:9, cinematic, no text.
```

---

## Hermes chat (natural language)

If tools are exposed, you can say:

> Call **partner_generate** on comfy-cloud with gpt-image-2, 16:9, quality medium, and this prompt: *[paste hero banner prompt above]*

## Verify MCP

```bash
hermes mcp test comfy-cloud
```

Auth: `COMFY_CLOUD_API_KEY` in Hermes `.env` must match `config.yaml` → `mcp_servers.comfy-cloud.headers.X-API-Key`, then `/reload-mcp`.

## Sizes (gpt-image-2)

| Use | `size` | `aspect_ratio` |
|-----|--------|----------------|
| GitHub / site hero | `1536x1024` | `16:9` |
| Logo / icon | `1024x1024` | `1:1` |
| Stream overlay | `1024x1536` | `9:16` |