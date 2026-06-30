# Comfy Cloud gpt-image-2 — art from **brain model smoke data**

NeuroGlyph art is **not** a generic hero banner. It is a **data painting** built from:

1. Fine-tuned checkpoint metadata (`val_acc`, task, 14ch)
2. **Smoke inference** on `data/processed/processed_hand.pt` (sample epochs)
3. **Eval** confusion matrix (left/right decoder errors)

## Generate prompt from live smoke tensors

```bash
python scripts/brain_art_prompt.py
# → data/brain_art_prompt.txt
# → assets/brand/comfy-partner_generate-brain-smoke.json
# → data/eval_hand.json
# → data/brain_art_bundle.json
```

## Comfy MCP (`partner_generate`)

Load JSON from `assets/brand/comfy-partner_generate-brain-smoke.json` or ask Hermes:

> comfy-cloud **partner_generate** using `comfy-partner_generate-brain-smoke.json` (gpt-image-2, 16:9)

Re-run `brain_art_prompt.py` after each fine-tune so the image prompt tracks **your** decoder state.

## Render locally (Hermes venv Python)

Pitfall: use `mcp.client.streamable_http`, not `streamablehttp`.

```bash
"C:\Users\MindExpander\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" scripts/comfy_partner_generate.py --payload assets/brand/comfy-partner_generate-brain-smoke.json
```

## When you have real EPOC sessions

Same script — point `--data` at your subject `processed_hand.pt`. The epoch brief uses **actual** channel energy + misclassified keystrokes.