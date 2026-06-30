# Hugging Face upload

Swarm stages under `data/uploads/` then pushes when authenticated.

```bash
hf auth login
# or: set HF_TOKEN=hf_... in environment (do not commit)

.venv/Scripts/python.exe scripts/push_model_hf.py --checkpoint checkpoints/tiny_b2q_hand.pt --push
.venv/Scripts/python.exe scripts/push_dataset_hf.py --raw data/raw --push
```

Repos (default):
- Model: `TheMindExpansionNetwork/NeuroGlyph-EPOC-Typing-v1`
- Dataset: `TheMindExpansionNetwork/neuroglyph-epoc-typing-en-v1`

Full local run:

```bash
.venv/Scripts/python.exe scripts/run_swarm_pipeline.py
```

Report: `data/swarm_report.json`