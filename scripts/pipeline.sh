#!/usr/bin/env bash
# Full local pipeline: synthetic or processed → train → eval → export
set -euo pipefail
TASK="${1:-hand}"
cd "$(dirname "$0")/.."
source .venv/Scripts/activate
python -m neuroglyph_data.make_epochs_cli --out data/processed --task "$TASK" --synthetic
python -m neuroglyph_train.train --task "$TASK" --epochs 8 --data data/processed
CKPT="checkpoints/tiny_b2q_${TASK}.pt"
python -m neuroglyph_train.evaluate --checkpoint "$CKPT" --data data/processed --output "checkpoints/eval_${TASK}.json"
python -m neuroglyph_train.export_model --checkpoint "$CKPT" --output "checkpoints/exported_${TASK}.ts"
echo "Pipeline complete: $CKPT"