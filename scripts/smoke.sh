#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m venv .venv
source .venv/Scripts/activate
pip install -U pip
pip install -e .
pytest tests/ -q
python -m neuroglyph_train.train --task hand --epochs 3
echo "NeuroGlyph Forge smoke pipeline OK"