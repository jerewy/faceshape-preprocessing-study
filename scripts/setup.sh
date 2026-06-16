#!/usr/bin/env bash
# Paper 2 training environment — Linux / macOS / Kaggle
# Run from the repo root:  bash scripts/setup.sh
set -euo pipefail

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

# --- PyTorch: choose ONE for your hardware ---
# NVIDIA GPU (CUDA 12.4 wheels):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU-only fallback:
# pip install torch torchvision

pip install -r requirements.txt

echo "Done. Activate with: source .venv/bin/activate"
python -c "import torch; print('cuda:', torch.cuda.is_available())"
