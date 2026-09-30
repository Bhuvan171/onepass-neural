#!/usr/bin/env bash
# Reproduce the OnePass neural-preview environment. Needs network ONCE; afterwards
# everything runs offline from .venv, .hf, .torch and third_party/.
set -euo pipefail
cd "$(dirname "$0")/.."

MAPANYTHING_COMMIT=3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
MODEL_REVISION=00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a

python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch==2.12.0 torchvision==0.27.0

if [ ! -d third_party/map-anything ]; then
  git clone https://github.com/facebookresearch/map-anything.git third_party/map-anything
fi
git -C third_party/map-anything checkout "$MAPANYTHING_COMMIT"
# Core package only. Do NOT install ".[all]" (pulls non-commercial external models).
.venv/bin/pip install -e third_party/map-anything
.venv/bin/pip install fastapi uvicorn python-multipart exifread pytest

if [ ! -d data/brighton_beach ]; then
  git clone --depth 1 https://github.com/pierotofy/drone_dataset_brighton_beach.git data/brighton_beach
fi

# Apache-2.0 checkpoint at a pinned revision, cached inside the project.
HF_HOME=$PWD/.hf .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/map-anything-apache', revision='$MODEL_REVISION')"

# Model construction fetches the DINOv2 *code* via torch.hub once; cache it in .torch.
HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 TORCH_HOME=$PWD/.torch .venv/bin/python -c "
from onepass_neural.infer import load_model; load_model()"

echo "Setup done. Run: scripts/serve.sh"
