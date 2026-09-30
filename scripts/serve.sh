#!/usr/bin/env bash
# Start the local OnePass neural-preview app on http://127.0.0.1:8765 (offline).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=$PWD/.hf HF_HUB_OFFLINE=1 TORCH_HOME=$PWD/.torch
exec .venv/bin/uvicorn onepass_neural.server:app --host 127.0.0.1 --port "${PORT:-8765}"
