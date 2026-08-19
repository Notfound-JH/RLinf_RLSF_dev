#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${1:-${PUBLIC_DATA_DIR:-/mnt/storage/public_data}/models/RLinf-Pi05-LIBERO-SFT}

if ! command -v hf >/dev/null 2>&1; then
  echo "hf command not found. Install it with: python -m pip install -U huggingface_hub[cli]" >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}"
echo "Downloading RLinf/RLinf-Pi05-LIBERO-SFT to ${MODEL_DIR}"
hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir "${MODEL_DIR}"
echo "Checkpoint is ready at ${MODEL_DIR}"
