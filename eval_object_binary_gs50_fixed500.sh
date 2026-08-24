#!/bin/bash
# Evaluate the LIBERO-Object binary-GRPO global_step_50 checkpoint on 500 fixed IDs.
#
# Usage:
#   bash eval_object_binary_gs50_fixed500.sh
#   GPUS=0,1 bash eval_object_binary_gs50_fixed500.sh

set -euo pipefail

WORKER_MODE=0
if [[ "${1:-}" == "--worker" ]]; then
  WORKER_MODE=1
  EVAL_LOG_FILE="$2"
  WORKER_GPU_IDS="$3"
  WORKER_RAY_PORT="$4"
  WORKER_RAY_NUM_CPUS="$5"
  WORKER_MEMORY_THRESHOLD="$6"
  WORKER_VENV_DIR="$7"
  WORKER_MODEL_DIR="$8"
fi

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${REPO_PATH}/$(basename "${BASH_SOURCE[0]}")"
GPU_IDS="${WORKER_GPU_IDS:-${GPUS:-${1:-0,1,2,3}}}"
RAY_PORT="${WORKER_RAY_PORT:-${RAY_PORT:-29502}}"
RAY_NUM_CPUS="${WORKER_RAY_NUM_CPUS:-${RAY_NUM_CPUS:-32}}"
RAY_MEMORY_USAGE_THRESHOLD="${WORKER_MEMORY_THRESHOLD:-${RAY_MEMORY_USAGE_THRESHOLD:-0.99}}"
TMUX_SESSION="${TMUX_SESSION:-rlinf_eval_object_gs50}"
VENV_DIR="${WORKER_VENV_DIR:-${VENV_DIR:-${REPO_PATH}/.venv}}"
MODEL_DIR="${WORKER_MODEL_DIR:-${RLINF_PI05_MODEL_PATH:-${HOME}/models/RLinf-Pi05-LIBERO-SFT}}"
CONFIG_NAME="libero_object_grpo_openpi05_eval"
CONFIG_DIR="${REPO_PATH}/denserwd/config/selected_vla/binary"
CHECKPOINT="/scratch/rlinf-checkpoints/openpi05_libero_object_binary_grpo_2xa100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"
RESULT_DIR="${REPO_PATH}/results/libero_object_binary_gs50_fixed500"
LOG_FILE="${EVAL_LOG_FILE:-${RESULT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log}"

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  echo "ERROR: virtual environment not found: ${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: base model not found: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"

if [[ "${WORKER_MODE}" == "1" ]]; then
  exec > >(tee -a "${LOG_FILE}") 2>&1

  cd "${REPO_PATH}"
  export PYTHONPATH="${PYTHONPATH:-}"
  source "${VENV_DIR}/bin/activate"

  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export RAY_NUM_CPUS
  export RAY_PORT
  export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}"
  export RAY_memory_monitor_refresh_ms=0
  export RAY_DISABLE_IMPORT_WARNING=1
  export RLINF_PI05_MODEL_PATH="${MODEL_DIR}"
  export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
  export MUJOCO_GL=egl
  export PYOPENGL_PLATFORM=egl
  export ROBOT_PLATFORM=LIBERO
  export HYDRA_FULL_ERROR=1
  export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"

  ray stop --force >/dev/null 2>&1 || true
  rm -rf /tmp/ray/* 2>/dev/null || true
  ray start --head \
    --port="${RAY_PORT}" \
    --num-cpus="${RAY_NUM_CPUS}" \
    --disable-usage-stats \
    --include-dashboard=false

  cd "${REPO_PATH}/denserwd"
  python eval_embodied_agent.py \
    --config-path "${CONFIG_DIR}" \
    --config-name "${CONFIG_NAME}"
  exit
fi

echo "============================================"
echo "  LIBERO-Object binary GRPO gs50 fixed500"
echo "============================================"
echo "GPUs       : ${GPU_IDS}"
echo "Config     : ${CONFIG_DIR}/${CONFIG_NAME}.yaml"
echo "Checkpoint : ${CHECKPOINT}"
echo "Base model : ${MODEL_DIR}"
echo "Venv       : ${VENV_DIR}"
echo "Results    : ${RESULT_DIR}"
echo "Log        : ${LOG_FILE}"
echo "tmux       : ${TMUX_SESSION}"
echo "============================================"

tmux kill-session -t "${TMUX_SESSION}" 2>/dev/null || true

tmux new-session -d -s "${TMUX_SESSION}" \
  "bash '${SCRIPT_PATH}' --worker '${LOG_FILE}' '${GPU_IDS}' '${RAY_PORT}' '${RAY_NUM_CPUS}' '${RAY_MEMORY_USAGE_THRESHOLD}' '${VENV_DIR}' '${MODEL_DIR}'"

echo "Started. Attach with: tmux attach -t ${TMUX_SESSION}"
echo "Log: ${LOG_FILE}"