#!/bin/bash
# LIBERO-10 full eval: 10 tasks × 50 init states = 500 episodes
# Checkpoint: GRPO global_step_50 (openpi_pytorch)
#
# Usage:
#   bash eval_gs50_full500.sh              # default GPUs 0,1
#   GPUS=0,1 bash eval_gs50_full500.sh
#   bash eval_gs50_full500.sh 0,1

set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPUS:-${1:-0,1}}"
RAY_PORT="${RAY_PORT:-29501}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
# Shared node is often near full RAM. Prefer 0.99 + disable monitor refresh.
RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.99}"
IMAGE="${IMAGE:-docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero}"
CONTAINER_NAME="${CONTAINER_NAME:-rlinf_eval_gs50}"
CONFIG_NAME="libero_10_grpo_openpi_pi05_gs50_eval500"
RESULT_DIR="${REPO_PATH}/results/libero_10_grpo_openpi_pi05_gs50_eval500"
LOG_FILE="${RESULT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
CKPT_PT="${REPO_PATH}/results/libero_10_grpo_openpi_pi05_bcfullshot_gs1500/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"

if [[ ! -f "${CKPT_PT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_PT}"
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "=============================="
echo "  gs50 full LIBERO-10 eval"
echo "=============================="
echo "GPU       : ${GPU_ID}"
echo "Config    : ${CONFIG_NAME}"
echo "Ckpt      : ${CKPT_PT}"
echo "Results   : ${RESULT_DIR}"
echo "Log       : ${LOG_FILE}"
echo "Container : ${CONTAINER_NAME}"
echo "tmux      : rlinf_eval_gs50"
echo "=============================="

tmux kill-session -t rlinf_eval_gs50 2>/dev/null || true

# Same pattern as resume_tmux_binary.sh: one quoted bash -c inside tmux.
tmux new-session -d -s rlinf_eval_gs50 "cd ${REPO_PATH} && docker rm -f ${CONTAINER_NAME} 2>/dev/null; docker run --rm --gpus '\"device=${GPU_ID}\"' -e NVIDIA_VISIBLE_DEVICES=${GPU_ID} -e RAY_NUM_CPUS=${RAY_NUM_CPUS} -e RAY_PORT=${RAY_PORT} -e RAY_memory_usage_threshold=${RAY_MEMORY_USAGE_THRESHOLD} -e RAY_memory_monitor_refresh_ms=0 -e RAY_DISABLE_IMPORT_WARNING=1 --shm-size 64g --network host --name ${CONTAINER_NAME} -v \$(pwd):/workspace/RLinf -v /mnt/storage/public_data:/mnt/storage/public_data ${IMAGE} bash -c '
set -e
cd /workspace/RLinf/denserwd && source switch_env openpi
export PYTHONPATH=/workspace/RLinf:\$PYTHONPATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export ROBOT_PLATFORM=LIBERO
export HYDRA_FULL_ERROR=1
export EMBODIED_PATH=/workspace/RLinf/denserwd
export REPO_PATH=/workspace/RLinf
export RAY_memory_usage_threshold=\${RAY_memory_usage_threshold}
export RAY_memory_monitor_refresh_ms=0

ray stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray/* 2>/dev/null || true
ray start --head \\
  --port=\${RAY_PORT} \\
  --num-cpus=\${RAY_NUM_CPUS} \\
  --disable-usage-stats \\
  --include-dashboard=false
ray status || true

python eval_embodied_agent.py \\
  --config-path \"\${EMBODIED_PATH}/config/vla\" \\
  --config-name ${CONFIG_NAME}
' 2>&1 | tee ${LOG_FILE}"

sleep 2
if tmux has-session -t rlinf_eval_gs50 2>/dev/null; then
  echo "Started. Attach: tmux attach -t rlinf_eval_gs50"
  echo "Log: ${LOG_FILE}"
else
  echo "ERROR: tmux session failed to start"
  exit 1
fi
