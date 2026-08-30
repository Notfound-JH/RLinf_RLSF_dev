#!/bin/bash
# LIBERO-10 full eval: 10 tasks × 50 init states = 500 episodes
# Checkpoint: HCRM global_step_50 (openpi_pytorch)
#
# Usage:
#   bash eval_gs50_full500.sh              # default GPUs 0,1,2,3,4
#   GPUS=0,1,2,3,4 bash eval_gs50_full500.sh
#   bash eval_gs50_full500.sh 0,1,2,3,4

set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPUS:-${1:-0,1,2,3,4}}"
RAY_PORT="${RAY_PORT:-29501}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-40}"
# Shared node is often near full RAM. Prefer 0.99 + disable monitor refresh.
RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.99}"
IMAGE="${IMAGE:-docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero}"
CONTAINER_NAME="${CONTAINER_NAME:-rlinf_eval_gs50}"
CONFIG_DIR="${CONFIG_DIR:-/workspace/RLinf/denserwd/config/selected_vla/hcrm}"
CONFIG_NAME="${CONFIG_NAME:-libero_10_grpo_openpi05_eval}"
RESULT_DIR="${RESULT_DIR:-${REPO_PATH}/results/libero_10_hcrm_gs50_eval500_5gpu}"
LOG_FILE="${RESULT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"
CKPT_PT="${CKPT_PT:-/scratch/rlinf-checkpoints/openpi05_libero_10_hcrm_grpo_2xa100_64env_gbs128_mbs16/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt}"
VENV_DIR="${VENV_DIR:-${REPO_PATH}/.venv}"
MODEL_DIR="${RLINF_PI05_MODEL_PATH:-${HOME}/models/RLinf-Pi05-LIBERO-SFT}"
USE_DOCKER="${USE_DOCKER:-auto}"

if [[ ! -f "${CKPT_PT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT_PT}"
  exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: base model not found: ${MODEL_DIR}"
  exit 1
fi
if [[ "${USE_DOCKER}" == "auto" ]]; then
  if command -v docker >/dev/null 2>&1; then
    USE_DOCKER=1
  else
    USE_DOCKER=0
  fi
fi
if [[ "${USE_DOCKER}" != "1" && ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "ERROR: virtual environment not found: ${VENV_DIR}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "=============================="
echo "  gs50 full LIBERO-10 eval"
echo "=============================="
echo "GPU       : ${GPU_ID}"
echo "Config    : ${CONFIG_DIR}/${CONFIG_NAME}.yaml"
echo "Ckpt      : ${CKPT_PT}"
echo "Results   : ${RESULT_DIR}"
echo "Log       : ${LOG_FILE}"
echo "Mode      : $([[ "${USE_DOCKER}" == "1" ]] && echo docker || echo native)"
echo "tmux      : rlinf_eval_gs50"
echo "=============================="

tmux kill-session -t rlinf_eval_gs50 2>/dev/null || true

if [[ "${USE_DOCKER}" == "1" ]]; then
  # Same pattern as resume_tmux_binary.sh: one quoted bash -c inside tmux.
  tmux new-session -d -s rlinf_eval_gs50 "cd ${REPO_PATH} && docker rm -f ${CONTAINER_NAME} 2>/dev/null; docker run --rm --gpus '\"device=${GPU_ID}\"' -e NVIDIA_VISIBLE_DEVICES=${GPU_ID} -e RAY_NUM_CPUS=${RAY_NUM_CPUS} -e RAY_PORT=${RAY_PORT} -e RAY_memory_usage_threshold=${RAY_MEMORY_USAGE_THRESHOLD} -e RAY_memory_monitor_refresh_ms=0 -e RAY_DISABLE_IMPORT_WARNING=1 -e RLINF_PI05_MODEL_PATH=/models/RLinf-Pi05-LIBERO-SFT --shm-size 64g --network host --name ${CONTAINER_NAME} -v \$(pwd):/workspace/RLinf -v /scratch:/scratch -v ${MODEL_DIR}:/models/RLinf-Pi05-LIBERO-SFT:ro ${IMAGE} bash -c '
set -e
cd /workspace/RLinf/denserwd && source switch_env openpi
export PYTHONPATH=/workspace/RLinf:\$PYTHONPATH
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export ROBOT_PLATFORM=LIBERO
export HYDRA_FULL_ERROR=1
export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
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
  --config-path "${CONFIG_DIR}" \
  --config-name "${CONFIG_NAME}" \
  runner.ckpt_path="${CKPT_PT}" \
  runner.logger.log_path="/workspace/RLinf/results/libero_10_hcrm_gs50_eval500_5gpu/output"
' 2>&1 | tee ${LOG_FILE}"
else
  CONFIG_DIR="${REPO_PATH}/denserwd/config/selected_vla/hcrm"
  tmux new-session -d -s rlinf_eval_gs50 "set -o pipefail; cd '${REPO_PATH}/denserwd' && \
export CUDA_VISIBLE_DEVICES='${GPU_ID}' RAY_NUM_CPUS='${RAY_NUM_CPUS}' RAY_PORT='${RAY_PORT}' \
RAY_memory_usage_threshold='${RAY_MEMORY_USAGE_THRESHOLD}' RAY_memory_monitor_refresh_ms=0 \
RAY_DISABLE_IMPORT_WARNING=1 RLINF_PI05_MODEL_PATH='${MODEL_DIR}' PYTHONPATH='${REPO_PATH}' \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO HYDRA_FULL_ERROR=1 \
EMBODIED_PATH='${REPO_PATH}/examples/embodiment'; \
'${VENV_DIR}/bin/ray' stop --force >/dev/null 2>&1 || true; rm -rf /tmp/ray/* 2>/dev/null || true; \
'${VENV_DIR}/bin/ray' start --head --port='${RAY_PORT}' --num-cpus='${RAY_NUM_CPUS}' --num-gpus=5 --disable-usage-stats --include-dashboard=false && \
'${VENV_DIR}/bin/python' eval_embodied_agent.py --config-path '${CONFIG_DIR}' --config-name '${CONFIG_NAME}' \
runner.ckpt_path='${CKPT_PT}' runner.logger.log_path='${RESULT_DIR}/output' 2>&1 | tee '${LOG_FILE}'"
fi

sleep 2
if tmux has-session -t rlinf_eval_gs50 2>/dev/null; then
  echo "Started. Attach: tmux attach -t rlinf_eval_gs50"
  echo "Log: ${LOG_FILE}"
else
  echo "ERROR: tmux session failed to start"
  exit 1
fi
