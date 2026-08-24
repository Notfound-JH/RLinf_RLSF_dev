#!/bin/bash
# Evaluate four binary-GRPO global_step_50 checkpoints in parallel.
# GPU pairs: Object 0-1, LIBERO-10 2-3, Goal 4-5, Spatial 6-7.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv}"
MODEL_DIR="${RLINF_PI05_MODEL_PATH:-${HOME}/models/RLinf-Pi05-LIBERO-SFT}"
CONFIG_DIR="${REPO_DIR}/denserwd/config/selected_vla/binary"
VENV_LIB_DIR="${VENV_DIR}/sysroot/usr/lib/x86_64-linux-gnu"
SYSTEM_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
RAY_PORT="${RAY_PORT:-29502}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-96}"
RAY_MEMORY_USAGE_THRESHOLD="${RAY_MEMORY_USAGE_THRESHOLD:-0.99}"
TMUX_PREFIX="${TMUX_PREFIX:-rlinf_eval_binary_gs50}"
LAUNCH_ID="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="${REPO_DIR}/results/binary_gs50_fixed500_parallel/${LAUNCH_ID}"

CONFIGS=(
  libero_object_grpo_openpi05_eval
  libero_10_grpo_openpi05_eval
  libero_goal_grpo_openpi05_eval
  libero_spatial_grpo_openpi05_eval
)
SUITES=(object libero10 goal spatial)
GPU_PAIRS=(0-1 2-3 4-5 6-7)

if [[ ! -x "${VENV_DIR}/bin/python" || ! -x "${VENV_DIR}/bin/ray" ]]; then
  echo "ERROR: incomplete virtual environment: ${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: base model not found: ${MODEL_DIR}" >&2
  exit 1
fi

for index in "${!CONFIGS[@]}"; do
  config_path="${CONFIG_DIR}/${CONFIGS[$index]}.yaml"
  if [[ ! -f "${config_path}" ]]; then
    echo "ERROR: config not found: ${config_path}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"
printf 'launch_id=%s\nray_address=127.0.0.1:%s\nmodel_dir=%s\n' \
  "${LAUNCH_ID}" "${RAY_PORT}" "${MODEL_DIR}" >"${LOG_DIR}/manifest.txt"

# Replace the earlier single-suite run before resetting its Ray cluster.
tmux kill-session -t rlinf_eval_object_gs50 2>/dev/null || true
for suite in "${SUITES[@]}"; do
  tmux kill-session -t "${TMUX_PREFIX}_${suite}" 2>/dev/null || true
done

"${VENV_DIR}/bin/ray" stop --force >/dev/null 2>&1 || true
rm -rf /tmp/ray/* 2>/dev/null || true
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  PYTHONPATH="${REPO_DIR}" \
  LD_LIBRARY_PATH="${VENV_LIB_DIR}:${SYSTEM_LD_LIBRARY_PATH}" \
  RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}" \
  "${VENV_DIR}/bin/ray" start --head \
    --port="${RAY_PORT}" \
    --num-cpus="${RAY_NUM_CPUS}" \
    --num-gpus=8 \
    --disable-usage-stats \
    --include-dashboard=false >/dev/null

started_sessions=()
cleanup_failed_launch() {
  local session
  for session in "${started_sessions[@]}"; do
    tmux kill-session -t "${session}" 2>/dev/null || true
  done
  "${VENV_DIR}/bin/ray" stop --force >/dev/null 2>&1 || true
}
trap cleanup_failed_launch ERR

for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$index]}"
  suite="${SUITES[$index]}"
  pair="${GPU_PAIRS[$index]}"
  session="${TMUX_PREFIX}_${suite}"
  namespace="RLinf_eval_${suite}_${LAUNCH_ID}"
  log_file="${LOG_DIR}/${suite}.log"

  printf 'suite=%s gpu_pair=%s session=%s namespace=%s config=%s log=%s\n' \
    "${suite}" "${pair}" "${session}" "${namespace}" "${config}" "${log_file}" \
    >>"${LOG_DIR}/manifest.txt"

  tmux new-session -d -s "${session}" \
    "set -o pipefail; cd '${REPO_DIR}/denserwd' && export CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' RAY_ADDRESS='127.0.0.1:${RAY_PORT}' RAY_memory_usage_threshold='${RAY_MEMORY_USAGE_THRESHOLD}' RAY_memory_monitor_refresh_ms=0 RAY_DISABLE_IMPORT_WARNING=1 RLINF_NAMESPACE='${namespace}' RLINF_PI05_MODEL_PATH='${MODEL_DIR}' PYTHONPATH='${REPO_DIR}' LD_LIBRARY_PATH='${VENV_LIB_DIR}:${SYSTEM_LD_LIBRARY_PATH}' PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO HYDRA_FULL_ERROR=1 EMBODIED_PATH='${REPO_DIR}/examples/embodiment' && '${VENV_DIR}/bin/python' eval_embodied_agent.py --config-path '${CONFIG_DIR}' --config-name '${config}' 2>&1 | tee -a '${log_file}'"
  started_sessions+=("${session}")
done

for session in "${started_sessions[@]}"; do
  tmux has-session -t "${session}"
done
trap - ERR

echo "Started four fixed500 evaluations:"
for index in "${!CONFIGS[@]}"; do
  echo "  ${TMUX_PREFIX}_${SUITES[$index]} -> GPUs ${GPU_PAIRS[$index]}"
done
echo "Logs and manifest: ${LOG_DIR}"