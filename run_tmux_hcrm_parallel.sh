#!/bin/bash
set -euo pipefail

# Launch four HCRM LIBERO trainings on one shared eight-GPU Ray cluster.
# Override example: RAY_NUM_CPUS=64 RAY_PORT=29501 bash run_tmux_hcrm_parallel.sh
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-96}
RAY_PORT=${RAY_PORT:-29500}
RAY_MEMORY_USAGE_THRESHOLD=${RAY_MEMORY_USAGE_THRESHOLD:-0.98}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=${RLINF_PI05_MODEL_PATH:-${HOME}/models/RLinf-Pi05-LIBERO-SFT}
VENV_DIR=${VENV_DIR:-${REPO_DIR}/.venv}
TMUX_PREFIX=${TMUX_PREFIX:-rlinf_hcrm}
KEEPALIVE_SESSION=${KEEPALIVE_SESSION:-gpu_keepalive_0_7}
CONFIG_DIR="${REPO_DIR}/denserwd/config/selected_vla/hcrm"
LAUNCH_ID=$(date +'%Y%m%d-%H%M%S')-$$
LAUNCH_LOG_DIR="${REPO_DIR}/logs/hcrm_parallel/${LAUNCH_ID}"

CONFIGS=(
  libero_10_grpo_openpi_pi05_hcrm
  libero_goal_grpo_openpi05_hcrm
  libero_object_grpo_openpi05_hcrm
  libero_spatial_grpo_openpi05_hcrm
)
RUN_IDS=(libero10 goal object spatial)
SESSIONS=(
  "${TMUX_PREFIX}_libero10"
  "${TMUX_PREFIX}_goal"
  "${TMUX_PREFIX}_object"
  "${TMUX_PREFIX}_spatial"
)

if [[ ! -x "${VENV_DIR}/bin/python" || ! -x "${VENV_DIR}/bin/ray" ]]; then
  echo "Missing environment: ${VENV_DIR}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Missing official checkpoint: ${MODEL_DIR}" >&2
  exit 1
fi
for index in "${!CONFIGS[@]}"; do
  if [[ ! -f "${CONFIG_DIR}/${CONFIGS[$index]}.yaml" ]]; then
    echo "Missing config: ${CONFIG_DIR}/${CONFIGS[$index]}.yaml" >&2
    exit 1
  fi
  if tmux has-session -t "${SESSIONS[$index]}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSIONS[$index]}" >&2
    exit 1
  fi
done

"${VENV_DIR}/bin/python" - "${REPO_DIR}" "${CONFIG_DIR}" "${CONFIGS[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

repo_dir = Path(sys.argv[1])
config_dir = Path(sys.argv[2])
config_names = sys.argv[3:]
occupied_gpus = set()
output_dirs = set()

for config_name in config_names:
    config = yaml.safe_load((config_dir / f"{config_name}.yaml").read_text())
    placement = config["cluster"]["component_placement"]
    task_gpus = {placement["actor"], placement["env,rollout"]}
    overlap = occupied_gpus & task_gpus
    if overlap:
        raise SystemExit(f"GPU placement overlap for {config_name}: {sorted(overlap)}")
    occupied_gpus.update(task_gpus)

    logger = config["runner"]["logger"]
    output_dir = (
        repo_dir / "denserwd" / logger["log_path"] / logger["experiment_name"]
    ).resolve()
    if output_dir in output_dirs:
        raise SystemExit(f"Duplicate log/checkpoint directory: {output_dir}")
    output_dirs.add(output_dir)
    print(f"PRECHECK {config_name}: GPUs={sorted(task_gpus)}, output={output_dir}")

if occupied_gpus != set(range(8)):
    raise SystemExit(f"Expected exact GPU coverage 0-7, got {sorted(occupied_gpus)}")
PY

if pgrep -f '[t]rain_embodied_agent_hcrm.py' >/dev/null; then
  echo "An embodied HCRM training process is already running; refusing to reset Ray." >&2
  exit 1
fi

mkdir -p "${LAUNCH_LOG_DIR}"
printf 'launch_id=%s\nray_address=127.0.0.1:%s\nmodel_dir=%s\n' \
  "${LAUNCH_ID}" "${RAY_PORT}" "${MODEL_DIR}" >"${LAUNCH_LOG_DIR}/manifest.txt"

"${VENV_DIR}/bin/ray" stop --force >/dev/null 2>&1 || true
CUDA_VISIBLE_DEVICES="${GPUS}" \
  PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" \
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
  config=${CONFIGS[$index]}
  run_id=${RUN_IDS[$index]}
  session=${SESSIONS[$index]}
  log_file="${LAUNCH_LOG_DIR}/${config}.log"
  namespace="RLinf_${run_id}_${LAUNCH_ID}"

  printf 'session=%s config=%s namespace=%s log=%s\n' \
    "${session}" "${config}" "${namespace}" "${log_file}" \
    >>"${LAUNCH_LOG_DIR}/manifest.txt"

  tmux new-session -d -s "${session}" \
    "set -o pipefail; cd \"${REPO_DIR}\" && source \"${VENV_DIR}/bin/activate\" && export CUDA_VISIBLE_DEVICES=\"${GPUS}\" RAY_ADDRESS=\"127.0.0.1:${RAY_PORT}\" RAY_memory_usage_threshold=\"${RAY_MEMORY_USAGE_THRESHOLD}\" RLINF_NAMESPACE=\"${namespace}\" RLINF_PI05_MODEL_PATH=\"${MODEL_DIR}\" RLINF_CONFIG_DIR=\"./config/selected_vla/hcrm\" PYTHONPATH=\"${REPO_DIR}:\${PYTHONPATH:-}\" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl && cd \"${REPO_DIR}/denserwd\" && bash run_embodiment_hcrm.sh \"${config}\" 2>&1 | tee -a \"${log_file}\""
  started_sessions+=("${session}")
done

if tmux has-session -t "${KEEPALIVE_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${KEEPALIVE_SESSION}"
  echo "Stopped GPU keepalive session: ${KEEPALIVE_SESSION}"
else
  echo "GPU keepalive session was not running: ${KEEPALIVE_SESSION}"
fi

for session in "${SESSIONS[@]}"; do
  tmux has-session -t "${session}"
done
trap - ERR

echo "Started four HCRM training sessions on non-overlapping GPU pairs:"
for index in "${!CONFIGS[@]}"; do
  echo "  ${SESSIONS[$index]} -> ${CONFIGS[$index]}"
done
echo "Launch logs and manifest: ${LAUNCH_LOG_DIR}"
