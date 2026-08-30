#!/bin/bash
set -euo pipefail

# Native eight-GPU launcher for LIBERO-10 Pi0.5 HCRM training.
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-96}
RAY_PORT=${RAY_PORT:-29500}
RAY_MEMORY_USAGE_THRESHOLD=${RAY_MEMORY_USAGE_THRESHOLD:-0.98}
CONFIG_NAME=${CONFIG_NAME:-libero_10_grpo_openpi_pi05_hcrm_8gpu}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=${RLINF_PI05_MODEL_PATH:-${HOME}/models/RLinf-Pi05-LIBERO-SFT}
VENV_DIR=${VENV_DIR:-${REPO_DIR}/.venv}
TMUX_SESSION=${TMUX_SESSION:-rlinf_train_hcrm_8gpu}
LOG_FILE=${LOG_FILE:-${REPO_DIR}/logs/${CONFIG_NAME}_$(date +'%Y%m%d_%H%M%S').log}

if [[ ! -x "${VENV_DIR}/bin/python" || ! -x "${VENV_DIR}/bin/ray" ]]; then
	echo "Missing environment: ${VENV_DIR}" >&2
	exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
	echo "Missing official checkpoint: ${MODEL_DIR}" >&2
	exit 1
fi
if [[ ! -f "${REPO_DIR}/denserwd/config/selected_vla/hcrm/${CONFIG_NAME}.yaml" ]]; then
	echo "Missing config: ${CONFIG_NAME}" >&2
	exit 1
fi
if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
	echo "tmux session already exists: ${TMUX_SESSION}" >&2
	exit 1
fi
if pgrep -f '[t]rain_embodied_agent_hcrm.py' >/dev/null; then
	echo "An HCRM training process is already running." >&2
	exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"
"${VENV_DIR}/bin/ray" stop --force >/dev/null 2>&1 || true
CUDA_VISIBLE_DEVICES="${GPUS}" \
	RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}" \
	"${VENV_DIR}/bin/ray" start --head \
		--port="${RAY_PORT}" \
		--num-cpus="${RAY_NUM_CPUS}" \
		--num-gpus=8 \
		--disable-usage-stats \
		--include-dashboard=false >/dev/null

tmux new-session -d -s "${TMUX_SESSION}" \
	"set -o pipefail; cd \"${REPO_DIR}\" && source \"${VENV_DIR}/bin/activate\" && export CUDA_VISIBLE_DEVICES=\"${GPUS}\" RAY_ADDRESS=\"127.0.0.1:${RAY_PORT}\" RAY_memory_usage_threshold=\"${RAY_MEMORY_USAGE_THRESHOLD}\" RLINF_PI05_MODEL_PATH=\"${MODEL_DIR}\" RLINF_CONFIG_DIR=\"./config/selected_vla/hcrm\" PYTHONPATH=\"${REPO_DIR}:\${PYTHONPATH:-}\" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl && cd \"${REPO_DIR}/denserwd\" && bash run_embodiment_hcrm.sh \"${CONFIG_NAME}\" 2>&1 | tee -a \"${LOG_FILE}\""

tmux has-session -t "${TMUX_SESSION}"
echo "Started tmux session: ${TMUX_SESSION}"
echo "Config: ${CONFIG_NAME}"
echo "GPUs: ${GPUS}"
echo "Log: ${LOG_FILE}"
echo "Attach with: tmux attach -t ${TMUX_SESSION}"
