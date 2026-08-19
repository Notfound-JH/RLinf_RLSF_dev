#!/bin/bash
# Host/Conda launcher for the official RLinf-Pi05-LIBERO-SFT checkpoint.
# Override example: GPUS=0,1 RAY_NUM_CPUS=32 RAY_PORT=29500 bash run_tmux_binary.sh
GPUS=${GPUS:-0,1}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}
RAY_PORT=${RAY_PORT:-29500}
RAY_MEMORY_USAGE_THRESHOLD=${RAY_MEMORY_USAGE_THRESHOLD:-0.98}
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PUBLIC_DATA_DIR=${PUBLIC_DATA_DIR:-/mnt/storage/public_data}
MODEL_DIR=${RLINF_PI05_MODEL_PATH:-${HOME}/models/RLinf-Pi05-LIBERO-SFT}
VENV_DIR=${VENV_DIR:-${REPO_DIR}/.venv}
TMUX_SESSION=${TMUX_SESSION:-rlinf_train}

if [ ! -f "${VENV_DIR}/bin/activate" ]; then
  echo "Missing environment: ${VENV_DIR}" >&2
  echo "Run: bash requirements/install.sh embodied --model openpi --env maniskill_libero --venv .venv" >&2
  exit 1
fi
if [ ! -d "${MODEL_DIR}" ]; then
  echo "Missing official checkpoint: ${MODEL_DIR}" >&2
  echo "Run: bash tools/prepare_pi05_libero.sh ${MODEL_DIR}" >&2
  exit 1
fi

mkdir -p "${REPO_DIR}/logs"
tmux new-session -d -s "${TMUX_SESSION}" "cd \"${REPO_DIR}\" && source \"${VENV_DIR}/bin/activate\" && export CUDA_VISIBLE_DEVICES=\"${GPUS}\" RAY_NUM_CPUS=\"${RAY_NUM_CPUS}\" RAY_PORT=\"${RAY_PORT}\" RAY_memory_usage_threshold=\"${RAY_MEMORY_USAGE_THRESHOLD}\" RLINF_PI05_MODEL_PATH=\"${MODEL_DIR}\" PYTHONPATH=\"${REPO_DIR}:\${PYTHONPATH:-}\" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl && ray stop --force >/dev/null 2>&1 || true; ray start --head --port=\"${RAY_PORT}\" --num-cpus=\"${RAY_NUM_CPUS}\" --disable-usage-stats --include-dashboard=false && cd \"${REPO_DIR}/denserwd\" && bash run_embodiment_hcrm.sh libero_10_grpo_openpi_hcrm 2>&1 | tee -a \"${REPO_DIR}/logs/rlinf_train.log\""

echo "Started tmux session: ${TMUX_SESSION}"
echo "Attach with: tmux attach -t ${TMUX_SESSION}"
