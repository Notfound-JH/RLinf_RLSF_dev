#!/bin/bash
# Evaluate the baseline pi0_libero model on LIBERO-10 (500 episodes: 10 tasks x 50 initial states)
#
# Usage:
#   bash eval_pi0_libero10_baseline.sh [GPU_ID]
#
# Examples:
#   bash eval_pi0_libero10_baseline.sh        # default: GPU 0
#   bash eval_pi0_libero10_baseline.sh 5      # use GPU 5

set -e

GPU_ID=${1:-0}
CONTAINER_NAME="rlinf_eval_pi0_libero10"
REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_DIR="${REPO_PATH}/data/results/pi0_libero_baseline_libero10_eval"
IMAGE="docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero"

echo "=============================="
echo "  pi0_libero  |  LIBERO-10"
echo "=============================="
echo "GPU      : ${GPU_ID}"
echo "Repo     : ${REPO_PATH}"
echo "Results  : ${RESULT_DIR}"
echo "Container: ${CONTAINER_NAME}"
echo "=============================="

mkdir -p "${RESULT_DIR}"

# Remove stale container if exists
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

LOG_FILE="${RESULT_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

docker run --rm \
  --gpus "device=${GPU_ID}" \
  -e NVIDIA_VISIBLE_DEVICES="${GPU_ID}" \
  --shm-size 32g \
  --network host \
  --name "${CONTAINER_NAME}" \
  -v "${REPO_PATH}:/workspace/RLinf" \
  -v /mnt/storage/public_data:/mnt/storage/public_data \
  "${IMAGE}" \
  bash -c '
    set -e

    cd /workspace/RLinf/denserwd
    source switch_env openpi

    export MUJOCO_GL="osmesa"
    export PYOPENGL_PLATFORM="osmesa"
    export ROBOT_PLATFORM="LIBERO"
    export HYDRA_FULL_ERROR=1
    export PYTHONPATH=/workspace/RLinf:${PYTHONPATH}

    export EMBODIED_PATH="/workspace/RLinf/denserwd"
    CONFIG_NAME="libero_10_sft_openpi05_one_shot_eval"

    echo ""
    echo "Python: $(which python)"
    echo "Config: ${EMBODIED_PATH}/config/vla/${CONFIG_NAME}.yaml"
    echo ""

    mkdir -p /workspace/RLinf/results

    python eval_embodied_agent.py \
      --config-path "${EMBODIED_PATH}/config/vla" \
      --config-name "${CONFIG_NAME}"
  ' 2>&1 | tee "${LOG_FILE}" | awk '
BEGIN {
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    RESET  = "\033[0m"
}
# per-episode result lines from env_worker.py — always show, highlighted
/\[eval\] episode/    { print GREEN $0 RESET; fflush(); next }
# RLinf high-level status (start, final metrics)
/\[INFO [^ ]+ RLinf\]/ { print YELLOW $0 RESET; fflush(); next }
# eval progress bar
/Evaluating Rollout/  { print $0; fflush(); next }
# errors and exceptions
/ERROR|Exception|Traceback|raise / { print RED $0 RESET; fflush(); next }
# suppress gym / robosuite / oneDNN noise
/Gym has been|Please upgrade|Users of this|See the migration/ { next }
/robosuite WARNING|WARNING:robosuite/ { next }
/oneDNN|cpu_feature_guard|absl::Initialize/ { next }
/UserWarning|FutureWarning/ { next }
/No OpenGL_accelerate/ { next }
# print everything else as-is
{ print; fflush() }
'

echo ""
echo "=============================="
echo "Done. Log saved to: ${LOG_FILE}"
echo "Results and videos: ${RESULT_DIR}"
echo "=============================="
