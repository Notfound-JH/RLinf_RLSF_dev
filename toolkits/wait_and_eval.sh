#!/bin/bash
# Wait for the training container (rlinf) to finish, then auto-discover ALL
# saved checkpoints under CKPT_ROOT and evaluate them (fast scan, no video,
# total_num_envs=4 each) using a pool of GPUs (default 2/3/4). If there are
# more checkpoints than GPUs, each GPU sequentially picks up the next
# pending checkpoint as soon as it finishes its current one.
#
# Usage: bash toolkits/wait_and_eval.sh [gpu_id1 gpu_id2 gpu_id3 ...]
#   (default GPUs: 2 3 4)

set -uo pipefail

REPO_PATH="/home/jhwang/storage/jhwang_data/RLinf-RLSF"
IMAGE="docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero"
TRAIN_CONTAINER="rlinf"
CKPT_ROOT="${REPO_PATH}/results/libero_10_grpo_openpi_pi05/checkpoints"
LOG_DIR="${REPO_PATH}/logs/eval_checkpoints"
mkdir -p "${LOG_DIR}"

GPUS=("$@")
if [[ ${#GPUS[@]} -eq 0 ]]; then
    GPUS=(2 3 4)
fi

WATCH_LOG="${LOG_DIR}/wait_and_eval_monitor.log"
echo "[$(date '+%F %T')] Start monitoring container '${TRAIN_CONTAINER}'. Worker GPUs: ${GPUS[*]}" | tee -a "${WATCH_LOG}"

# Poll until the training container stops running.
while true; do
    status=$(docker inspect -f '{{.State.Running}}' "${TRAIN_CONTAINER}" 2>/dev/null)
    if [[ "${status}" != "true" ]]; then
        echo "[$(date '+%F %T')] Training container '${TRAIN_CONTAINER}' is no longer running (status=${status}). Proceeding to eval." | tee -a "${WATCH_LOG}"
        break
    fi
    echo "[$(date '+%F %T')] Training container still running, waiting 60s..." >> "${WATCH_LOG}"
    sleep 60
done

# Extra safety margin: give the process a bit of time to fully release GPU memory.
sleep 15

# Auto-discover all checkpoints available right now (no hardcoding).
mapfile -t STEPS < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -name "global_step_*" \
    -exec basename {} \; | sed 's/global_step_//' | sort -n)

if [[ ${#STEPS[@]} -eq 0 ]]; then
    echo "[$(date '+%F %T')] ERROR: no checkpoints found under ${CKPT_ROOT}" | tee -a "${WATCH_LOG}"
    exit 1
fi

echo "[$(date '+%F %T')] Discovered checkpoints: ${STEPS[*]} (total ${#STEPS[@]}), distributing over GPUs: ${GPUS[*]}" | tee -a "${WATCH_LOG}"

run_one_eval() {
    local gpu_id=$1
    local step=$2
    local container_name="rlinf-eval-gpu${gpu_id}"
    local log_file="${LOG_DIR}/wait_and_eval_gpu${gpu_id}_step${step}.log"

    docker rm -f "${container_name}" >/dev/null 2>&1

    echo "[$(date '+%F %T')] [GPU ${gpu_id}] evaluating step ${step} -> ${log_file}" | tee -a "${WATCH_LOG}"

    docker run --rm -i=false --gpus "\"device=${gpu_id}\"" \
        -e NVIDIA_VISIBLE_DEVICES=${gpu_id} \
        -e EMBODIED_PATH=/workspace/RLinf/denserwd \
        --shm-size 50g \
        --name "${container_name}" \
        -v "${REPO_PATH}:/workspace/RLinf" \
        -v /mnt/storage/public_data:/mnt/storage/public_data \
        "${IMAGE}" \
        bash -c "cd /workspace/RLinf/denserwd && source switch_env openpi && cd /workspace/RLinf && python toolkits/eval_checkpoints.py --steps ${step} --eval-envs 4 --skip-video-run --single-gpu" \
        < /dev/null > "${log_file}" 2>&1

    echo "[$(date '+%F %T')] [GPU ${gpu_id}] finished step ${step} (exit=$?)" | tee -a "${WATCH_LOG}"
}

# Worker-pool: each GPU sequentially pulls the next pending step from a shared
# FIFO queue file, so this scales to any number of checkpoints/GPUs.
QUEUE_FILE=$(mktemp)
printf '%s\n' "${STEPS[@]}" > "${QUEUE_FILE}"
LOCK_FILE="${QUEUE_FILE}.lock"

worker_loop() {
    local gpu_id=$1
    while true; do
        local step=""
        (
            flock -x 200
            step=$(head -n 1 "${QUEUE_FILE}" 2>/dev/null)
            if [[ -n "${step}" ]]; then
                sed -i '1d' "${QUEUE_FILE}"
            fi
            echo "${step}" > "${QUEUE_FILE}.next_${gpu_id}"
        ) 200>"${LOCK_FILE}"
        step=$(cat "${QUEUE_FILE}.next_${gpu_id}")
        rm -f "${QUEUE_FILE}.next_${gpu_id}"
        if [[ -z "${step}" ]]; then
            break
        fi
        run_one_eval "${gpu_id}" "${step}"
    done
    echo "[$(date '+%F %T')] [GPU ${gpu_id}] worker done, no more pending checkpoints." | tee -a "${WATCH_LOG}"
}

for gpu_id in "${GPUS[@]}"; do
    worker_loop "${gpu_id}" &
done

wait
rm -f "${QUEUE_FILE}" "${LOCK_FILE}"

echo "[$(date '+%F %T')] All checkpoints evaluated." | tee -a "${WATCH_LOG}"
echo "=== Final per-checkpoint tails ===" | tee -a "${WATCH_LOG}"
for step in "${STEPS[@]}"; do
    f=$(ls "${LOG_DIR}"/wait_and_eval_gpu*_step${step}.log 2>/dev/null | head -n 1)
    if [[ -n "${f}" ]]; then
        echo "--- step ${step} (${f}) ---" | tee -a "${WATCH_LOG}"
        tail -n 20 "${f}" | tee -a "${WATCH_LOG}"
    fi
done
