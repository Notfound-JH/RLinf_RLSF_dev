#!/bin/bash
# Evaluate ALL saved checkpoints (global_step_10 .. global_step_50) for the
# libero_10_grpo_openpi_pi05 run, using GPUs 2/3/4 as a worker pool.
#
# Run this yourself whenever you want (does NOT wait for any training
# container). Each GPU picks up the next pending checkpoint from a shared
# queue as soon as it finishes the previous one, so 5 checkpoints over 3
# GPUs means 2 rounds (3 then 2).
#
# By default this script launches itself inside a detached tmux session
# (named "rlinf_eval" by default) so it keeps running even if you close
# your terminal / SSH session. Attach anytime with:
#   tmux attach -t rlinf_eval
# Detach again with Ctrl-b d (the session keeps running in the background).
#
# Usage:
#   bash toolkits/run_full_eval.sh                # uses GPUs 2 3 4, all checkpoints
#   bash toolkits/run_full_eval.sh 2 3 4           # explicit GPU list
#   bash toolkits/run_full_eval.sh 2 3 4 -- 10 20  # only eval steps 10 and 20
#
# Env vars:
#   TMUX_SESSION=rlinf_eval   name of the tmux session to (re)create
#   NO_TMUX=1                 skip tmux and run directly in the current shell
#   EVAL_ENVS=4               parallel envs per checkpoint (passed to eval_checkpoints.py --eval-envs)
#   EXTRA_OVERRIDES="..."     extra space-separated Hydra overrides forwarded to every eval run,
#                             e.g. EXTRA_OVERRIDES="algorithm.eval_rollout_epoch=5" to run multiple
#                             rollout epochs per checkpoint (more total episodes without raising
#                             parallelism / peak CPU-mem).

set -uo pipefail

REPO_PATH="/home/jhwang/storage/jhwang_data/RLinf-RLSF"
SCRIPT_PATH="${REPO_PATH}/toolkits/run_full_eval.sh"
IMAGE="docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-maniskill_libero"
CKPT_ROOT="${REPO_PATH}/results/libero_10_grpo_openpi_pi05/checkpoints"
LOG_DIR="${REPO_PATH}/logs/eval_checkpoints"
EVAL_ENVS=${EVAL_ENVS:-4}
mkdir -p "${LOG_DIR}"

# Parse EXTRA_OVERRIDES (space-separated) into an array for --extra-override.
EXTRA_OVERRIDE_ARGS=()
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    read -r -a EXTRA_OVERRIDE_ARGS <<< "${EXTRA_OVERRIDES}"
fi

TMUX_SESSION=${TMUX_SESSION:-rlinf_eval}

# Re-exec ourselves inside a detached tmux session unless we're already
# inside one (TMUX env var set) or the caller explicitly opted out.
if [[ -z "${NO_TMUX:-}" && -z "${TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "[warn] tmux not found, running directly in this shell instead." >&2
    else
        if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
            echo "[info] tmux session '${TMUX_SESSION}' already exists. Attach with:"
            echo "         tmux attach -t ${TMUX_SESSION}"
            exit 0
        fi
        # Re-quote all original args for the inner shell command.
        inner_args=""
        for a in "$@"; do
            inner_args+=" $(printf '%q' "${a}")"
        done
        # Forward the env vars the user may have set (EVAL_ENVS,
        # EXTRA_OVERRIDES) into the detached tmux session -- otherwise they
        # are silently lost and the inner script falls back to defaults.
        tmux new-session -d -s "${TMUX_SESSION}" \
            "NO_TMUX=1 EVAL_ENVS=$(printf '%q' "${EVAL_ENVS:-4}") EXTRA_OVERRIDES=$(printf '%q' "${EXTRA_OVERRIDES:-}") bash '${SCRIPT_PATH}'${inner_args}"
        echo "[info] Launched in detached tmux session '${TMUX_SESSION}'."
        echo "       Attach with:  tmux attach -t ${TMUX_SESSION}"
        echo "       Detach with:  Ctrl-b d"
        echo "       Logs also at: ${LOG_DIR}/run_full_eval.log"
        exit 0
    fi
fi



# Parse args: GPU list, optionally followed by "--" and an explicit step list.
GPUS=()
STEPS_ARG=()
parsing_steps=false
for arg in "$@"; do
    if [[ "${arg}" == "--" ]]; then
        parsing_steps=true
        continue
    fi
    if [[ "${parsing_steps}" == true ]]; then
        STEPS_ARG+=("${arg}")
    else
        GPUS+=("${arg}")
    fi
done
if [[ ${#GPUS[@]} -eq 0 ]]; then
    GPUS=(2 3 4)
fi

MAIN_LOG="${LOG_DIR}/run_full_eval.log"
: > "${MAIN_LOG}"
echo "[$(date '+%F %T')] Worker GPUs: ${GPUS[*]}" | tee -a "${MAIN_LOG}"

if [[ ${#STEPS_ARG[@]} -gt 0 ]]; then
    STEPS=("${STEPS_ARG[@]}")
else
    mapfile -t STEPS < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -name "global_step_*" \
        -exec basename {} \; | sed 's/global_step_//' | sort -n)
fi

if [[ ${#STEPS[@]} -eq 0 ]]; then
    echo "[$(date '+%F %T')] ERROR: no checkpoints found under ${CKPT_ROOT}" | tee -a "${MAIN_LOG}"
    exit 1
fi
echo "[$(date '+%F %T')] Checkpoints to evaluate: ${STEPS[*]} (total ${#STEPS[@]})" | tee -a "${MAIN_LOG}"

# Sanity check: warn (but don't block) if a requested GPU looks busy.
for gpu_id in "${GPUS[@]}"; do
    used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_id}" 2>/dev/null | tr -d ' ')
    if [[ -n "${used_mib}" && "${used_mib}" -gt 500 ]]; then
        echo "[$(date '+%F %T')] WARNING: GPU ${gpu_id} currently has ${used_mib} MiB used -- someone else may be using it!" | tee -a "${MAIN_LOG}"
    fi
done

run_one_eval() {
    local gpu_id=$1
    local step=$2
    local container_name="rlinf-eval-gpu${gpu_id}"
    local log_file="${LOG_DIR}/run_full_eval_gpu${gpu_id}_step${step}.log"

    docker rm -f "${container_name}" >/dev/null 2>&1

    echo "[$(date '+%F %T')] [GPU ${gpu_id}] evaluating step ${step} -> ${log_file}" | tee -a "${MAIN_LOG}"

    local extra_override_flags=""
    for ov in "${EXTRA_OVERRIDE_ARGS[@]}"; do
        extra_override_flags+=" --extra-override $(printf '%q' "${ov}")"
    done

    docker run --rm -i=false --gpus "\"device=${gpu_id}\"" \
        -e NVIDIA_VISIBLE_DEVICES=${gpu_id} \
        -e EMBODIED_PATH=/workspace/RLinf/denserwd \
        --shm-size 50g \
        --name "${container_name}" \
        -v "${REPO_PATH}:/workspace/RLinf" \
        -v /mnt/storage/public_data:/mnt/storage/public_data \
        "${IMAGE}" \
        bash -c "cd /workspace/RLinf/denserwd && source switch_env openpi && cd /workspace/RLinf && python toolkits/eval_checkpoints.py --steps ${step} --eval-envs ${EVAL_ENVS} --skip-video-run --single-gpu${extra_override_flags}" \
        < /dev/null > "${log_file}" 2>&1
    local rc=$?

    echo "[$(date '+%F %T')] [GPU ${gpu_id}] finished step ${step} (exit=${rc})" | tee -a "${MAIN_LOG}"
}

# Worker pool: each GPU sequentially pulls the next pending step from a
# shared queue file (protected by flock), so this scales to any number of
# checkpoints/GPUs.
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
    echo "[$(date '+%F %T')] [GPU ${gpu_id}] worker done, no more pending checkpoints." | tee -a "${MAIN_LOG}"
}

for gpu_id in "${GPUS[@]}"; do
    worker_loop "${gpu_id}" &
done

wait
rm -f "${QUEUE_FILE}" "${LOCK_FILE}"

echo "[$(date '+%F %T')] All checkpoints evaluated." | tee -a "${MAIN_LOG}"
echo "" | tee -a "${MAIN_LOG}"
echo "=== Summary (success_rate per step) ===" | tee -a "${MAIN_LOG}"
for step in "${STEPS[@]}"; do
    f=$(ls "${LOG_DIR}"/run_full_eval_gpu*_step${step}.log 2>/dev/null | head -n 1)
    if [[ -n "${f}" ]]; then
        line=$(grep -E "^${step}[[:space:]]*\|" "${f}" | tail -n 1)
        echo "step ${step}: ${line:-<no metrics parsed, see ${f}>}" | tee -a "${MAIN_LOG}"
    fi
done
