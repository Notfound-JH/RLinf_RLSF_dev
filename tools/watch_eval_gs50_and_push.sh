#!/usr/bin/env bash

set -uo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR=${VENV_DIR:-${REPO_DIR}/.venv}
MODEL_DIR=${RLINF_PI05_MODEL_PATH:-${HOME}/models/RLinf-Pi05-LIBERO-SFT}
EXPERIMENT=${EXPERIMENT:-openpi05_libero_10_binary_grpo_8xa100_128env_gbs128_mbs16_gs50}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${REPO_DIR}/results/${EXPERIMENT}/checkpoints/global_step_50}
CHECKPOINT_FILE=${CHECKPOINT_DIR}/actor/model_state_dict/full_weights.pt
STEP_FILE=${CHECKPOINT_DIR}/global_step.txt
CONFIG_DIR=${REPO_DIR}/denserwd/config/selected_vla/binary
CONFIG_NAME=${CONFIG_NAME:-libero_10_grpo_openpi05_eval}
TRAIN_SESSION=${TRAIN_SESSION:-rlinf_libero10_8gpu}
GPU_IDS=${GPU_IDS:-0,1,2,3,4}
GPU_COUNT=${GPU_COUNT:-5}
RAY_PORT=${RAY_PORT:-29502}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-96}
POLL_SECONDS=${POLL_SECONDS:-60}
RESULT_ROOT=${RESULT_ROOT:-${REPO_DIR}/results/libero10_128env_gbs128_mbs16_gs50_eval500}
SUMMARY_REL=${SUMMARY_REL:-eval_summaries/libero10_128env_gbs128_mbs16_global_step_50.md}
SUMMARY_FILE=${REPO_DIR}/${SUMMARY_REL}
WATCH_LOG=${WATCH_LOG:-${REPO_DIR}/logs/watch_eval_gs50_and_push.log}

mkdir -p "$(dirname "${WATCH_LOG}")"
exec >>"${WATCH_LOG}" 2>&1

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

check_prerequisites() {
  local failed=0

  for required in \
    "${VENV_DIR}/bin/python" \
    "${VENV_DIR}/bin/ray" \
    "${CONFIG_DIR}/${CONFIG_NAME}.yaml"; do
    if [[ ! -e "${required}" ]]; then
      log "Missing required path: ${required}"
      failed=1
    fi
  done
  if [[ ! -d "${MODEL_DIR}" ]]; then
    log "Missing base model: ${MODEL_DIR}"
    failed=1
  fi
  if ! command -v git >/dev/null 2>&1; then
    log "git is not available."
    failed=1
  fi
  if ! git -C "${REPO_DIR}" remote get-url origin >/dev/null 2>&1; then
    log "Git remote origin is not configured."
    failed=1
  fi
  return "${failed}"
}

checkpoint_ready() {
  [[ -s "${CHECKPOINT_FILE}" ]] || return 1
  [[ -f "${STEP_FILE}" ]] || return 1
  [[ "$(<"${STEP_FILE}")" == "50" ]]
}

wait_for_checkpoint() {
  log "Watching for complete checkpoint: ${CHECKPOINT_DIR}"
  while ! checkpoint_ready; do
    log "global_step_50 is not complete; checking again in ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
  done
  log "Complete global_step_50 checkpoint detected."
}

wait_for_training_exit() {
  while tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null || \
    pgrep -f '[t]rain_embodied_agent_hcrm.py' >/dev/null; do
    log "Training still owns the Ray cluster; checking again in ${POLL_SECONDS}s."
    sleep "${POLL_SECONDS}"
  done
  log "Training has exited; evaluation may take over the Ray cluster."
}

run_evaluation() {
  local eval_log=$1
  local tensorboard_dir=$2
  local rc

  "${VENV_DIR}/bin/ray" stop --force >/dev/null 2>&1 || true
  rm -rf /tmp/ray/* 2>/dev/null || true
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    PATH="${VENV_DIR}/bin:${PATH}" \
    PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" \
    RAY_memory_usage_threshold=0.99 \
    "${VENV_DIR}/bin/ray" start --head \
      --port="${RAY_PORT}" \
      --num-cpus="${RAY_NUM_CPUS}" \
      --num-gpus="${GPU_COUNT}" \
      --disable-usage-stats \
      --include-dashboard=false

  log "Starting LIBERO-10 fixed500 evaluation on GPUs ${GPU_IDS}."
  set +e
  (
    cd "${REPO_DIR}/denserwd" || exit 1
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
      RAY_ADDRESS="127.0.0.1:${RAY_PORT}" \
      RAY_memory_usage_threshold=0.99 \
      RAY_memory_monitor_refresh_ms=0 \
      RAY_DISABLE_IMPORT_WARNING=1 \
      RLINF_NAMESPACE="RLinf_eval_libero10_gs50_$(date +%s)" \
      RLINF_PI05_MODEL_PATH="${MODEL_DIR}" \
      PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}" \
      LD_LIBRARY_PATH="${VENV_DIR}/sysroot/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      MUJOCO_GL=egl \
      PYOPENGL_PLATFORM=egl \
      ROBOT_PLATFORM=LIBERO \
      HYDRA_FULL_ERROR=1 \
      EMBODIED_PATH="${REPO_DIR}/examples/embodiment" \
      "${VENV_DIR}/bin/python" eval_embodied_agent.py \
        --config-path "${CONFIG_DIR}" \
        --config-name "${CONFIG_NAME}" \
        "runner.ckpt_path=${CHECKPOINT_FILE}" \
        "runner.logger.log_path=${tensorboard_dir}"
  ) 2>&1 | tee "${eval_log}"
  rc=${PIPESTATUS[0]}
  set -e

  "${VENV_DIR}/bin/ray" stop --force >/dev/null 2>&1 || true
  return "${rc}"
}

write_summary() {
  local eval_log=$1
  local eval_rc=$2
  local started_at=$3
  local finished_at=$4
  local counts total success_once return_positive success_rate return_rate status final_metrics

  counts=$(awk '
    /\[eval\] episode=/ {
      total += 1
      if ($0 ~ /success=True/) success_once += 1
      if (match($0, /return=-?[0-9]+([.][0-9]+)?/)) {
        value = substr($0, RSTART + 7, RLENGTH - 7)
        if ((value + 0) > 0) return_positive += 1
      }
    }
    END { printf "%d %d %d", total, success_once, return_positive }
  ' "${eval_log}")
  read -r total success_once return_positive <<<"${counts}"
  success_rate=$(awk -v n="${success_once}" -v d="${total}" 'BEGIN { printf "%.4f", d ? n / d : 0 }')
  return_rate=$(awk -v n="${return_positive}" -v d="${total}" 'BEGIN { printf "%.4f", d ? n / d : 0 }')
  if [[ "${eval_rc}" == "0" && "${total}" == "500" ]]; then
    status=completed
  elif [[ "${eval_rc}" == "0" ]]; then
    status=incomplete
  else
    status=failed
  fi
  final_metrics=$(grep -E "\{'eval/" "${eval_log}" | tail -n 1 | sed $'s/\033\[[0-9;]*m//g')

  mkdir -p "$(dirname "${SUMMARY_FILE}")"
  cat >"${SUMMARY_FILE}" <<EOF
# LIBERO-10 Global Step 50 Evaluation

| Field | Value |
| --- | --- |
| Status | ${status} |
| Checkpoint | ${EXPERIMENT}/global_step_50 |
| Evaluation config | ${CONFIG_NAME} |
| Episodes expected | 500 |
| Episodes parsed | ${total} |
| Success once | ${success_once} |
| Success rate | ${success_rate} |
| Positive final return | ${return_positive} |
| Positive-return rate | ${return_rate} |
| Exit code | ${eval_rc} |
| Started (UTC) | ${started_at} |
| Finished (UTC) | ${finished_at} |

## Aggregated Metrics

\`\`\`text
${final_metrics:-No aggregated metric line was found.}
\`\`\`
EOF
  log "Wrote evaluation summary: ${SUMMARY_FILE}"
}

push_summary() {
  git -C "${REPO_DIR}" add -- "${SUMMARY_REL}"
  if git -C "${REPO_DIR}" diff --cached --quiet -- "${SUMMARY_REL}"; then
    log "Evaluation summary is unchanged; no Git commit needed."
    return 0
  fi
  git -C "${REPO_DIR}" commit --only -s \
    -m "results: add LIBERO-10 step 50 evaluation" -- "${SUMMARY_REL}"

  for attempt in 1 2 3 4 5; do
    if git -C "${REPO_DIR}" push origin HEAD:main; then
      log "Evaluation summary pushed to GitHub."
      return 0
    fi
    log "GitHub push attempt ${attempt}/5 failed; retrying in 60s."
    sleep 60
  done
  log "Failed to push evaluation summary after five attempts."
  return 1
}

main() {
  local run_id result_dir eval_log tensorboard_dir started_at finished_at eval_rc

  log "Watcher started independently of VS Code."
  check_prerequisites || return 1
  if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
    log "Prerequisite check passed."
    return 0
  fi

  wait_for_checkpoint
  wait_for_training_exit

  run_id=$(date -u +'%Y%m%dT%H%M%SZ')
  result_dir=${RESULT_ROOT}/${run_id}
  eval_log=${result_dir}/eval.log
  tensorboard_dir=${result_dir}/tensorboard
  mkdir -p "${tensorboard_dir}"
  started_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  if run_evaluation "${eval_log}" "${tensorboard_dir}"; then
    eval_rc=0
  else
    eval_rc=$?
  fi
  finished_at=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

  write_summary "${eval_log}" "${eval_rc}" "${started_at}" "${finished_at}"
  push_summary
  log "Watcher finished with evaluation exit code ${eval_rc}."
  return "${eval_rc}"
}

main "$@"