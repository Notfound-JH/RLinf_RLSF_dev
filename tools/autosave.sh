#!/usr/bin/env bash

set -uo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_FILE=${RLINF_AUTOSAVE_ENV_FILE:-${HOME}/.config/rlinf-autosave/env}
STATE_DIR=${XDG_STATE_HOME:-${HOME}/.local/state}/rlinf-autosave
LOG_FILE=${STATE_DIR}/autosave.log
SESSION_NAME=${RLINF_AUTOSAVE_SESSION:-rlinf_autosave}
MAX_GIT_FILE_BYTES=${MAX_GIT_FILE_BYTES:-90000000}
EXPECTED_CONTAINER_URL=https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus
export PATH="${HOME}/.local/bin:${PATH}"

timestamp() {
  date -u +'%Y-%m-%dT%H:%M:%SZ'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

load_config() {
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    log "Missing ${CONFIG_FILE}; run tools/configure_autosave.sh first."
    return 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  set +a

  if [[ -z "${AZURE_BLOB_SAS_URL:-}" ]]; then
    log "AZURE_BLOB_SAS_URL is not configured."
    return 1
  fi
  if [[ "${AZURE_BLOB_SAS_URL}" != *\?* ]]; then
    log "AZURE_BLOB_SAS_URL does not contain a SAS query string."
    return 1
  fi
}

save_code() {
  local path size

  if ! git -C "${REPO_DIR}" add --all; then
    log "Git staging failed."
    return 1
  fi

  while IFS= read -r -d '' path; do
    [[ -f "${REPO_DIR}/${path}" ]] || continue
    size=$(stat -c '%s' "${REPO_DIR}/${path}")
    if (( size > MAX_GIT_FILE_BYTES )); then
      git -C "${REPO_DIR}" restore --staged -- "${path}"
      log "Skipped oversized Git file (${size} bytes): ${path}"
    fi
  done < <(git -C "${REPO_DIR}" diff --cached --name-only --diff-filter=ACMR -z)

  if git -C "${REPO_DIR}" diff --cached --quiet; then
    log "Code is already saved."
    return 0
  fi

  if ! git -C "${REPO_DIR}" commit -m "autosave: $(timestamp)"; then
    log "Git commit failed."
    return 1
  fi
  if ! git -C "${REPO_DIR}" push origin HEAD; then
    log "Git push failed; the commit remains saved locally for the next retry."
    return 1
  fi

  log "Code committed and pushed to origin."
}

sync_artifacts() {
  local base_url sas_token host_id prefix relative source target experiment checkpoint signature signature_file
  local -a paths

  load_config || return 1
  if [[ "${AUTOSAVE_BLOB_ENABLED:-1}" != "1" ]]; then
    log "Azure Blob sync is disabled."
    return 0
  fi
  if ! command -v azcopy >/dev/null 2>&1; then
    log "azcopy is not installed or not on PATH."
    return 1
  fi

  base_url=${AZURE_BLOB_SAS_URL%%\?*}
  sas_token=${AZURE_BLOB_SAS_URL#*\?}
  prefix=${AZURE_BLOB_PREFIX:-jinghe/rlinf-rlsf}
  if [[ "${base_url%/}" != "${EXPECTED_CONTAINER_URL}" ]]; then
    log "Refusing to sync outside ${EXPECTED_CONTAINER_URL}."
    return 1
  fi
  if [[ "${prefix}" != "jinghe" && "${prefix}" != jinghe/* ]]; then
    log "Refusing to sync outside the jinghe prefix."
    return 1
  fi
  host_id=$(hostname -s | tr -cd 'A-Za-z0-9._-')
  IFS=':' read -r -a paths <<<"${AUTOSAVE_PATHS:-logs:results:data/models:models_ckpts:checkpoints}"

  for relative in "${paths[@]}"; do
    if [[ -z "${relative}" || "${relative}" == /* || "${relative}" == *..* ]]; then
      log "Skipped unsafe autosave path: ${relative}"
      continue
    fi
    source=${REPO_DIR}/${relative}
    [[ -d "${source}" ]] || continue
    target="${base_url%/}/${prefix}/${host_id}/${relative}?${sas_token}"
    if azcopy sync "${source}" "${target}" \
      --recursive=true \
      --delete-destination=false \
      --output-level=essential; then
      log "Synced ${relative} to Azure Blob."
    else
      log "Azure Blob sync failed for ${relative}."
      return 1
    fi
  done

  source=${AUTOSAVE_EXTERNAL_RESULTS_DIR:-/home/aiscuser/work/results}
  if [[ "${source}" != "/home/aiscuser/work/results" ]]; then
    log "Refusing unexpected external results path: ${source}"
    return 1
  fi
  [[ -d "${source}" ]] || return 0

  mkdir -p "${STATE_DIR}/checkpoint-signatures"
  for experiment in "${source}"/*; do
    [[ -d "${experiment}/checkpoints" ]] || continue
    checkpoint=$(find "${experiment}/checkpoints" -mindepth 1 -maxdepth 1 -type d \
      -name 'global_step_*' -printf '%f\n' | sort -V | tail -n 1)
    [[ -n "${checkpoint}" ]] || continue
    source=${experiment}/checkpoints/${checkpoint}
    signature=$(find "${source}" -type f -printf '%P\t%s\t%T@\n' | sort | sha256sum | cut -d' ' -f1)
    signature_file=${STATE_DIR}/checkpoint-signatures/$(basename "${experiment}").sha256
    if [[ -f "${signature_file}" ]] && [[ "$(cat "${signature_file}")" == "${checkpoint} ${signature}" ]]; then
      continue
    fi

    target="${base_url%/}/${prefix}/${host_id}/training-results/$(basename "${experiment}")/checkpoints/${checkpoint}?${sas_token}"
    if azcopy sync "${source}" "${target}" \
      --recursive=true \
      --delete-destination=false \
      --output-level=essential; then
      printf '%s %s\n' "${checkpoint}" "${signature}" >"${signature_file}"
      log "Synced latest checkpoint: $(basename "${experiment}")/${checkpoint}."
    else
      log "Azure Blob sync failed for $(basename "${experiment}")/${checkpoint}."
      return 1
    fi
  done
}

run_once() {
  save_code || true
  sync_artifacts || true
}

run_daemon() {
  local interval=${AUTOSAVE_INTERVAL_SECONDS:-300}

  mkdir -p "${STATE_DIR}"
  exec >>"${LOG_FILE}" 2>&1
  log "Autosave daemon started for ${REPO_DIR}."
  while true; do
    run_once
    sleep "${interval}"
  done
}

start_daemon() {
  load_config || return 1
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    log "Autosave is already running in tmux session ${SESSION_NAME}."
    return 0
  fi
  mkdir -p "${STATE_DIR}"
  tmux new-session -d -s "${SESSION_NAME}" "exec bash '${REPO_DIR}/tools/autosave.sh' daemon"
  log "Autosave started in tmux session ${SESSION_NAME}."
  log "Status: bash tools/autosave.sh status"
}

show_status() {
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "running (${SESSION_NAME})"
  else
    echo "stopped"
  fi
  if [[ -f "${LOG_FILE}" ]]; then
    tail -n 20 "${LOG_FILE}"
  fi
}

case "${1:-}" in
  code) save_code ;;
  logs) sync_artifacts ;;
  once) run_once ;;
  daemon) run_daemon ;;
  start) start_daemon ;;
  status) show_status ;;
  stop)
    tmux kill-session -t "${SESSION_NAME}" 2>/dev/null || true
    log "Autosave stopped."
    ;;
  *)
    echo "Usage: $0 {start|status|stop|once|code|logs}" >&2
    exit 2
    ;;
esac