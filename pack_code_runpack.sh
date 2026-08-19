#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(dirname "${SCRIPT_DIR}")
REPO_NAME=$(basename "${SCRIPT_DIR}")
OUTPUT=${1:-${PARENT_DIR}/${REPO_NAME}-code-runpack.tar.gz}

case "${OUTPUT}" in
  /*) ;;
  *) OUTPUT=$(cd "$(dirname "${OUTPUT}")" && pwd)/$(basename "${OUTPUT}") ;;
esac

mkdir -p "$(dirname "${OUTPUT}")"
rm -f "${OUTPUT}"

echo "Creating ${OUTPUT}"
tar -C "${PARENT_DIR}" -czf "${OUTPUT}" \
  --exclude="${REPO_NAME}/.git" \
  --exclude="${REPO_NAME}/.venv" \
  --exclude="${REPO_NAME}/docker" \
  --exclude="${REPO_NAME}/Dockerfile" \
  --exclude="${REPO_NAME}/run_docker.sh" \
  --exclude="${REPO_NAME}/run_docker_train.sh" \
  --exclude="${REPO_NAME}/run_docker_gnode3.sh" \
  --exclude="${REPO_NAME}/models_ckpts" \
  --exclude="${REPO_NAME}/data" \
  --exclude="${REPO_NAME}/logs" \
  --exclude="${REPO_NAME}/results" \
  --exclude="${REPO_NAME}/assets" \
  --exclude="${REPO_NAME}/**/__pycache__" \
  --exclude="${REPO_NAME}/**/.pytest_cache" \
  --exclude="${REPO_NAME}/**/.ruff_cache" \
  --exclude="${REPO_NAME}/**/*.log" \
  "${REPO_NAME}"

echo "Created: ${OUTPUT}"
ls -lh "${OUTPUT}"
