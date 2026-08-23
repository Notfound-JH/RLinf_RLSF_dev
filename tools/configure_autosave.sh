#!/usr/bin/env bash

set -euo pipefail

CONFIG_FILE=${RLINF_AUTOSAVE_ENV_FILE:-${HOME}/.config/rlinf-autosave/env}
EXPECTED_CONTAINER_URL=https://cloudaigcrdataeus.blob.core.windows.net/tunnel-data-code-eus
AZURE_BLOB_PREFIX=jinghe/rlinf-rlsf

printf 'Paste the Azure Blob container SAS URL (input is hidden): ' >/dev/tty
IFS= read -r -s SAS_URL </dev/tty
printf '\n' >/dev/tty

BASE_URL=${SAS_URL%%\?*}
QUERY=${SAS_URL#*\?}
SAS_PERMISSIONS=$(printf '%s' "${QUERY}" | tr '&' '\n' | sed -n 's/^sp=//p')

if [[ "${BASE_URL%/}" != "${EXPECTED_CONTAINER_URL}" ]]; then
  echo "Expected a SAS URL for ${EXPECTED_CONTAINER_URL}." >&2
  exit 1
fi
for permission in r c w l; do
  if [[ "${SAS_PERMISSIONS}" != *"${permission}"* ]]; then
    echo "SAS permission '${permission}' is required (expected at least rcwl)." >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${CONFIG_FILE}")"
umask 077
{
  printf 'AZURE_BLOB_SAS_URL=%q\n' "${SAS_URL}"
  printf 'AZURE_BLOB_PREFIX=%q\n' "${AZURE_BLOB_PREFIX}"
  printf 'AUTOSAVE_BLOB_ENABLED=%q\n' "1"
  printf 'AUTOSAVE_INTERVAL_SECONDS=%q\n' "300"
  printf 'AUTOSAVE_PATHS=%q\n' "logs:results:data/models:models_ckpts:checkpoints"
  printf 'AUTOSAVE_EXTERNAL_RESULTS_DIR=%q\n' "/home/aiscuser/works/results"
} >"${CONFIG_FILE}"
chmod 600 "${CONFIG_FILE}"
unset SAS_URL QUERY

echo "Saved autosave configuration to ${CONFIG_FILE} (mode 600)."
echo "Azure destination: ${EXPECTED_CONTAINER_URL}/${AZURE_BLOB_PREFIX}/"
