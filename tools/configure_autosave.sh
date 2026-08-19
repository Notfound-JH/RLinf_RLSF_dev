#!/usr/bin/env bash

set -euo pipefail

CONFIG_FILE=${RLINF_AUTOSAVE_ENV_FILE:-${HOME}/.config/rlinf-autosave/env}

printf 'Paste the Azure Blob container SAS URL (input is hidden): ' >/dev/tty
IFS= read -r -s SAS_URL </dev/tty
printf '\n' >/dev/tty

case "${SAS_URL}" in
  https://*.blob.core.windows.net/*\?*) ;;
  *)
    echo "Expected an HTTPS Azure Blob container SAS URL." >&2
    exit 1
    ;;
esac

mkdir -p "$(dirname "${CONFIG_FILE}")"
umask 077
{
  printf 'AZURE_BLOB_SAS_URL=%q\n' "${SAS_URL}"
  printf 'AZURE_BLOB_PREFIX=%q\n' "rlinf-rlsf"
  printf 'AUTOSAVE_BLOB_ENABLED=%q\n' "1"
  printf 'AUTOSAVE_INTERVAL_SECONDS=%q\n' "300"
  printf 'AUTOSAVE_PATHS=%q\n' "logs:results:denserwd/data/models:models_ckpts:checkpoints"
} >"${CONFIG_FILE}"
chmod 600 "${CONFIG_FILE}"
unset SAS_URL

echo "Saved autosave configuration to ${CONFIG_FILE} (mode 600)."
