#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
PIPELINE_ROOT="${PIPELINE_ROOT:-}"
LATEST_PATH="outputs/experiments/rq123_corrected_latest.env"
if [[ -z "${PIPELINE_ROOT}" && -f "${LATEST_PATH}" ]]; then
  PIPELINE_ROOT="$(awk -F= '$1=="PIPELINE_ROOT" {print substr($0, index($0, "=")+1)}' "${LATEST_PATH}")"
fi
if [[ -z "${PIPELINE_ROOT}" || ! -d "${PIPELINE_ROOT}" ]]; then
  echo "Set PIPELINE_ROOT to a corrected RQ1-RQ3 experiment." >&2
  exit 1
fi

conda run -n "${ENV_NAME}" python scripts/rq123/corrected_pipeline.py monitor \
  --pipeline-root "${PIPELINE_ROOT}"

log_path="$(find "${PIPELINE_ROOT}/logs/driver" -maxdepth 1 -type f \
  -name 'corrected_pipeline_*.log' -printf '%T@ %p\n' 2>/dev/null |
  sort -n | tail -n 1 | cut -d' ' -f2-)"
if [[ -n "${log_path}" ]]; then
  echo
  echo "Latest log: ${log_path}"
  tail -40 "${log_path}"
fi
