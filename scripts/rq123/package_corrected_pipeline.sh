#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
PIPELINE_ROOT="${PIPELINE_ROOT:-}"
OUTPUT="${OUTPUT:-}"
if [[ -z "${PIPELINE_ROOT}" ]]; then
  echo "Set PIPELINE_ROOT to the corrected RQ1-RQ3 experiment." >&2
  exit 1
fi

command=(
  conda run -n "${ENV_NAME}" python
  scripts/rq123/corrected_pipeline.py package
  --pipeline-root "${PIPELINE_ROOT}"
)
if [[ -n "${OUTPUT}" ]]; then
  command+=(--output "${OUTPUT}")
fi
if [[ "${COMPRESS:-0}" == "1" ]]; then
  command+=(--compress)
fi
"${command[@]}"
