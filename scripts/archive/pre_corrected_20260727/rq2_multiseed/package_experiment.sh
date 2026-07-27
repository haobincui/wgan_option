#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/rq2_multiseed_textbase_20260627}"
PACKAGE_PATH="${PACKAGE_PATH:-${EXP_ROOT}.zip}"

if [[ ! -d "${EXP_ROOT}" ]]; then
  echo "Experiment root not found: ${EXP_ROOT}" >&2
  exit 1
fi

if [[ "${PACKAGE_PATH}" = /* ]]; then
  PACKAGE_ABS="${PACKAGE_PATH}"
else
  PACKAGE_ABS="${REPO_ROOT}/${PACKAGE_PATH}"
fi

mkdir -p "$(dirname "${PACKAGE_ABS}")"
if command -v zip >/dev/null 2>&1; then
  (cd "$(dirname "${EXP_ROOT}")" && zip -r "${PACKAGE_ABS}" "$(basename "${EXP_ROOT}")")
  echo "Wrote zip package: ${PACKAGE_ABS}"
else
  tar_path="${PACKAGE_ABS%.zip}.tar.gz"
  tar -czf "${tar_path}" -C "$(dirname "${EXP_ROOT}")" "$(basename "${EXP_ROOT}")"
  echo "zip command not available; wrote tar package: ${tar_path}"
fi
