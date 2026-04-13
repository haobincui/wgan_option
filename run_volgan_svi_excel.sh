#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ENV_NAME="${ENV_NAME:-py312}"
TRAIN_CONFIG_PATH="${1:-${TRAIN_CONFIG_PATH:-}}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
WORK_DIR="${WORK_DIR:-logs/volgan/${RUN_TS}}"
LOG_FILE="${WORK_DIR}/run.log"

activate_env() {
  if command -v conda >/dev/null 2>&1; then
    local conda_base
    conda_base="$(conda info --base)"
    # shellcheck disable=SC1090
    source "${conda_base}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    return
  fi
  # shellcheck disable=SC1091
  source activate "${ENV_NAME}"
}

if [[ -z "${TRAIN_CONFIG_PATH}" ]]; then
  echo "Usage: $0 <train-config.yaml>" >&2
  echo "Or set TRAIN_CONFIG_PATH=/path/to/train-config.yaml" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_CONFIG_PATH}" ]]; then
  echo "Train config not found: ${TRAIN_CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"

activate_env
python -V >/dev/null

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Root dir: ${SCRIPT_DIR}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Train config: ${TRAIN_CONFIG_PATH}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Work dir: ${WORK_DIR}"
} | tee "${LOG_FILE}"

python -V | tee -a "${LOG_FILE}"
python scripts/volgan/main.py pipeline --train-config "${TRAIN_CONFIG_PATH}" | tee -a "${LOG_FILE}"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Log file: ${LOG_FILE}"
