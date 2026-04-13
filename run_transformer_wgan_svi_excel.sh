#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ENV_NAME="${ENV_NAME:-py312}"
TRAIN_CONFIG_PATH="${1:-${TRAIN_CONFIG_PATH:-}}"
shift $(( $# > 0 ? 1 : 0 ))

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
WORK_DIR="${WORK_DIR:-logs/transformer_wgan/${RUN_TS}}"
LOG_FILE="${WORK_DIR}/run.log"
PID_FILE="${WORK_DIR}/run.pid"

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
  echo "Usage: $0 <train-config.yaml> [extra-cli-args...]" >&2
  echo "Example: $0 configs/transformer_wgan/train_lp.yaml" >&2
  exit 1
fi

if [[ ! -f "${TRAIN_CONFIG_PATH}" ]]; then
  echo "Train config not found: ${TRAIN_CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"

activate_env
python -V >/dev/null

nohup python scripts/transformer_wgan/main.py train --config "${TRAIN_CONFIG_PATH}" "$@" >"${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "=========================================="
echo "  Transformer WGAN job launched"
echo "=========================================="
echo "  Train config : ${TRAIN_CONFIG_PATH}"
echo "  PID          : ${PID}"
echo "  PID file     : ${PID_FILE}"
echo "  Log file     : ${LOG_FILE}"
echo "=========================================="
echo ""
echo "Follow logs:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "Check status:"
echo "  kill -0 ${PID} 2>/dev/null && echo running || echo stopped"
echo ""
echo "Stop training:"
echo "  kill ${PID}"
