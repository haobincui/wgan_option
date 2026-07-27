#!/usr/bin/env bash
# Launch the RQ1 no-text FiLM WGAN short-ATM baseline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-configs/film_wgan/train_lp_exp_F5.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/training/film_wgan_notext/svi-excel}"

ENV_NAME="${ENV_NAME:-py312}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
RUN_TS="${RUN_TS:-film_notext_shortatm_$(date -u +%Y%m%d-%H%M%S)}"
WORK_DIR="${WORK_DIR:-logs/film_wgan_notext/${RUN_TS}}"
LOG_FILE="${WORK_DIR}/run.log"
PID_FILE="${WORK_DIR}/run.pid"

export ENV_NAME CUDA_VISIBLE_DEVICES RUN_TS WORK_DIR

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

if [[ ! -f "${TRAIN_CONFIG_PATH}" ]]; then
  echo "Train config not found: ${TRAIN_CONFIG_PATH}" >&2
  exit 1
fi

mkdir -p "${WORK_DIR}"

# RQ1 no-text baseline aligned with the current best short-ATM FiLM setup.
# text_embedding_mode=none keeps the FiLM architecture and feeds a fixed zero text channel.
activate_env
python -V >/dev/null

nohup python scripts/film_wgan/main.py train --config "${TRAIN_CONFIG_PATH}" \
  --set "text_embedding_mode=none" \
  --set "normalize_text_embedding=false" \
  --set "output_root=${OUTPUT_ROOT}" \
  --set "lambda_atm_short=30.0" \
  --set "atm_short_range=0.06" \
  --set "atm_short_max_days=90.0" \
  "$@" >"${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "=========================================="
echo "  RQ1 no-text FiLM WGAN job launched"
echo "=========================================="
echo "  Train config : ${TRAIN_CONFIG_PATH}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  CUDA device  : ${CUDA_VISIBLE_DEVICES}"
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
