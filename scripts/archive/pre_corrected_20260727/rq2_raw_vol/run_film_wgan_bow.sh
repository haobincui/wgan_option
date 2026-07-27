#!/usr/bin/env bash
# Launch the RQ2 BoW FiLM WGAN on raw-vol interpolation surfaces.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-configs/film_wgan/train_raw_vol_textbase.yaml}"
DATA_PATH="${DATA_PATH:-data/processed/raw-excel/rq_raw_vol_selected/merged_vol_rq2_text.xlsx}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/training/film_wgan_raw_vol/bow}"
ENV_NAME="${ENV_NAME:-py312}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
RUN_TS="${RUN_TS:-raw_vol_bow_$(date -u +%Y%m%d-%H%M%S)}"
WORK_DIR="${WORK_DIR:-logs/film_wgan_raw_vol/${RUN_TS}}"
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

[[ -f "${TRAIN_CONFIG_PATH}" ]] || { echo "Train config not found: ${TRAIN_CONFIG_PATH}" >&2; exit 1; }
[[ -f "${DATA_PATH}" ]] || { echo "Raw-vol RQ workbook not found: ${DATA_PATH}" >&2; exit 1; }
mkdir -p "${WORK_DIR}"

activate_env
nohup python scripts/film_wgan/main.py train --config "${TRAIN_CONFIG_PATH}" \
  --set "data_path=${DATA_PATH}" \
  --set "text_embedding_mode=bow" \
  --set "normalize_text_embedding=true" \
  --set "output_root=${OUTPUT_ROOT}" \
  "$@" >"${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

echo "Raw-vol RQ2 BoW FiLM WGAN launched"
echo "PID: ${PID}"
echo "Log: ${LOG_FILE}"
