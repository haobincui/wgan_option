#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_film_wgan_svi_excel.sh"

TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-configs/film_wgan/train_lp_exp_F5.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/training/film_wgan_notext/svi-excel}"

ENV_NAME="${ENV_NAME:-py312}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
RUN_TS="${RUN_TS:-film_notext_shortatm_$(date -u +%Y%m%d-%H%M%S)}"
WORK_DIR="${WORK_DIR:-logs/film_wgan_notext/${RUN_TS}}"

export ENV_NAME CUDA_VISIBLE_DEVICES RUN_TS WORK_DIR

if [[ ! -f "${BASE_RUNNER}" ]]; then
  echo "Base runner not found: ${BASE_RUNNER}" >&2
  exit 1
fi

# RQ1 no-text baseline aligned with the current best short-ATM FiLM setup.
# text_embedding_mode=none keeps the FiLM architecture and feeds a fixed zero text channel.
bash "${BASE_RUNNER}" "${TRAIN_CONFIG_PATH}" \
  --set "text_embedding_mode=none" \
  --set "normalize_text_embedding=false" \
  --set "output_root=${OUTPUT_ROOT}" \
  --set "lambda_atm_short=30.0" \
  --set "atm_short_range=0.06" \
  --set "atm_short_max_days=90.0" \
  "$@"
