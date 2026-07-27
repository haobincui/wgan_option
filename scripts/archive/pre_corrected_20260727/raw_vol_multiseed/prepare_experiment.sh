#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/raw_vol_rq1_rq2_$(date -u +%Y%m%d)}"
CONFIG_PATH="${CONFIG_PATH:-configs/film_wgan/train_raw_vol_textbase.yaml}"
DATA_PATH="${DATA_PATH:-data/processed/raw-excel/rq_raw_vol_selected/merged_vol_rq2_text.xlsx}"
FEATURE_DIR="${FEATURE_DIR:-data/processed/text_features/rq2/20260625-075653}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/rq2_multiseed/rq2_multiseed.py prepare \
  --exp-root "${EXP_ROOT}" \
  --frozen-config "${CONFIG_PATH}" \
  --data-path "${DATA_PATH}" \
  --feature-dir "${FEATURE_DIR}" \
  --experiment-title "Raw-Vol RQ1/RQ2 Multi-Seed Controlled Experiment"
