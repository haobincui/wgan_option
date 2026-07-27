#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/raw_vol_rq1_rq2_$(date -u +%Y%m%d)}"
DATA_PATH="${DATA_PATH:-data/processed/raw-excel/rq_raw_vol_selected/merged_vol_rq2_text.xlsx}"
SEEDS="${SEEDS_CSV:-42,101,202,303,404}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/rq2_multiseed/rq2_multiseed.py collect \
  --exp-root "${EXP_ROOT}" \
  --seeds "${SEEDS}" \
  --data-path "${DATA_PATH}"
