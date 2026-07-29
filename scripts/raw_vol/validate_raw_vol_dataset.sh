#!/usr/bin/env bash
# Validate one raw-vol merged workbook and write raw_vol_dataset_validation.json.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
DATASET_DIR="${1:-${DATASET_DIR:-}}"
MIN_USABLE_PAIRS="${MIN_USABLE_PAIRS:-100}"
WARN_USABLE_PAIRS="${WARN_USABLE_PAIRS:-1000}"
WINDOW_MINUTES="${WINDOW_MINUTES:-5}"
MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY:-2}"

if [[ -z "${DATASET_DIR}" ]]; then
  echo "Usage: $0 <data/processed/raw-excel/run_dir>" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  conda run -n "${ENV_NAME}" python scripts/raw_vol/raw_vol_pipeline.py validate \
    --dataset-dir "${DATASET_DIR}" \
    --window-minutes "${WINDOW_MINUTES}" \
    --min-strikes-per-expiry "${MIN_STRIKES_PER_EXPIRY}" \
    --min-usable-pairs "${MIN_USABLE_PAIRS}" \
    --warn-usable-pairs "${WARN_USABLE_PAIRS}"
else
  python scripts/raw_vol/raw_vol_pipeline.py validate \
    --dataset-dir "${DATASET_DIR}" \
    --window-minutes "${WINDOW_MINUTES}" \
    --min-strikes-per-expiry "${MIN_STRIKES_PER_EXPIRY}" \
    --min-usable-pairs "${MIN_USABLE_PAIRS}" \
    --warn-usable-pairs "${WARN_USABLE_PAIRS}"
fi
