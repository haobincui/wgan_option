#!/usr/bin/env bash
# Build one raw-vol interpolation dataset from raw option IV points.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
DEVICE="${DEVICE:-cpu}"
CONFIG_PATH="${CONFIG_PATH:-configs/surface_builder/raw/generate_surface-raw-excel-rq.yaml}"
SOURCE_TIMEZONE="${SOURCE_TIMEZONE:-Europe/London}"
RUN_TS="${RUN_TS:-raw_vol_$(date -u +%Y%m%d-%H%M%S)}"
WINDOW_MINUTES="${WINDOW_MINUTES:-3}"
MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY:-3}"
MIN_USABLE_PAIRS="${MIN_USABLE_PAIRS:-100}"
WARN_USABLE_PAIRS="${WARN_USABLE_PAIRS:-1000}"
DATASET_DIR="data/processed/raw-excel/${RUN_TS}"

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

activate_env

python scripts/generate_surface/main.py generate_surface \
  --device "${DEVICE}" \
  --config "${CONFIG_PATH}" \
  --model raw \
  --data-range excel \
  --run-ts "${RUN_TS}" \
  --source-timezone "${SOURCE_TIMEZONE}" \
  --window-minutes "${WINDOW_MINUTES}" \
  --min-strikes-per-expiry "${MIN_STRIKES_PER_EXPIRY}"

python scripts/merge_file/merge_vol.py \
  --input-dir "${DATASET_DIR}" \
  --source-timezone "${SOURCE_TIMEZONE}"

python scripts/raw_vol/raw_vol_pipeline.py validate \
  --dataset-dir "${DATASET_DIR}" \
  --window-minutes "${WINDOW_MINUTES}" \
  --min-strikes-per-expiry "${MIN_STRIKES_PER_EXPIRY}" \
  --source-timezone "${SOURCE_TIMEZONE}" \
  --min-usable-pairs "${MIN_USABLE_PAIRS}" \
  --warn-usable-pairs "${WARN_USABLE_PAIRS}"

echo "Raw-vol dataset ready: ${DATASET_DIR}"
