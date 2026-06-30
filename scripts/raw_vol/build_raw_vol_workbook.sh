#!/usr/bin/env bash
# Rebuild merged_vol.xlsx for an existing raw-vol surface generation directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
DATASET_DIR="${1:-${DATASET_DIR:-}}"
MIN_USABLE_PAIRS="${MIN_USABLE_PAIRS:-100}"
WARN_USABLE_PAIRS="${WARN_USABLE_PAIRS:-1000}"

if [[ -z "${DATASET_DIR}" ]]; then
  echo "Usage: $0 <data/processed/raw-excel/run_dir>" >&2
  exit 1
fi

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
python scripts/merge_file/merge_vol.py --input-dir "${DATASET_DIR}"
python scripts/raw_vol/raw_vol_pipeline.py validate \
  --dataset-dir "${DATASET_DIR}" \
  --min-usable-pairs "${MIN_USABLE_PAIRS}" \
  --warn-usable-pairs "${WARN_USABLE_PAIRS}"
