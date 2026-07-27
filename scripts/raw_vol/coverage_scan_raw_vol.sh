#!/usr/bin/env bash
# Scan raw-vol coverage over window/min-strike settings and select the best dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
DEVICE="${DEVICE:-cpu}"
CONFIG_PATH="${CONFIG_PATH:-configs/surface_builder/raw/generate_surface-raw-excel-rq.yaml}"
SOURCE_TIMEZONE="${SOURCE_TIMEZONE:-Europe/London}"
SCAN_TS="${SCAN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
SCAN_ROOT="${SCAN_ROOT:-data/processed/raw-excel/coverage_scan_${SCAN_TS}}"
WINDOW_CANDIDATES="${WINDOW_CANDIDATES:-3 5 10 15 30 60}"
MIN_USABLE_PAIRS="${MIN_USABLE_PAIRS:-100}"
WARN_USABLE_PAIRS="${WARN_USABLE_PAIRS:-1000}"
COVERAGE_CSV="${COVERAGE_CSV:-${SCAN_ROOT}/raw_vol_coverage_scan.csv}"
BEST_JSON="${BEST_JSON:-${SCAN_ROOT}/raw_vol_selected_dataset.json}"
UPDATE_SELECTED_LINK="${UPDATE_SELECTED_LINK:-1}"

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

run_one() {
  local window="$1"
  local min_strikes="$2"
  local run_ts="raw_vol_w${window}_s${min_strikes}_${SCAN_TS}"
  local dataset_dir="data/processed/raw-excel/${run_ts}"

  echo "Scanning window=${window}, min_strikes_per_expiry=${min_strikes}"
  python scripts/generate_surface/main.py generate_surface \
    --device "${DEVICE}" \
    --config "${CONFIG_PATH}" \
    --model raw \
    --data-range excel \
    --run-ts "${run_ts}" \
    --source-timezone "${SOURCE_TIMEZONE}" \
    --window-minutes "${window}" \
    --min-strikes-per-expiry "${min_strikes}"

  python scripts/merge_file/merge_vol.py \
    --input-dir "${dataset_dir}" \
    --source-timezone "${SOURCE_TIMEZONE}"
  python scripts/raw_vol/raw_vol_pipeline.py validate \
    --dataset-dir "${dataset_dir}" \
    --window-minutes "${window}" \
    --min-strikes-per-expiry "${min_strikes}" \
    --source-timezone "${SOURCE_TIMEZONE}" \
    --min-usable-pairs "${MIN_USABLE_PAIRS}" \
    --warn-usable-pairs "${WARN_USABLE_PAIRS}" \
    --coverage-csv "${COVERAGE_CSV}" \
    --no-fail
}

activate_env
mkdir -p "${SCAN_ROOT}"

for window in ${WINDOW_CANDIDATES}; do
  run_one "${window}" 3
done

best_usable="$(python - <<PY
import pandas as pd
from pathlib import Path
p=Path("${COVERAGE_CSV}")
df=pd.read_csv(p)
print(int(pd.to_numeric(df["usable_pairs"], errors="coerce").fillna(0).max()))
PY
)"

if [[ "${best_usable}" -lt "${MIN_USABLE_PAIRS}" ]]; then
  echo "Best min_strikes=3 usable_pairs=${best_usable}; scanning min_strikes=2"
  for window in ${WINDOW_CANDIDATES}; do
    run_one "${window}" 2
  done
fi

python scripts/raw_vol/raw_vol_pipeline.py select-best \
  --coverage-csv "${COVERAGE_CSV}" \
  --output-json "${BEST_JSON}"

if [[ "${UPDATE_SELECTED_LINK}" == "1" ]]; then
  selected_dataset="$(python - <<PY
import json
print(json.load(open("${BEST_JSON}", "r", encoding="utf-8"))["selected_dataset_dir"])
PY
)"
  selected_link="data/processed/raw-excel/rq_raw_vol_selected"
  if [[ -e "${selected_link}" && ! -L "${selected_link}" ]]; then
    echo "Selected link target exists and is not a symlink; leaving unchanged: ${selected_link}" >&2
  else
    ln -sfn "$(basename "${selected_dataset}")" "${selected_link}"
    echo "Updated selected raw-vol symlink: ${selected_link} -> $(basename "${selected_dataset}")"
  fi
fi

echo "Coverage scan CSV: ${COVERAGE_CSV}"
echo "Best selection JSON: ${BEST_JSON}"
