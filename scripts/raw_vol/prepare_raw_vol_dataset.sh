#!/usr/bin/env bash
# Build one raw-vol interpolation dataset from raw option IV points.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
DEVICE="${DEVICE:-cpu}"
CONFIG_PATH="${CONFIG_PATH:-configs/surface_builder/raw/generate_surface-raw-excel-rq.yaml}"
OPTION_DATA_GLOB="${OPTION_DATA_GLOB:-data/raw/option_data/0#TY+/0#TY+_202[23]-*.csv.gz}"
TARGET_XLSX="${TARGET_XLSX:-data/raw/text_embedding/news_with_openai_embeddings_large.xlsx}"
RATE_CURVE_PATH="${RATE_CURVE_PATH:-data/reference/us_treasury_par_yield_curve_2022_2023.csv}"
SOURCE_TIMEZONE="${SOURCE_TIMEZONE:-Europe/London}"
PUBLICATION_AVAILABILITY_LAG_MINUTES="${PUBLICATION_AVAILABILITY_LAG_MINUTES:-0}"
RUN_TS="${RUN_TS:-raw_vol_$(date -u +%Y%m%d-%H%M%S)}"
WINDOW_MINUTES="${WINDOW_MINUTES:-5}"
MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY:-2}"
OPTION_FILTER_MODE="${OPTION_FILTER_MODE:-otm_preferred_itm_fallback}"
MAX_ITM_MONEYNESS_DISTANCE="${MAX_ITM_MONEYNESS_DISTANCE:-0.05}"
MIN_USABLE_PAIRS="${MIN_USABLE_PAIRS:-100}"
WARN_USABLE_PAIRS="${WARN_USABLE_PAIRS:-1000}"
NUMERICAL_THREADS_PER_WORKER="${NUMERICAL_THREADS_PER_WORKER:-1}"
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

# ProcessPool workers inherit numerical-library defaults. Cap each worker so
# calibration_workers does not multiply into thousands of BLAS/OpenMP threads.
export OMP_NUM_THREADS="${NUMERICAL_THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${NUMERICAL_THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${NUMERICAL_THREADS_PER_WORKER}"
export NUMEXPR_NUM_THREADS="${NUMERICAL_THREADS_PER_WORKER}"

python scripts/generate_surface/main.py generate_surface \
  --device "${DEVICE}" \
  --config "${CONFIG_PATH}" \
  --input-glob "${OPTION_DATA_GLOB}" \
  --model raw \
  --data-range excel \
  --run-ts "${RUN_TS}" \
  --target-xlsx "${TARGET_XLSX}" \
  --rate-curve-path "${RATE_CURVE_PATH}" \
  --source-timezone "${SOURCE_TIMEZONE}" \
  --publication-availability-lag-minutes "${PUBLICATION_AVAILABILITY_LAG_MINUTES}" \
  --window-minutes "${WINDOW_MINUTES}" \
  --min-strikes-per-expiry "${MIN_STRIKES_PER_EXPIRY}" \
  --option-filter-mode "${OPTION_FILTER_MODE}" \
  --max-itm-moneyness-distance "${MAX_ITM_MONEYNESS_DISTANCE}"

python scripts/merge_file/merge_vol.py \
  --input-dir "${DATASET_DIR}" \
  --source-timezone "${SOURCE_TIMEZONE}" \
  --publication-availability-lag-minutes "${PUBLICATION_AVAILABILITY_LAG_MINUTES}"

python scripts/raw_vol/raw_vol_pipeline.py validate \
  --dataset-dir "${DATASET_DIR}" \
  --window-minutes "${WINDOW_MINUTES}" \
  --min-strikes-per-expiry "${MIN_STRIKES_PER_EXPIRY}" \
  --source-timezone "${SOURCE_TIMEZONE}" \
  --publication-availability-lag-minutes "${PUBLICATION_AVAILABILITY_LAG_MINUTES}" \
  --min-usable-pairs "${MIN_USABLE_PAIRS}" \
  --warn-usable-pairs "${WARN_USABLE_PAIRS}"

echo "Raw-vol dataset ready: ${DATASET_DIR}"
