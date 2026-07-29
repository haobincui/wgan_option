#!/usr/bin/env bash
# Build the relaxed news-time dataset and optionally run the full RQ1-RQ3 matrix.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
ENV_NAME="${ENV_NAME:-py312}"
INDEX_ID="${INDEX_ID:-raw_2022_2023_london_itm5_v1}"
INDEX_DIR="${INDEX_DIR:-data/processed/raw-market-index/${INDEX_ID}}"
DATASET_DIR="${DATASET_DIR:-data/processed/raw-excel-relaxed/${RUN_TS}}"
PIPELINE_ROOT="${PIPELINE_ROOT:-outputs/experiments/rq123_relaxed_time_${RUN_TS}}"
OPTION_DATA_GLOB="${OPTION_DATA_GLOB:-data/raw/option_data/0#TY+/0#TY+_202[23]-*.csv.gz}"
NEWS_WORKBOOK="${NEWS_WORKBOOK:-data/raw/text_embedding/news_with_openai_embeddings_large.xlsx}"
FEATURE_ROOT="${FEATURE_ROOT:-data/processed/text_features/rq2/20260625-075653}"
STRICT_BASELINE_WORKBOOK="${STRICT_BASELINE_WORKBOOK:-data/processed/raw-excel/rq123_corrected_lag0_s2_itm5_20260728-114931/merged_vol.xlsx}"
RAW_CONFIG="${RAW_CONFIG:-configs/surface_builder/raw/generate_surface-raw-market-index.yaml}"
RATE_CURVE_PATH="${RATE_CURVE_PATH:-data/reference/us_treasury_par_yield_curve_2022_2023.csv}"
SOURCE_TIMEZONE="${SOURCE_TIMEZONE:-Europe/London}"
PUBLICATION_AVAILABILITY_LAG_MINUTES="${PUBLICATION_AVAILABILITY_LAG_MINUTES:-0}"
WINDOW_MINUTES="${WINDOW_MINUTES:-5}"
INTRADAY_TOLERANCE_MINUTES="${INTRADAY_TOLERANCE_MINUTES:-15}"
MAX_SESSION_SHIFT_MINUTES="${MAX_SESSION_SHIFT_MINUTES:-4320}"
CALIBRATION_WORKERS="${CALIBRATION_WORKERS:-32}"
MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY:-2}"
MIN_EXPIRIES_PER_MINUTE="${MIN_EXPIRIES_PER_MINUTE:-2}"
OPTION_FILTER_MODE="${OPTION_FILTER_MODE:-otm_preferred_itm_fallback}"
MAX_ITM_MONEYNESS_DISTANCE="${MAX_ITM_MONEYNESS_DISTANCE:-0.05}"
GPU_IDS="${GPU_IDS:-0 1}"
RUNS_PER_GPU="${RUNS_PER_GPU:-2}"
RUN_DOWNSTREAM="${RUN_DOWNSTREAM:-1}"
KEEP_BATCH_ARTIFACTS="${KEEP_BATCH_ARTIFACTS:-0}"

PYTHON=(conda run --no-capture-output -n "${ENV_NAME}" python)

mkdir -p "${DATASET_DIR}" "${PIPELINE_ROOT}/inputs"
{
  printf 'run_ts=%s\n' "${RUN_TS}"
  printf 'index_dir=%s\n' "${INDEX_DIR}"
  printf 'dataset_dir=%s\n' "${DATASET_DIR}"
  printf 'pipeline_root=%s\n' "${PIPELINE_ROOT}"
  printf 'source_timezone=%s\n' "${SOURCE_TIMEZONE}"
  printf 'publication_availability_lag_minutes=%s\n' \
    "${PUBLICATION_AVAILABILITY_LAG_MINUTES}"
  printf 'window_minutes=%s\n' "${WINDOW_MINUTES}"
  printf 'intraday_tolerance_minutes=%s\n' \
    "${INTRADAY_TOLERANCE_MINUTES}"
  printf 'max_session_shift_minutes=%s\n' \
    "${MAX_SESSION_SHIFT_MINUTES}"
  printf 'calibration_workers=%s\n' "${CALIBRATION_WORKERS}"
  printf 'run_downstream=%s\n' "${RUN_DOWNSTREAM}"
} >"${PIPELINE_ROOT}/inputs/relaxed_time_parameters.env"

index_args=(
  scripts/raw_vol/relaxed_time_pipeline.py run
  --index-dir "${INDEX_DIR}"
  --output-dir "${DATASET_DIR}"
  --input-glob "${OPTION_DATA_GLOB}"
  --config "${RAW_CONFIG}"
  --news-xlsx "${NEWS_WORKBOOK}"
  --strict-baseline-workbook "${STRICT_BASELINE_WORKBOOK}"
  --rate-curve-path "${RATE_CURVE_PATH}"
  --source-timezone "${SOURCE_TIMEZONE}"
  --publication-availability-lag-minutes
  "${PUBLICATION_AVAILABILITY_LAG_MINUTES}"
  --window-minutes "${WINDOW_MINUTES}"
  --intraday-tolerance-minutes "${INTRADAY_TOLERANCE_MINUTES}"
  --max-session-shift-minutes "${MAX_SESSION_SHIFT_MINUTES}"
  --min-strikes-per-expiry "${MIN_STRIKES_PER_EXPIRY}"
  --min-expiries-per-minute "${MIN_EXPIRIES_PER_MINUTE}"
  --option-filter-mode "${OPTION_FILTER_MODE}"
  --max-itm-moneyness-distance "${MAX_ITM_MONEYNESS_DISTANCE}"
  --calibration-workers "${CALIBRATION_WORKERS}"
)
if [[ "${KEEP_BATCH_ARTIFACTS}" == "1" ]]; then
  index_args+=(--keep-batch-artifacts)
fi
"${PYTHON[@]}" "${index_args[@]}"

ENV_NAME="${ENV_NAME}" \
FEATURE_DIR="${FEATURE_ROOT}" \
SOURCE_TIMEZONE="${SOURCE_TIMEZONE}" \
PUBLICATION_AVAILABILITY_LAG_MINUTES="${PUBLICATION_AVAILABILITY_LAG_MINUTES}" \
WINDOW_MINUTES="${WINDOW_MINUTES}" \
MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY}" \
  bash scripts/raw_vol/enrich_raw_vol_rq2_text.sh "${DATASET_DIR}"

"${PYTHON[@]}" scripts/raw_vol/raw_vol_pipeline.py validate \
  --dataset-dir "${DATASET_DIR}" \
  --workbook "${DATASET_DIR}/merged_vol_rq2_text.xlsx" \
  --window-minutes "${WINDOW_MINUTES}" \
  --min-strikes-per-expiry "${MIN_STRIKES_PER_EXPIRY}" \
  --source-timezone "${SOURCE_TIMEZONE}" \
  --publication-availability-lag-minutes \
  "${PUBLICATION_AVAILABILITY_LAG_MINUTES}" \
  --min-usable-pairs 100 \
  --warn-usable-pairs 1000

if [[ "${RUN_DOWNSTREAM}" == "1" ]]; then
  RUN_TS="${RUN_TS}" \
  PIPELINE_ROOT="${PIPELINE_ROOT}" \
  DATASET_DIR="${DATASET_DIR}" \
  BUILD_DATASET=0 \
  ENV_NAME="${ENV_NAME}" \
  OPTION_DATA_GLOB="${OPTION_DATA_GLOB}" \
  NEWS_WORKBOOK="${NEWS_WORKBOOK}" \
  FEATURE_ROOT="${FEATURE_ROOT}" \
  RAW_CONFIG="${RAW_CONFIG}" \
  RATE_CURVE_PATH="${RATE_CURVE_PATH}" \
  SOURCE_TIMEZONE="${SOURCE_TIMEZONE}" \
  PUBLICATION_AVAILABILITY_LAG_MINUTES="${PUBLICATION_AVAILABILITY_LAG_MINUTES}" \
  WINDOW_MINUTES="${WINDOW_MINUTES}" \
  MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY}" \
  OPTION_FILTER_MODE="${OPTION_FILTER_MODE}" \
  MAX_ITM_MONEYNESS_DISTANCE="${MAX_ITM_MONEYNESS_DISTANCE}" \
  GPU_IDS="${GPU_IDS}" \
  RUNS_PER_GPU="${RUNS_PER_GPU}" \
    bash scripts/rq123/run_corrected_pipeline.sh
fi

echo "Relaxed-time pipeline complete: ${PIPELINE_ROOT}"
