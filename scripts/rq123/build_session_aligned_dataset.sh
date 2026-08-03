#!/usr/bin/env bash
# Rebuild raw-vol/news pairs under the frozen CME Treasury session policy.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
ENV_NAME="${ENV_NAME:-py312}"
INDEX_DIR="${INDEX_DIR:-data/processed/raw-market-index/raw_2022_2023_london_itm5_relaxed_v2}"
DATASET_DIR="${DATASET_DIR:-data/processed/raw-excel-session/rq123_cme_session_${RUN_TS}}"
NEWS_WORKBOOK="${NEWS_WORKBOOK:-data/raw/text_embedding/news_with_openai_embeddings_large.xlsx}"
FEATURE_DIR="${FEATURE_DIR:-data/processed/text_features/rq2/20260625-075653}"
STRICT_BASELINE_WORKBOOK="${STRICT_BASELINE_WORKBOOK:-data/processed/raw-excel/rq123_corrected_lag0_s2_itm5_20260728-114931/merged_vol.xlsx}"
RATE_CURVE_PATH="${RATE_CURVE_PATH:-data/reference/us_treasury_par_yield_curve_2022_2023.csv}"
SESSION_CALENDAR_PATH="${SESSION_CALENDAR_PATH:-data/reference/cme_treasury_globex_closures_2022_2023.csv}"
SOURCE_TIMEZONE="${SOURCE_TIMEZONE:-Europe/London}"
PUBLICATION_AVAILABILITY_LAG_MINUTES="${PUBLICATION_AVAILABILITY_LAG_MINUTES:-0}"
WINDOW_MINUTES="${WINDOW_MINUTES:-5}"
ORIGIN_TOLERANCE_MINUTES="${ORIGIN_TOLERANCE_MINUTES:-5}"
MIN_STRIKES_PER_EXPIRY="${MIN_STRIKES_PER_EXPIRY:-2}"
MIN_EXPIRIES_PER_MINUTE="${MIN_EXPIRIES_PER_MINUTE:-2}"
OPTION_FILTER_MODE="${OPTION_FILTER_MODE:-otm_preferred_itm_fallback}"
MAX_ITM_MONEYNESS_DISTANCE="${MAX_ITM_MONEYNESS_DISTANCE:-0.05}"

PYTHON=(conda run --no-capture-output -n "${ENV_NAME}" python)

mkdir -p "${DATASET_DIR}"
for required_path in \
  "${INDEX_DIR}/market_surface_index.sqlite" \
  "${NEWS_WORKBOOK}" \
  "${FEATURE_DIR}/bow_features.xlsx" \
  "${FEATURE_DIR}/llm_sentiment_features.xlsx" \
  "${STRICT_BASELINE_WORKBOOK}" \
  "${RATE_CURVE_PATH}" \
  "${SESSION_CALENDAR_PATH}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required input does not exist: ${required_path}" >&2
    exit 1
  fi
done

{
  printf 'run_ts=%s\n' "${RUN_TS}"
  printf 'dataset_dir=%s\n' "${DATASET_DIR}"
  printf 'index_dir=%s\n' "${INDEX_DIR}"
  printf 'news_workbook=%s\n' "${NEWS_WORKBOOK}"
  printf 'feature_dir=%s\n' "${FEATURE_DIR}"
  printf 'strict_baseline_workbook=%s\n' \
    "${STRICT_BASELINE_WORKBOOK}"
  printf 'rate_curve_path=%s\n' "${RATE_CURVE_PATH}"
  printf 'alignment_mode=exchange_session\n'
  printf 'source_timezone=%s\n' "${SOURCE_TIMEZONE}"
  printf 'publication_availability_lag_minutes=%s\n' \
    "${PUBLICATION_AVAILABILITY_LAG_MINUTES}"
  printf 'window_minutes=%s\n' "${WINDOW_MINUTES}"
  printf 'origin_tolerance_minutes=%s\n' "${ORIGIN_TOLERANCE_MINUTES}"
  printf 'session_calendar_path=%s\n' "${SESSION_CALENDAR_PATH}"
  printf 'closed_news_rule=next_cme_continuous_session_open\n'
  printf 'open_news_rule=publication_minute\n'
} >"${DATASET_DIR}/run_manifest.env"

sha256sum \
  "${FEATURE_DIR}/bow_features.xlsx" \
  "${FEATURE_DIR}/llm_sentiment_features.xlsx" \
  >"${DATASET_DIR}/rq2_feature_input_sha256.txt"

"${PYTHON[@]}" scripts/raw_vol/relaxed_time_pipeline.py build-dataset \
  --index-dir "${INDEX_DIR}" \
  --output-dir "${DATASET_DIR}" \
  --news-xlsx "${NEWS_WORKBOOK}" \
  --strict-baseline-workbook "${STRICT_BASELINE_WORKBOOK}" \
  --rate-curve-path "${RATE_CURVE_PATH}" \
  --source-timezone "${SOURCE_TIMEZONE}" \
  --publication-availability-lag-minutes \
  "${PUBLICATION_AVAILABILITY_LAG_MINUTES}" \
  --alignment-mode exchange_session \
  --session-calendar-path "${SESSION_CALENDAR_PATH}" \
  --origin-tolerance-minutes "${ORIGIN_TOLERANCE_MINUTES}" \
  --window-minutes "${WINDOW_MINUTES}" \
  --min-strikes-per-expiry "${MIN_STRIKES_PER_EXPIRY}" \
  --min-expiries-per-minute "${MIN_EXPIRIES_PER_MINUTE}" \
  --option-filter-mode "${OPTION_FILTER_MODE}" \
  --max-itm-moneyness-distance "${MAX_ITM_MONEYNESS_DISTANCE}"

"${PYTHON[@]}" scripts/rq2/enrich_merged_vol.py \
  --merged-vol "${DATASET_DIR}/merged_vol.xlsx" \
  --bow-features "${FEATURE_DIR}/bow_features.xlsx" \
  --sentiment-features "${FEATURE_DIR}/llm_sentiment_features.xlsx" \
  --output "${DATASET_DIR}/merged_vol_rq2_text.xlsx"

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

sha256sum \
  "${DATASET_DIR}/news_market_session_alignment.csv" \
  "${DATASET_DIR}/cme_session_calendar_snapshot.csv" \
  "${DATASET_DIR}/merged_vol.xlsx" \
  "${DATASET_DIR}/merged_vol_rq2_text.xlsx" \
  >"${DATASET_DIR}/dataset_output_sha256.txt"

echo "Session-aligned raw-vol dataset ready: ${DATASET_DIR}"
echo "Coverage: ${DATASET_DIR}/alignment_coverage_summary.csv"
echo "Validation: ${DATASET_DIR}/raw_vol_dataset_validation.json"
