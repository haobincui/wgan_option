#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/experiments/rq1_event30_prior5m_raw_vol_${RUN_TS}}"
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:-outputs/experiments/rq1_intraday_sequence_raw_vol_20260803-135818}"
SOURCE_DATABASE="${SOURCE_DATABASE:-${SOURCE_EXPERIMENT}/inputs/data/market_surface_index.sqlite}"
SOURCE_NEWS="${SOURCE_NEWS:-${SOURCE_EXPERIMENT}/inputs/data/news_with_openai_embeddings_large.xlsx}"
SOURCE_CALENDAR="${SOURCE_CALENDAR:-${SOURCE_EXPERIMENT}/inputs/data/cme_treasury_globex_closures_2022_2023.csv}"
BASE_CONFIG="${BASE_CONFIG:-configs/film_wgan/train_rq1_event30_textbase.yaml}"

for path in "${SOURCE_DATABASE}" "${SOURCE_NEWS}" "${SOURCE_CALENDAR}" "${BASE_CONFIG}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Missing required input: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${EXPERIMENT_ROOT}/inputs/data" "${EXPERIMENT_ROOT}/inputs/configs"
cp --reflink=auto --preserve=timestamps "${SOURCE_DATABASE}" \
  "${EXPERIMENT_ROOT}/inputs/data/market_surface_index.sqlite"
cp --reflink=auto --preserve=timestamps "${SOURCE_NEWS}" \
  "${EXPERIMENT_ROOT}/inputs/data/news_with_openai_embeddings_large.xlsx"
cp --reflink=auto --preserve=timestamps "${SOURCE_CALENDAR}" \
  "${EXPERIMENT_ROOT}/inputs/data/cme_treasury_globex_closures_2022_2023.csv"
cp --reflink=auto --preserve=timestamps "${BASE_CONFIG}" \
  "${EXPERIMENT_ROOT}/inputs/configs/train_rq1_event30_textbase.yaml"

conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/rq1_event30/event30_dataset.py \
  --database "${EXPERIMENT_ROOT}/inputs/data/market_surface_index.sqlite" \
  --news-workbook "${EXPERIMENT_ROOT}/inputs/data/news_with_openai_embeddings_large.xlsx" \
  --session-calendar "${EXPERIMENT_ROOT}/inputs/data/cme_treasury_globex_closures_2022_2023.csv" \
  --base-config "${EXPERIMENT_ROOT}/inputs/configs/train_rq1_event30_textbase.yaml" \
  --output-root "${EXPERIMENT_ROOT}" \
  --news-lookback-minutes "${NEWS_LOOKBACK_MINUTES:-5}" \
  --exclude-origin-minute-news

printf '%s\n' "${EXPERIMENT_ROOT}" > outputs/experiments/rq1_event30_prior5m_latest.txt
echo "experiment_root=${EXPERIMENT_ROOT}"
