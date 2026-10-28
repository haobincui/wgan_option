#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
SOURCE_REPO_ROOT="${SOURCE_REPO_ROOT:-/home/haobin_cui/research_files_space_2/wgan_option-rq123-london-timezone}"
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/experiments/rq1_intraday_sequence_raw_vol_${RUN_TS}}"
DATABASE_PATH="${DATABASE_PATH:-${SOURCE_REPO_ROOT}/data/processed/raw-market-index/raw_2022_2023_london_itm5_relaxed_v2/market_surface_index.sqlite}"
NEWS_WORKBOOK="${NEWS_WORKBOOK:-${SOURCE_REPO_ROOT}/data/raw/text_embedding/news_with_openai_embeddings_large.xlsx}"
SESSION_CALENDAR="${SESSION_CALENDAR:-data/reference/cme_treasury_globex_closures_2022_2023.csv}"
SEEDS="${SEEDS:-42 202 404}"
read -r -a SEED_ARGS <<<"${SEEDS}"

conda run --no-capture-output -n "${ENV_NAME}" \
  python scripts/rq1_sequence/rq1_sequence_experiment.py prepare \
  --experiment-root "${EXPERIMENT_ROOT}" \
  --database "${DATABASE_PATH}" \
  --news-workbook "${NEWS_WORKBOOK}" \
  --session-calendar "${SESSION_CALENDAR}" \
  --seeds "${SEED_ARGS[@]}"

printf '%s\n' "${EXPERIMENT_ROOT}" > outputs/experiments/rq1_intraday_sequence_latest.txt
echo "experiment_root=${EXPERIMENT_ROOT}"

