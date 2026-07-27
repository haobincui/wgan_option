#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-py312}"
CONFIG_PATH="${CONFIG_PATH:-configs/rq3/scheduled_news_regime_raw_vol.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RQ1_EXPERIMENT="${RQ1_EXPERIMENT:-}"
RQ2_EXPERIMENT="${RQ2_EXPERIMENT:-}"
EVENT_CALENDAR_PATH="${EVENT_CALENDAR_PATH:-}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing RQ3 config: ${CONFIG_PATH}" >&2
  exit 1
fi

COMMAND=(
  conda run -n "${ENV_NAME}"
  python scripts/rq3/main.py scheduled-news-regime
  --config "${CONFIG_PATH}"
)
if [[ -n "${OUTPUT_DIR}" ]]; then
  COMMAND+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${RQ1_EXPERIMENT}" ]]; then
  COMMAND+=(--rq1-experiment "${RQ1_EXPERIMENT}")
fi
if [[ -n "${RQ2_EXPERIMENT}" ]]; then
  COMMAND+=(--rq2-experiment "${RQ2_EXPERIMENT}")
fi
if [[ -n "${EVENT_CALENDAR_PATH}" ]]; then
  COMMAND+=(--event-calendar "${EVENT_CALENDAR_PATH}")
fi

echo "config=${CONFIG_PATH}"
echo "output_dir=${OUTPUT_DIR:-auto}"
echo "environment=${ENV_NAME}"
echo "rq1_experiment=${RQ1_EXPERIMENT:-from_config}"
echo "rq2_experiment=${RQ2_EXPERIMENT:-from_config}"
echo "event_calendar=${EVENT_CALENDAR_PATH:-from_config}"
"${COMMAND[@]}"
