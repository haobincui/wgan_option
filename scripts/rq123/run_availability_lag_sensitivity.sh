#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

SERIES_TS="${SERIES_TS:-$(date -u +%Y%m%d-%H%M%S)}"
LAGS="${LAGS:-1 2 5}"
GPU_IDS="${GPU_IDS:-0 1}"
RUNS_PER_GPU="${RUNS_PER_GPU:-2}"
SERIES_ROOT="${SERIES_ROOT:-outputs/experiments/rq123_availability_lag_${SERIES_TS}}"
mkdir -p "${SERIES_ROOT}/logs"

for lag in ${LAGS}; do
  child_root="${SERIES_ROOT}/lag_${lag}"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start lag=${lag}"
  RUN_TS="${SERIES_TS}_lag${lag}" \
  PIPELINE_ROOT="${child_root}" \
  DATASET_RUN_TS="rq123_corrected_lag${lag}_${SERIES_TS}" \
  PUBLICATION_AVAILABILITY_LAG_MINUTES="${lag}" \
  GPU_IDS="${GPU_IDS}" \
  RUNS_PER_GPU="${RUNS_PER_GPU}" \
  PACKAGE_EXPERIMENT=0 \
    bash scripts/rq123/run_corrected_pipeline.sh \
    2>&1 | tee "${SERIES_ROOT}/logs/lag_${lag}.log"
done

conda run -n "${ENV_NAME:-py312}" python \
  scripts/rq123/corrected_pipeline.py summarize-lags \
  --series-root "${SERIES_ROOT}"
echo "availability_lag_series_root=${SERIES_ROOT}"
