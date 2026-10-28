#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_TS="${RUN_TS:-$(date -u +%Y%m%d-%H%M%S)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-outputs/experiments/rq1_event30_prior5m_raw_vol_${RUN_TS}}"
LOG_DIR="outputs/experiments/logs"
LOG_PATH="${LOG_DIR}/rq1_event30_prepare_${RUN_TS}.log"
PID_PATH="${LOG_DIR}/rq1_event30_prepare_${RUN_TS}.pid"
LATEST_ENV="outputs/experiments/rq1_event30_prepare_latest.env"
mkdir -p "${LOG_DIR}"

setsid env \
  RUN_TS="${RUN_TS}" \
  EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
  ENV_NAME="${ENV_NAME:-py312}" \
  NEWS_LOOKBACK_MINUTES="${NEWS_LOOKBACK_MINUTES:-5}" \
  bash scripts/rq1_event30/prepare_experiment.sh \
  >"${LOG_PATH}" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "${PID}" > "${PID_PATH}"
cat > "${LATEST_ENV}" <<EOF
RUN_TS=${RUN_TS}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT}
LOG_PATH=${LOG_PATH}
PID_PATH=${PID_PATH}
PID=${PID}
EOF

echo "pid=${PID}"
echo "log=${LOG_PATH}"
echo "experiment_root=${EXPERIMENT_ROOT}"
