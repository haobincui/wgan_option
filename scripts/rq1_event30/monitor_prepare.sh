#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

LATEST_ENV="outputs/experiments/rq1_event30_prepare_latest.env"
if [[ ! -f "${LATEST_ENV}" ]]; then
  echo "No RQ1 event30 background preparation has been registered." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${LATEST_ENV}"

if kill -0 "${PID}" 2>/dev/null; then
  echo "status=running pid=${PID}"
else
  echo "status=stopped pid=${PID}"
fi
echo "experiment_root=${EXPERIMENT_ROOT}"
echo "log=${LOG_PATH}"
tail -n "${TAIL_LINES:-40}" "${LOG_PATH}" 2>/dev/null || true

if [[ -f "${EXPERIMENT_ROOT}/event30_dataset_validation.json" ]]; then
  python - <<PY
import json
from pathlib import Path
p=Path("${EXPERIMENT_ROOT}/event30_dataset_validation.json")
d=json.loads(p.read_text())
print("validation_status=", d["status"])
print("all_origin_candidates=", d["all_origin_candidate_count"])
print("news_alignment_mode=", d.get("news_alignment_mode"))
print("news_conditioned_candidates=", d.get("news_conditioned_candidate_count"))
for fold, row in d["folds"].items():
    print(
        fold,
        "all=", row["all_origin_split_counts"],
        "events=", row.get("news_conditioned_split_counts"),
    )
PY
fi
