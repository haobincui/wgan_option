#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/raw_vol_rq1_rq2_$(date -u +%Y%m%d)}"
ZIP_PATH="${ZIP_PATH:-${EXP_ROOT%/}.zip}"

zip -rq "${ZIP_PATH}" "${EXP_ROOT}"
zip -T "${ZIP_PATH}"
du -sh "${ZIP_PATH}"
