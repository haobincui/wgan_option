#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EXP_ROOT="${EXP_ROOT:-outputs/experiments/rq2_multiseed_textbase_20260627}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" scripts/rq2_multiseed/rq2_multiseed.py prepare --exp-root "${EXP_ROOT}"
