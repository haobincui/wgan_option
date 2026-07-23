#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
ENV_NAME="${ENV_NAME:-py312}"

conda run -n "${ENV_NAME}" python scripts/rq1/rq1_experiment.py prepare "$@"
