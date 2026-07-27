#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
conda run -n py312 python scripts/rq2_pair/rq2_pair_experiment.py package "$@"
