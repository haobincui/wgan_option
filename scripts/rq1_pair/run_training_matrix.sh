#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
conda run -n py312 python scripts/rq1_pair/rq1_pair_experiment.py train-matrix "$@"
