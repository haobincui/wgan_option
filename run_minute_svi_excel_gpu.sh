#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

python scripts/generate_surface/main.py \
  minute-svi-excel \
  --device gpu \
  --config configs/surface_builder/minute-svi-excel.yaml \
  "$@"
