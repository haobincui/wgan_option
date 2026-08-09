#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
RQ1 v2 pilot launch is disabled: its shuffled arm used an asymmetric negative
definition that can include the target's native text as a negative. Existing v2
artifacts remain readable for audit only and must not be resumed or extended.

Use scripts/rq1_pair/start_v3_pilot_gpu0.sh with a new, empty experiment root.
EOF
exit 2
