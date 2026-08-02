#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/User8/xland_joint_wam}"
NANOWM_ENV="${NANOWM_ENV:-nanowm-xland}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/runs}"
RUN_NAME="${RUN_NAME:-xland_joint_128}"
PORT="${PORT:-6008}"

exec conda run -n "$NANOWM_ENV" --no-capture-output \
  tensorboard \
  --logdir "$RESULTS_ROOT/$RUN_NAME" \
  --host 0.0.0.0 \
  --port "$PORT"
