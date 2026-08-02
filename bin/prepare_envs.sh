#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/User8/xland_joint_wam}"
NANOWM_ENV="${NANOWM_ENV:-nanowm-xland}"
XLAND_ENV="${XLAND_ENV:-xland-five-object}"
NANOWM_REPO="$ROOT/repos/nano-world-model"
XLAND_REPO="$ROOT/repos/xland-minigrid-five-object"

if ! conda env list | awk '{print $1}' | grep -Fxq "$NANOWM_ENV"; then
  conda env create -n "$NANOWM_ENV" -f "$NANOWM_REPO/environment.yml"
else
  echo "Conda environment already exists: $NANOWM_ENV"
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$XLAND_ENV"; then
  conda create -y -n "$XLAND_ENV" python=3.10
else
  echo "Conda environment already exists: $XLAND_ENV"
fi

conda run -n "$XLAND_ENV" \
  python -m pip install -e "$XLAND_REPO[baselines]" tensorboardX

echo "Prepared NanoWM env: $NANOWM_ENV"
echo "Prepared XLand env: $XLAND_ENV"
