#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/User8/xland_joint_wam}"
NANOWM_ENV="${NANOWM_ENV:-nanowm-xland}"
GPU="${GPU:-0}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/data}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT/runs}"
RUN_NAME="${RUN_NAME:-xland_joint_128}"
REPO="$ROOT/repos/nano-world-model"
DATASET="$DATASET_ROOT/exactcraft_easy_10k_128/train"
RUN_DIR="$RESULTS_ROOT/$RUN_NAME"

if [[ ! -d "$DATASET" ]]; then
  echo "Dataset is absent: $DATASET"
  echo "Copy or mount the external 128px train trajectories there."
  exit 2
fi
if ! find "$DATASET" -type f -name 'episode_train_*.npz' -print -quit | grep -q .; then
  echo "No episode_train_*.npz files found below: $DATASET"
  exit 3
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Run directory already exists; refusing to overwrite: $RUN_DIR"
  exit 4
fi

mkdir -p "$RESULTS_ROOT" "$ROOT/logs"
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU"
export DATASET_DIR="$DATASET_ROOT"
export RESULTS_DIR="$RESULTS_ROOT"
export WANDB_MODE="${WANDB_MODE:-disabled}"

exec conda run -n "$NANOWM_ENV" --no-capture-output \
  python src/main.py \
  model=nanowm_xland_s4 \
  dataset=xland/exactcraft_easy \
  experiment=xland_joint \
  logger.name=tensorboard \
  wandb.enabled=false \
  "hydra.run.dir=$RUN_DIR"
