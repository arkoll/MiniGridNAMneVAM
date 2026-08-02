#!/usr/bin/env bash
# Base fixed-compute scaling: 1k/3k/10k/30k/100k, video+action vs action-only.
set -euo pipefail

ROOT="${ROOT:-/home/user8_3/xland/xland_joint_wam}"
PY="/home/user8_3/.conda/envs/nanowm-baba/bin/python"
REPO="$ROOT/repos/nano-world-model"
SOURCE="$ROOT/data/exactcraft_easy_100k_128/train"
DATA_ROOT="$ROOT/data"
RUN_ROOT="$ROOT/runs/xland_scaling_base_15k_dual_loss"
LOG_ROOT="/home/user8_3/.quarterdeck-jobs"
MAX_STEPS=15000
BATCH_SIZE=8
VAL_EVERY=5000
CKPT_EVERY=5000
SIZES=(1000 3000 10000 30000 100000)

wait_for_source() {
  while true; do
    local count
    count="$(find "$SOURCE" -maxdepth 1 -type f -name 'episode_train_*.npz' | wc -l | tr -d ' ')"
    if [[ "$count" == "100000" ]]; then
      echo "100k source dataset already validated; proceeding"
      return 0
    fi
    echo "waiting for 100k source dataset: current=$count"
    sleep 60
  done
}

make_subset() {
  local size="$1"
  local target_root="$DATA_ROOT/exactcraft_easy_scaling_${size}_128"
  local destination="$target_root/train"
  local building="$DATA_ROOT/.exactcraft_easy_scaling_${size}_128_building"
  if [[ -d "$destination" ]]; then
    local existing
    existing="$(find "$destination" -maxdepth 1 -type f -name 'episode_train_*.npz' | wc -l | tr -d ' ')"
    if [[ "$existing" == "$size" ]]; then
      echo "subset already ready: size=$size"
      return 0
    fi
    echo "refusing to alter existing incomplete subset $destination (files=$existing)" >&2
    return 1
  fi
  if [[ -e "$building" ]]; then
    local partial_count
    partial_count="$(find "$building/train" -maxdepth 1 -type f -name 'episode_train_*.npz' | wc -l | tr -d ' ')"
    if [[ "$partial_count" == "$size" ]]; then
      mv "$building" "$target_root"
      echo "recovered completed subset: size=$size path=$destination"
      return 0
    fi
    echo "refusing to alter incomplete build directory $building (files=$partial_count)" >&2
    return 1
  fi
  mkdir -p "$building/train"
  set +o pipefail
  find "$SOURCE" -maxdepth 1 -type f -name 'episode_train_*.npz' -print0 | sort -z | head -z -n "$size" | xargs -0 -r -n 200 cp -t "$building/train"
  local copy_status=${PIPESTATUS[3]}
  set -o pipefail
  [[ "$copy_status" == "0" ]]
  local copied
  copied="$(find "$building/train" -maxdepth 1 -type f -name 'episode_train_*.npz' | wc -l | tr -d ' ')"
  [[ "$copied" == "$size" ]]
  mv "$building" "$target_root"
  echo "subset ready: size=$size path=$destination"
}

start_watcher() {
  local run_name="$1"
  local gpu="$2"
  ROOT="$ROOT" RUN_NAME="xland_scaling_base_15k_dual_loss/$run_name" ACTION_GPU="$gpu" EVAL_EPISODES=30 \
    "$ROOT/bin/watch_closed_loop_eval.sh" >"$LOG_ROOT/scaling15k-${run_name}-closed-loop.log" 2>&1 &
  echo "watcher pid=$! run=$run_name action_gpu=$gpu"
}

run_one() {
  local size="$1"
  local variant="$2"
  local gpu="$3"
  local video_weight="1.0"
  if [[ "$variant" == "action_only" ]]; then
    video_weight="0.0"
  fi
  local run_name="base_${size}_${variant}"
  local dataset="$DATA_ROOT/exactcraft_easy_scaling_${size}_128/train"
  local run_dir="$RUN_ROOT/$run_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES="$gpu" DATASET_DIR="$DATA_ROOT" RESULTS_DIR="$ROOT/runs" \
    "$PY" "$REPO/src/main.py" \
      model=nanowm_xland_b4 dataset=xland/exactcraft_easy experiment=xland_joint \
      logger.name=tensorboard wandb.enabled=false \
      experiment.training.max_steps="$MAX_STEPS" \
      experiment.training.val_every_n_steps="$VAL_EVERY" \
      experiment.training.checkpointing.across_timesteps.every_n_train_steps="$CKPT_EVERY" \
      experiment.joint_action.video_loss_weight="$video_weight" \
      experiment.joint_action.visualize_every_n_steps=0 \
      "experiment.training.batch_size=$BATCH_SIZE" \
      "dataset.loader.data_path=$dataset" \
      "hydra.run.dir=$run_dir" >"$LOG_ROOT/scaling15k-${run_name}.log" 2>&1 &
  local train_pid=$!
  echo "train pid=$train_pid run=$run_name gpu=$gpu size=$size variant=$variant"
  start_watcher "$run_name" "$gpu"
  wait "$train_pid"
  echo "finished run=$run_name"
}

worker_gpu1() {
  run_one 1000 video 1
  run_one 3000 action_only 1
  run_one 10000 action_only 1
  run_one 30000 action_only 1
}
worker_gpu2() {
  run_one 1000 action_only 2
  run_one 10000 video 2
  run_one 100000 video 2
}
worker_gpu3() {
  run_one 3000 video 3
  run_one 30000 video 3
  run_one 100000 action_only 3
}

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
wait_for_source
for size in "${SIZES[@]}"; do
  make_subset "$size"
done
worker_gpu1 &
pid1=$!
worker_gpu2 &
pid2=$!
worker_gpu3 &
pid3=$!
wait "$pid1"
wait "$pid2"
wait "$pid3"
echo "all scaling15k runs finished"
