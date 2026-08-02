#!/usr/bin/env bash
# Fixed-compute Base scaling study: 1k, 3k, 10k, 30k, 100k trajectories.
set -euo pipefail

ROOT="${ROOT:-/home/user8_3/xland/xland_joint_wam}"
PY="/home/user8_3/.conda/envs/nanowm-baba/bin/python"
REPO="$ROOT/repos/nano-world-model"
SOURCE="$ROOT/data/exactcraft_easy_100k_128/train"
DATA_ROOT="$ROOT/data"
RUN_ROOT="$ROOT/runs/xland_scaling_base_30k"
LOG_ROOT="/home/user8_3/.quarterdeck-jobs"
MAX_STEPS=30000
VAL_EVERY=5000
CKPT_EVERY=5000
SIZES=(1000 3000 10000 30000 100000)

wait_for_prior_queue() {
  while pgrep -f "$ROOT/bin/run_xland_30k_comparison_queue.sh" >/dev/null; do
    echo "waiting for current model-comparison queue to release GPUs 2/3"
    sleep 60
  done
}

wait_for_source() {
  while true; do
    count="$(find "$SOURCE" -maxdepth 1 -type f -name 'episode_train_*.npz' | wc -l | tr -d ' ')"
    if [[ "$count" == "100000" ]]; then
      "$PY" "$ROOT/src/validate_xland_dataset.py" --data-dir "$SOURCE" --expected-count 100000
      return 0
    fi
    echo "waiting for 100k source dataset: current=$count"
    sleep 60
  done
}

make_subset() {
  local size="$1"
  local destination="$DATA_ROOT/exactcraft_easy_scaling_${size}_128/train"
  local building="$DATA_ROOT/.exactcraft_easy_scaling_${size}_128_building"
  if [[ -d "$destination" ]]; then
    local existing
    existing="$(find "$destination" -maxdepth 1 -type f -name 'episode_train_*.npz' | wc -l | tr -d ' ')"
    if [[ "$existing" == "$size" ]]; then
      echo "subset already ready: size=$size"
      return 0
    fi
    echo "refusing to alter incomplete subset $destination (files=$existing)" >&2
    return 1
  fi
  if [[ -e "$building" ]]; then
    echo "refusing to reuse stale build directory $building" >&2
    return 1
  fi
  mkdir -p "$building/train"
  find "$SOURCE" -maxdepth 1 -type f -name 'episode_train_*.npz' -print0 | sort -z | head -z -n "$size" | xargs -0 -r -n 200 cp -t "$building/train"
  local copied
  copied="$(find "$building/train" -maxdepth 1 -type f -name 'episode_train_*.npz' | wc -l | tr -d ' ')"
  [[ "$copied" == "$size" ]]
  mv "$building" "$DATA_ROOT/exactcraft_easy_scaling_${size}_128"
  echo "subset ready: size=$size path=$destination"
}

start_watcher() {
  local run_name="$1"
  ROOT="$ROOT" RUN_NAME="xland_scaling_base_30k/$run_name" ACTION_GPU=0 EVAL_EPISODES=30 \
    "$ROOT/bin/watch_closed_loop_eval.sh" >"$LOG_ROOT/scaling-${run_name}-closed-loop.log" 2>&1 &
  echo "watcher pid=$! run=$run_name"
}

start_train() {
  local size="$1"
  local gpu="$2"
  local run_name="base_${size}"
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
      experiment.joint_action.video_loss_weight=1.0 \
      experiment.joint_action.visualize_every_n_steps=0 \
      "dataset.loader.data_path=$dataset" \
      "hydra.run.dir=$run_dir" >"$LOG_ROOT/scaling-${run_name}.log" 2>&1 &
  TRAIN_PID=$!
  echo "train pid=$TRAIN_PID run=$run_name gpu=$gpu dataset_size=$size"
  start_watcher "$run_name"
}

run_pair() {
  local first="$1"
  local second="${2:-}"
  start_train "$first" 2
  local first_pid="$TRAIN_PID"
  if [[ -n "$second" ]]; then
    start_train "$second" 3
    local second_pid="$TRAIN_PID"
    wait "$first_pid"
    wait "$second_pid"
  else
    wait "$first_pid"
  fi
}

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
wait_for_prior_queue
wait_for_source
for size in "${SIZES[@]}"; do
  make_subset "$size"
done
run_pair 1000 3000
run_pair 10000 30000
run_pair 100000
echo "scaling queue finished"
