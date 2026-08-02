#!/usr/bin/env bash
# Sequential 30k ExactCraft comparison: Base, Large, Small; video+action vs action-only.
set -euo pipefail

ROOT="${ROOT:-/home/user8_3/xland/xland_joint_wam}"
PY="/home/user8_3/.conda/envs/nanowm-baba/bin/python"
REPO="$ROOT/repos/nano-world-model"
DATASET="$ROOT/data/exactcraft_easy_30k_128/train"
RUN_ROOT="$ROOT/runs/xland_30k_compare"
LOG_ROOT="/home/user8_3/.quarterdeck-jobs"
MAX_STEPS=60000
VAL_EVERY=5000
CKPT_EVERY=5000

wait_for_dataset() {
  while true; do
    count="$(find "$DATASET" -maxdepth 1 -type f -name 'episode_train_*.npz' | wc -l | tr -d ' ')"
    if [[ "$count" == "30000" ]]; then
      "$PY" "$ROOT/src/validate_xland_dataset.py" --data-dir "$DATASET" --expected-count 30000
      return 0
    fi
    echo "waiting for 30k dataset: current=$count"
    sleep 60
  done
}

start_watcher() {
  local run_name="$1"
  local action_gpu="$2"
  ROOT="$ROOT" RUN_NAME="xland_30k_compare/$run_name" ACTION_GPU="$action_gpu" EVAL_EPISODES=30 \
    "$ROOT/bin/watch_closed_loop_eval.sh" >"$LOG_ROOT/xland30k-${run_name}-closed-loop.log" 2>&1 &
  echo "watcher pid=$! run=$run_name action_gpu=$action_gpu"
}

start_train() {
  local run_name="$1"
  local model="$2"
  local gpu="$3"
  local video_loss_weight="$4"
  local run_dir="$RUN_ROOT/$run_name"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES="$gpu" DATASET_DIR="$ROOT/data" RESULTS_DIR="$ROOT/runs" \
    "$PY" "$REPO/src/main.py" \
      model="$model" dataset=xland/exactcraft_easy experiment=xland_joint \
      logger.name=tensorboard wandb.enabled=false \
      experiment.training.max_steps="$MAX_STEPS" \
      experiment.training.val_every_n_steps="$VAL_EVERY" \
      experiment.training.checkpointing.across_timesteps.every_n_train_steps="$CKPT_EVERY" \
      experiment.joint_action.video_loss_weight="$video_loss_weight" \
      experiment.joint_action.visualize_every_n_steps=0 \
      "dataset.loader.data_path=$DATASET" \
      "hydra.run.dir=$run_dir" >"$LOG_ROOT/xland30k-${run_name}.log" 2>&1 &
  TRAIN_PID=$!
  echo "train pid=$TRAIN_PID run=$run_name gpu=$gpu video_loss_weight=$video_loss_weight"
}

run_stage() {
  local stage="$1"
  local model="$2"
  echo "starting stage=$stage model=$model"
  local video_pid
  local action_pid
  start_train "${stage}_video" "$model" 2 1.0
  video_pid="$TRAIN_PID"
  start_train "${stage}_action_only" "$model" 3 0.0
  action_pid="$TRAIN_PID"
  start_watcher "${stage}_video" 0
  start_watcher "${stage}_action_only" 1
  wait "$video_pid"
  wait "$action_pid"
  echo "finished stage=$stage"
}

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
wait_for_dataset
run_stage base nanowm_xland_b4
run_stage large nanowm_xland_l4
run_stage small nanowm_xland_s4
echo "all three comparison stages finished"
