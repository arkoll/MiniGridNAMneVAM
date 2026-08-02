#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/User9/xland_joint_wam
XLAND_REPO=/home/User9/xland-minigrid-five-object
CHECKPOINT="$XLAND_REPO/runs/privileged_rl_easy_20260730/exact_easy/checkpoints/best_val/train_state.msgpack"
COLLECTOR="$ROOT/src/collect_xland_ppo.py"
DATA_ROOT="$ROOT/data/exactcraft_easy_10k"
LOG_ROOT="$ROOT/logs/collection_10k"

mkdir -p "$DATA_ROOT/train" "$DATA_ROOT/val" "$LOG_ROOT"

existing_count="$(
  find "$DATA_ROOT" -type f -name 'episode_*.npz' | wc -l | tr -d ' '
)"
if [[ "$existing_count" != "0" ]]; then
  echo "Refusing to mix a new collection with $existing_count existing episodes."
  exit 2
fi

pids=()

run_worker() {
  local split="$1"
  local worker="$2"
  local episodes="$3"
  local offset="$4"
  local seed_start="$5"
  local gpu="$6"
  local stochastic_flag="$7"
  local shard
  shard="$(printf 'shard_%02d' "$worker")"
  mkdir -p "$DATA_ROOT/$split/$shard" "$LOG_ROOT/tensorboard/$split/$shard"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.20
    conda run -n xland-five-object --no-capture-output \
      python "$COLLECTOR" \
      --xland-repo "$XLAND_REPO" \
      --checkpoint "$CHECKPOINT" \
      --output-dir "$DATA_ROOT/$split/$shard" \
      --tb-dir "$LOG_ROOT/tensorboard/$split/$shard" \
      --split "$split" \
      --episodes "$episodes" \
      --episode-offset "$offset" \
      --seed-start "$seed_start" \
      --log-every 50 \
      $stochastic_flag
  ) >"$LOG_ROOT/${split}_${shard}.log" 2>&1 &
  pids+=("$!")
  echo "launched split=$split worker=$worker episodes=$episodes gpu=$gpu pid=${pids[-1]}"
}

# Nine train workers: 1000 trajectories each, balanced over six training tasks.
for worker in $(seq 0 8); do
  offset=$((worker * 1000))
  run_worker train "$worker" 1000 "$offset" "$((1000000 + offset))" "$((worker % 3))" "--stochastic"
done

# Three validation workers: 334 + 333 + 333 held-out-composition trajectories.
val_offset=0
for worker in $(seq 0 2); do
  episodes=333
  if [[ "$worker" == "0" ]]; then
    episodes=334
  fi
  run_worker val "$worker" "$episodes" "$val_offset" "$((2000000 + val_offset))" "$worker" ""
  val_offset=$((val_offset + episodes))
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

train_count="$(
  find "$DATA_ROOT/train" -type f -name 'episode_train_*.npz' | wc -l | tr -d ' '
)"
val_count="$(
  find "$DATA_ROOT/val" -type f -name 'episode_val_*.npz' | wc -l | tr -d ' '
)"
echo "finished train=$train_count val=$val_count status=$status"

if [[ "$train_count" != "9000" || "$val_count" != "1000" ]]; then
  exit 3
fi
exit "$status"
