#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/user8_3/xland/xland_joint_wam}"
RUN_NAME="${RUN_NAME:-xland_joint_128_gpu2_60k}"
ACTION_GPU="${ACTION_GPU:-1}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
NANOWM_PY="/home/user8_3/.conda/envs/nanowm-baba/bin/python"
XLAND_PY="/home/user8_3/miniconda3/envs/vam/bin/python"
RUN_DIR="$ROOT/runs/$RUN_NAME"
NANOWM_REPO="$ROOT/repos/nano-world-model"
XLAND_REPO="/home/user8_3/xland/xland-minigrid-five-object"
EVAL_DIR="$RUN_DIR/closed_loop_eval"
TB_DIR="$RUN_DIR/closed_loop_tb"
LOG_DIR="$RUN_DIR/closed_loop_logs"
RUN_SLUG="${RUN_NAME//\//_}"
SOCKET="/tmp/user8_3_${RUN_SLUG}_action.sock"

mkdir -p "$EVAL_DIR" "$TB_DIR" "$LOG_DIR"

evaluate_split() {
  local checkpoint="$1"
  local step="$2"
  local split="$3"
  local seed_start="$4"
  local strategy="$5"
  local temperature="$6"
  local result="$EVAL_DIR/${split}_temperature_${temperature}_step_$(printf '%07d' "$step").json"
  [[ -f "$result" ]] && return 0

  rm -f "$SOCKET"
  CUDA_VISIBLE_DEVICES="$ACTION_GPU" DATASET_DIR="$ROOT/data" RESULTS_DIR="$ROOT/runs" \
    "$NANOWM_PY" "$ROOT/src/xland_action_server.py" \
      --nanowm-repo "$NANOWM_REPO" --config "$RUN_DIR/config.yaml" \
      --checkpoint "$checkpoint" --socket "$SOCKET" --device cuda \
      --strategy "$strategy" --temperature "$temperature" --sampling-seed "$((seed_start + step))" \
      >"$LOG_DIR/server_${split}_temperature_${temperature}_step_${step}.log" 2>&1 &
  local server_pid="$!"

  set +e
  JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" \
    "$XLAND_PY" "$ROOT/src/evaluate_xland_closed_loop.py" \
      --xland-repo "$XLAND_REPO" --socket "$SOCKET" --output-dir "$EVAL_DIR" --tb-dir "$TB_DIR" \
      --global-step "$step" --split "$split" --strategy "$strategy" --temperature "$temperature" \
      --episodes "$EVAL_EPISODES" --seed-start "$seed_start" --image-size 128 \
      >"$LOG_DIR/${split}_temperature_${temperature}_step_${step}.log" 2>&1
  local status="$?"
  set -e

  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  [[ "$status" -eq 0 ]]
}

while true; do
  while IFS= read -r checkpoint; do
    filename="$(basename "$checkpoint")"
    step="${filename##*step=}"
    step="${step%.ckpt}"
    evaluate_split "$checkpoint" "$step" train 3000000 greedy 0
    evaluate_split "$checkpoint" "$step" val_ood 4000000 greedy 0
    evaluate_split "$checkpoint" "$step" train 3000000 sample 1
    evaluate_split "$checkpoint" "$step" val_ood 4000000 sample 1
  done < <(find "$RUN_DIR/checkpoints/across_timesteps" -maxdepth 1 -type f -name 'xland-joint-*.ckpt' 2>/dev/null | sort -V)
  sleep 30
done
