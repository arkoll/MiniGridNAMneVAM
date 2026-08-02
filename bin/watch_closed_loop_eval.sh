#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/User8/xland_joint_wam}"
RUN_NAME="${RUN_NAME:-xland_joint_128}"
NANOWM_ENV="${NANOWM_ENV:-nanowm-xland}"
XLAND_ENV="${XLAND_ENV:-xland-five-object}"
ACTION_GPU="${ACTION_GPU:-2}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
RUN_DIR="$ROOT/runs/$RUN_NAME"
NANOWM_REPO="$ROOT/repos/nano-world-model"
XLAND_REPO="$ROOT/repos/xland-minigrid-five-object"
EVAL_DIR="$RUN_DIR/closed_loop_eval"
TB_DIR="$RUN_DIR/closed_loop_tb"
LOG_DIR="$RUN_DIR/closed_loop_logs"
SOCKET="/tmp/${USER}_${RUN_NAME}_action.sock"

mkdir -p "$EVAL_DIR" "$TB_DIR" "$LOG_DIR"

while true; do
  while IFS= read -r checkpoint; do
    filename="$(basename "$checkpoint")"
    step="$(printf '%s' "$filename" | sed -n 's/.*step=\([0-9][0-9]*\).ckpt/\1/p')"
    [[ -n "$step" ]] || continue
    result="$EVAL_DIR/closed_loop_step_$(printf '%07d' "$step").json"
    [[ ! -f "$result" ]] || continue

    rm -f "$SOCKET"
    (
      export CUDA_VISIBLE_DEVICES="$ACTION_GPU"
      export DATASET_DIR="$ROOT/data"
      export RESULTS_DIR="$ROOT/runs"
      conda run -n "$NANOWM_ENV" --no-capture-output \
        python "$ROOT/src/xland_action_server.py" \
        --nanowm-repo "$NANOWM_REPO" \
        --config "$RUN_DIR/config.yaml" \
        --checkpoint "$checkpoint" \
        --socket "$SOCKET" \
        --device cuda
    ) >"$LOG_DIR/server_step_${step}.log" 2>&1 &
    server_pid="$!"

    set +e
    (
      export JAX_PLATFORMS=cpu
      export JAX_PLATFORM_NAME=cpu
      export CUDA_VISIBLE_DEVICES=""
      conda run -n "$XLAND_ENV" --no-capture-output \
        python "$ROOT/src/evaluate_xland_closed_loop.py" \
        --xland-repo "$XLAND_REPO" \
        --socket "$SOCKET" \
        --output-dir "$EVAL_DIR" \
        --tb-dir "$TB_DIR" \
        --global-step "$step" \
        --episodes "$EVAL_EPISODES" \
        --seed-start 3000000 \
        --image-size 128
    ) >"$LOG_DIR/eval_step_${step}.log" 2>&1
    status="$?"
    set -e

    if [[ "$status" == "0" ]]; then
      wait "$server_pid" || status="$?"
    else
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
    [[ "$status" == "0" ]] || exit "$status"
  done < <(
    find "$RUN_DIR/checkpoints/across_timesteps" \
      -maxdepth 1 -type f -name 'xland-joint-*.ckpt' \
      2>/dev/null | sort -V
  )

  [[ ! -f "$EVAL_DIR/closed_loop_step_0030000.json" ]] || exit 0
  sleep 30
done
