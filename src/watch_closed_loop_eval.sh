#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/User9/xland_joint_wam
RUN_DIR="$ROOT/runs/xland_joint_128"
NANOWM_REPO="$ROOT/repos/nano-world-model"
XLAND_REPO=/home/User9/xland-minigrid-five-object
EVAL_DIR="$RUN_DIR/closed_loop_eval"
TB_DIR="$RUN_DIR/closed_loop_tb"
LOG_DIR="$RUN_DIR/closed_loop_logs"
SOCKET=/tmp/xland_joint_action_128.sock

mkdir -p "$EVAL_DIR" "$TB_DIR" "$LOG_DIR"

while true; do
  found_checkpoint=false
  while IFS= read -r checkpoint; do
    found_checkpoint=true
    filename="$(basename "$checkpoint")"
    step="$(printf '%s' "$filename" | sed -n 's/.*step=\([0-9][0-9]*\).ckpt/\1/p')"
    if [[ -z "$step" ]]; then
      continue
    fi
    result="$EVAL_DIR/closed_loop_step_$(printf '%07d' "$step").json"
    if [[ -f "$result" ]]; then
      continue
    fi

    rm -f "$SOCKET"
    (
      export CUDA_VISIBLE_DEVICES=2
      export DATASET_DIR="$ROOT/data"
      export RESULTS_DIR="$ROOT/runs"
      conda run -n baba-wam --no-capture-output \
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
      conda run -n xland-five-object --no-capture-output \
        python "$ROOT/src/evaluate_xland_closed_loop.py" \
        --xland-repo "$XLAND_REPO" \
        --socket "$SOCKET" \
        --output-dir "$EVAL_DIR" \
        --tb-dir "$TB_DIR" \
        --global-step "$step" \
        --episodes 30 \
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
    if [[ "$status" != "0" ]]; then
      echo "closed-loop evaluation failed at step=$step status=$status"
      exit "$status"
    fi
    echo "closed-loop evaluation finished at step=$step"
  done < <(
    find "$RUN_DIR/checkpoints/across_timesteps" \
      -maxdepth 1 \
      -type f \
      -name 'xland-joint-*.ckpt' \
      2>/dev/null \
      | sort -V
  )

  if [[ -f "$EVAL_DIR/closed_loop_step_0030000.json" ]]; then
    echo "final checkpoint evaluated"
    exit 0
  fi
  if [[ "$found_checkpoint" == "false" ]]; then
    echo "waiting for first checkpoint"
  fi
  sleep 30
done
