#!/usr/bin/env bash
# Re-evaluate final (step 60000) model-size comparison checkpoints on 200 episodes.
set -euo pipefail

ROOT="/home/user8_3/xland/xland_joint_wam"
NANOWM_PY="/home/user8_3/.conda/envs/nanowm-baba/bin/python"
XLAND_PY="/home/user8_3/miniconda3/envs/vam/bin/python"
NANOWM_REPO="$ROOT/repos/nano-world-model"
XLAND_REPO="/home/user8_3/xland/xland-minigrid-five-object"
SOURCE_RUNS="$ROOT/runs/xland_30k_compare"
OUT_ROOT="$ROOT/runs/xland_model_size_final_200eval"
LOG_ROOT="/home/user8_3/.quarterdeck-jobs"
EPISODES=200
STEP=60000

evaluate_one() {
  local run="$1"
  local gpu="$2"
  local split="$3"
  local strategy="$4"
  local temperature="$5"
  local seed_start="$6"
  local run_dir="$SOURCE_RUNS/$run"
  local checkpoint="$run_dir/checkpoints/across_timesteps/xland-joint-epoch=17-step=60000.ckpt"
  local result_dir="$OUT_ROOT/results/$run"
  local tb_dir="$OUT_ROOT/tensorboard/$run"
  local log_dir="$OUT_ROOT/logs/$run"
  local socket="/tmp/user8_3_modelsize200_${run}_${split}_t${temperature}.sock"
  mkdir -p "$result_dir" "$tb_dir" "$log_dir"
  local result="$result_dir/${split}_temperature_${temperature}_step_0060000.json"
  if [[ -f "$result" ]]; then
    echo "skip existing result=$result"
    return 0
  fi
  rm -f "$socket"
  CUDA_VISIBLE_DEVICES="$gpu" DATASET_DIR="$ROOT/data" RESULTS_DIR="$ROOT/runs" \
    "$NANOWM_PY" "$ROOT/src/xland_action_server.py" \
      --nanowm-repo "$NANOWM_REPO" --config "$run_dir/config.yaml" \
      --checkpoint "$checkpoint" --socket "$socket" --device cuda \
      --strategy "$strategy" --temperature "$temperature" --sampling-seed "$((seed_start + STEP))" \
      >"$log_dir/server_${split}_t${temperature}.log" 2>&1 &
  local server_pid=$!
  local ready=0
  for _ in $(seq 1 240); do
    [[ -S "$socket" ]] && { ready=1; break; }
    sleep 1
  done
  if [[ "$ready" != "1" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    echo "action server did not become ready: run=$run split=$split t=$temperature" >&2
    return 1
  fi
  JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" \
    "$XLAND_PY" "$ROOT/src/evaluate_xland_closed_loop.py" \
      --xland-repo "$XLAND_REPO" --socket "$socket" --output-dir "$result_dir" --tb-dir "$tb_dir" \
      --global-step "$STEP" --split "$split" --strategy "$strategy" --temperature "$temperature" \
      --episodes "$EPISODES" --seed-start "$seed_start" --image-size 128 \
      >"$log_dir/${split}_t${temperature}.log" 2>&1
  local status=$?
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  [[ "$status" -eq 0 ]]
}

evaluate_model() {
  local run="$1"
  local gpu="$2"
  echo "evaluating run=$run gpu=$gpu"
  evaluate_one "$run" "$gpu" train greedy 0 3000000
  evaluate_one "$run" "$gpu" val_ood greedy 0 4000000
  evaluate_one "$run" "$gpu" train sample 1 3000000
  evaluate_one "$run" "$gpu" val_ood sample 1 4000000
  echo "finished run=$run"
}

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
(
  evaluate_model base_video 1
  evaluate_model base_action_only 1
) >"$LOG_ROOT/modelsize200-gpu1.log" 2>&1 &
pid1=$!
(
  evaluate_model large_video 2
  evaluate_model large_action_only 2
) >"$LOG_ROOT/modelsize200-gpu2.log" 2>&1 &
pid2=$!
(
  evaluate_model small_video 3
  evaluate_model small_action_only 3
) >"$LOG_ROOT/modelsize200-gpu3.log" 2>&1 &
pid3=$!
wait "$pid1"
wait "$pid2"
wait "$pid3"
"$NANOWM_PY" "$ROOT/src/plot_xland_model_size_200eval.py" --results-root "$OUT_ROOT/results" --output-dir "$OUT_ROOT/plots"
echo "all final model-size evaluations and plots finished"
