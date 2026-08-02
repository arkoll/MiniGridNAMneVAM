#!/usr/bin/env bash
# Запись пары эпизодов для гифки: сервер действий (torch, GPU) + среда (jax, CPU).
# Чекпоинт и температура те же, что в замере для рис. 1: final_v4b, шаг 15000, T = 1.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
REPO=$ROOT/joint_wam/repos/nano-world-model
RUN_DIR=${RUN_DIR:-$ROOT/runs/final_v4b}
CKPT=${CKPT:-$RUN_DIR/checkpoints/across_timesteps/xland-joint-epoch=2-step=15000.ckpt}
GPU=${GPU:-2}
OUT=${OUT:-$ROOT/reports/xland/video_h1}
TREE=${TREE:-1}
SEED_BASE=${SEED_BASE:-41000000}
CANDIDATES=${CANDIDATES:-12}
TEMPERATURE=${TEMPERATURE:-1.0}
POLICY_SEED=${POLICY_SEED:-1234}
SOCK=/tmp/h1_video_$$.sock

mkdir -p "$OUT"
echo "$(date '+%H:%M:%S') | чекпоинт: $CKPT"
echo "$(date '+%H:%M:%S') | дерево $TREE, сиды от $SEED_BASE, T = $TEMPERATURE, карта $GPU"

CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 $ROOT/.venv-nanowm/bin/python \
  $ROOT/joint_wam/src/xland_action_server.py \
  --nanowm-repo "$REPO" \
  --config "$RUN_DIR/config.yaml" \
  --checkpoint "$CKPT" \
  --temperature "$TEMPERATURE" \
  --epsilon 0 \
  --seed "$POLICY_SEED" \
  --socket "$SOCK" > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null' EXIT

for i in $(seq 1 240); do
  grep -q "READY" "$OUT/server.log" 2>/dev/null && break
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "сервер действий упал:"; tail -20 "$OUT/server.log"; exit 1
  fi
  sleep 1
done
echo "$(date '+%H:%M:%S') | сервер поднят"

CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $ROOT/.venv-xland/bin/python \
  $S/record_h1_pair.py \
  --socket "$SOCK" --tree "$TREE" --seed-base "$SEED_BASE" \
  --candidates "$CANDIDATES" --out "$OUT"

kill $SERVER_PID 2>/dev/null
echo "$(date '+%H:%M:%S') | готово: $OUT"
