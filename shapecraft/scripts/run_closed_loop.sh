#!/usr/bin/env bash
# Замкнутый контур: поднимаем сервер действий (torch), гоняем среду (jax), считаем долю побед.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
RUN_DIR=${RUN_DIR:-$ROOT/runs/shapecraft_expert_128}
REPO=$ROOT/joint_wam/repos/nano-world-model
GPU=${GPU:-2}
EPISODES=${EPISODES:-200}
OUT=${OUT:-$ROOT/reports/xland/closed_loop}
SOCK=/tmp/shapecraft_wam_$$.sock

mkdir -p "$OUT"
CKPT=${CKPT:-$(ls -t "$RUN_DIR"/checkpoints/latest/*.ckpt | head -1)}
echo "$(date '+%H:%M:%S') | чекпоинт: $CKPT"
echo "$(date '+%H:%M:%S') | эпизодов на банк: $EPISODES"

# по умолчанию нули — это прежнее поведение (argmax), ничего в старых замерах не меняется
TEMPERATURE=${TEMPERATURE:-0}
EPSILON=${EPSILON:-0}
POLICY_SEED=${POLICY_SEED:-1234}

CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 $ROOT/.venv-nanowm/bin/python \
  $ROOT/joint_wam/src/xland_action_server.py \
  --nanowm-repo "$REPO" \
  --config "$RUN_DIR/config.yaml" \
  --checkpoint "$CKPT" \
  --temperature "$TEMPERATURE" \
  --epsilon "$EPSILON" \
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

# банк задаётся снаружи: сервер держит одно соединение за запуск
BANK=${BANK:-val}
SEED=${SEED:-22000000}
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $ROOT/.venv-xland/bin/python \
  $S/eval_closed_loop.py \
  --socket "$SOCK" --bank "$BANK" --episodes "$EPISODES" \
  --seed-base "$SEED" --out "$OUT/closed_loop_$BANK.json"

kill $SERVER_PID 2>/dev/null
echo "$(date '+%H:%M:%S') | готово: $OUT/closed_loop_$BANK.json"
