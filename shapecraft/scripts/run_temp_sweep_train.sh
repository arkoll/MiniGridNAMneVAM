#!/usr/bin/env bash
# Та же развёртка по температуре, но на банке ОБУЧАЮЩИХ задач (сиды 21 млн).
#
# Соответствие названий: банк train с невиданными сидами — это val_id (знакомая привязка
# цвета к форме, новая раскладка), банк val — это val_ood (комплементарные привязки).
# Точка T=0 уже посчитана раньше: argmax на этом же чекпоинте дал 0.470.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
GPU=${GPU:-0}
EPISODES=${EPISODES:-200}
RUN_DIR=$ROOT/runs/shapecraft_dagger_128
CKPT=$RUN_DIR/checkpoints/across_timesteps/xland-joint-epoch=31-step=120000.ckpt
BASE=$ROOT/reports/xland/stochastic
mkdir -p "$BASE"
say() { echo "$(date '+%m-%d %H:%M:%S') | $*"; }

while pgrep -f "run_temp_sweep2.sh" > /dev/null; do say "жду предыдущую развёртку"; sleep 60; done

say "банк train (=val_id), сиды 21 млн, чекпоинт 120000, $EPISODES эпизодов"

for temp in 1.0 2.0 3.0 4.0 6.0; do
  out=$BASE/train_temp_$temp
  if [ -f "$out/closed_loop_train.json" ]; then say "T=$temp — уже есть"; continue; fi
  mkdir -p "$out"
  say "WAM, банк train, температура $temp"
  RUN_DIR=$RUN_DIR CKPT=$CKPT BANK=train SEED=21000000 EPISODES=$EPISODES GPU=$GPU OUT=$out \
    TEMPERATURE=$temp EPSILON=0 POLICY_SEED=1234 \
    bash $S/run_closed_loop.sh > "$out/run.log" 2>&1
  if [ -f "$out/closed_loop_train.json" ]; then
    say "  $(grep -o '\"success_rate\": [0-9.]*' "$out/closed_loop_train.json" | head -1), $(grep -o '\"mean_length\": [0-9.]*' "$out/closed_loop_train.json" | head -1)"
  else
    say "  ОШИБКА, смотри $out/run.log"
  fi
done

say "TEMP_SWEEP_TRAIN_STATUS=OK"
