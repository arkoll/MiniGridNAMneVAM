#!/usr/bin/env bash
# Продолжение развёртки по температуре: ищем вершину и делаем сравнение с экспертом честным.
#
# 1. WAM при T = 3, 4, 6. Кривая обязана где-то развернуться: при бесконечной температуре
#    это равномерно случайная политика, а у неё доля побед ровно 0.000 (мерено при сборе).
#    Сейчас мы на восходящей ветке (0.465 → 0.795 → 0.905) и вершину не нашли.
# 2. Эксперт по той же сетке температур. Его эталон 0.845 получен ЖАДНО, то есть с той же
#    ручкой, которая нашей модели стоила 0.33 — сравнивать так нельзя.
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

# ждём, пока освободится карта от предыдущей развёртки
while pgrep -f "run_stochastic_sweep.sh" > /dev/null; do
  say "жду окончания первой развёртки"
  sleep 60
done

say "чекпоинт: шаг 120000; банк val; $EPISODES эпизодов; сиды 22 млн"

run_wam() {  # $1 температура
  local temp=$1
  local out=$BASE/temp_$temp
  if [ -f "$out/closed_loop_val.json" ]; then say "WAM T=$temp — уже есть, пропускаю"; return 0; fi
  mkdir -p "$out"
  say "WAM, температура $temp"
  RUN_DIR=$RUN_DIR CKPT=$CKPT BANK=val SEED=22000000 EPISODES=$EPISODES GPU=$GPU OUT=$out \
    TEMPERATURE=$temp EPSILON=0 POLICY_SEED=1234 \
    bash $S/run_closed_loop.sh > "$out/run.log" 2>&1
  if [ -f "$out/closed_loop_val.json" ]; then
    say "  $(grep -o '\"success_rate\": [0-9.]*' "$out/closed_loop_val.json" | head -1), $(grep -o '\"mean_length\": [0-9.]*' "$out/closed_loop_val.json" | head -1)"
  else
    say "  ОШИБКА, смотри $out/run.log"
  fi
}

run_wam 3.0
run_wam 4.0
run_wam 6.0

say "теперь эксперт по той же сетке (на JAX, секунды на точку)"
for t in 0.0 0.5 1.0 1.5 2.0 3.0 4.0 6.0; do
  f=$BASE/expert_T$t.json
  [ -f "$f" ] && { say "эксперт T=$t — уже есть"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    $ROOT/.venv-xland/bin/python $S/expert_reference_sr.py \
    --episodes "$EPISODES" --temperature "$t" --seed 1234 --out "$f" 2>&1 \
    | grep -E "^эксперт" | while read -r l; do say "  $l"; done
done

say "СВОДКА (банк val, 200 эпизодов, сиды 22 млн):"
$ROOT/.venv-nanowm/bin/python $S/summarize_stochastic.py
say "TEMP_SWEEP2_STATUS=OK"
