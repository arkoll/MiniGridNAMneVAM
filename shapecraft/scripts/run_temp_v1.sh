#!/usr/bin/env bash
# Развёртка по температуре для модели, обученной на ПОЛНОСТЬЮ экспертном датасете
# (вариант 1 итогового сравнения: экспертные состояния + экспертные действия).
#
# Сетка температур и число эпизодов те же, что у прежнего графика temp_success_rate.png,
# и банки те же (сиды 21 и 22 млн), чтобы две картинки читались рядом.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
GPU=${GPU:-0}
EPISODES=${EPISODES:-200}
RUN_DIR=$ROOT/runs/final_v1
CKPT=$RUN_DIR/checkpoints/across_timesteps/xland-joint-epoch=2-step=30000.ckpt
BASE=$ROOT/reports/xland/temp_v1
mkdir -p "$BASE"
say() { echo "$(date '+%m-%d %H:%M:%S') | $*"; }

say "модель: полностью экспертный датасет, 30 000 шагов, $EPISODES эпизодов на точку"

for t in 0 1.0 2.0 3.0 4.0 6.0; do
  for bs in "val|22000000" "train|21000000"; do
    bank=${bs%%|*}; seed=${bs##*|}
    out=$BASE/T$t
    [ -f "$out/closed_loop_$bank.json" ] && { say "T=$t $bank — уже есть"; continue; }
    mkdir -p "$out"
    say "T=$t, банк $bank"
    RUN_DIR=$RUN_DIR CKPT=$CKPT BANK=$bank SEED=$seed EPISODES=$EPISODES GPU=$GPU OUT=$out \
      TEMPERATURE=$t EPSILON=0 POLICY_SEED=1234 \
      bash $S/run_closed_loop.sh > "$out/run_$bank.log" 2>&1
    say "  $(grep -o '\"success_rate\": [0-9.]*' "$out/closed_loop_$bank.json" 2>/dev/null | head -1 || echo ОШИБКА)"
  done
  $ROOT/.venv-nanowm/bin/python $S/plot_temp_v1.py > /dev/null 2>&1
done

say "строю график"
$ROOT/.venv-nanowm/bin/python $S/plot_temp_v1.py 2>&1 | grep -v Warning | tail -3
say "TEMP_V1_STATUS=OK"
