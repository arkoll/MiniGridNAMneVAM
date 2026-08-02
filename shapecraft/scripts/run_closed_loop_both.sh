#!/usr/bin/env bash
# Оба банка задач в замкнутом контуре плюс эталон экспертной политики на тех же сидах.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
OUT=${OUT:-$ROOT/reports/xland/closed_loop}
GPU=${GPU:-2}
EPISODES=${EPISODES:-200}
mkdir -p "$OUT"
say() { echo "$(date '+%H:%M:%S') | $*"; }

say "эталон экспертной политики на тех же банках и сидах"
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
  $ROOT/.venv-xland/bin/python $S/expert_reference_sr.py \
  --episodes "$EPISODES" --out "$OUT/expert_reference.json" 2>&1 | grep -viE "warn|deprecat"

for bank in val train; do
  seed=22000000
  [ "$bank" = "train" ] && seed=21000000
  if [ -f "$OUT/closed_loop_$bank.json" ]; then
    say "банк $bank — уже посчитан, пропускаю"
    continue
  fi
  say "замкнутый контур, банк $bank, $EPISODES эпизодов"
  BANK=$bank SEED=$seed EPISODES=$EPISODES GPU=$GPU OUT=$OUT \
    bash $S/run_closed_loop.sh 2>&1 | grep -viE "warn|deprecat" | tail -6
done

say "CLOSED_LOOP_BOTH_STATUS=OK"
