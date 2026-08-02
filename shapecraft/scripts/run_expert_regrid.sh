#!/usr/bin/env bash
# Пересчёт сетки эксперта с поэпизодным выводом: нужен для исходной награды среды и её разброса.
# Ждёт освобождения карты от развёртки WAM. На JAX это меньше минуты на точку.
set -uo pipefail
ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
GPU=${GPU:-0}
BASE=$ROOT/reports/xland/stochastic
say() { echo "$(date '+%m-%d %H:%M:%S') | $*"; }

while pgrep -f "run_temp_sweep_train.sh" > /dev/null; do say "жду развёртку WAM"; sleep 60; done

for t in 0.0 0.5 1.0 1.5 2.0 3.0 4.0 6.0; do
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    $ROOT/.venv-xland/bin/python $S/expert_reference_sr.py \
    --episodes 200 --temperature "$t" --seed 1234 --out "$BASE/expert_T$t.json" 2>&1 \
    | grep -E "^эксперт" | while read -r l; do say "  $l"; done
done

say "собираю данные графика"
$ROOT/.venv-nanowm/bin/python $S/build_temp_chart_data.py
say "EXPERT_REGRID_STATUS=OK"
