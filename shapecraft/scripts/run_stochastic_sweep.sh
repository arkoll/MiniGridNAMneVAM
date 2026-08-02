#!/usr/bin/env bash
# Разделяющий опыт: ломает ли случайность предельный цикл.
#
# Гипотеза. Все 664 проигранных эпизода из 664 упёрлись ровно в лимит шагов, а выигранные
# идут за 27 шагов — как у эксперта. Значит модель не «играет слабо», а застревает: жадная
# политика без памяти после неверной догадки о правиле возвращается в то же состояние.
#
# Если это так, любая случайность в выборе действия должна заметно поднять долю побед,
# НИЧЕГО не меняя в модели. Если не поднимет — гипотеза неверна.
#
# Три режима на одном и том же чекпоинте и тех же сидах среды, что и эталон argmax=0.465:
#   T=1.0    сэмплирование по собственной неуверенности модели
#   T=2.0    та же неуверенность, растянутая (голова уверенная, t0=0.93)
#   eps=0.15 принудительный шум, работает независимо от уверенности
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
GPU=${GPU:-0}
EPISODES=${EPISODES:-200}
RUN_DIR=${RUN_DIR:-$ROOT/runs/shapecraft_dagger_128}
CKPT=${CKPT:-$RUN_DIR/checkpoints/across_timesteps/xland-joint-epoch=31-step=120000.ckpt}
BASE=$ROOT/reports/xland/stochastic
mkdir -p "$BASE"
say() { echo "$(date '+%m-%d %H:%M:%S') | $*"; }

say "чекпоинт: $CKPT"
say "эталон для сравнения: argmax, 200 эпизодов, доля побед 0.465"

run_one() {  # $1 имя, $2 температура, $3 эпсилон
  local name=$1 temp=$2 eps=$3
  local out=$BASE/$name
  if [ -f "$out/closed_loop_val.json" ]; then say "$name — уже посчитан, пропускаю"; return 0; fi
  mkdir -p "$out"
  say "режим $name (температура $temp, эпсилон $eps), $EPISODES эпизодов"
  RUN_DIR=$RUN_DIR CKPT=$CKPT BANK=val SEED=22000000 EPISODES=$EPISODES GPU=$GPU OUT=$out \
    TEMPERATURE=$temp EPSILON=$eps POLICY_SEED=1234 \
    bash $S/run_closed_loop.sh > "$out/run.log" 2>&1
  if [ -f "$out/closed_loop_val.json" ]; then
    say "  $(grep -o '\"success_rate\": [0-9.]*' "$out/closed_loop_val.json" | head -1), $(grep -o '\"mean_length\": [0-9.]*' "$out/closed_loop_val.json" | head -1)"
  else
    say "  ОШИБКА, смотри $out/run.log"; tail -5 "$out/run.log" | while read -r l; do say "    $l"; done
  fi
}

run_one temp_1.0 1.0 0
run_one temp_2.0 2.0 0
run_one eps_0.15 0   0.15

say "итог:"
python3 $S/why_fail.py 2>/dev/null | head -1
for d in "$BASE"/*/; do
  n=$(basename "$d")
  [ -f "$d/closed_loop_val.json" ] || continue
  say "  $n: $(grep -o '\"success_rate\": [0-9.]*' "$d/closed_loop_val.json" | head -1) $(grep -o '\"mean_length\": [0-9.]*' "$d/closed_loop_val.json" | head -1)"
done
say "STOCHASTIC_SWEEP_STATUS=OK"
