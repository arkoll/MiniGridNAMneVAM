#!/usr/bin/env bash
# Ночная очередь: ворота -> обучение PPO -> профиль температуры -> сбор четырёх наборов.
# Очередь возобновляемая: готовые единицы работы пропускаются по наличию отчёта.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
PY=$ROOT/.venv-xland/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
RUN=$ROOT/runs/xland-night-$STAMP
DATA=$ROOT/data/xland-shape
STATUS=$RUN/STATUS.txt

PPO_STEPS=${PPO_STEPS:-50000000}
EP_TRAIN=${EP_TRAIN:-12000}
EP_VAL=${EP_VAL:-3000}
GZIP=${GZIP:-4}

mkdir -p "$RUN/code" "$DATA"
cp $ROOT/scripts/xland/*.py "$RUN/code/" 2>/dev/null
sha256sum $ROOT/scripts/xland/*.py > "$RUN/code/sha256.txt"

say() { echo "$(date '+%m-%d %H:%M:%S') | $*" | tee -a "$STATUS"; }

say "старт, карта $CUDA_VISIBLE_DEVICES, прогон $RUN"
say "параметры: PPO_STEPS=$PPO_STEPS EP_TRAIN=$EP_TRAIN EP_VAL=$EP_VAL GZIP=$GZIP"

# ------------------------------------------------------------------ 1. ворота
if grep -q "VALIDATE_STATUS=OK" "$RUN/validate.log" 2>/dev/null; then
  say "1/7 ворота среды — уже пройдены, пропускаю"
else
  say "1/7 ворота среды"
  cd $ROOT && $PY scripts/xland/validate_env.py > "$RUN/validate.log" 2>&1
  if ! grep -q "VALIDATE_STATUS=OK" "$RUN/validate.log"; then
    say "ВОРОТА НЕ ПРОШЛИ — очередь остановлена, смотри $RUN/validate.log"
    say "NIGHT_STATUS=FAIL"
    exit 1
  fi
  say "1/7 ворота пройдены"
fi

# -------------------------------------------------------------- 2. обучение
if [ -f "$RUN/ppo/ppo_report.json" ]; then
  say "2/7 PPO — уже обучен, пропускаю"
else
  say "2/7 обучение PPO на $PPO_STEPS переходов"
  cd $ROOT && $PY scripts/xland/ppo_xland.py --total-timesteps "$PPO_STEPS" --out "$RUN/ppo" > "$RUN/ppo.log" 2>&1
  if ! grep -q "PPO_STATUS=OK" "$RUN/ppo.log"; then
    say "PPO УПАЛ — очередь остановлена, смотри $RUN/ppo.log"
    say "NIGHT_STATUS=FAIL"
    exit 1
  fi
  say "2/7 PPO обучен: $(grep 'тыс. шагов/с' "$RUN/ppo.log" | tail -1)"
  grep -A0 'жадно success' "$RUN/ppo.log" | while read -r l; do say "      $l"; done
fi
POLICY="$RUN/ppo/policy.pkl"

# ------------------------------------------------- 3. рабочая точка сбора
if [ -f "$DATA/temperature_profile_train.json" ]; then
  say "3/7 профиль температуры — уже есть, пропускаю"
else
  say "3/7 профиль температуры"
  cd $ROOT && $PY scripts/xland/collect_xland.py --benchmark train --episodes 1 --batch 512 \
      --profile-temperature --policy "$POLICY" --out-root "$DATA" > "$RUN/temperature.log" 2>&1
  grep -E "T=|CHOSEN" "$RUN/temperature.log" | while read -r l; do say "      $l"; done
fi
TEMP=$($PY -c "import json;print(json.load(open('$DATA/temperature_profile_train.json'))['chosen']['temperature'])")
say "3/7 рабочая температура: $TEMP"

# --------------------------------------------------------------- 4-7. сбор
collect() {
  local bench=$1 eps=$2 step=$3
  if [ -f "$DATA/report_$bench.json" ]; then
    say "$step/7 сбор $bench — уже собран, пропускаю"
    return 0
  fi
  say "$step/7 сбор $bench: $eps эпизодов, T=$TEMP"
  cd $ROOT && $PY scripts/xland/collect_xland.py --benchmark "$bench" --episodes "$eps" --batch 256 \
      --temperature "$TEMP" --policy "$POLICY" --out-root "$DATA" --gzip "$GZIP" > "$RUN/collect_$bench.log" 2>&1
  if ! grep -q "COLLECT_${bench^^}_STATUS=OK" "$RUN/collect_$bench.log"; then
    say "$step/7 $bench УПАЛ — смотри $RUN/collect_$bench.log"
    return 1
  fi
  say "$step/7 $bench собран: $(tail -2 "$RUN/collect_$bench.log" | head -1)"

  # ворота по датасету: кадры на диске обязаны совпадать с рендером состояний
  cd $ROOT && $PY scripts/xland/check_dataset.py --root "$DATA" --benchmark "$bench" \
      > "$RUN/check_$bench.log" 2>&1
  if grep -q "DATASET_STATUS=OK" "$RUN/check_$bench.log"; then
    say "$step/7 $bench ворота датасета пройдены: $(grep 'сводка' "$RUN/check_$bench.log")"
  else
    say "$step/7 $bench ВОРОТА ДАТАСЕТА НЕ ПРОШЛИ — смотри $RUN/check_$bench.log"
    return 1
  fi
}

FAILED=0
collect train        "$EP_TRAIN" 4 || FAILED=1
collect val_seen     "$EP_VAL"   5 || FAILED=1
collect val_ood_chain "$EP_VAL"  6 || FAILED=1
collect val_ood_all  "$EP_VAL"   7 || FAILED=1

ln -sfn "$RUN" "$ROOT/runs/xland-night-latest"
say "итог: $(du -sh "$DATA" | cut -f1) в $DATA"
if [ "$FAILED" -eq 0 ]; then
  say "NIGHT_STATUS=OK"
else
  say "NIGHT_STATUS=PARTIAL — часть наборов не собралась"
fi
