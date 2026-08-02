#!/usr/bin/env bash
# Сторож доли побед: следит за появлением чекпоинтов и гоняет по ним замкнутый контур.
#
# Работает на той же карте, что и обучение: свободна ровно одна, занимать чужие нельзя.
# Замер латентный (по одному кадру за запрос), карту не насыщает, но обучение притормаживает —
# это осознанный размен ради живой кривой.
#
# Порядок: сначала добираем недостающие точки первого прогона, потом идём за новыми
# чекпоинтами продолжения. Каждая точка пишется строкой в sr_curve.jsonl сразу, поэтому
# кривую видно, не дожидаясь конца.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
LIVE=$ROOT/live
GPU=${GPU:-0}
EPISODES=${EPISODES:-150}
BANK=${BANK:-val}
SEED=${SEED:-22000000}
RUN1=${RUN1:-$ROOT/runs/shapecraft_dagger_128}
RUN2=${RUN2:-$ROOT/runs/shapecraft_dagger_128_c2}
BACKFILL=${BACKFILL:-"20000 40000 80000 100000"}
EVERY=${EVERY:-20000}

mkdir -p "$LIVE"
CURVE=$LIVE/sr_curve.jsonl
touch "$CURVE"
say() { echo "$(date '+%m-%d %H:%M:%S') | $*"; }
state() { echo "$*" > "$LIVE/watcher_state.txt"; }

say "сторож запущен: карта $GPU, $EPISODES эпизодов на точку, банк $BANK"
say "добор первого прогона: $BACKFILL; продолжение: каждые $EVERY шагов"

have_step() { grep -q "\"step\": $1," "$CURVE" 2>/dev/null; }

measure() {  # $1 шаг, $2 путь к чекпоинту, $3 каталог прогона, $4 пометка
  local st=$1 ck=$2 rundir=$3 origin=$4
  local out=$LIVE/eval/step_$st
  mkdir -p "$out"
  state "считаю шаг $st ($origin), $EPISODES эпизодов, начато $(date '+%H:%M')"
  say "замер шага $st ($origin)"
  rm -f "$out/closed_loop_$BANK.json"
  RUN_DIR=$rundir CKPT=$ck BANK=$BANK SEED=$SEED EPISODES=$EPISODES GPU=$GPU OUT=$out \
    bash $S/run_closed_loop.sh > "$out/run.log" 2>&1
  if [ ! -f "$out/closed_loop_$BANK.json" ]; then
    say "  замер шага $st УПАЛ, смотри $out/run.log"
    state "замер шага $st упал, жду следующий чекпоинт"
    return 1
  fi
  $ROOT/.venv-nanowm/bin/python - "$out/closed_loop_$BANK.json" "$st" "$origin" "$CURVE" <<'PY'
import json, sys
src, step, origin, curve = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
d = json.load(open(src))
row = {
    "step": step,
    "success_rate": round(d["success_rate"], 4),
    "stderr": round(d.get("success_rate_stderr", 0.0), 4),
    "episodes": d.get("episodes"),
    "mean_length": round(d.get("mean_length", 0.0), 1),
    "origin": origin,
}
with open(curve, "a") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("SR", step, row["success_rate"])
PY
  say "  готово: $(grep -o '\"success_rate\": [0-9.]*' "$out/closed_loop_$BANK.json" | head -1)"
  return 0
}

step_of() { local b=${1##*step=}; echo "${b%.ckpt}"; }

while : ; do
  if [ -f "$LIVE/STOP" ]; then say "вижу STOP, выхожу"; state "остановлен по STOP"; break; fi

  did=0

  # 1. добор точек первого прогона
  for st in $BACKFILL; do
    [ -f "$LIVE/STOP" ] && break
    have_step "$st" && continue
    ck=$(ls "$RUN1"/checkpoints/across_timesteps/*step=$st.ckpt 2>/dev/null | head -1)
    [ -z "$ck" ] && continue
    measure "$st" "$ck" "$RUN1" "первый прогон"
    did=1
    break
  done

  # 2. новые чекпоинты продолжения, по возрастанию шага
  if [ "$did" -eq 0 ]; then
    for ck in $(ls "$RUN2"/checkpoints/across_timesteps/*.ckpt 2>/dev/null | sort -t= -k3 -n); do
      [ -f "$LIVE/STOP" ] && break
      st=$(step_of "$ck")
      [ -z "$st" ] && continue
      [ $((st % EVERY)) -ne 0 ] && continue
      have_step "$st" && continue
      measure "$st" "$ck" "$RUN2" "продолжение"
      did=1
      break
    done
  fi

  if [ "$did" -eq 0 ] && [ -f "$LIVE/TRAIN_DONE" ]; then say "обучение закончено и все точки посчитаны, выхожу"; state "все точки посчитаны"; break; fi
  if [ "$did" -eq 0 ]; then
    n=$(wc -l < "$CURVE")
    state "точек в кривой: $n, жду следующий чекпоинт (проверка каждые 3 мин)"
    sleep 180
  fi
done

say "SR_WATCHER_STATUS=DONE"
