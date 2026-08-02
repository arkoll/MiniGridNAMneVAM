#!/usr/bin/env bash
# Вариант 4 отдельно, на второй карте, параллельно основному конвейеру.
#
# Берём именно его, потому что он ПОСЛЕДНИЙ в очереди основного конвейера: когда тот до
# него доберётся, чекпоинты и замеры уже будут на диске, и его собственные проверки
# «уже есть — пропускаю» сработают штатно. Гонки за один и тот же прогон не возникает.
#
# Воркеров загрузки данных меньше, чем у основного (5 против 8): ядер 16, и две обучалки
# с восемью воркерами каждая начали бы толкаться за процессор и замедлили бы обе.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
PYT=$ROOT/.venv-nanowm/bin/python
LIVE=$ROOT/live-final
OUT=$ROOT/reports/xland/final
GPU=${GPU:-0}
HALF=${HALF:-15000}
EPISODES=${EPISODES:-150}
WORKERS=${WORKERS:-5}

say() { echo "$(date '+%m-%d %H:%M:%S') | [карта $GPU] $*" | tee -a "$LIVE/STATUS.txt"; }
stopped() { [ -f "$LIVE/STOP" ]; }

say "вариант 4 на отдельной карте: фаза A и фаза B по $HALF шагов"

train_one() {
  local name=$1 cfg=$2 steps=$3 envs=$4 extra=$5
  if [ -n "$(ls "$ROOT/runs/$name"/checkpoints/across_timesteps/*.ckpt 2>/dev/null)" ]; then
    say "$name — уже есть, пропускаю"; return 0
  fi
  say "обучение $name ($cfg, $steps шагов)"
  local t0=$(date +%s)
  timeout -k 5m -s INT 170m env $envs \
    GPU=$GPU RUN_NAME=$name MAX_STEPS=$steps DATASET_CONFIG="xland/$cfg" \
    EXTRA_ARGS="experiment.infra.num_workers=$WORKERS experiment.training.val_every_n_steps=$((steps/3)) experiment.training.checkpointing.across_timesteps.every_n_train_steps=$((steps/6)) experiment.training.checkpointing.latest.every_n_train_steps=2000 experiment.joint_action.visualize_every_n_steps=$steps $extra" \
    bash $S/run_train_joint_ours.sh > "$LIVE/train_$name.log" 2>&1
  local rc=$?; say "  $name готово за $((($(date +%s)-t0)/60)) мин, код $rc, $(tail -c 400 "$LIVE/train_$name.log" | tr '\r' '\n' | grep -o 'step=[0-9]*' | tail -1)"
}

eval_one() {
  local m=$1 t=$2 bank=$3 seed=$4
  local o=$OUT/${m}_T${t}
  [ -f "$o/closed_loop_$bank.json" ] && { say "$m T=$t $bank — уже есть"; return 0; }
  local ck=$(ls -t "$ROOT/runs/$m"/checkpoints/across_timesteps/*.ckpt 2>/dev/null | head -1)
  [ -z "$ck" ] && { say "$m: нет чекпоинта"; return 1; }
  mkdir -p "$o"
  say "замер $m, T=$t, банк $bank"
  RUN_DIR=$ROOT/runs/$m CKPT=$ck BANK=$bank SEED=$seed EPISODES=$EPISODES GPU=$GPU OUT=$o \
    TEMPERATURE=$t EPSILON=0 POLICY_SEED=1234 \
    bash $S/run_closed_loop.sh > "$o/run_$bank.log" 2>&1
  say "  $(grep -o '"success_rate": [0-9.]*' "$o/closed_loop_$bank.json" 2>/dev/null | head -1 || echo ОШИБКА)"
}

stopped || train_one final_v4a final_v4a "$HALF" "XLAND_ACTION_FIELD=executed" ""

if ! stopped; then
  PRE=$(ls -t "$ROOT/runs/final_v4a"/checkpoints/across_timesteps/*.ckpt 2>/dev/null | head -1)
  if [ -z "$PRE" ]; then say "фаза A без чекпоинта, стоп"; exit 1; fi
  # своя ссылка, чтобы не столкнуться с основным конвейером; «=» в имени ломает hydra
  ln -sf "$PRE" "$LIVE/v4a_pretrained_par.ckpt"
  train_one final_v4b final_v4b "$HALF" "" "experiment.pretrained=$LIVE/v4a_pretrained_par.ckpt"
fi

for t in 1.0 0 2.0; do
  stopped && break
  eval_one final_v4b "$t" val 22000000
  [ "$t" = "2.0" ] && continue
  eval_one final_v4b "$t" train 21000000
done

$PYT $S/plot_final_bars.py > /dev/null 2>&1
say "V4_PARALLEL_STATUS=OK"
