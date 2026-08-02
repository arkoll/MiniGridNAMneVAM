#!/usr/bin/env bash
# Продолжение прогона на ещё ~120 тысяч шагов с живым статусом и мягкой остановкой.
#
# Настоящее продолжение, а не новый прогон: чекпоинт уходит в fit(ckpt_path=...), поэтому
# восстанавливаются и оптимизатор, и счётчик шагов. Параметры МОДЕЛИ не трогаются.
#
# Каталог прогона новый, чтобы не затирать первый: TensorBoard всё равно продолжит нумерацию
# шагов с 122 380, и склеенная кривая читается как одна.
#
# Стена и файл STOP делаются одним и тем же сторожевым циклом: он же раз в минуту
# переписывает live/LIVE.md.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
LIVE=$ROOT/live

GPU=${GPU:-0}
RUN_NAME=${RUN_NAME:-shapecraft_dagger_128_c2}
FROM_RUN=${FROM_RUN:-$ROOT/runs/shapecraft_dagger_128}
MAX_STEPS=${MAX_STEPS:-242000}
WALL_HOURS=${WALL_HOURS:-11.5}

RUN_DIR=$ROOT/runs/$RUN_NAME
LOG=$LIVE/train.log
mkdir -p "$LIVE"
rm -f "$LIVE/STOP" "$LIVE/TRAIN_DONE"
say() { echo "$(date '+%m-%d %H:%M:%S') | $*"; }

CKPT=$(ls -t "$FROM_RUN"/checkpoints/across_timesteps/*.ckpt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then say "нет чекпоинта в $FROM_RUN"; exit 1; fi
STEP0=$(b=${CKPT##*step=}; echo "${b%.ckpt}")

# hydra ломается на «=» внутри значения, а в имени чекпоинта их два.
# Ссылка без «=» снимает вопрос об экранировании целиком.
RESUME=$LIVE/resume_from.ckpt
ln -sf "$CKPT" "$RESUME"

say "продолжаю с $CKPT (шаг $STEP0)"
say "цель $MAX_STEPS шагов, стена ${WALL_HOURS} ч, карта $GPU, каталог $RUN_DIR"

T0=$(date +%s)
WALL_SEC=$(awk -v h="$WALL_HOURS" 'BEGIN{printf "%d", h*3600}')

cat > "$LIVE/meta.json" <<META
{
  "run1": "$FROM_RUN",
  "run2": "$RUN_DIR",
  "train_log": "$LOG",
  "max_steps": $MAX_STEPS,
  "step0": $STEP0,
  "t0": $T0,
  "wall_seconds": $WALL_SEC
}
META

GPU=$GPU RUN_NAME=$RUN_NAME MAX_STEPS=$MAX_STEPS \
  DATASET_CONFIG=xland/shapecraft_dagger \
  EXTRA_ARGS="experiment.resume_from_checkpoint=$RESUME experiment.training.val_every_n_steps=10000 experiment.training.checkpointing.across_timesteps.every_n_train_steps=10000 experiment.training.checkpointing.latest.every_n_train_steps=2000 experiment.joint_action.visualize_every_n_steps=40000" \
  bash $S/run_train_joint_ours.sh > "$LOG" 2>&1 &
TPID=$!
say "обучение поднято, pid $TPID"

asked=0
while kill -0 "$TPID" 2>/dev/null; do
  now=$(date +%s)
  reason=""
  [ -f "$LIVE/STOP" ] && reason="файл STOP"
  [ $((now - T0)) -ge "$WALL_SEC" ] && reason="${reason:-стена по времени}"
  if [ -n "$reason" ] && [ "$asked" -eq 0 ]; then
    say "останавливаю мягко: $reason"
    kill -INT "$TPID" 2>/dev/null
    asked=$(date +%s)
  fi
  # если за 6 минут не завершился — добиваем, чекпоинты каждые 2000 шагов уже на диске
  if [ "$asked" -ne 0 ] && [ $((now - asked)) -ge 360 ]; then
    say "мягкая остановка не сработала, снимаю жёстко"
    kill -9 "$TPID" 2>/dev/null
    break
  fi
  $ROOT/.venv-nanowm/bin/python $S/live_status.py 2>/dev/null
  sleep 60
done

wait "$TPID" 2>/dev/null
rc=$?
touch "$LIVE/TRAIN_DONE"
say "обучение закончилось, код $rc"
tail -3 "$LOG" | tr '\r' '\n' | tail -2 | while read -r l; do say "  $l"; done
say "чекпоинтов: $(ls "$RUN_DIR"/checkpoints/across_timesteps/*.ckpt 2>/dev/null | wc -l)"
$ROOT/.venv-nanowm/bin/python $S/live_status.py 2>/dev/null
say "DAGGER_CONT_STATUS=OK"
