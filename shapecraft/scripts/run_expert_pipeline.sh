#!/usr/bin/env bash
# Экспертный датасет (жадная политика, только успехи) и следом обучение WAM на нём.
# Очередь возобновляемая: раскатка пропускается по наличию отчёта, шарды — по отметкам.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
PYX=$ROOT/.venv-xland/bin/python
DST=$ROOT/data/xland-expert
GPU=${GPU:-2}
WORKERS=${WORKERS:-12}
PER_TASK_TRAIN=${PER_TASK_TRAIN:-31}
PER_TASK_VAL=${PER_TASK_VAL:-15}
RUN_NAME=${RUN_NAME:-shapecraft_expert_128}

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
RUN=$ROOT/runs/xland-expert-$STAMP
mkdir -p "$RUN/code"
cp $S/expert_*.py $S/shape_common.py $S/run_expert_pipeline.sh "$RUN/code/" 2>/dev/null
sha256sum $S/expert_*.py $S/shape_common.py > "$RUN/code/sha256.txt"
STATUS=$RUN/STATUS.txt
say() { echo "$(date '+%m-%d %H:%M:%S') | $*" | tee -a "$STATUS"; }

say "старт: экспертный сбор, карта $GPU, воркеров записи $WORKERS"
say "квоты: train $PER_TASK_TRAIN на задачу, валидации по $PER_TASK_VAL"

# ------------------------------------------------------------ 1. раскатка
for s in train val_id val_ood; do
  if [ -f "$DST/rollout_$s.json" ]; then
    say "раскатка $s — уже есть, пропускаю"
    continue
  fi
  q=$PER_TASK_VAL
  [ "$s" = "train" ] && q=$PER_TASK_TRAIN
  say "раскатка $s (квота $q на задачу)"
  (cd $S && CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    $PYX expert_rollout.py --set "$s" --per-task "$q") > "$RUN/rollout_$s.log" 2>&1
  if grep -q "EXPERT_ROLLOUT_${s^^}_STATUS=OK" "$RUN/rollout_$s.log"; then
    say "  $(grep -o '{.*}' "$RUN/rollout_$s.log" | tail -1 | cut -c1-260)"
  else
    say "раскатка $s УПАЛА — смотри $RUN/rollout_$s.log"; say "EXPERT_STATUS=FAIL"; exit 1
  fi
done

# --------------------------------------------------- 2. рендер и запись
for s in train val_id val_ood; do
  say "запись $s: $WORKERS воркеров"
  pids=""
  for w in $(seq 0 $((WORKERS - 1))); do
    (cd $S && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
      $PYX expert_write.py --set "$s" --worker "$w" --workers "$WORKERS") \
      > "$RUN/write_${s}_w${w}.log" 2>&1 &
    pids="$pids $!"
  done
  fail=0
  for pid in $pids; do wait "$pid" || fail=1; done
  ok=$(grep -l "EXPERT_WRITE_${s^^}_W.*_STATUS=OK" "$RUN"/write_${s}_w*.log 2>/dev/null | wc -l)
  if [ "$ok" -ne "$WORKERS" ] || [ "$fail" -ne 0 ]; then
    say "запись $s НЕПОЛНАЯ ($ok из $WORKERS)"; say "EXPERT_STATUS=FAIL"; exit 1
  fi
  cat "$DST/$s"/_index_part_*.jsonl > "$DST/$s/_index.jsonl" 2>/dev/null
  rm -f "$DST/$s"/_index_part_*.jsonl
  n=$(ls "$DST/$s" | grep -c '^episode_train')
  say "запись $s готова: $n эпизодов, $(du -sh "$DST/$s" | cut -f1)"
done

# ------------------------------------------- 3. склейка валидаций
say "собираю val_both"
mkdir -p "$DST/val_both"
find "$DST/val_both" -type l -delete
for s in val_id val_ood; do
  for f in "$DST/$s"/episode_train_*.npz; do ln -sf "$f" "$DST/val_both/$(basename "$f")"; done
done
say "val_both: $(ls "$DST/val_both" | wc -l) ссылок"
say "данные готовы: $(du -sh --dereference "$DST" 2>/dev/null | cut -f1)"

# ---------------------------------------------------- 4. обучение WAM
say "запускаю обучение WAM: $RUN_NAME"
(cd $ROOT && GPU=$GPU RUN_NAME=$RUN_NAME DATASET_CONFIG=xland/shapecraft_expert \
  bash $S/run_train_joint_ours.sh) > "$RUN/train.log" 2>&1
if grep -qE "^EXPERT|Done|`echo`" /dev/null; then :; fi
tail -3 "$RUN/train.log" | while read -r l; do say "  $l"; done
say "EXPERT_STATUS=OK (обучение см. $RUN/train.log и runs/$RUN_NAME)"
