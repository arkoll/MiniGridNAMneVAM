#!/usr/bin/env bash
# Ночной прогон: смешанный набор с экспертной разметкой -> длинное обучение -> замкнутый контур.
#
# 1. Раскатка: по каждому набору две половины (эксперт / случайная), метка всегда экспертная.
# 2. Рендер в 128 и запись npz.
# 3. Короткий смоук обучения на новых данных: ловим проводку до того, как занять ночь.
# 4. Обучение joint-модели. Параметры МОДЕЛИ не трогаются: только длина прогона и периоды.
# 5. Замкнутый контур на обоих банках задач для двух чекпоинтов (середина и конец).
#
# Очередь возобновляемая: раскатка пропускается по наличию отчёта, шарды — по отметкам.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
PYX=$ROOT/.venv-xland/bin/python
DST=$ROOT/data/xland-dagger

GPU=${GPU:-2}
WORKERS=${WORKERS:-12}
PER_TASK_TRAIN=${PER_TASK_TRAIN:-47}   # на каждую половину: 47*324*2 = 30456 эпизодов
PER_TASK_VAL=${PER_TASK_VAL:-8}        # на каждую половину: 8*324*2 = 5184 эпизода на набор
RUN_NAME=${RUN_NAME:-shapecraft_dagger_128}
MAX_STEPS=${MAX_STEPS:-150000}         # реальную длину ограничит стена по времени
TRAIN_WALL=${TRAIN_WALL:-9h}
EPISODES=${EPISODES:-200}

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
RUN=$ROOT/runs/xland-dagger-$STAMP
mkdir -p "$RUN/code"
cp $S/dagger_*.py $S/shape_common.py $S/run_dagger_pipeline.sh "$RUN/code/" 2>/dev/null
sha256sum $S/dagger_*.py $S/shape_common.py > "$RUN/code/sha256.txt"
STATUS=$RUN/STATUS.txt
say() { echo "$(date '+%m-%d %H:%M:%S') | $*" | tee -a "$STATUS"; }

say "старт: смешанный сбор с экспертной разметкой, карта $GPU, воркеров записи $WORKERS"
say "квоты на половину: train $PER_TASK_TRAIN на задачу, валидации по $PER_TASK_VAL"
say "обучение: $RUN_NAME, потолок $MAX_STEPS шагов, стена $TRAIN_WALL"

# ------------------------------------------------------------ 1. раскатка
for s in train val_id val_ood; do
  for pol in expert random; do
    if [ -f "$DST/rollout_${s}_${pol}.json" ]; then
      say "раскатка $s/$pol — уже есть, пропускаю"
      continue
    fi
    q=$PER_TASK_VAL
    [ "$s" = "train" ] && q=$PER_TASK_TRAIN
    say "раскатка $s/$pol (квота $q на задачу)"
    (cd $S && CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
      $PYX dagger_rollout.py --set "$s" --policy "$pol" --per-task "$q") \
      > "$RUN/rollout_${s}_${pol}.log" 2>&1
    if grep -q "DAGGER_ROLLOUT_${s^^}_${pol^^}_STATUS=OK" "$RUN/rollout_${s}_${pol}.log"; then
      say "  $(grep -o '{.*}' "$RUN/rollout_${s}_${pol}.log" | tail -1 | cut -c1-300)"
    else
      say "раскатка $s/$pol УПАЛА — смотри $RUN/rollout_${s}_${pol}.log"
      say "DAGGER_STATUS=FAIL"; exit 1
    fi
  done
done

# --------------------------------------------------- 2. рендер и запись
for s in train val_id val_ood; do
  for pol in expert random; do
    say "запись $s/$pol: $WORKERS воркеров"
    pids=""
    for w in $(seq 0 $((WORKERS - 1))); do
      (cd $S && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
        $PYX dagger_write.py --set "$s" --policy "$pol" --worker "$w" --workers "$WORKERS") \
        > "$RUN/write_${s}_${pol}_w${w}.log" 2>&1 &
      pids="$pids $!"
    done
    fail=0
    for pid in $pids; do wait "$pid" || fail=1; done
    ok=$(grep -l "DAGGER_WRITE_${s^^}_${pol^^}_W.*_STATUS=OK" "$RUN"/write_${s}_${pol}_w*.log 2>/dev/null | wc -l)
    if [ "$ok" -ne "$WORKERS" ] || [ "$fail" -ne 0 ]; then
      say "запись $s/$pol НЕПОЛНАЯ ($ok из $WORKERS)"; say "DAGGER_STATUS=FAIL"; exit 1
    fi
  done
  cat "$DST/$s"/_index_part_*.jsonl > "$DST/$s/_index.jsonl" 2>/dev/null
  rm -f "$DST/$s"/_index_part_*.jsonl
  n=$(ls "$DST/$s" | grep -c '^episode_train')
  say "набор $s готов: $n эпизодов, $(du -sh "$DST/$s" | cut -f1)"
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

# ------------------------------------ 4. смоук обучения на новых данных
say "смоук обучения на новых данных (30 шагов)"
GPU=$GPU SMOKE=1 RUN_NAME=${RUN_NAME}_smoke MAX_STEPS=30 \
  DATASET_CONFIG=xland/shapecraft_dagger \
  bash $S/run_train_joint_ours.sh > "$RUN/train_smoke.log" 2>&1
if ! grep -q "step=0000030" "$RUN/train_smoke.log"; then
  say "смоук обучения УПАЛ — смотри $RUN/train_smoke.log"
  tail -15 "$RUN/train_smoke.log" | while read -r l; do say "  $l"; done
  say "DAGGER_STATUS=FAIL"; exit 1
fi
say "  $(grep 'ShapeCraft' "$RUN/train_smoke.log" | tail -2 | tr '\n' ' ')"

# ---------------------------------------------------- 5. обучение WAM
say "запускаю обучение WAM: $RUN_NAME (стена $TRAIN_WALL, дальше эвал в любом случае)"
timeout -k 5m -s INT "$TRAIN_WALL" \
  env GPU=$GPU RUN_NAME=$RUN_NAME MAX_STEPS=$MAX_STEPS \
  DATASET_CONFIG=xland/shapecraft_dagger \
  EXTRA_ARGS="experiment.training.val_every_n_steps=10000 experiment.training.checkpointing.across_timesteps.every_n_train_steps=10000 experiment.training.checkpointing.latest.every_n_train_steps=2000 experiment.joint_action.visualize_every_n_steps=20000" \
  bash $S/run_train_joint_ours.sh > "$RUN/train.log" 2>&1
rc=$?
say "обучение завершено (код $rc), последние строки:"
tail -3 "$RUN/train.log" | while read -r l; do say "  $l"; done

RUN_DIR=$ROOT/runs/$RUN_NAME
n_ck=$(ls "$RUN_DIR"/checkpoints/across_timesteps/*.ckpt 2>/dev/null | wc -l)
say "чекпоинтов сохранено: $n_ck"
ls "$RUN_DIR/checkpoints/across_timesteps" 2>/dev/null | while read -r l; do say "  $l"; done

# --------------------------------------- 6. замкнутый контур, два чекпоинта
LAST=$(ls -t "$RUN_DIR"/checkpoints/across_timesteps/*.ckpt 2>/dev/null | head -1)
MID=$(ls -tr "$RUN_DIR"/checkpoints/across_timesteps/*.ckpt 2>/dev/null | awk -v n="$n_ck" 'NR==int((n+1)/2)')
say "финальный чекпоинт: ${LAST:-НЕТ}"
say "средний чекпоинт:   ${MID:-НЕТ}"

if [ -z "$LAST" ]; then
  say "чекпоинтов нет, эвал невозможен"; say "DAGGER_STATUS=FAIL"; exit 1
fi

say "эталон экспертной политики на тех же банках и сидах"
mkdir -p "$ROOT/reports/xland/closed_loop_dagger"
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
  $PYX $S/expert_reference_sr.py --episodes "$EPISODES" \
  --out "$ROOT/reports/xland/closed_loop_dagger/expert_reference.json" \
  > "$RUN/expert_reference.log" 2>&1

for tag in final mid; do
  ck=$LAST
  [ "$tag" = "mid" ] && ck=$MID
  [ -z "$ck" ] && continue
  if [ "$tag" = "mid" ] && [ "$ck" = "$LAST" ]; then
    say "средний чекпоинт совпал с финальным, пропускаю"; continue
  fi
  for bank in val train; do
    seed=22000000
    [ "$bank" = "train" ] && seed=21000000
    out=$ROOT/reports/xland/closed_loop_dagger/$tag
    if [ -f "$out/closed_loop_$bank.json" ]; then
      say "$tag/$bank — уже посчитан, пропускаю"; continue
    fi
    say "замкнутый контур [$tag] банк $bank, $EPISODES эпизодов"
    RUN_DIR=$RUN_DIR CKPT=$ck BANK=$bank SEED=$seed EPISODES=$EPISODES GPU=$GPU OUT=$out \
      bash $S/run_closed_loop.sh > "$RUN/closed_loop_${tag}_${bank}.log" 2>&1
    if [ -f "$out/closed_loop_$bank.json" ]; then
      say "  $(grep -o '\"success_rate\"[^,]*' "$out/closed_loop_$bank.json" | head -1)"
    else
      say "  ОШИБКА, смотри $RUN/closed_loop_${tag}_${bank}.log"
    fi
  done
done

say "DAGGER_STATUS=OK"
say "данные: $DST | прогон: $RUN_DIR | контур: $ROOT/reports/xland/closed_loop_dagger"
