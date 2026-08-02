#!/usr/bin/env bash
# Сбор датасетов ShapeCraft Easy: раскатка на GPU, рендер и запись на CPU, потом ворота.
# Очередь возобновляемая: раскатка пропускается по наличию отчёта, шарды — по отметкам.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
PY=$ROOT/.venv-xland/bin/python
S=$ROOT/scripts/xland
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
RUN=$ROOT/runs/xland-shape-collect-$STAMP
WORKERS=${WORKERS:-10}
STATUS=$RUN/STATUS.txt

mkdir -p "$RUN/code"
cp $S/*.py $S/*.sh "$RUN/code/" 2>/dev/null
sha256sum $S/*.py > "$RUN/code/sha256.txt"

say() { echo "$(date '+%m-%d %H:%M:%S') | $*" | tee -a "$STATUS"; }

say "старт сбора, карта $CUDA_VISIBLE_DEVICES, воркеров записи $WORKERS"
say "прогон $RUN"

SETS="train val_id val_ood"

# ------------------------------------------------------ 1. раскатка на GPU
for s in $SETS; do
  if [ -f "$ROOT/data/xland-shape-craft/_shards/rollout_$s.json" ]; then
    say "раскатка $s — уже есть, пропускаю"
    continue
  fi
  say "раскатка $s"
  (cd $S && $PY collect_rollout.py --set "$s") > "$RUN/rollout_$s.log" 2>&1
  if grep -q "ROLLOUT_${s^^}_STATUS=OK" "$RUN/rollout_$s.log"; then
    say "  $(grep -o '{.*}' "$RUN/rollout_$s.log" | tail -1 | cut -c1-200)"
  else
    say "раскатка $s УПАЛА — смотри $RUN/rollout_$s.log"
    say "COLLECT_STATUS=FAIL"
    exit 1
  fi
done

# --------------------------------------- 2. рендер и запись, только CPU
for s in $SETS; do
  say "запись $s: $WORKERS воркеров"
  pids=""
  for w in $(seq 0 $((WORKERS - 1))); do
    (cd $S && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
      $PY collect_write.py --set "$s" --worker "$w" --workers "$WORKERS") \
      > "$RUN/write_${s}_w${w}.log" 2>&1 &
    pids="$pids $!"
  done
  fail=0
  for pid in $pids; do wait "$pid" || fail=1; done
  ok=$(grep -l "WRITE_${s^^}_W.*_STATUS=OK" "$RUN"/write_${s}_w*.log 2>/dev/null | wc -l)
  say "запись $s: успешно завершилось воркеров $ok из $WORKERS"
  if [ "$ok" -ne "$WORKERS" ] || [ "$fail" -ne 0 ]; then
    say "запись $s НЕПОЛНАЯ — смотри $RUN/write_${s}_w*.log"
    say "COLLECT_STATUS=FAIL"
    exit 1
  fi
  say "  $(du -sh $ROOT/data/xland-shape-craft/$s | cut -f1) в наборе $s"
done

# ------------------------------------------ 3. ворота, манифесты, утечки
say "ворота, манифесты и отчёт по утечкам"
(cd $S && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu $PY finalize_datasets.py) > "$RUN/finalize.log" 2>&1
grep -E "^\[|^манифест|^утечки|^ *train:|^ *val" "$RUN/finalize.log" | while read -r l; do say "  $l"; done

if grep -q "FINALIZE_STATUS=OK" "$RUN/finalize.log"; then
  ln -sfn "$RUN" "$ROOT/runs/xland-shape-collect-latest"
  say "итог: $(du -sh $ROOT/data/xland-shape-craft | cut -f1)"
  say "COLLECT_STATUS=OK"
else
  say "ВОРОТА НЕ ПРОШЛИ — смотри $RUN/finalize.log"
  say "COLLECT_STATUS=FAIL"
  exit 1
fi
