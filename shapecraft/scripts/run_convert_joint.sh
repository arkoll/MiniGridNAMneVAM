#!/usr/bin/env bash
# Конвертация наших наборов в формат joint-модели: hdf5 256 -> npz 128.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
PY=$ROOT/.venv-xland/bin/python
S=$ROOT/scripts/xland
DST=$ROOT/data/xland-shape-joint128
WORKERS=${WORKERS:-12}
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu

say() { echo "$(date '+%H:%M:%S') | $*"; }

for set_name in train val_id val_ood; do
  if [ -f "$DST/$set_name/_index.jsonl" ]; then
    say "$set_name — уже сконвертирован, пропускаю"
    continue
  fi
  say "конвертирую $set_name на $WORKERS воркерах"
  pids=""
  for w in $(seq 0 $((WORKERS - 1))); do
    (cd $S && $PY to_joint_npz.py --set "$set_name" --worker "$w" --workers "$WORKERS") &
    pids="$pids $!"
  done
  fail=0
  for pid in $pids; do wait "$pid" || fail=1; done
  if [ "$fail" -ne 0 ]; then
    say "$set_name УПАЛ"
    exit 1
  fi
  cat "$DST/$set_name"/_index_part_*.jsonl > "$DST/$set_name/_index.jsonl"
  rm -f "$DST/$set_name"/_index_part_*.jsonl
  n=$(ls "$DST/$set_name" | grep -c '^episode_train')
  say "$set_name готов: $n эпизодов, $(du -sh "$DST/$set_name" | cut -f1)"
done

# склейка валидаций в один каталог символическими ссылками: один даталоадер на оба набора,
# а раздельные числа считает ShapeCraftSplitActionMetrics по именам файлов
say "собираю val_both"
mkdir -p "$DST/val_both"
find "$DST/val_both" -type l -delete
for set_name in val_id val_ood; do
  for f in "$DST/$set_name"/episode_train_*.npz; do
    ln -sf "$f" "$DST/val_both/$(basename "$f")"
  done
done
say "val_both: $(ls "$DST/val_both" | wc -l) ссылок"

say "итог: $(du -sh --dereference "$DST" 2>/dev/null | cut -f1) в $DST"
say "CONVERT_STATUS=OK"
