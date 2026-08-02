#!/usr/bin/env bash
# Ждёт окончания основной очереди и только потом запускает исследовательский сбор.
# Отдельный процесс, чтобы не править работающий скрипт основной очереди.
set -uo pipefail

MAIN_STATUS=$1
POLICY=$2
ROOT=/home/user8_2/AIRI_WAM

echo "$(date '+%H:%M:%S') ждём NIGHT_STATUS в $MAIN_STATUS"
for _ in $(seq 1 480); do   # максимум 8 часов ожидания
  if grep -q "NIGHT_STATUS=" "$MAIN_STATUS" 2>/dev/null; then break; fi
  sleep 60
done

if ! grep -q "NIGHT_STATUS=" "$MAIN_STATUS" 2>/dev/null; then
  echo "$(date '+%H:%M:%S') основная очередь так и не закончилась, исследовательский сбор не запускаю"
  exit 1
fi

# Версия сборщика с epsilon подменяется ТОЛЬКО сейчас: пока шла основная очередь, все
# четыре экспертных набора должны были собираться одним и тем же кодом. Подмена
# переименованием, чтобы не портить работающие процессы.
if [ -f "$ROOT/scripts/xland/collect_xland_eps.py" ]; then
  mv -f "$ROOT/scripts/xland/collect_xland_eps.py" "$ROOT/scripts/xland/collect_xland.py"
  echo "$(date '+%H:%M:%S') сборщик обновлён до версии с epsilon"
fi

# отчёт по утечкам для экспертного набора — он к этому моменту уже собран целиком
echo "$(date '+%H:%M:%S') отчёт по утечкам экспертного набора"
cd $ROOT && $ROOT/.venv-xland/bin/python scripts/xland/leak_report.py \
    --root $ROOT/data/xland-shape > "$(dirname "$MAIN_STATUS")/leak.log" 2>&1
tail -8 "$(dirname "$MAIN_STATUS")/leak.log"

echo "$(date '+%H:%M:%S') запускаю исследовательский сбор"
exec env POLICY="$POLICY" EPSILON=0.3 EP_TRAIN=12000 EP_VAL=3000 \
     bash $ROOT/scripts/xland/run_night_explore.sh
