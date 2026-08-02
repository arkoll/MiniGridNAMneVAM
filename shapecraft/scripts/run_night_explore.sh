#!/usr/bin/env bash
# Второй проход: тот же полигон, но политика с epsilon-жадностью.
#
# Зачем. Обученная политика решает уровень за 8 шагов и настолько уверена, что
# температура на неё не действует (при T=3 успех остаётся 1.000). Значит в «экспертном»
# датасете нет ни одного провала и почти нет шагов, где агент прошёл рядом с ИНЕРТНОЙ
# фигурой и ничего не случилось. Без таких отрицательных примеров world-модель не может
# выучить, что реагируют не все формы, и вывод «правило по форме» будет не на чем основывать.
#
# Epsilon применяется одинаково ко всем четырём наборам, поэтому сравнение
# «знакомые цвета против невиданных» не портится.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
PY=$ROOT/.venv-xland/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export XLA_PYTHON_CLIENT_PREALLOCATE=false

STAMP=${STAMP:-$(date +%Y%m%d-%H%M%S)}
RUN=$ROOT/runs/xland-explore-$STAMP
DATA=$ROOT/data/xland-shape-explore
POLICY=${POLICY:?нужен путь к policy.pkl}
STATUS=$RUN/STATUS.txt

EPSILON=${EPSILON:-0.3}
EP_TRAIN=${EP_TRAIN:-12000}
EP_VAL=${EP_VAL:-3000}
GZIP=${GZIP:-4}

mkdir -p "$RUN/code" "$DATA"
cp $ROOT/scripts/xland/*.py "$RUN/code/" 2>/dev/null
sha256sum $ROOT/scripts/xland/*.py > "$RUN/code/sha256.txt"
echo "$POLICY" > "$RUN/policy_source.txt"

say() { echo "$(date '+%m-%d %H:%M:%S') | $*" | tee -a "$STATUS"; }

say "старт исследовательского сбора, карта $CUDA_VISIBLE_DEVICES, eps=$EPSILON"
say "политика: $POLICY"

# профиль epsilon: сколько провалов и какой длины эпизоды даёт каждое значение
if [ -f "$DATA/epsilon_profile_train.json" ]; then
  say "профиль epsilon — уже есть, пропускаю"
else
  say "профиль epsilon"
  cd $ROOT && $PY scripts/xland/collect_xland.py --benchmark train --episodes 1 --batch 512 \
      --profile-epsilon --policy "$POLICY" --out-root "$DATA" > "$RUN/epsilon.log" 2>&1
  grep -E "eps=|CHOSEN" "$RUN/epsilon.log" | while read -r l; do say "      $l"; done
fi

collect() {
  local bench=$1 eps_count=$2
  if [ -f "$DATA/report_$bench.json" ]; then
    say "сбор $bench — уже собран, пропускаю"
    return 0
  fi
  say "сбор $bench: $eps_count эпизодов, eps=$EPSILON"
  cd $ROOT && $PY scripts/xland/collect_xland.py --benchmark "$bench" --episodes "$eps_count" --batch 256 \
      --temperature 1.0 --epsilon "$EPSILON" --policy "$POLICY" --out-root "$DATA" --gzip "$GZIP" \
      > "$RUN/collect_$bench.log" 2>&1
  if ! grep -q "COLLECT_${bench^^}_STATUS=OK" "$RUN/collect_$bench.log"; then
    say "$bench УПАЛ — смотри $RUN/collect_$bench.log"
    return 1
  fi
  say "$bench собран: $(tail -2 "$RUN/collect_$bench.log" | head -1)"
  cd $ROOT && $PY scripts/xland/check_dataset.py --root "$DATA" --benchmark "$bench" \
      > "$RUN/check_$bench.log" 2>&1
  if grep -q "DATASET_STATUS=OK" "$RUN/check_$bench.log"; then
    say "$bench ворота пройдены: $(grep 'сводка' "$RUN/check_$bench.log")"
    say "      $(grep 'покрытие' "$RUN/check_$bench.log")"
  else
    say "$bench ВОРОТА НЕ ПРОШЛИ — смотри $RUN/check_$bench.log"
    return 1
  fi
}

FAILED=0
collect train "$EP_TRAIN" || FAILED=1
collect val_seen "$EP_VAL" || FAILED=1
collect val_ood_chain "$EP_VAL" || FAILED=1
collect val_ood_all "$EP_VAL" || FAILED=1

say "отчёт по утечкам"
cd $ROOT && $PY scripts/xland/leak_report.py --root "$DATA" > "$RUN/leak.log" 2>&1
grep -E "стартов|повторов" "$RUN/leak.log" | while read -r l; do say "      $l"; done

ln -sfn "$RUN" "$ROOT/runs/xland-explore-latest"
say "итог: $(du -sh "$DATA" | cut -f1) в $DATA"
if [ "$FAILED" -eq 0 ]; then say "EXPLORE_STATUS=OK"; else say "EXPLORE_STATUS=PARTIAL"; fi
