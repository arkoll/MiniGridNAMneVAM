#!/usr/bin/env bash
# Пересчёт сравнения трёх датасетов под СЭМПЛИРОВАНИЕМ вместо argmax.
#
# Зачем. Все прежние числа (0.335 экспертный, 0.380 смешанный, 0.465 нынешний) получены
# жадной раскаткой, а она в этой среде стоит 0.44 доли побед даже привилегированному
# эксперту. На таких числах мы принимали решение о том, как собирать данные.
#
# Честность протокола: температура ОДНА для всех, 2.0, выбрана один раз и не подбирается
# под модель. Сиды среды и число эпизодов те же, что всегда.
#
# Шаги обучения выровнены: у всех трёх берём чекпоинт 30 000 шагов. У нынешнего датасета
# есть и 120 000, он уже посчитан отдельно (0.905 / 0.925) — сравнивать его с чужими
# тридцатью тысячами было бы нечестно.
#
# Порядок: сначала ВСЕ три на банке отложенных задач (это главный столбец), потом на
# банке знакомых. Если прогон прервать на середине, главное сравнение уже будет целым.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
GPU=${GPU:-1}
EPISODES=${EPISODES:-200}
TEMP=${TEMP:-2.0}
BASE=$ROOT/reports/xland/datasets_T2
mkdir -p "$BASE"
say() { echo "$(date '+%m-%d %H:%M:%S') | $*"; }

# имя | каталог прогона | чекпоинт
MODELS=(
  "mixed_random_labels|$ROOT/runs/shapecraft_joint_128|xland-joint-epoch=12-step=30000.ckpt"
  "expert_only|$ROOT/runs/shapecraft_expert_128|xland-joint-epoch=23-step=30000.ckpt"
  "mixed_expert_labels|$ROOT/runs/shapecraft_dagger_128|xland-joint-epoch=7-step=30000.ckpt"
)

say "температура $TEMP для всех, $EPISODES эпизодов, чекпоинты 30 000 шагов, карта $GPU"

run_one() {  # $1 имя, $2 каталог, $3 чекпоинт, $4 банк, $5 сид
  local name=$1 rundir=$2 ckpt=$3 bank=$4 seed=$5
  local out=$BASE/$name
  mkdir -p "$out"
  if [ -f "$out/closed_loop_$bank.json" ]; then say "$name / $bank — уже есть"; return 0; fi
  if [ ! -f "$rundir/checkpoints/across_timesteps/$ckpt" ]; then
    say "$name: НЕТ чекпоинта $ckpt"; return 1
  fi
  say "$name, банк $bank"
  RUN_DIR=$rundir CKPT="$rundir/checkpoints/across_timesteps/$ckpt" \
    BANK=$bank SEED=$seed EPISODES=$EPISODES GPU=$GPU OUT=$out \
    TEMPERATURE=$TEMP EPSILON=0 POLICY_SEED=1234 \
    bash $S/run_closed_loop.sh > "$out/run_$bank.log" 2>&1
  if [ -f "$out/closed_loop_$bank.json" ]; then
    say "  $(grep -o '\"success_rate\": [0-9.]*' "$out/closed_loop_$bank.json" | head -1), $(grep -o '\"mean_length\": [0-9.]*' "$out/closed_loop_$bank.json" | head -1)"
  else
    say "  ОШИБКА, смотри $out/run_$bank.log"; tail -4 "$out/run_$bank.log" | while read -r l; do say "    $l"; done
  fi
}

for bank_seed in "val|22000000" "train|21000000"; do
  bank=${bank_seed%%|*}; seed=${bank_seed##*|}
  say "--- банк $bank ---"
  for m in "${MODELS[@]}"; do
    IFS='|' read -r name rundir ckpt <<< "$m"
    run_one "$name" "$rundir" "$ckpt" "$bank" "$seed"
  done
done

say "--- сводка ---"
$ROOT/.venv-nanowm/bin/python $S/summarize_datasets_T2.py
say "DATASETS_T2_STATUS=OK"
