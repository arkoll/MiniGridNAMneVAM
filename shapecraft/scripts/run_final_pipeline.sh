#!/usr/bin/env bash
# ИТОГОВОЕ сравнение четырёх подходов к составу датасета. Одна карта, одна ночь.
#
# Все четыре прогона отличаются РОВНО составом обучающих данных. Совпадают: модель,
# оптимизатор, число шагов, сиды, валидация, число ОКОН в обучающем пуле.
#
#   1  только экспертные состояния и экспертные действия
#   2  половина окон экспертных, половина случайных; метка везде экспертная
#   3  те же кадры, метка — фактически выполненное действие
#   4  те же данные, двухфазно: сначала случайные (15k шагов), затем голова действий
#      создаётся заново и учится на экспертных (15k шагов). Суммарный бюджет тот же.
#
# ЧТО КОНТРОЛИРУЕТСЯ. Число окон в пуле у всех ~3.05 млн, но обучение успевает посмотреть
# только 240 000 окон (30k шагов x batch 8), поэтому пул нигде не исчерпывается и повторов
# нет ни у кого. Реальный контроль — одинаковое число ШАГОВ. Равенство пулов нужно, чтобы
# ни один вариант не начал повторять данные; это фиксируется, а не выдаётся за главное.
#
# Случайная половина режется отрезками. 190 окон с одного случайного эпизода — почти
# дубликаты: случайная политика в ShapeCraft не крафтит ни разу, три объекта стоят на месте.
# Поэтому вместо 8 100 эпизодов целиком берём 59 616 эпизодов по отрезку в 29 кадров:
# то же число окон, в 7.4 раза больше РАЗНЫХ раскладок, начало отрезка равномерно по эпизоду.
#
# Замер: замкнутый контур через настоящую среду, два банка задач. Температуры считаются
# в порядке 1 -> 0 -> 2, чтобы при нехватке ночи главный график был целым. Двойка нужна
# как страховка: состав данных меняет энтропию головы по построению, поэтому одна
# номинальная температура ставит варианты в разные точки их собственных кривых.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
S=$ROOT/scripts/xland
PYX=$ROOT/.venv-xland/bin/python
PYT=$ROOT/.venv-nanowm/bin/python
DST=$ROOT/data/final
LIVE=$ROOT/live-final

GPU=${GPU:-0}
WORKERS=${WORKERS:-12}
STEPS=${STEPS:-30000}
HALF=$((STEPS / 2))
EPISODES=${EPISODES:-150}
TRAIN_WALL=${TRAIN_WALL:-170m}
SLICE=${SLICE:-29}

EXPERT_PER_TASK=${EXPERT_PER_TASK:-368}              # 119 232 эпизода ~ 3.01 млн окон
RANDOM_PER_TASK=${RANDOM_PER_TASK:-184}              #  59 616 отрезков ~ 1.55 млн окон
MIXED_EXPERT_PER_TASK=${MIXED_EXPERT_PER_TASK:-184}  #  59 616 эпизодов ~ 1.51 млн окон
VAL_EXPERT_PER_TASK=${VAL_EXPERT_PER_TASK:-6}
VAL_RANDOM_PER_TASK=${VAL_RANDOM_PER_TASK:-6}

mkdir -p "$LIVE"
rm -f "$LIVE/STOP"
STATUS=$LIVE/STATUS.txt
say() { echo "$(date '+%m-%d %H:%M:%S') | $*" | tee -a "$STATUS"; }
stopped() { [ -f "$LIVE/STOP" ]; }

say "=== ИТОГОВЫЙ ПРОГОН === карта $GPU, $STEPS шагов на вариант, $EPISODES эпизодов на замер"

# ------------------------------------------------------------------ 1. раскатка
export DAGGER_DST=$DST
roll() {
  if [ -f "$DST/rollout_$1_$2.json" ]; then say "раскатка $1/$2 — уже есть"; return 0; fi
  say "раскатка $1/$2, квота $3 на задачу"
  (cd $S && CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false \
    $PYX dagger_rollout.py --set "$1" --policy "$2" --per-task "$3" --batch 1296 --max-rounds 600) \
    > "$LIVE/rollout_$1_$2.log" 2>&1
  if ! grep -q "DAGGER_ROLLOUT_${1^^}_${2^^}_STATUS=OK" "$LIVE/rollout_$1_$2.log"; then
    say "раскатка $1/$2 УПАЛА — $LIVE/rollout_$1_$2.log"; exit 1
  fi
  say "  $(grep -o '\"episodes\": [0-9]*' "$LIVE/rollout_$1_$2.log" | tail -1) эпизодов"
}
roll train   expert "$EXPERT_PER_TASK"
roll train   random "$RANDOM_PER_TASK"
roll val_id  expert "$VAL_EXPERT_PER_TASK"
roll val_id  random "$VAL_RANDOM_PER_TASK"
roll val_ood expert "$VAL_EXPERT_PER_TASK"
roll val_ood random "$VAL_RANDOM_PER_TASK"

# --------------------------------------------------------------- 2. рендер
for s in train val_id val_ood; do
  # Без этой проверки перезапуск ломает всё: воркеры пропустят шарды по отметкам, напишут
  # ПУСТЫЕ куски индекса, и склейка затрёт готовый _index.jsonl, по которому строятся
  # представления. Один раз уже наступили.
  if [ -s "$DST/$s/_index.jsonl" ]; then
    say "набор $s — индекс на месте ($(wc -l < "$DST/$s/_index.jsonl") строк), рендер пропускаю"
    continue
  fi
  for pol in expert random; do
    sl=0; [ "$pol" = "random" ] && sl=$SLICE
    say "рендер $s/$pol, $WORKERS воркеров, отрезок ${sl:-целиком}"
    pids=""
    for w in $(seq 0 $((WORKERS - 1))); do
      (cd $S && CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu \
        $PYX dagger_write.py --set "$s" --policy "$pol" --worker "$w" --workers "$WORKERS" \
        --slice-frames "$sl") > "$LIVE/write_${s}_${pol}_w${w}.log" 2>&1 &
      pids="$pids $!"
    done
    fail=0; for pid in $pids; do wait "$pid" || fail=1; done
    ok=$(grep -l "DAGGER_WRITE_${s^^}_${pol^^}_W.*_STATUS=OK" "$LIVE"/write_${s}_${pol}_w*.log 2>/dev/null | wc -l)
    if [ "$ok" -ne "$WORKERS" ] || [ "$fail" -ne 0 ]; then
      say "рендер $s/$pol НЕПОЛНЫЙ ($ok из $WORKERS)"; exit 1
    fi
  done
  cat "$DST/$s"/_index_part_*.jsonl > "$DST/$s/_index.jsonl" 2>/dev/null
  rm -f "$DST/$s"/_index_part_*.jsonl
  say "набор $s готов: $(ls "$DST/$s" | grep -c '^episode_train') эпизодов, $(du -sh "$DST/$s" | cut -f1)"
done

# ------------------------------------------------------- 3. представления и конфиги
say "собираю представления"
$PYT $S/build_final_views.py --root "$DST" --expert-per-task "$MIXED_EXPERT_PER_TASK" \
  2>&1 | while read -r l; do say "  $l"; done
bash $S/make_final_configs.sh | while read -r l; do say "  $l"; done

# ------------------------------------------------------------------ 4. обучение
train_one() {
  local name=$1 cfg=$2 steps=$3 envs=$4 extra=$5
  if [ -n "$(ls "$ROOT/runs/$name"/checkpoints/across_timesteps/*.ckpt 2>/dev/null)" ]; then
    say "обучение $name — уже есть, пропускаю"; return 0
  fi
  say "обучение $name ($cfg, $steps шагов)"
  local t0=$(date +%s)
  timeout -k 5m -s INT "$TRAIN_WALL" env $envs \
    GPU=$GPU RUN_NAME=$name MAX_STEPS=$steps DATASET_CONFIG="xland/$cfg" \
    EXTRA_ARGS="experiment.training.val_every_n_steps=$((steps/3)) experiment.training.checkpointing.across_timesteps.every_n_train_steps=$((steps/6)) experiment.training.checkpointing.latest.every_n_train_steps=2000 experiment.joint_action.visualize_every_n_steps=$steps $extra" \
    bash $S/run_train_joint_ours.sh > "$LIVE/train_$name.log" 2>&1
  local rc=$?
  say "  $name готово за $((($(date +%s)-t0)/60)) мин, код $rc, $(tail -c 400 "$LIVE/train_$name.log" | tr '\r' '\n' | grep -o 'step=[0-9]*' | tail -1)"
}

OUT=$ROOT/reports/xland/final
mkdir -p "$OUT"
eval_one() {
  local m=$1 t=$2 bank=$3 seed=$4
  local o=$OUT/${m}_T${t}
  [ -f "$o/closed_loop_$bank.json" ] && { say "$m T=$t $bank — уже есть"; return 0; }
  local ck=$(ls -t "$ROOT/runs/$m"/checkpoints/across_timesteps/*.ckpt 2>/dev/null | head -1)
  [ -z "$ck" ] && { say "$m: нет чекпоинта, пропускаю"; return 1; }
  mkdir -p "$o"
  say "замер $m, T=$t, банк $bank"
  RUN_DIR=$ROOT/runs/$m CKPT=$ck BANK=$bank SEED=$seed EPISODES=$EPISODES GPU=$GPU OUT=$o \
    TEMPERATURE=$t EPSILON=0 POLICY_SEED=1234 \
    bash $S/run_closed_loop.sh > "$o/run_$bank.log" 2>&1
  if [ -f "$o/closed_loop_$bank.json" ]; then
    say "  $(grep -o '"success_rate": [0-9.]*' "$o/closed_loop_$bank.json" | head -1)"
  else
    say "  ОШИБКА, $o/run_$bank.log"
  fi
}

# Обучение и замер ЧЕРЕДУЮТСЯ: вариант измеряется сразу, как обучится, и графики
# перестраиваются. Если время кончится раньше конвейера, готовые варианты уже будут
# на графике целиком, а не окажутся все недомеренными.
measure_model() {
  local m=$1
  for t in 1.0 0; do
    stopped && return 0
    eval_one "$m" "$t" val   22000000
    eval_one "$m" "$t" train 21000000
  done
  $PYT $S/plot_final_bars.py > /dev/null 2>&1
  say "  графики перестроены после $m"
}

stopped || train_one final_v1  final_v1  "$STEPS" "" ""
stopped || measure_model final_v1
stopped || train_one final_v2  final_v2  "$STEPS" "" ""
stopped || measure_model final_v2
stopped || train_one final_v3  final_v2  "$STEPS" "XLAND_ACTION_FIELD=executed" ""
stopped || measure_model final_v3
stopped || train_one final_v4a final_v4a "$HALF"  "XLAND_ACTION_FIELD=executed" ""
if ! stopped; then
  PRE=$(ls -t "$ROOT/runs/final_v4a"/checkpoints/across_timesteps/*.ckpt 2>/dev/null | head -1)
  if [ -n "$PRE" ]; then
    ln -sf "$PRE" "$LIVE/v4a_pretrained.ckpt"   # в имени чекпоинта есть «=», hydra на нём спотыкается
    train_one final_v4b final_v4b "$HALF" "" "experiment.pretrained=$LIVE/v4a_pretrained.ckpt"
    measure_model final_v4b
  else
    say "фаза A не дала чекпоинта, вариант 4 пропущен"
  fi
fi

MODELS="final_v1 final_v2 final_v3 final_v4b"
say "--- температура 2.0, только отложенные (страховка) ---"
for m in $MODELS; do
  stopped && break
  eval_one "$m" 2.0 val 22000000
done

# --------------------------------------------- 6. энтропия головы у каждого варианта
# Состав данных меняет энтропию по построению, и без этого числа столбики нечем защитить.
say "энтропия головы на общем наборе состояний"
ST=$ROOT/reports/xland/entropy/states_val_ood.npz
if [ -f "$ST" ]; then
  for m in $MODELS; do
    ck=$(ls -t "$ROOT/runs/$m"/checkpoints/across_timesteps/*.ckpt 2>/dev/null | head -1)
    [ -z "$ck" ] && continue
    (cd $ROOT/joint_wam/repos/nano-world-model && CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 \
      $PYT $S/entropy_measure.py --states "$ST" --run-dir "$ROOT/runs/$m" --checkpoint "$ck") \
      > "$OUT/entropy_$m.log" 2>&1
    say "  $m: $(grep 'Вся выборка' "$OUT/entropy_$m.log" | head -1)"
  done
else
  say "  набора состояний нет, пропускаю"
fi

# ------------------------------------------------------------------ 7. графики
say "строю графики"
$PYT $S/plot_final_bars.py 2>&1 | while read -r l; do say "  $l"; done
say "FINAL_PIPELINE_STATUS=OK"
say "числа: $OUT | графики: $OUT/final_T1.png и $OUT/final_T0.png"
