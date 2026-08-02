"""Чередует обучение и замер: каждый вариант измеряется сразу, как обучится.

Иначе при нехватке времени не будет ни одного полного столбика — все замеры
стоят в конце очереди.
"""

p = "/home/user8_2/AIRI_WAM/scripts/xland/run_final_pipeline.sh"
s = open(p).read()
start = s.index("stopped || train_one final_v1")
end = s.index("# --------------------------------------------- 6. энтропия")

new = r'''OUT=$ROOT/reports/xland/final
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

'''

open(p, "w").write(s[:start] + new + s[end:])
print("переписано, длина", len(s[:start] + new + s[end:]))
