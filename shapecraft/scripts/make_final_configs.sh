#!/usr/bin/env bash
# Конфиги датасетов для итогового сравнения. Отличается ТОЛЬКО data_path_train;
# валидация у всех одна и та же, иначе метрики между прогонами несопоставимы.
set -uo pipefail
CFG=/home/user8_2/AIRI_WAM/joint_wam/repos/nano-world-model/src/configs/dataset/xland

write_one() {  # $1 имя, $2 каталог обучения, $3 комментарий
  cat > "$CFG/$1.yaml" <<YAML
# @package dataset
#
# $3
# Валидация общая для всех четырёх вариантов: data/final/val_both.
name: "xland"
frame_interval: 1
loader:
  n_rollout: null
  data_path: "\${dataset_dir}/final/$2"
  data_path_train: "\${dataset_dir}/final/$2"
  data_path_val: "\${dataset_dir}/final/val_both"
  split_ratio: 0.9
  validation_size: null
  normalize_state: false
  normalize_action: false
  normalize_pixel: true
  train_slice_mode: "random"
  val_slice_mode: "exhaustive"
  stride: 1
  resize_mode: "stretch"
  random_seed: 42
  validation_fixed_subset_path: null
  validation_fixed_subset_size: 1024
  validation_fixed_subset_seed: 4242
spec:
  action_dim: 6
YAML
  echo "  $CFG/$1.yaml"
}

write_one final_v1 v1_expert  "Вариант 1: только экспертные состояния и экспертные действия."
write_one final_v2 v2_mixed   "Варианты 2 и 3: половина окон из экспертных эпизодов, половина из случайных. Вариант 3 читает то же самое, но с XLAND_ACTION_FIELD=executed."
write_one final_v4a v4_random "Вариант 4, фаза A: только случайные эпизоды (обучается модель мира)."
write_one final_v4b v4_expert "Вариант 4, фаза B: только экспертные эпизоды, голова действий новая."
echo "CONFIGS_STATUS=OK"
