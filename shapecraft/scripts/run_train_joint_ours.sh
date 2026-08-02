#!/usr/bin/env bash
# Обучение joint-модели ментора на НАШИХ наборах ShapeCraft.
#
# Параметры модели не трогаются вообще: model=nanowm_xland_s4 как есть.
# Меняются только данные (наши три набора вместо его сбора) и состав метрик валидации.
#
# Запускаем нашим .venv-nanowm, а не его conda-окружением: этот venv уже проверен на этом
# же фреймворке и на этих картах, а conda-окружение пришлось бы собирать с нуля по
# environment.yml со старыми пинами.
set -uo pipefail

ROOT=/home/user8_2/AIRI_WAM
REPO=$ROOT/joint_wam/repos/nano-world-model
PY=$ROOT/.venv-nanowm/bin/python

GPU=${GPU:-1}
RUN_NAME=${RUN_NAME:-shapecraft_joint_128}
DATASET_CONFIG=${DATASET_CONFIG:-xland/shapecraft_ours}
MAX_STEPS=${MAX_STEPS:-}
# произвольные добавочные переопределения hydra, через пробел; параметры МОДЕЛИ сюда не кладём
EXTRA_ARGS=${EXTRA_ARGS:-}
SMOKE=${SMOKE:-0}

export CUDA_VISIBLE_DEVICES="$GPU"
export DATASET_DIR=$ROOT/data
export RESULTS_DIR=$ROOT/runs
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

RUN_DIR=$RESULTS_DIR/$RUN_NAME
if [ -e "$RUN_DIR" ] && [ "$SMOKE" != "1" ]; then
  echo "каталог прогона уже существует, не перезаписываю: $RUN_DIR"
  exit 4
fi
mkdir -p "$RESULTS_DIR"

EXTRA=()
if [ -n "$MAX_STEPS" ]; then
  EXTRA+=("experiment.training.max_steps=$MAX_STEPS")
fi
if [ -n "$EXTRA_ARGS" ]; then
  # намеренно без кавычек: слова разбиваются в отдельные аргументы hydra
  # shellcheck disable=SC2206
  EXTRA+=($EXTRA_ARGS)
fi
if [ "$SMOKE" = "1" ]; then
  rm -rf "$RUN_DIR"
  EXTRA+=(
    "experiment.training.val_every_n_steps=20"
    "experiment.training.log_every=5"
    # периоды у двух чекпоинтеров обязаны различаться: при одинаковых настройках
    # Lightning не может различить их состояния и падает
    "experiment.training.checkpointing.latest.every_n_train_steps=10"
    "experiment.training.checkpointing.across_timesteps.every_n_train_steps=25"
    "experiment.joint_action.visualize_every_n_steps=20"
    "experiment.joint_action.visualize_samples=2"
    "dataset.loader.validation_fixed_subset_size=32"
  )
fi

cd "$REPO"
exec $PY src/main.py \
  model=nanowm_xland_s4 \
  dataset="$DATASET_CONFIG" \
  experiment=xland_joint_ours \
  logger.name=tensorboard \
  wandb.enabled=false \
  "hydra.run.dir=$RUN_DIR" \
  "${EXTRA[@]}"
