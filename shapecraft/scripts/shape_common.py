"""Общее для сбора датасетов ShapeCraft Easy: среда, политика, рендер, состав наборов.

Собираем три набора:

  train    — обучающие задачи, диапазон сидов A;
  val_id   — ТЕ ЖЕ задачи, диапазон сидов B. Контроль «новая раскладка при знакомой
             привязке цвета к форме»: показывает обычное обобщение, без OOD;
  val_ood  — валидационные задачи (комплементарные привязки), диапазон сидов C.

Разница между val_id и val_ood и есть чистый эффект привязки: всё остальное совпадает.

Смесь политик одинакова во всех трёх наборах, иначе сравнение потеряет смысл.
Ровно каждый шестой эпизод собирается СЛУЧАЙНОЙ политикой по допустимым действиям,
остальные — чекпоинтом train_sr_50.

Почему именно train_sr_50, а не best_val: замер лестницы чекпоинтов показал, что
best_val играет на валидации заметно хуже, чем на трейне (29 кадров на эпизод против
48, 6.6 манипуляций объектами против 24.4), и эта разница попала бы в датасет поверх
задуманной. У train_sr_50 статистики совпадают до третьего знака: успех 0.951/0.953,
длина 49/49, события 14.2/14.2, кадры 57/57. Случайная политика симметрична столь же
точно (0.000/0.000, 15.0/15.1).

Почему каждый шестой: успешный эпизод даёт около 50 кадров, провальный всегда 193.
При доле случайных 1/6 получается 0.79 успешных эпизодов и ровно половина КАДРОВ из
успешных — то есть баланс по кадрам, а именно окна видит world-модель.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

REPO = "/home/User8/xland-minigrid-five-object"


def bootstrap():
    """Их чекаут должен победить установленный xminigrid, а tensorboardX не нужен вовсе."""
    for p in (os.path.join(REPO, "training"), os.path.join(REPO, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "tensorboardX" not in sys.modules:
        stub = types.ModuleType("tensorboardX")

        class _SW:
            def __init__(self, *a, **k):
                pass

            def add_scalar(self, *a, **k):
                pass

            def close(self):
                pass

        stub.SummaryWriter = _SW
        sys.modules["tensorboardX"] = stub


bootstrap()

ENV_ID = "XLand-MiniGrid-ShapeCraft-Easy-8x8-v1"
CKPT = f"{REPO}/runs/privileged_rl_easy_20260730/shape_easy/checkpoints/train_sr_50/train_state.msgpack"
NUM_TASKS = 324
RANDOM_EVERY = 6  # каждый шестой круг — случайная политика

# круги по задачам: каждая из 324 задач получает поровну эпизодов.
# число кругов кратно RANDOM_EVERY, чтобы доля случайных была ТОЧНО одинаковой в наборах.
SETS = {
    "train": {"tasks": "train", "rounds": 60, "seed_base": 1_000_000},
    "val_id": {"tasks": "train", "rounds": 18, "seed_base": 2_000_000},
    "val_ood": {"tasks": "val", "rounds": 18, "seed_base": 3_000_000},
}

DATA_ROOT = "/home/user8_2/AIRI_WAM/data/xland-shape-craft"
SHARD_ROOT = os.path.join(DATA_ROOT, "_shards")


def episodes_in(set_name: str) -> int:
    return SETS[set_name]["rounds"] * NUM_TASKS


def episode_plan(set_name: str, index: int):
    """Задача, номер круга и признак «случайная политика» для эпизода с номером index."""
    task = index % NUM_TASKS
    rnd = index // NUM_TASKS
    return task, rnd, (rnd % RANDOM_EVERY == 0)


# ------------------------------------------------------------------- рендер ---


def make_render_fn(params):
    """Рендер всего поля из символьного состояния — та же функция, что у них в env.render.

    Их `render` читает только `timestep.state`, поэтому кадр восстанавливается из
    (сетка, позиция, направление, карман) и полный TimeStep собирать не нужно.
    """
    from xminigrid.envs.five_object_crafting import _render_agent_with_pocket
    from xminigrid.rendering.rgb_render import render as rgb_render

    view_size = int(params.view_size)
    height = int(params.height)

    def render(grid, pos, direction, pocket):
        grid = np.asarray(grid)
        image = rgb_render(grid, agent=None, view_size=view_size)
        tile_size = int(image.shape[0] // height)
        y, x = int(pos[0]), int(pos[1])
        tile = _render_agent_with_pocket(
            grid_tile=grid[y, x],
            pocket=np.asarray(pocket),
            direction=int(direction),
            tile_size=tile_size,
        )
        image[y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size] = tile
        return np.asarray(image, dtype=np.uint8)

    return render


# ------------------------------------------------------- среда и политика ---


def make_env_and_policy(load_weights: bool = True):
    import jax
    import jax.tree_util as jtu
    from flax import serialization

    import train_privileged_crafting_ppo as T

    train_bench, val_bench = T.task_banks(ENV_ID)
    env, base_params = T.make_env(ENV_ID, autoreset=False)
    _, dir_count = T.embodiment_spec(ENV_ID)
    net = T.PrivilegedActorCritic(
        num_actions=env.num_actions(base_params), agent_direction_classes=dir_count + 1
    )
    policy_params = None
    if load_weights:
        init_params = base_params.replace(ruleset=train_bench.get_ruleset(0))
        init_ts = env.reset(init_params, jax.random.key(0))
        template = net.init(jax.random.key(0), jtu.tree_map(lambda x: x[None], init_ts.observation))
        with open(CKPT, "rb") as f:
            policy_params = serialization.from_state_dict(
                template, serialization.msgpack_restore(f.read())["params"]
            )
    return env, base_params, net, policy_params, {"train": train_bench, "val": val_bench}


def task_meta(tasks_split: str, task_index0: int) -> dict:
    """Человекочитаемое описание задачи: дерево, палитра, правила, цвета ролей."""
    from xminigrid.envs.shape_crafting_easy import RULE_TEXT, get_shape_easy_task

    t = get_shape_easy_task(tasks_split, task_index0 + 1)
    return {
        "uid": t.uid,
        "tree_id": int(t.tree_id),
        "palette_id": int(t.palette_id),
        "rule_ids": list(t.rule_ids),
        "rules": [RULE_TEXT[r] for r in t.rule_ids],
        "colors": dict(zip(("ball", "square", "pyramid", "hexagon"), t.palette_color_names)),
    }
