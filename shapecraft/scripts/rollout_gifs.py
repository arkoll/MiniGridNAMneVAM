"""Гифки полных эпизодов ShapeCraft Easy обученной политикой коллеги.

Берём их чекаут, их обёртки наблюдения и их сеть — иначе привилегированное наблюдение
не соберётся и веса не подойдут. Раскатываем жадно (`dist.mode()`), как в их же
процедуре оценки, и пишем каждый кадр полного поля.

Из чекпоинта берём только `params`: восстанавливать состояние оптимизатора не нужно,
а значит не нужно и повторять расписание learning rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types

import numpy as np

REPO = "/home/User8/xland-minigrid-five-object"
# их src должен победить установленный в venv xminigrid
sys.path.insert(0, os.path.join(REPO, "training"))
sys.path.insert(0, os.path.join(REPO, "src"))

# tensorboardX нужен их скрипту только для логов обучения; ставить пакет ради импорта
# незачем — подсовываем заглушку
if "tensorboardX" not in sys.modules:
    stub = types.ModuleType("tensorboardX")

    class _SummaryWriter:
        def __init__(self, *a, **k):
            pass

        def add_scalar(self, *a, **k):
            pass

        def close(self):
            pass

    stub.SummaryWriter = _SummaryWriter
    sys.modules["tensorboardX"] = stub

import imageio.v3 as iio  # noqa: E402
import jax  # noqa: E402
import jax.tree_util as jtu  # noqa: E402
from flax import serialization  # noqa: E402

import train_privileged_crafting_ppo as T  # noqa: E402
from xminigrid.envs.shape_crafting_easy import RULE_TEXT, get_shape_easy_task  # noqa: E402

ENV_ID = "XLand-MiniGrid-ShapeCraft-Easy-8x8-v1"
PALETTES_PER_TREE = 36
SCALE = 2  # 256 -> 512, иначе подпись не помещается по ширине


def _font(size=13):
    try:
        import matplotlib
        from PIL import ImageFont

        path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
        return ImageFont.truetype(path, size)
    except Exception:
        return None


def caption(lines, width, height=104):
    """Полоса с подписью над кадром. Без PIL просто возвращаем пустую полосу."""
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return strip
    img = Image.fromarray(strip)
    draw = ImageDraw.Draw(img)
    font = _font()
    y = 6
    for line in lines:
        draw.text((10, y), line, fill=(235, 235, 235), font=font)
        y += 20 if font else 12
    return np.asarray(img)


def upscale(frame, factor=SCALE):
    return np.repeat(np.repeat(frame, factor, axis=0), factor, axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        default=f"{REPO}/runs/privileged_rl_easy_20260730/shape_easy/checkpoints/best_val/train_state.msgpack",
    )
    p.add_argument("--out", default="/home/user8_2/AIRI_WAM/reports/xland/gifs")
    p.add_argument("--per-split", type=int, default=4)
    p.add_argument("--fps", type=float, default=4.0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_bench, val_bench = T.task_banks(ENV_ID)
    env, base_params = T.make_env(ENV_ID, autoreset=False)
    _, direction_count = T.embodiment_spec(ENV_ID)
    network = T.PrivilegedActorCritic(
        num_actions=env.num_actions(base_params),
        agent_direction_classes=direction_count + 1,
    )
    print(f"среда: {ENV_ID}, max_steps={base_params.max_steps}, действий {env.num_actions(base_params)}")
    print(f"задач: train {train_bench.num_rulesets()}, val {val_bench.num_rulesets()}")

    # шаблон параметров той же формы, что у обученных весов
    init_params = base_params.replace(ruleset=train_bench.get_ruleset(0))
    init_ts = env.reset(init_params, jax.random.key(0))
    template = network.init(jax.random.key(0), jtu.tree_map(lambda x: x[None], init_ts.observation))

    with open(args.checkpoint, "rb") as f:
        raw = serialization.msgpack_restore(f.read())
    policy_params = serialization.from_state_dict(template, raw["params"])
    print(f"чекпоинт загружен: {args.checkpoint}")

    @jax.jit
    def act(observation):
        dist, _ = network.apply(policy_params, jtu.tree_map(lambda x: x[None], observation))
        return dist.mode()[0]

    step_fn = jax.jit(env.step)
    reset_fn = jax.jit(env.reset)

    # разные деревья и разные палитры, чтобы видеть и разные правила, и разные цвета
    picks = [(1, 1), (3, 12), (6, 23), (9, 34)][: args.per_split]
    summary = []

    for split, bench in (("train", train_bench), ("val", val_bench)):
        for tree_id, palette_id in picks:
            task_index = (tree_id - 1) * PALETTES_PER_TREE + palette_id
            task = get_shape_easy_task(split, task_index)
            params = base_params.replace(ruleset=bench.get_ruleset(task_index - 1))
            ts = reset_fn(params, jax.random.key(1000 + task_index))

            frames = [upscale(np.asarray(env.render(params, ts), dtype=np.uint8))]
            actions = []
            steps = 0
            success = False
            while steps < int(base_params.max_steps):
                action = act(ts.observation)
                ts = step_fn(params, ts, action)
                actions.append(int(action))
                frames.append(upscale(np.asarray(env.render(params, ts), dtype=np.uint8)))
                steps += 1
                if bool(ts.last()):
                    success = bool(ts.reward >= 0.9)
                    break

            w = frames[0].shape[1]
            colors = ",  ".join(f"{n}={c}" for n, c in zip(("ball", "square", "pyramid", "hexagon"), task.palette_color_names))
            head = caption(
                [
                    f"{split.upper()}   {task.uid}   tree {task.tree_id}",
                    f"rules:  {'   |   '.join(RULE_TEXT[r] for r in task.rule_ids)}",
                    f"colors: {colors}",
                    f"steps: {steps}    outcome: {'SUCCESS' if success else 'timeout'}",
                ],
                width=w,
            )
            with_head = [np.concatenate([head, f], axis=0) for f in frames]
            with_head += [with_head[-1]] * 6  # задержка на последнем кадре

            name = f"{split}_tree{task.tree_id}_pal{task.palette_id}_{'ok' if success else 'timeout'}.gif"
            path = os.path.join(args.out, name)
            iio.imwrite(path, np.stack(with_head), extension=".gif",
                        duration=int(1000 / args.fps), loop=0)
            size_kb = os.path.getsize(path) / 1024
            print(f"  {name}: {steps} шагов, {'успех' if success else 'таймаут'}, {size_kb:.0f} КБ")
            summary.append(
                {
                    "split": split,
                    "uid": task.uid,
                    "tree_id": task.tree_id,
                    "palette_id": task.palette_id,
                    "rules": [RULE_TEXT[r] for r in task.rule_ids],
                    "colors": dict(zip(("ball", "square", "pyramid", "hexagon"), task.palette_color_names)),
                    "steps": steps,
                    "success": success,
                    "actions": actions,
                    "gif": name,
                }
            )

    with open(os.path.join(args.out, "episodes.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    ok = sum(1 for s in summary if s["success"])
    print(f"\nитог: {ok} из {len(summary)} эпизодов успешны")
    print("GIFS_STATUS=OK")


if __name__ == "__main__":
    main()
