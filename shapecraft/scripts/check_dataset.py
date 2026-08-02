"""Ворота по собранному датасету: то, что лежит на диске, обязано соответствовать состояниям.

Проверяется на случайной выборке эпизодов:
  1. длины согласованы: кадров ровно на один больше, чем действий;
  2. каждый кадр ПОБАЙТОВО совпадает с рендером сохранённого символического состояния;
  3. у успешных эпизодов агент в последнем кадре стоит на зелёной клетке-цели;
  4. индексный файл сходится с тем, что реально лежит на диске;
  5. в обучающем наборе нет отложенных цветов, а в OOD-наборах они есть.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import h5py
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xland_common as xc
from xminigrid.core.constants import Colors, Tiles
from xminigrid.types import AgentState

FAILS = []


def gate(name, ok, detail=""):
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/user8_2/AIRI_WAM/data/xland-shape")
    p.add_argument("--benchmark", required=True)
    p.add_argument("--sample", type=int, default=24)
    args = p.parse_args()

    render = jax.jit(xc.make_render_fn(xc.DATASET_TILE_SIZE))
    files = sorted(glob.glob(os.path.join(args.root, args.benchmark, "ep_*.h5")))
    gate(f"файлы эпизодов найдены ({args.benchmark})", len(files) > 0, f"{len(files)} шт.")
    if not files:
        sys.exit(1)

    index_path = os.path.join(args.root, f"index_{args.benchmark}.jsonl")
    with open(index_path) as f:
        index = [json.loads(line) for line in f]
    gate("индекс сходится с числом файлов", len(index) == len(files), f"индекс {len(index)}, файлов {len(files)}")

    rng = np.random.default_rng(0)
    picks = rng.choice(len(files), size=min(args.sample, len(files)), replace=False)

    len_ok = frames_ok = goal_ok = True
    checked_frames = 0
    colors_seen = set()
    max_diff = 0
    for i in picks:
        with h5py.File(files[i], "r") as h:
            frames = h["frames"][:]
            actions = h["actions"][:]
            grid = h["grid"][:]
            pos = h["agent_pos"][:]
            direction = h["agent_dir"][:]
            outcome = h.attrs["outcome"]
            steps = int(h.attrs["steps"])

        if not (frames.shape[0] == actions.shape[0] + 1 == grid.shape[0] == steps + 1):
            len_ok = False

        # Побайтовое совпадение кадра с рендером сохранённого состояния.
        # Длину добиваем до постоянной, иначе jit перекомпилируется на каждую
        # встреченную длину эпизода и проверка тянется минутами.
        n = grid.shape[0]
        pad = xc.MAX_STEPS + 1 - n
        gp = np.concatenate([grid, np.repeat(grid[-1:], pad, axis=0)]) if pad else grid
        pp_ = np.concatenate([pos, np.repeat(pos[-1:], pad, axis=0)]) if pad else pos
        dp = np.concatenate([direction, np.repeat(direction[-1:], pad, axis=0)]) if pad else direction
        ref_full = jax.vmap(render)(
            jnp.asarray(gp),
            AgentState(
                position=jnp.asarray(pp_),
                direction=jnp.asarray(dp),
                pocket=jnp.zeros((gp.shape[0], 2), dtype=jnp.uint8),
            ),
        )
        ref = np.asarray(ref_full[:n], dtype=np.uint8)
        if not np.array_equal(ref, frames):
            frames_ok = False
            max_diff = max(max_diff, int(np.abs(ref.astype(int) - frames.astype(int)).max()))
        checked_frames += frames.shape[0]

        if outcome == "success":
            y, x = int(pos[-1][0]), int(pos[-1][1])
            if not (grid[-1, y, x, 0] == Tiles.GOAL and grid[-1, y, x, 1] == Colors.GREEN):
                goal_ok = False

        colors_seen |= set(int(c) for c in grid[0][:, :, 1].reshape(-1))

    gate("длины согласованы: кадров = действий + 1", len_ok)
    gate(
        "кадры побайтово совпадают с рендером состояния",
        frames_ok,
        f"проверено {checked_frames} кадров" if frames_ok else f"максимальное расхождение {max_diff}",
    )
    gate("у успешных эпизодов агент финиширует на зелёной цели", goal_ok)

    obj_colors = colors_seen - {Colors.EMPTY, Colors.BLACK, Colors.GREY, Colors.GREEN, Colors.RED}
    held = obj_colors & set(xc.HELD_OUT_COLORS)
    if args.benchmark in ("train", "val_seen"):
        gate("отложенных цветов в наборе нет", not held, "цвета: " + ", ".join(sorted(xc.COLOR_NAMES[c] for c in obj_colors)))
    else:
        gate("отложенные цвета в наборе есть", bool(held), "цвета: " + ", ".join(sorted(xc.COLOR_NAMES[c] for c in obj_colors)))

    # ------------------------------------------------------------ покрытие ---
    # Политика решает уровень за 8 шагов, поэтому надо проверить, что в данных вообще
    # есть ОТРИЦАТЕЛЬНЫЕ примеры: агент шагнул вперёд, а рядом инертная фигура, и ничего
    # не произошло. Без них модель не может выучить, что реагируют не все формы, и
    # вывод про «правило по форме» будет не на чем основывать.
    fired = adjacent_inert = adjacent_chain = forward_steps = 0
    fired_by_color: dict[int, int] = {}
    react_shapes = {int(Tiles.SQUARE), int(Tiles.BALL), int(Tiles.PYRAMID)}
    inert_shapes = set(int(s) for s in xc.INERT_SHAPES)
    for i in picks:
        with h5py.File(files[i], "r") as h:
            grid = h["grid"][:]
            pos = h["agent_pos"][:]
            actions = h["actions"][:]
        for t in range(actions.shape[0]):
            if int(actions[t]) != 0:
                continue
            forward_steps += 1
            y, x = int(pos[t + 1][0]), int(pos[t + 1][1])
            neigh = []
            for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid.shape[1] and 0 <= nx < grid.shape[2]:
                    neigh.append((int(grid[t, ny, nx, 0]), int(grid[t, ny, nx, 1])))
            if any(s in inert_shapes for s, _ in neigh):
                adjacent_inert += 1
            if any(s in react_shapes for s, _ in neigh):
                adjacent_chain += 1
            if not np.array_equal(grid[t], grid[t + 1]):
                fired += 1
                for s, c in neigh:
                    if s in react_shapes:
                        fired_by_color[c] = fired_by_color.get(c, 0) + 1
                        break

    gate(
        "в данных есть отрицательные примеры: шаг вперёд рядом с инертной фигурой",
        adjacent_inert > 0,
        f"{adjacent_inert} из {forward_steps} шагов вперёд ({adjacent_inert / max(forward_steps, 1):.3f})",
    )
    gate(
        "срабатывания правил представлены всеми цветами набора",
        len(fired_by_color) >= (len(xc.HELD_OUT_COLORS) if "ood" in args.benchmark else len(xc.TRAIN_COLORS)),
        ", ".join(f"{xc.COLOR_NAMES.get(c, c)}={n}" for c, n in sorted(fired_by_color.items())),
    )
    print(
        f"покрытие на выборке: шагов вперёд {forward_steps}, из них срабатываний правила {fired} "
        f"({fired / max(forward_steps, 1):.3f}), рядом реагирующая фигура {adjacent_chain}, "
        f"рядом инертная {adjacent_inert}"
    )

    n_success = sum(1 for r in index if r["outcome"] == "success")
    frames_total = sum(r["num_frames"] for r in index)
    print(
        f"\nсводка: {len(index)} эпизодов, {frames_total} кадров, "
        f"успех {n_success / len(index):.3f}, кадров на эпизод {frames_total / len(index):.1f}"
    )

    if FAILS:
        print("DATASET_STATUS=FAIL: " + ", ".join(FAILS))
        sys.exit(1)
    print("DATASET_STATUS=OK")


if __name__ == "__main__":
    main()
