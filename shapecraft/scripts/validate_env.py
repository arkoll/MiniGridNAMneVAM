"""Ворота по среде: прогоняются ДО обучения, ни один следующий шаг без них не начинается.

Что проверяется:
  1. наш быстрый рендер побайтово совпадает с эталонной сборкой плиток;
  2. картинка — детерминированная функция символического состояния;
  3. цепочка правил НЕ схлопывается: за один шаг срабатывает ровно одно звено;
  4. инертные объекты не меняются;
  5. отложенные цвета отсутствуют в обучающем наборе и присутствуют в OOD-наборах;
  6. награда считается по ожидаемой формуле.

Заодно рисует картинки для глазной проверки и для показа коллегам.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import xland_common as xc
from xminigrid.core.constants import DIRECTIONS, TILES_REGISTRY, Colors, Tiles
from xminigrid.rendering.rgb_render import render_tile

FIG_DIR = os.environ.get("XLAND_FIG_DIR", "/home/user8_2/AIRI_WAM/reports/xland/figures")
os.makedirs(FIG_DIR, exist_ok=True)

FAILS = []


def gate(name: str, ok: bool, detail: str = ""):
    mark = "OK  " if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


# ---------------------------------------------------------------- подготовка ---

env, env_params = xc.make_env(tile_size=xc.DATASET_TILE_SIZE)
benchmarks = xc.all_benchmarks()
render_fn = jax.jit(xc.make_render_fn(xc.DATASET_TILE_SIZE))

print(f"среда: {xc.ENV_ID}, поле {env_params.height}x{env_params.width}, max_steps={env_params.max_steps}")
print(f"кадр: {env.observation_shape(env_params)}")
print("наборы задач: " + ", ".join(f"{k}={b.num_rulesets()}" for k, b in benchmarks.items()))
print(f"правил в наборе: {int(benchmarks['train'].num_rules[0])}\n")


def reset_with(bench_name: str, ruleset_id: int, seed_index: int):
    ruleset = benchmarks[bench_name].get_ruleset(ruleset_id)
    params = env_params.replace(ruleset=ruleset)
    timestep = jax.jit(env.reset)(params, xc.episode_seed(bench_name, seed_index))
    return params, timestep


# ------------------------------------------------------ 1-2. ворота рендера ---

params, ts = reset_with("train", 0, 0)
ours = np.asarray(render_fn(ts.state.grid, ts.state.agent))
ref = xc.render_reference(ts.state.grid, ts.state.agent, xc.DATASET_TILE_SIZE)
gate("рендер совпадает с эталонной сборкой плиток", np.array_equal(ours, ref), f"кадр {ours.shape}, dtype {ours.dtype}")

again = np.asarray(render_fn(ts.state.grid, ts.state.agent))
gate("картинка детерминирована по состоянию", np.array_equal(ours, again))

gate(
    "кадр ровно 256x256 и делится на плитки",
    ours.shape == (256, 256, 3) and 256 % xc.DATASET_TILE_SIZE == 0,
    f"{ours.shape}, плитка {xc.DATASET_TILE_SIZE}",
)

# наблюдение среды — это тот же самый кадр
gate("наблюдение среды == кадр всего поля", np.array_equal(np.asarray(ts.observation), ours))


# ------------------------------------------- 3-4. ворота по цепочке правил ---


def neighbours(pos):
    for d in range(4):
        yield d, (int(pos[0]) + int(DIRECTIONS[d][0]), int(pos[1]) + int(DIRECTIONS[d][1]))


def put_agent_facing(state, target_yx):
    """Ставит агента на свободную клетку рядом с target и разворачивает на неё."""
    grid = np.asarray(state.grid)
    for d, (ny, nx) in neighbours(jnp.asarray(target_yx)):
        if not (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]):
            continue
        if grid[ny, nx, 0] != Tiles.FLOOR:
            continue
        # агент стоит в (ny, nx), смотреть должен в сторону target
        back = (d + 2) % 4
        agent = state.agent.replace(position=jnp.asarray((ny, nx)), direction=jnp.asarray(back))
        return state.replace(agent=agent)
    raise RuntimeError("не нашлось свободной клетки рядом с объектом")


def chain_demo(bench_name: str, ruleset_id: int, seed_index: int):
    """Ставим агента вплотную к квадрату и жмём «вперёд». Возвращает кадры и историю плитки."""
    params, ts = reset_with(bench_name, ruleset_id, seed_index)
    grid = np.asarray(ts.state.grid)
    sq = np.argwhere(grid[:, :, 0] == Tiles.SQUARE)
    assert len(sq) == 1, f"ожидался ровно один квадрат, найдено {len(sq)}"
    sq = tuple(sq[0])

    ts = ts.replace(state=put_agent_facing(ts.state, sq))
    frames = [np.asarray(render_fn(ts.state.grid, ts.state.agent))]
    tiles = [tuple(int(v) for v in np.asarray(ts.state.grid)[sq[0], sq[1]])]
    rewards, terminated = [], []

    step = jax.jit(env.step, static_argnums=())
    for _ in range(4):
        ts = step(params, ts, 0)  # действие 0 — «вперёд»
        g = np.asarray(ts.state.grid)
        frames.append(np.asarray(render_fn(ts.state.grid, ts.state.agent)))
        tiles.append(tuple(int(v) for v in g[sq[0], sq[1]]))
        rewards.append(float(ts.reward))
        terminated.append(bool(ts.discount == 0.0))
    return frames, tiles, rewards, terminated, sq


frames, tiles, rewards, terminated, sq = chain_demo("train", 0, 0)
start_color = tiles[0][1]
expected = [
    (Tiles.SQUARE, start_color),
    (Tiles.BALL, start_color),
    (Tiles.PYRAMID, start_color),
    (Tiles.GOAL, Colors.GREEN),
]
gate(
    "цепочка идёт по одному звену за шаг, а не схлопывается",
    tiles[:4] == expected,
    " -> ".join(f"{xc.TILE_NAMES.get(t, t)}/{xc.COLOR_NAMES.get(c, c)}" for t, c in tiles[:4]),
)
gate("цвет переносится по цепочке", all(c == start_color for _, c in tiles[1:3]))
gate(
    "цель достигается шагом на зелёную клетку",
    terminated[-1] and rewards[-1] > 0,
    f"награда {rewards[-1]:.4f} на шаге {len(rewards)}",
)
expected_reward = 1.0 - 0.9 * (len(rewards) / xc.MAX_STEPS)
gate(
    "награда совпадает с формулой 1 - 0.9*t/max_steps",
    abs(rewards[-1] - expected_reward) < 1e-5,
    f"получено {rewards[-1]:.5f}, ожидалось {expected_reward:.5f}",
)

# инертные объекты не меняются за всю демонстрацию
params0, ts0 = reset_with("train", 0, 0)
g0 = np.asarray(ts0.state.grid)
inert_positions = [tuple(p) for p in np.argwhere(np.isin(g0[:, :, 0], list(xc.INERT_SHAPES)))]
params, ts = reset_with("train", 0, 0)
ts = ts.replace(state=put_agent_facing(ts.state, sq))
step = jax.jit(env.step)
inert_stable = True
for _ in range(6):
    ts = step(params, ts, 0)
    g = np.asarray(ts.state.grid)
    for p in inert_positions:
        if tuple(g[p[0], p[1]]) != tuple(g0[p[0], p[1]]):
            inert_stable = False
gate("инертные объекты не меняются", inert_stable, f"{len(inert_positions)} шт.")


# ------------------------------------------------- 5. ворота по цветам ------


def colors_in(bench_name: str) -> set:
    tiles_arr = np.asarray(benchmarks[bench_name].init_tiles)
    return set(int(c) for c in tiles_arr[:, :, 1].reshape(-1))


train_colors = colors_in("train")
gate(
    "в обучающем наборе нет отложенных цветов",
    not (train_colors & set(xc.HELD_OUT_COLORS)),
    "цвета: " + ", ".join(sorted(xc.COLOR_NAMES[c] for c in train_colors)),
)
ood_chain_colors = set(int(c) for c in np.asarray(benchmarks["val_ood_chain"].init_tiles)[:, 0, 1])
gate(
    "в val_ood_chain квадрат всегда отложенного цвета",
    ood_chain_colors == set(xc.HELD_OUT_COLORS),
    "цвета квадрата: " + ", ".join(sorted(xc.COLOR_NAMES[c] for c in ood_chain_colors)),
)
gate(
    "в val_ood_all знакомых цветов нет вообще",
    not (colors_in("val_ood_all") & set(xc.TRAIN_COLORS)),
)
# каждый цвет палитры обязан реально появиться в кадре нужным RGB: ловит и опечатки
# в палитре, и тихую порчу цвета при приведении типов
from xminigrid.rendering.rgb_render import COLORS_MAP  # noqa: E402

palette_ok, palette_detail = True, []
for c in xc.ALL_COLORS:
    t = np.asarray(
        render_tile(tile=(int(Tiles.SQUARE), int(c)), agent_direction=None, highlight=False, tile_size=32),
        dtype=np.uint8,
    )
    want = np.asarray(COLORS_MAP[c], dtype=np.uint8)
    if not (t.reshape(-1, 3) == want).all(axis=1).any():
        palette_ok = False
        palette_detail.append(xc.COLOR_NAMES[c])
gate("каждый цвет палитры даёт свой RGB в кадре", palette_ok, "сломаны: " + ", ".join(palette_detail) if palette_detail else "7 цветов")

# все цвета палитры попарно различимы по RGB — иначе символьное чтение кадра невозможно
rgbs = {c: tuple(int(v) for v in COLORS_MAP[c]) for c in xc.ALL_COLORS}
gate(
    "цвета палитры попарно различны",
    len(set(rgbs.values())) == len(rgbs),
    ", ".join(f"{xc.COLOR_NAMES[c]}={rgbs[c]}" for c in xc.ALL_COLORS),
)

gate(
    "цвет агента не занят объектами",
    Colors.RED not in xc.ALL_COLORS,
    "агент рисуется красным треугольником, пирамида — тоже треугольник",
)
gate(
    "цвет цели не занят объектами",
    Colors.GREEN not in xc.ALL_COLORS,
)
gate(
    "правила заведены на все цвета (среда одинакова везде)",
    int(benchmarks["train"].num_rules[0]) == xc.CHAIN_DEPTH * len(xc.ALL_COLORS),
    f"{int(benchmarks['train'].num_rules[0])} правил = {xc.CHAIN_DEPTH} звена x {len(xc.ALL_COLORS)} цветов",
)


# --------------------------------------------------------------- картинки ---


def tile_img(tile, color, size=48):
    # ВАЖНО: render_tile отдаёт float64 в диапазоне 0..255 (виноват downsample со средним),
    # а matplotlib для вещественных картинок ждёт 0..1 и молча обрезает всё выше единицы:
    # фиолетовый, коричневый и розовый превращаются в белый. Приводим к uint8 явно.
    return np.asarray(
        render_tile(tile=(int(tile), int(color)), agent_direction=None, highlight=False, tile_size=size),
        dtype=np.uint8,
    )


def draw_chain_figure():
    fig, axes = plt.subplots(1, len(frames), figsize=(3.0 * len(frames), 3.6))
    titles = ["старт: агент вплотную к квадрату"] + [f"шаг {i + 1}: «вперёд»" for i in range(len(frames) - 1)]
    for ax, img, title, tl in zip(axes, frames, titles, tiles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(f"клетка: {xc.TILE_NAMES.get(tl[0], tl[0])} / {xc.COLOR_NAMES.get(tl[1], tl[1])}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Цепочка правил срабатывает по одному звену за шаг (проверка на схлопывание)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "chain_steps.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def draw_levels_figure():
    rows = [("train", "обучение WAM"), ("val_ood_chain", "OOD: новый цвет квадрата"), ("val_ood_all", "OOD: все цвета новые")]
    n = 4
    fig, axes = plt.subplots(len(rows), n, figsize=(3.0 * n, 3.2 * len(rows)))
    rng = np.random.default_rng(0)
    for r, (bench, label) in enumerate(rows):
        b = benchmarks[bench]
        ids = rng.choice(b.num_rulesets(), size=n, replace=False)
        for c in range(n):
            _, ts = reset_with(bench, int(ids[c]), 1000 + r * 100 + c)
            axes[r, c].imshow(np.asarray(render_fn(ts.state.grid, ts.state.agent)))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            sq_c = int(np.asarray(b.init_tiles)[int(ids[c]), 0, 1])
            axes[r, c].set_title(f"квадрат: {xc.COLOR_NAMES[sq_c]}", fontsize=9)
        axes[r, 0].set_ylabel(label, fontsize=10)
    fig.suptitle("Примеры уровней: правила и формы одинаковые, различаются только цвета", fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "levels.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def draw_design_figure():
    """Картинка для коллег: палитра, цепочка правил, что где новое."""
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.25, 1.5], hspace=0.42)

    # --- палитра ---
    ax = fig.add_subplot(gs[0])
    ax.set_title("Палитра: какие цвета встречаются в обучении WAM, а какие только на валидации", fontsize=12, pad=14)
    for i, c in enumerate(xc.TRAIN_COLORS):
        ax.imshow(tile_img(Tiles.SQUARE, c), extent=(i * 1.2, i * 1.2 + 1, 1.15, 2.15))
        ax.text(i * 1.2 + 0.5, 1.0, xc.COLOR_NAMES[c], ha="center", fontsize=9)
    for i, c in enumerate(xc.HELD_OUT_COLORS):
        x0 = (len(xc.TRAIN_COLORS) + 1 + i) * 1.2
        ax.imshow(tile_img(Tiles.SQUARE, c), extent=(x0, x0 + 1, 1.15, 2.15))
        ax.text(x0 + 0.5, 1.0, xc.COLOR_NAMES[c], ha="center", fontsize=9)
    ax.text(len(xc.TRAIN_COLORS) * 1.2 / 2 - 0.3, 2.45, "в обучении WAM", ha="center", fontsize=11, weight="bold")
    ax.text(
        (len(xc.TRAIN_COLORS) + 1.5) * 1.2,
        2.45,
        "только на валидации (OOD)",
        ha="center",
        fontsize=11,
        weight="bold",
        color="crimson",
    )
    ax.set_xlim(-0.3, (len(xc.ALL_COLORS) + 1.6) * 1.2)
    ax.set_ylim(0.6, 2.8)
    ax.axis("off")

    # --- цепочка правил ---
    ax = fig.add_subplot(gs[1])
    ax.set_title(
        "Правило зависит только от ФОРМЫ. Цвет переносится по цепочке и схлопывается на последнем звене",
        fontsize=12,
        pad=14,
    )
    demo_color = xc.HELD_OUT_COLORS[0]
    seq = [(Tiles.SQUARE, demo_color), (Tiles.BALL, demo_color), (Tiles.PYRAMID, demo_color), xc.TERMINAL_TILE]
    labels = ["квадрат", "круг", "пирамида", "цель"]
    for i, ((t, c), lab) in enumerate(zip(seq, labels)):
        ax.imshow(tile_img(t, c), extent=(i * 2.4, i * 2.4 + 1.2, 0.6, 1.8))
        ax.text(i * 2.4 + 0.6, 0.3, f"{lab}\n{xc.COLOR_NAMES[c]}", ha="center", fontsize=9)
        if i < len(seq) - 1:
            ax.annotate(
                "", xy=((i + 1) * 2.4 - 0.05, 1.0), xytext=(i * 2.4 + 1.3, 1.0), arrowprops=dict(arrowstyle="->", lw=2)
            )
            ax.text(i * 2.4 + 1.85, 1.15, "шаг вперёд\nрядом", ha="center", fontsize=8)
    ax.text(
        len(seq) * 2.4 + 0.2,
        1.0,
        "цель эпизода:\nвстать на зелёную клетку",
        fontsize=10,
        va="center",
    )
    ax.set_xlim(-0.3, len(seq) * 2.4 + 4.2)
    ax.set_ylim(-0.15, 2.0)
    ax.axis("off")

    # --- инертные и уровни ---
    ax = fig.add_subplot(gs[2])
    ax.set_title("Уровень 8x8: один реагирующий квадрат, два инертных объекта, три действия", fontsize=12, pad=14)
    _, ts_demo = reset_with("val_ood_chain", 0, 7)
    ax.imshow(np.asarray(render_fn(ts_demo.state.grid, ts_demo.state.agent)), extent=(0, 4, 0, 4))
    for i, s in enumerate(xc.INERT_SHAPES):
        ax.imshow(tile_img(s, xc.TRAIN_COLORS[i]), extent=(4.7 + i * 1.4, 5.7 + i * 1.4, 2.6, 3.6))
        ax.text(5.2 + i * 1.4, 2.35, xc.TILE_NAMES[s], ha="center", fontsize=9)
    ax.text(4.7, 3.9, "инертные формы: не участвуют ни в одном правиле", fontsize=10)
    ax.text(
        4.7,
        1.9,
        "действия: вперёд, поворот по часовой, поворот против\n"
        "pick_up и put_down не используются\n"
        f"лимит шагов: {xc.MAX_STEPS}\n"
        f"награда: 1 - 0.9 * шаг / {xc.MAX_STEPS} в момент достижения цели, иначе 0",
        fontsize=10,
        va="top",
    )
    ax.set_xlim(-0.2, 11)
    ax.set_ylim(-0.2, 4.4)
    ax.axis("off")

    fig.suptitle(
        "Эксперимент: выучит ли world-модель правило по форме и перенесёт ли его на невиданный цвет",
        fontsize=14,
        y=0.985,
    )
    path = os.path.join(FIG_DIR, "design.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


print()
for p in (draw_chain_figure(), draw_levels_figure(), draw_design_figure()):
    print("картинка:", p)

print()
if FAILS:
    print("VALIDATE_STATUS=FAIL: " + ", ".join(FAILS))
    sys.exit(1)
print("VALIDATE_STATUS=OK")
