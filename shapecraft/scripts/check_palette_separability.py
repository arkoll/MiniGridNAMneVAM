"""Различимость плиток ВНУТРИ реальной сцены ShapeCraft Easy.

Важная поправка: палитра сцены назначает каждой роли СВОЙ цвет, поэтому пары вида
«зелёный шар против зелёного шестиугольника» в одной сцене не встречаются. Меряем то,
что реально может оказаться в одном кадре: четыре ролевые плитки конкретной палитры,
плюс синий ключ, жёлтая звезда и агент.

Сплит задаётся чётностью: для роли i в train индекс цвета c удовлетворяет (c - i) % 2 == 0,
в val — == 1. Ровно как в shape_crafting_easy.py.
"""

import itertools

import numpy as np
from xminigrid.core.constants import Colors, Tiles
from xminigrid.rendering.rgb_render import render_tile

TS = 32
ROLE_SHAPES = (Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID, Tiles.HEX)
ROLE_NAMES = ("ball", "square", "pyramid", "hex")
COLOR_IDS = (Colors.RED, Colors.GREEN, Colors.ORANGE, Colors.PURPLE, Colors.PINK, Colors.WHITE)
COLOR_NAMES = ("red", "green", "orange", "purple", "pink", "white")


def tile(shape, color, agent_dir=None):
    return np.asarray(
        render_tile(tile=(int(shape), int(color)), agent_direction=agent_dir, highlight=False, tile_size=TS),
        dtype=np.float64,
    )


def palettes(split):
    parity = 0 if split == "train" else 1
    return [
        a
        for a in itertools.product(range(6), repeat=4)
        if len(set(a)) == 4 and all((c - i) % 2 == parity for i, c in enumerate(a))
    ]


BLUE_KEY = ("blue key", tile(Tiles.KEY, Colors.BLUE))
YELLOW_STAR = ("yellow star", tile(Tiles.STAR, Colors.YELLOW))
AGENTS = [(f"agent dir{d}", tile(Tiles.FLOOR, Colors.BLACK, agent_dir=d)) for d in range(4)]

for split in ("train", "val"):
    pals = palettes(split)
    worst = (1e9, None)
    per_palette_min = []
    for pal in pals:
        items = [(f"{COLOR_NAMES[c]} {ROLE_NAMES[i]}", tile(ROLE_SHAPES[i], COLOR_IDS[c])) for i, c in enumerate(pal)]
        items += [BLUE_KEY, YELLOW_STAR] + AGENTS
        best = (1e9, None)
        for (n1, t1), (n2, t2) in itertools.combinations(items, 2):
            if n1.startswith("agent") and n2.startswith("agent"):
                continue
            d = float(np.abs(t1 - t2).mean())
            if d < best[0]:
                best = (d, f"{n1} ~ {n2}")
        per_palette_min.append(best[0])
        if best[0] < worst[0]:
            worst = (best[0], best[1], pal)
    print(f"{split}: палитр {len(pals)}")
    print(f"  худшая палитра: {worst[0]:.3f}  ({worst[1]})")
    print(f"  минимум по палитрам: медиана {np.median(per_palette_min):.3f}, "
          f"худший {min(per_palette_min):.3f}, лучший {max(per_palette_min):.3f}")
    print(f"  палитр с минимумом ниже 10: {sum(1 for v in per_palette_min if v < 10)}")
    print()
