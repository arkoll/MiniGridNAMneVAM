"""Самые похожие пары плиток в палитре ShapeCraft — до кодировщика.

Если две разные плитки уже в сыром кадре почти неразличимы, символьное чтение кадра
(METHODS.md §11) на них сломается, а после SD-VAE станет только хуже.
"""

import itertools

import numpy as np
from xminigrid.core.constants import Colors, Tiles
from xminigrid.rendering.rgb_render import render_tile

TS = 32
SHAPES = {"ball": Tiles.BALL, "square": Tiles.SQUARE, "pyramid": Tiles.PYRAMID, "hex": Tiles.HEX}
PALETTE = {"red": Colors.RED, "green": Colors.GREEN, "orange": Colors.ORANGE,
           "purple": Colors.PURPLE, "pink": Colors.PINK, "white": Colors.WHITE}


def tile(shape, color):
    return np.asarray(
        render_tile(tile=(int(shape), int(color)), agent_direction=None, highlight=False, tile_size=TS),
        dtype=np.float64,
    )


items = {f"{cn} {sn}": tile(s, c) for sn, s in SHAPES.items() for cn, c in PALETTE.items()}
pairs = sorted(
    (float(np.abs(items[a] - items[b]).mean()), a, b) for a, b in itertools.combinations(items, 2)
)

print("самые похожие пары объектов (среднее по пикселям, 0..255):")
for d, a, b in pairs[:12]:
    print(f"  {d:7.3f}  {a}  ~  {b}")
print(f"\nвсего пар: {len(pairs)}, медиана {np.median([p[0] for p in pairs]):.2f}")
print(f"пар ближе 10: {sum(1 for p in pairs if p[0] < 10)}")
print(f"пар ближе 5:  {sum(1 for p in pairs if p[0] < 5)}")
