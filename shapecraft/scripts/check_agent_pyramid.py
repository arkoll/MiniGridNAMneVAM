"""Насколько красная пирамида отличима от агента в кадре.

В палитре ShapeCraft есть RED, а агент рисуется КРАСНЫМ ТРЕУГОЛЬНИКОМ. Пирамида —
тоже треугольник. Считаем расстояние между плитками и сравниваем с типичным
расстоянием между разными объектами: если оно того же порядка, различимость под
вопросом, и это надо проверять уже после SD-VAE (ворота §11 METHODS.md).

Геометрия спрайтов агента и пирамиды в форке не менялась, поэтому меряем
на штатном xminigrid.
"""

import itertools

import numpy as np
from xminigrid.core.constants import Colors, Tiles
from xminigrid.rendering.rgb_render import render_tile

TS = 32
SHAPES = {"ball": Tiles.BALL, "square": Tiles.SQUARE, "pyramid": Tiles.PYRAMID, "hex": Tiles.HEX}
PALETTE = {"red": Colors.RED, "green": Colors.GREEN, "orange": Colors.ORANGE,
           "purple": Colors.PURPLE, "pink": Colors.PINK, "white": Colors.WHITE}


def tile(shape, color, agent_dir=None):
    return np.asarray(
        render_tile(tile=(int(shape), int(color)), agent_direction=agent_dir, highlight=False, tile_size=TS),
        dtype=np.float64,
    )


def dist(a, b):
    return float(np.abs(a - b).mean())


# агент на пустом полу, все четыре направления
agent = {d: tile(Tiles.FLOOR, Colors.BLACK, agent_dir=d) for d in range(4)}
floor = tile(Tiles.FLOOR, Colors.BLACK)

print("расстояние агента (по направлениям) до объектов, среднее по пикселям 0..255:")
rows = []
for sname, s in SHAPES.items():
    for cname, c in PALETTE.items():
        obj = tile(s, c)
        d_min = min(dist(agent[d], obj) for d in range(4))
        rows.append((d_min, f"{cname} {sname}"))
rows.sort()
for d, name in rows[:8]:
    print(f"  {d:7.3f}  {name}")
print("  ...")
for d, name in rows[-3:]:
    print(f"  {d:7.3f}  {name}")

print()
print(f"агент против пустого пола: {min(dist(agent[d], floor) for d in range(4)):7.3f}")

pairs = []
for (s1, c1), (s2, c2) in itertools.combinations(
    [(s, c) for s in SHAPES.values() for c in PALETTE.values()], 2
):
    pairs.append(dist(tile(s1, c1), tile(s2, c2)))
print(f"минимальное расстояние между двумя РАЗНЫМИ объектами: {min(pairs):7.3f}")
print(f"медиана по парам объектов: {float(np.median(pairs)):7.3f}")

worst = rows[0]
print()
print(f"ХУДШИЙ СЛУЧАЙ: агент против «{worst[1]}» = {worst[0]:.3f}")
print(f"это {worst[0] / min(pairs):.2f} от минимального расстояния между объектами")
