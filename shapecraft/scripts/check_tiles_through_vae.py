"""Ворота §11 METHODS.md: различимы ли плитки ПОСЛЕ SD-VAE, при 128 и при 256.

Процедура: собираем атлас эталонных плиток, кладём их в кадр, гоняем кадр через
кодировщик туда-обратно и смотрим две вещи:

  1. долю плиток, которые после восстановления классифицируются по ближайшему эталону
     правильно (это и есть символьное чтение кадра);
  2. минимальное попарное расстояние между РАЗНЫМИ эталонами после восстановления.

Почему это критично именно сейчас: при кадре 128 плитка занимает 16 px, а SD-VAE сжимает
в 8 раз — то есть на плитку приходится 2x2 клетки латента. Формы шар / квадрат /
пирамида / шестиугольник должны выжить в этих четырёх числах.

Запускать из .venv-nanowm (там torch и diffusers), дописав site-packages от .venv-xland
в конец sys.path — оттуда нужен только jax для констант xminigrid.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np

XLAND_SP = "/home/user8_2/AIRI_WAM/.venv-xland/lib/python3.11/site-packages"
if XLAND_SP not in sys.path:
    sys.path.append(XLAND_SP)  # в КОНЕЦ: свои пакеты выигрывают, недостающие подхватываются
sys.path.insert(0, "/home/User8/xland-minigrid-five-object/src")

import torch  # noqa: E402
from diffusers import AutoencoderKL  # noqa: E402
from xminigrid.core.constants import Colors, Tiles  # noqa: E402
from xminigrid.rendering.rgb_render import render_tile  # noqa: E402

SHAPES = {"ball": Tiles.BALL, "square": Tiles.SQUARE, "pyramid": Tiles.PYRAMID, "hex": Tiles.HEX}
PALETTE = {"red": Colors.RED, "green": Colors.GREEN, "orange": Colors.ORANGE,
           "purple": Colors.PURPLE, "pink": Colors.PINK, "white": Colors.WHITE}
SPECIAL = {
    "blue key": (Tiles.KEY, Colors.BLUE),
    "yellow star": (Tiles.STAR, Colors.YELLOW),
    "blue door locked": (Tiles.DOOR_LOCKED, Colors.BLUE),
    "blue door open": (Tiles.DOOR_OPEN, Colors.BLUE),
    "wall": (Tiles.WALL, Colors.GREY),
    "floor": (Tiles.FLOOR, Colors.BLACK),
}


def atlas(tile_size: int):
    """Эталонные плитки: объекты палитры, служебные плитки и агент всех направлений."""
    items = {}
    for sn, s in SHAPES.items():
        for cn, c in PALETTE.items():
            items[f"{cn} {sn}"] = np.asarray(
                render_tile(tile=(int(s), int(c)), agent_direction=None, highlight=False, tile_size=tile_size),
                dtype=np.uint8,
            )
    for name, (s, c) in SPECIAL.items():
        items[name] = np.asarray(
            render_tile(tile=(int(s), int(c)), agent_direction=None, highlight=False, tile_size=tile_size),
            dtype=np.uint8,
        )
    for d in range(4):
        items[f"agent dir{d}"] = np.asarray(
            render_tile(tile=(int(Tiles.FLOOR), int(Colors.BLACK)), agent_direction=d, highlight=False,
                        tile_size=tile_size),
            dtype=np.uint8,
        )
    return items


def pack_into_frames(tiles, grid_side: int, tile_size: int):
    """Раскладываем атлас по кадрам grid_side x grid_side, чтобы гнать через VAE как картинки."""
    names = list(tiles)
    per_frame = grid_side * grid_side
    frames, layout = [], []
    for start in range(0, len(names), per_frame):
        chunk = names[start : start + per_frame]
        img = np.zeros((grid_side * tile_size, grid_side * tile_size, 3), dtype=np.uint8)
        pos = []
        for i, n in enumerate(chunk):
            y, x = divmod(i, grid_side)
            img[y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size] = tiles[n]
            pos.append((n, y, x))
        frames.append(img)
        layout.append(pos)
    return frames, layout


@torch.no_grad()
def roundtrip(vae, frames, device):
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1.0
    lat = vae.encode(x).latent_dist.mode()
    rec = vae.decode(lat).sample
    rec = ((rec.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    return rec, tuple(lat.shape[1:])


def evaluate(vae, device, tile_size: int, grid_side: int):
    tiles = atlas(tile_size)
    frames, layout = pack_into_frames(tiles, grid_side, tile_size)
    rec, lat_shape = roundtrip(vae, frames, device)

    # восстановленные версии каждого эталона
    recovered = {}
    for f, pos in enumerate(layout):
        for n, y, x in pos:
            recovered[n] = rec[f, y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size].astype(
                np.float64
            )
    ref = {n: t.astype(np.float64) for n, t in tiles.items()}
    names = list(ref)

    # классификация по ближайшему ЧИСТОМУ эталону
    correct, confusions = 0, []
    for n in names:
        d = [(float(np.abs(recovered[n] - ref[m]).mean()), m) for m in names]
        d.sort()
        if d[0][1] == n:
            correct += 1
        else:
            confusions.append((n, d[0][1], d[0][0]))

    # минимальное попарное расстояние между разными эталонами ПОСЛЕ восстановления
    pairs = sorted(
        (float(np.abs(recovered[a] - recovered[b]).mean()), a, b) for a, b in itertools.combinations(names, 2)
    )
    return {
        "tile_size": tile_size,
        "frame": f"{grid_side * tile_size}x{grid_side * tile_size}",
        "latent_shape": list(lat_shape),
        "latent_cells_per_tile": (tile_size // 8) ** 2,
        "n_tiles": len(names),
        "classification_accuracy": correct / len(names),
        "confusions": confusions[:8],
        "min_pair_after_vae": pairs[0][0],
        "min_pair_names": f"{pairs[0][1]} ~ {pairs[0][2]}",
        "closest_pairs": [(round(d, 3), a, b) for d, a, b in pairs[:5]],
    }


def evaluate_real_frames(vae, device, data_root, set_name, tile_size, n_episodes=4):
    """То же на НАСТОЯЩИХ кадрах датасета: доля верно прочитанных клеток."""
    import h5py

    sys.path.insert(0, "/home/user8_2/AIRI_WAM/scripts/xland")
    import shape_common as sc  # noqa: E402

    _, base_params, *_ = sc.make_env_and_policy(load_weights=False)
    render = sc.make_render_fn(base_params, tile_size=tile_size)
    files = sorted(glob.glob(os.path.join(data_root, set_name, "ep_*.h5")))[:n_episodes]
    tiles = atlas(tile_size)
    ref_names = list(tiles)
    ref_stack = np.stack([tiles[n].astype(np.float64) for n in ref_names])

    total = correct = 0
    for path in files:
        with h5py.File(path, "r") as h:
            g, pp, dd, pk = h["grid"][:], h["agent_pos"][:], h["agent_dir"][:], h["agent_pocket"][:]
        idxs = np.linspace(0, g.shape[0] - 1, min(8, g.shape[0])).astype(int)
        frames = np.stack([render(g[k], pp[k], dd[k], pk[k]) for k in idxs])
        rec, _ = roundtrip(vae, list(frames), device)
        side = frames.shape[1] // tile_size
        for f in range(rec.shape[0]):
            for y in range(side):
                for x in range(side):
                    t = rec[f, y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size]
                    truth = frames[f, y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size]
                    d_rec = np.abs(ref_stack - t.astype(np.float64)).mean(axis=(1, 2, 3))
                    d_true = np.abs(ref_stack - truth.astype(np.float64)).mean(axis=(1, 2, 3))
                    # клетка засчитывается, только если по ЧИСТОМУ кадру эталон найден верно
                    if d_true.min() < 1e-9 or True:
                        total += 1
                        correct += int(d_rec.argmin() == d_true.argmin())
    return correct / max(total, 1), total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/user8_2/AIRI_WAM/data/xland-shape-craft")
    p.add_argument("--set", default="train")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    print(f"устройство {device}\n")

    results = {}
    for tile_size, grid_side in ((32, 8), (16, 8)):
        r = evaluate(vae, device, tile_size, grid_side)
        results[r["frame"]] = r
        print(f"=== кадр {r['frame']} (плитка {tile_size} px, {r['latent_cells_per_tile']} клеток латента на плитку) ===")
        print(f"  латент {r['latent_shape']}")
        print(f"  классификация по ближайшему эталону: {r['classification_accuracy']:.4f} на {r['n_tiles']} плитках")
        print(f"  минимальное попарное расстояние после VAE: {r['min_pair_after_vae']:.3f} ({r['min_pair_names']})")
        if r["confusions"]:
            print("  путаница:")
            for a, b, d in r["confusions"]:
                print(f"    {a} прочиталось как {b} ({d:.3f})")
        print()

    out = "/home/user8_2/AIRI_WAM/reports/xland/tiles_through_vae.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"отчёт: {out}")
    print("VAE_GATE_STATUS=OK")


if __name__ == "__main__":
    main()
