"""То же, но на НАСТОЯЩИХ кадрах собранного датасета: 256 против 128.

Атласный тест кладёт 34 разные плитки вплотную — это заведомо тяжелее реального кадра,
где объекты стоят поодиночке на чёрном полу. Здесь читаем настоящие кадры и считаем долю
верно прочитанных клеток отдельно по клеткам С ОБЪЕКТАМИ (они и решают дело).
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

XLAND_SP = "/home/user8_2/AIRI_WAM/.venv-xland/lib/python3.11/site-packages"
if XLAND_SP not in sys.path:
    sys.path.append(XLAND_SP)
sys.path.insert(0, "/home/User8/xland-minigrid-five-object/src")

import h5py  # noqa: E402
import torch  # noqa: E402
from diffusers import AutoencoderKL  # noqa: E402
from xminigrid.core.constants import Colors, Tiles  # noqa: E402
from xminigrid.envs.five_object_crafting import _render_agent_with_pocket  # noqa: E402
from xminigrid.rendering.rgb_render import render as rgb_render  # noqa: E402
from xminigrid.rendering.rgb_render import render_tile  # noqa: E402

OBJECT_SHAPES = {int(Tiles.BALL), int(Tiles.SQUARE), int(Tiles.PYRAMID), int(Tiles.HEX), int(Tiles.KEY), int(Tiles.STAR)}


def render_at(grid, pos, direction, pocket, tile_size):
    grid = np.asarray(grid)
    img = rgb_render(grid, agent=None, view_size=5, tile_size=tile_size)
    y, x = int(pos[0]), int(pos[1])
    tile = _render_agent_with_pocket(
        grid_tile=grid[y, x], pocket=np.asarray(pocket), direction=int(direction), tile_size=tile_size
    )
    img[y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size] = tile
    return np.asarray(img, dtype=np.uint8)


def atlas(tile_size):
    """Все плитки, которые могут встретиться: объекты, служебные и агент."""
    items = {}
    shapes = {"ball": Tiles.BALL, "square": Tiles.SQUARE, "pyramid": Tiles.PYRAMID, "hex": Tiles.HEX}
    palette = {"red": Colors.RED, "green": Colors.GREEN, "orange": Colors.ORANGE,
               "purple": Colors.PURPLE, "pink": Colors.PINK, "white": Colors.WHITE}
    for sn, s in shapes.items():
        for cn, c in palette.items():
            items[f"{cn} {sn}"] = (int(s), int(c))
    items["blue key"] = (int(Tiles.KEY), int(Colors.BLUE))
    items["yellow star"] = (int(Tiles.STAR), int(Colors.YELLOW))
    items["door locked"] = (int(Tiles.DOOR_LOCKED), int(Colors.BLUE))
    items["door open"] = (int(Tiles.DOOR_OPEN), int(Colors.BLUE))
    items["wall"] = (int(Tiles.WALL), int(Colors.GREY))
    items["floor"] = (int(Tiles.FLOOR), int(Colors.BLACK))
    names, imgs, tiles = [], [], []
    for n, (s, c) in items.items():
        names.append(n)
        tiles.append((s, c))
        imgs.append(
            np.asarray(render_tile(tile=(s, c), agent_direction=None, highlight=False, tile_size=tile_size),
                       dtype=np.float64)
        )
    return names, tiles, np.stack(imgs)


@torch.no_grad()
def roundtrip(vae, frames, device):
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().to(device) / 127.5 - 1.0
    lat = vae.encode(x).latent_dist.mode()
    rec = vae.decode(lat).sample
    return ((rec.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    root = "/home/user8_2/AIRI_WAM/data/xland-shape-craft/train"
    files = sorted(glob.glob(os.path.join(root, "ep_*.h5")))[:12]
    print(f"эпизодов в проверке: {len(files)}, устройство {device}\n")

    out = {}
    for tile_size in (32, 16):
        names, tiles, ref = atlas(tile_size)
        by_tile = {t: i for i, t in enumerate(tiles)}
        tot_obj = ok_obj = tot_all = ok_all = 0
        conf = {}
        for path in files:
            with h5py.File(path, "r") as h:
                g, pp, dd, pk = h["grid"][:], h["agent_pos"][:], h["agent_dir"][:], h["agent_pocket"][:]
            idxs = np.linspace(0, g.shape[0] - 1, min(6, g.shape[0])).astype(int)
            frames = [render_at(g[k], pp[k], dd[k], pk[k], tile_size) for k in idxs]
            rec = roundtrip(vae, frames, device)
            side = frames[0].shape[0] // tile_size
            for fi, k in enumerate(idxs):
                for y in range(side):
                    for x in range(side):
                        if (y, x) == (int(pp[k][0]), int(pp[k][1])):
                            continue  # клетка агента: в атласе её нет, у неё свой спрайт с карманом
                        truth = (int(g[k, y, x, 0]), int(g[k, y, x, 1]))
                        if truth not in by_tile:
                            continue
                        patch = rec[fi, y * tile_size : (y + 1) * tile_size, x * tile_size : (x + 1) * tile_size]
                        pred = int(np.abs(ref - patch.astype(np.float64)).mean(axis=(1, 2, 3)).argmin())
                        good = pred == by_tile[truth]
                        tot_all += 1
                        ok_all += int(good)
                        if truth[0] in OBJECT_SHAPES:
                            tot_obj += 1
                            ok_obj += int(good)
                            if not good:
                                key = f"{names[by_tile[truth]]} -> {names[pred]}"
                                conf[key] = conf.get(key, 0) + 1
        acc_all = ok_all / max(tot_all, 1)
        acc_obj = ok_obj / max(tot_obj, 1)
        out[f"{side * tile_size}x{side * tile_size}"] = {
            "tile_size": tile_size,
            "cells_checked": tot_all,
            "accuracy_all_cells": acc_all,
            "object_cells_checked": tot_obj,
            "accuracy_object_cells": acc_obj,
            "confusions": sorted(conf.items(), key=lambda kv: -kv[1])[:8],
        }
        print(f"=== кадр {side * tile_size}x{side * tile_size} (плитка {tile_size} px) ===")
        print(f"  все клетки:      {acc_all:.4f} на {tot_all}")
        print(f"  клетки с объектами: {acc_obj:.4f} на {tot_obj}")
        for k, v in sorted(conf.items(), key=lambda kv: -kv[1])[:6]:
            print(f"    {k}: {v}")
        print()

    with open("/home/user8_2/AIRI_WAM/reports/xland/real_frames_vae.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("REAL_FRAMES_VAE_STATUS=OK")


if __name__ == "__main__":
    main()
