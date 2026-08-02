"""Наши наборы ShapeCraft (hdf5, 256) -> формат joint-модели ментора (npz, 128).

Сбор пропускаем: его скрипт собирает ExactCraft, а гипотеза наша — ShapeCraft.
Начинаем сразу с ужатия, как он и делает отдельным проходом.

Ужатие ровно двукратное усреднением 2x2, целочисленное — тем же выражением, что у него
в замкнутом контуре (`(sum + 2) // 4`). Это важно: если обучать на кадрах, полученных
одним способом, а в контуре подавать полученные другим, модель встретит чужие пиксели.

Раскладка на выходе:
  train/   episode_train_XXXXXXX.npz       — весь обучающий набор
  val_id/  episode_train_id_XXXXXXX.npz    — контроль: знакомые привязки, новые раскладки
  val_ood/ episode_train_ood_XXXXXXX.npz   — тест: невиданные привязки цвета к форме
  val_both/                                — символические ссылки на оба val, один даталоадер

Валидационные наборы берутся ПО МАНИФЕСТАМ: там они подрезаны до совпадения по числу
успешных и провальных эпизодов в каждой задаче.

Правила и цель в npz не кладём вовсе — ни в каком виде.
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np

SRC_ROOT = "/home/user8_2/AIRI_WAM/data/xland-shape-craft"
DST_ROOT = "/home/user8_2/AIRI_WAM/data/xland-shape-joint128"
PREFIX = {"train": "episode_train", "val_id": "episode_train_id", "val_ood": "episode_train_ood"}


def downsample_half(rgb: np.ndarray) -> np.ndarray:
    """Точное двукратное усреднение 2x2, целочисленное — как в замкнутом контуре ментора."""
    if rgb.shape[1] != 256 or rgb.shape[2] != 256:
        raise ValueError(f"ожидались кадры 256x256, получено {rgb.shape}")
    s = rgb.astype(np.uint16)
    summed = s[:, 0::2, 0::2] + s[:, 0::2, 1::2] + s[:, 1::2, 0::2] + s[:, 1::2, 1::2]
    return ((summed + 2) // 4).astype(np.uint8)


def episodes_of(set_name: str):
    """Список эпизодов набора. Для валидаций — по манифесту, для трейна — весь индекс."""
    if set_name == "train":
        path = os.path.join(SRC_ROOT, "index_train.jsonl")
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    manifest = os.path.join(SRC_ROOT, "manifests", f"{set_name}.txt")
    with open(manifest) as f:
        keep = {line.strip() for line in f if line.strip()}
    with open(os.path.join(SRC_ROOT, f"index_{set_name}.jsonl")) as f:
        return [r for r in (json.loads(line) for line in f if line.strip()) if r["path"] in keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, choices=list(PREFIX))
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--min-frames", type=int, default=4, help="окно модели — 4 кадра")
    args = p.parse_args()

    rows = episodes_of(args.set)
    mine = [r for i, r in enumerate(rows) if i % args.workers == args.worker]
    out_dir = os.path.join(DST_ROOT, args.set)
    os.makedirs(out_dir, exist_ok=True)
    print(f"воркер {args.worker}/{args.workers}: {len(mine)} эпизодов из {len(rows)}", flush=True)

    n_ok = n_short = n_frames = n_bytes = 0
    index = []
    for r in mine:
        with h5py.File(os.path.join(SRC_ROOT, r["path"]), "r") as h:
            frames = h["frames"][:]
            actions = h["actions"][:]
            attrs = dict(h.attrs)
        if frames.shape[0] != actions.shape[0] + 1:
            raise ValueError(f"рассогласование в {r['path']}: {frames.shape[0]} кадров, {actions.shape[0]} действий")
        if frames.shape[0] < args.min_frames:
            n_short += 1
            continue

        small = downsample_half(frames)
        name = f"{PREFIX[args.set]}_{int(r['id']):07d}.npz"
        path = os.path.join(out_dir, name)
        # метаданные только для аудита: правила и цель не кладём ни в каком виде
        meta = {
            "dataset": args.set,
            "episode_id": int(r["id"]),
            "task_index": int(attrs["task_index"]),
            "tree_id": int(attrs["tree_id"]),
            "palette_id": int(attrs["palette_id"]),
            "outcome": str(attrs["outcome"]),
            "policy": str(attrs["policy"]),
            "steps": int(attrs["steps"]),
            "source_image_size": 256,
            "image_size": 128,
            "resize_method": "exact_2x_area_int",
        }
        tmp = path + ".tmp.npz"
        np.savez_compressed(
            tmp,
            rgb=small,
            actions=actions.astype(np.int64),
            metadata=np.asarray(json.dumps(meta, sort_keys=True)),
        )
        os.replace(tmp, path)
        n_bytes += os.path.getsize(path)
        n_frames += small.shape[0]
        n_ok += 1
        index.append({**meta, "file": name, "num_frames": int(small.shape[0])})

    with open(os.path.join(out_dir, f"_index_part_{args.worker:02d}.jsonl"), "w") as f:
        for row in index:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        f"воркер {args.worker}: записано {n_ok}, пропущено коротких {n_short}, "
        f"кадров {n_frames}, {n_bytes / 1e6:.0f} МБ ({n_bytes / max(n_frames, 1) / 1024:.2f} КБ/кадр)",
        flush=True,
    )
    print(f"CONVERT_{args.set.upper()}_W{args.worker}_STATUS=OK")


if __name__ == "__main__":
    main()
