"""Отчёт по утечкам между наборами (METHODS.md §7): пересечение измеряется, а не предполагается.

Диапазоны сидов у наборов не пересекаются, но этого мало: разные сиды регулярно дают одну и ту же
раскладку, потому что пространство состояний поля 8x8 мало. Считаем два числа:

  1. доля эпизодов валидации, чья СТАРТОВАЯ раскладка встречается в обучающем наборе;
  2. доля отдельных КАДРОВ валидации, побайтово совпадающих с кадром из обучающего набора.

Второе число неустранимо: разные старты сходятся к одному состоянию за несколько шагов. Для
сравнения наборов между собой это безвредно, но абсолютный уровень метрик завышает, поэтому
число надо привести явно.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import h5py
import numpy as np


def grid_hash(g: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(g).tobytes(), digest_size=16).hexdigest()


def scan(root: str, bench: str, frame_stride: int):
    files = sorted(glob.glob(os.path.join(root, bench, "ep_*.h5")))
    starts, frames = [], set()
    for path in files:
        with h5py.File(path, "r") as h:
            grid = h["grid"][:]
            pos = h["agent_pos"][:]
            direction = h["agent_dir"][:]
        # раскладка = сетка плюс поза агента: картинка зависит и от того, и от другого
        starts.append(grid_hash(np.concatenate([grid[0].reshape(-1), pos[0], [direction[0]]])))
        for k in range(0, grid.shape[0], frame_stride):
            frames.add(grid_hash(np.concatenate([grid[k].reshape(-1), pos[k], [direction[k]]])))
    return files, starts, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/user8_2/AIRI_WAM/data/xland-shape")
    p.add_argument("--frame-stride", type=int, default=1)
    args = p.parse_args()

    benches = ["train", "val_seen", "val_ood_chain", "val_ood_all"]
    data = {}
    for b in benches:
        files, starts, frames = scan(args.root, b, args.frame_stride)
        data[b] = {"n": len(files), "starts": starts, "frames": frames}
        print(f"{b}: {len(files)} эпизодов, уникальных стартов {len(set(starts))}, состояний {len(frames)}")

    train_starts = set(data["train"]["starts"])
    train_frames = data["train"]["frames"]
    report = {"frame_stride": args.frame_stride}

    print()
    for b in benches:
        if b == "train":
            dup = len(data[b]["starts"]) - len(set(data[b]["starts"]))
            report[b] = {
                "episodes": data[b]["n"],
                "unique_starts": len(set(data[b]["starts"])),
                "repeated_starts": dup,
                "repeated_starts_share": dup / max(data[b]["n"], 1),
            }
            print(f"{b}: повторов стартовой раскладки внутри набора {dup} ({dup / max(data[b]['n'], 1):.4f})")
            continue
        overlap_starts = sum(1 for s in data[b]["starts"] if s in train_starts)
        overlap_frames = len(data[b]["frames"] & train_frames)
        report[b] = {
            "episodes": data[b]["n"],
            "unique_starts": len(set(data[b]["starts"])),
            "start_overlap_with_train": overlap_starts,
            "start_overlap_share": overlap_starts / max(data[b]["n"], 1),
            "state_overlap_with_train": overlap_frames,
            "state_overlap_share": overlap_frames / max(len(data[b]["frames"]), 1),
        }
        print(
            f"{b}: стартов, встречающихся в train — {overlap_starts} "
            f"({overlap_starts / max(data[b]['n'], 1):.4f}); "
            f"состояний, встречающихся в train — {overlap_frames} "
            f"({overlap_frames / max(len(data[b]['frames']), 1):.4f})"
        )

    out = os.path.join(args.root, "leak_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nотчёт: {out}")
    print("LEAK_REPORT_STATUS=OK")


if __name__ == "__main__":
    main()
