"""Восстановление _index.jsonl из шардов — кадры уже отрендерены, индекс потерялся.

Повторяет ровно ту арифметику, что делал dagger_write.py: имя файла от сида, число кадров
= длина эпизода плюс один, а для случайной половины — обрезано до отрезка.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

PREFIX = {"train": "episode_train", "val_id": "episode_train_id", "val_ood": "episode_train_ood"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/user8_2/AIRI_WAM/data/final")
    p.add_argument("--slice-frames", type=int, default=29)
    args = p.parse_args()

    for s in PREFIX:
        rows = []
        for pol in ("expert", "random"):
            shard_dir = os.path.join(args.root, "_shards", f"{s}_{pol}")
            if not os.path.isdir(shard_dir):
                continue
            for path in sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz"))):
                z = np.load(path)
                for e in range(z["task"].shape[0]):
                    L = int(z["length"][e])
                    n = L + 1
                    if pol == "random" and args.slice_frames and n > args.slice_frames:
                        n = args.slice_frames
                    seed = int(z["seed"][e])
                    rows.append({
                        "dataset": s,
                        "task_index": int(z["task"][e]),
                        "seed": seed,
                        "steps": n - 1,
                        "outcome": "success" if bool(z["success"][e]) else "fail",
                        "policy": "expert_greedy" if pol == "expert" else "random_masked",
                        "label_source": "expert_greedy_best_val",
                        "image_size": 128,
                        "file": f"{PREFIX[s]}_{seed:09d}.npz",
                        "num_frames": n,
                    })
        out = os.path.join(args.root, s, "_index.jsonl")
        # сверяем с тем, что реально лежит на диске
        have = {f for f in os.listdir(os.path.join(args.root, s)) if f.startswith("episode_train")}
        rows = [r for r in rows if r["file"] in have]
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        exp = sum(1 for r in rows if r["policy"] == "expert_greedy")
        print(f"{s}: строк {len(rows)} (экспертных {exp}, случайных {len(rows)-exp}), "
              f"файлов на диске {len(have)}, окон {sum(r['num_frames'] - 3 for r in rows)}")
    print("REBUILD_INDEX_STATUS=OK")


if __name__ == "__main__":
    main()
