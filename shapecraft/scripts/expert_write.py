"""Рендер экспертных эпизодов сразу в формат joint-модели: npz с кадрами 128.

Промежуточного hdf5 на 256 тут нет — он нужен был предыдущему пайплайну. Рисуем поле в 256
(их же функция рендера) и тут же ужимаем ровно вдвое целочисленным усреднением 2x2 — тем
же выражением, что у ментора в замкнутом контуре.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np

import shape_common as sc

DST = "/home/user8_2/AIRI_WAM/data/xland-expert"
PREFIX = {"train": "episode_train", "val_id": "episode_train_id", "val_ood": "episode_train_ood"}


def downsample_half(rgb: np.ndarray) -> np.ndarray:
    s = rgb.astype(np.uint16)
    summed = s[:, 0::2, 0::2] + s[:, 0::2, 1::2] + s[:, 1::2, 0::2] + s[:, 1::2, 1::2]
    return ((summed + 2) // 4).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, choices=list(PREFIX))
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    env, base_params, *_ = sc.make_env_and_policy(load_weights=False)
    render = sc.make_render_fn(base_params)
    out_dir = os.path.join(DST, args.set)
    done_dir = os.path.join(DST, "_shards", args.set, "_written")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(DST, "_shards", args.set, "shard_*.npz")))
    mine = [s for i, s in enumerate(shards) if i % args.workers == args.worker]
    print(f"воркер {args.worker}/{args.workers}: {len(mine)} шардов из {len(shards)}", flush=True)

    n_ep = n_frames = n_bytes = 0
    index = []
    t0 = time.time()
    for path in mine:
        marker = os.path.join(done_dir, os.path.basename(path) + f".w{args.worker}.done")
        if os.path.exists(marker):
            continue
        z = np.load(path)
        for e in range(z["task"].shape[0]):
            L = int(z["length"][e])
            g = np.concatenate([z["grid"][e], z["final_grid"][e][None]], axis=0)[: L + 1]
            pp = np.concatenate([z["pos"][e], z["final_pos"][e][None]], axis=0)[: L + 1]
            dd = np.concatenate([z["dir"][e], z["final_dir"][e][None]], axis=0)[: L + 1]
            pk = np.concatenate([z["pocket"][e], z["final_pocket"][e][None]], axis=0)[: L + 1]

            frames256 = np.stack([render(g[k], pp[k], dd[k], pk[k]) for k in range(L + 1)])
            frames = downsample_half(frames256)

            seed = int(z["seed"][e])
            meta = {
                "dataset": args.set,
                "task_index": int(z["task"][e]),
                "seed": seed,
                "steps": L,
                "outcome": "success",
                "policy": "best_val_greedy",
                "image_size": 128,
                "resize_method": "exact_2x_area_int",
            }
            name = f"{PREFIX[args.set]}_{seed:09d}.npz"
            fpath = os.path.join(out_dir, name)
            tmp = fpath + ".tmp.npz"
            np.savez_compressed(
                tmp,
                rgb=frames,
                actions=z["action"][e, :L].astype(np.int64),
                metadata=np.asarray(json.dumps(meta, sort_keys=True)),
            )
            os.replace(tmp, fpath)
            n_bytes += os.path.getsize(fpath)
            n_frames += frames.shape[0]
            n_ep += 1
            index.append({**meta, "file": name, "num_frames": int(frames.shape[0])})
        open(marker, "w").close()
        el = time.time() - t0
        print(
            f"  воркер {args.worker}: {n_ep} эпизодов, {n_frames} кадров, "
            f"{n_bytes / 1e6:.0f} МБ, {n_frames / max(el, 1e-9):.0f} кадров/с",
            flush=True,
        )

    with open(os.path.join(out_dir, f"_index_part_{args.worker:02d}.jsonl"), "w") as f:
        for row in index:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"EXPERT_WRITE_{args.set.upper()}_W{args.worker}_STATUS=OK")


if __name__ == "__main__":
    main()
