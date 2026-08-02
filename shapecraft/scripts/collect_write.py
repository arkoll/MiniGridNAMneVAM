"""Этап 2: рендер кадров и запись эпизодов в hdf5. Только CPU, запускается пачкой процессов.

Соглашение о действиях (METHODS.md §10): `actions[t]` — действие, выполненное ИЗ кадра `t`
и приводящее к кадру `t+1`. Поэтому кадров ровно на один больше, чем действий.

Возобновляемость: шард пропускается, если его отметка уже лежит рядом.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import h5py
import numpy as np

import shape_common as sc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, choices=list(sc.SETS))
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--gzip", type=int, default=4)
    args = p.parse_args()

    cfg = sc.SETS[args.set]
    shard_dir = os.path.join(sc.SHARD_ROOT, args.set)
    out_dir = os.path.join(sc.DATA_ROOT, args.set)
    done_dir = os.path.join(shard_dir, "_written")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)

    # среда нужна только ради параметров рендера, веса не грузим
    env, base_params, *_ = sc.make_env_and_policy(load_weights=False)
    render = sc.make_render_fn(base_params)
    max_steps = int(base_params.max_steps)

    shards = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    mine = [s for i, s in enumerate(shards) if i % args.workers == args.worker]
    index_path = os.path.join(shard_dir, f"index_part_{args.worker:02d}.jsonl")
    print(f"воркер {args.worker}/{args.workers}: {len(mine)} шардов из {len(shards)}", flush=True)

    t0 = time.time()
    n_ep = n_frames = n_bytes = 0
    with open(index_path, "a") as index_file:
        for path in mine:
            marker = os.path.join(done_dir, os.path.basename(path) + f".w{args.worker}.done")
            if os.path.exists(marker):
                continue
            z = np.load(path)
            grids, poss, dirs, pockets = z["grid"], z["pos"], z["dir"], z["pocket"]
            actions, rewards = z["action"], z["reward"]
            lengths, successes = z["length"], z["success"]
            ep_ids, tasks, rounds, use_random = z["episode_id"], z["task"], z["round"], z["use_random"]
            final = (z["final_grid"], z["final_pos"], z["final_dir"], z["final_pocket"])

            for e in range(len(ep_ids)):
                L = int(lengths[e])
                # состояние после k шагов лежит в grid[e, k]; после конца эпизода среда
                # заморожена, поэтому grid[e, L] уже терминальное. Хвост нужен только
                # когда эпизод упёрся в лимит и записей ровно max_steps.
                g = np.concatenate([grids[e], final[0][e][None]], axis=0)[: L + 1]
                pp = np.concatenate([poss[e], final[1][e][None]], axis=0)[: L + 1]
                dd = np.concatenate([dirs[e], final[2][e][None]], axis=0)[: L + 1]
                pk = np.concatenate([pockets[e], final[3][e][None]], axis=0)[: L + 1]

                frames = np.stack([render(g[k], pp[k], dd[k], pk[k]) for k in range(L + 1)])

                meta = sc.task_meta(cfg["tasks"], int(tasks[e]))
                ep_id = int(ep_ids[e])
                fpath = os.path.join(out_dir, f"ep_{ep_id:07d}.h5")
                with h5py.File(fpath, "w") as h:
                    h.create_dataset("frames", data=frames, compression="gzip", compression_opts=args.gzip)
                    h.create_dataset("actions", data=actions[e, :L].astype(np.uint8))
                    h.create_dataset("rewards", data=rewards[e, :L].astype(np.float32))
                    h.create_dataset("grid", data=g, compression="gzip", compression_opts=args.gzip)
                    h.create_dataset("agent_pos", data=pp)
                    h.create_dataset("agent_dir", data=dd)
                    h.create_dataset("agent_pocket", data=pk)
                    h.attrs.update(
                        dataset=args.set,
                        tasks_split=cfg["tasks"],
                        episode_id=ep_id,
                        task_index=int(tasks[e]),
                        task_uid=meta["uid"],
                        tree_id=meta["tree_id"],
                        palette_id=meta["palette_id"],
                        rules=" | ".join(meta["rules"]),
                        colors=json.dumps(meta["colors"], ensure_ascii=False),
                        steps=L,
                        num_frames=L + 1,
                        outcome="success" if bool(successes[e]) else "timeout",
                        final_reward=float(z["final_reward"][e]),
                        policy="random" if bool(use_random[e]) else "train_sr_50",
                        reset_seed=int(cfg["seed_base"]) + ep_id,
                        max_steps=max_steps,
                    )
                size = os.path.getsize(fpath)
                n_bytes += size
                n_frames += L + 1
                n_ep += 1
                index_file.write(
                    json.dumps(
                        {
                            "id": ep_id,
                            "path": os.path.relpath(fpath, sc.DATA_ROOT),
                            "steps": L,
                            "num_frames": L + 1,
                            "outcome": "success" if bool(successes[e]) else "timeout",
                            "policy": "random" if bool(use_random[e]) else "train_sr_50",
                            "task_index": int(tasks[e]),
                            "tree_id": meta["tree_id"],
                            "palette_id": meta["palette_id"],
                            "round": int(rounds[e]),
                            "final_reward": float(z["final_reward"][e]),
                        }
                    )
                    + "\n"
                )
            index_file.flush()
            open(marker, "w").close()
            el = time.time() - t0
            print(
                f"  воркер {args.worker}: {n_ep} эпизодов, {n_frames} кадров, "
                f"{n_bytes / 1e6:.0f} МБ ({n_bytes / max(n_frames, 1) / 1024:.1f} КБ/кадр), "
                f"{n_frames / max(el, 1e-9):.0f} кадров/с",
                flush=True,
            )

    print(f"WRITE_{args.set.upper()}_W{args.worker}_STATUS=OK")


if __name__ == "__main__":
    main()
