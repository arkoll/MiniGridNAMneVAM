"""Сбор траекторий обученной политикой в hdf5 — по одному файлу на эпизод.

Эпизод = одно испытание (решение от 2026-07-29): авто-сброс не используется, каждый слот
среды проходит ровно один эпизод. Побочно это убирает конфаунд «номер испытания в цепочке»:
политика без памяти, каждый эпизод — свежий уровень.

Соглашение о действиях (METHODS.md §10): `actions[t]` — действие, выполненное ИЗ кадра `t`
и приводящее к кадру `t+1`. Поэтому кадров на один больше, чем действий.

Кадры пишутся в разрешении 256x256 (плитка 32 px) — это то, что увидит world-модель.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import distrax
import h5py
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xland_common as xc
from ppo_xland import ActorCritic


def build_rollout(env, policy_tile_render, net, max_steps):
    """Один эпизод на каждый слот среды. Возвращает символические состояния и действия."""

    def rollout(params, reset_key, net_params, temperature, rng):
        timestep = jax.vmap(env.reset)(params, reset_key)

        def body(carry, _):
            rng, ts = carry
            rng, k = jax.random.split(rng)
            obs_small = jax.vmap(policy_tile_render)(ts.state.grid, ts.state.agent)
            logits, _ = net.apply(net_params, obs_small)
            action = distrax.Categorical(logits=logits / temperature).sample(seed=k)
            new_ts = jax.vmap(env.step)(params, ts, action)
            rec = dict(
                grid=ts.state.grid,
                pos=ts.state.agent.position,
                dir=ts.state.agent.direction,
                action=action,
                reward=new_ts.reward,
                done=new_ts.last(),
                terminated=new_ts.discount == 0.0,
            )
            return (rng, new_ts), rec

        (rng, final_ts), rec = jax.lax.scan(body, (rng, timestep), None, max_steps)
        rec["final_grid"] = final_ts.state.grid
        rec["final_pos"] = final_ts.state.agent.position
        rec["final_dir"] = final_ts.state.agent.direction
        return rec

    return rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True, choices=["train", "val_seen", "val_ood_chain", "val_ood_all"])
    p.add_argument("--episodes", type=int, required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--policy", type=str, default="/home/user8_2/AIRI_WAM/runs/xland-ppo/policy.pkl")
    p.add_argument("--out-root", type=str, default="/home/user8_2/AIRI_WAM/data/xland-shape")
    p.add_argument("--gzip", type=int, default=4)
    p.add_argument("--profile-temperature", action="store_true")
    args = p.parse_args()

    # rgb_obs=False: кадры рендерим сами и только для сохраняемых шагов
    env, base_params = xc.make_env(tile_size=xc.DATASET_TILE_SIZE, auto_reset=False, rgb_obs=False)
    benchmarks = xc.all_benchmarks()
    bench = benchmarks[args.benchmark]
    net = ActorCritic(num_actions=len(xc.ACTIONS))
    with open(args.policy, "rb") as f:
        net_params = pickle.load(f)

    policy_render = xc.make_render_fn(xc.POLICY_TILE_SIZE)
    dataset_render = jax.jit(jax.vmap(xc.make_render_fn(xc.DATASET_TILE_SIZE)))
    rollout = jax.jit(build_rollout(env, policy_render, net, base_params.max_steps))

    if args.profile_temperature:
        print(f"профилирование температуры на наборе {args.benchmark}:")
        rows = []
        for temp in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
            ids = jnp.arange(args.batch) % bench.num_rulesets()
            params = base_params.replace(ruleset=jax.vmap(bench.get_ruleset)(ids))
            keys = jax.vmap(lambda i: xc.episode_seed(args.benchmark, i))(jnp.arange(args.batch))
            rec = rollout(params, keys, net_params, temp, jax.random.key(0))
            done = np.asarray(rec["done"])
            rew = np.asarray(rec["reward"])
            first = np.argmax(done, axis=0)
            finished = done.any(axis=0)
            success = np.array([rew[first[i], i] > 0 if finished[i] else False for i in range(args.batch)])
            lengths = np.where(finished, first + 1, base_params.max_steps)
            rows.append(
                {
                    "temperature": temp,
                    "success_rate": float(success.mean()),
                    "mean_length": float(lengths.mean()),
                    "frames_per_episode": float(lengths.mean() + 1),
                }
            )
            print(
                f"  T={temp:<5}: успех {success.mean():.3f}, провал {1 - success.mean():.3f}, "
                f"средняя длина {lengths.mean():.1f}, кадров на эпизод {lengths.mean() + 1:.1f}"
            )
        os.makedirs(args.out_root, exist_ok=True)
        # рабочая точка — по max min(доля успеха, доля провала), как требует METHODS.md §3:
        # обе корзины должны быть непустыми, а не «успеха побольше»
        best = max(rows, key=lambda r: min(r["success_rate"], 1 - r["success_rate"]))
        with open(os.path.join(args.out_root, f"temperature_profile_{args.benchmark}.json"), "w") as f:
            json.dump({"rows": rows, "chosen": best}, f, indent=2, ensure_ascii=False)
        print(f"CHOSEN_TEMPERATURE={best['temperature']}")
        return

    out_dir = os.path.join(args.out_root, args.benchmark)
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(args.out_root, f"index_{args.benchmark}.jsonl")

    seed_offset = 0
    written = 0
    t0 = time.time()
    total_frames = 0
    total_bytes = 0
    n_success = 0
    index_file = open(index_path, "w")

    while written < args.episodes:
        b = min(args.batch, args.episodes - written)
        idx = jnp.arange(seed_offset, seed_offset + b)
        ruleset_ids = idx % bench.num_rulesets()
        params = base_params.replace(ruleset=jax.vmap(bench.get_ruleset)(ruleset_ids))
        keys = jax.vmap(lambda i: xc.episode_seed(args.benchmark, i))(idx)
        rec = rollout(params, keys, net_params, args.temperature, jax.random.key(int(seed_offset)))

        done = np.asarray(rec["done"])
        rew = np.asarray(rec["reward"])
        term = np.asarray(rec["terminated"])
        acts = np.asarray(rec["action"])
        grids = np.asarray(rec["grid"])
        poss = np.asarray(rec["pos"])
        dirs = np.asarray(rec["dir"])
        fin_grid = np.asarray(rec["final_grid"])
        fin_pos = np.asarray(rec["final_pos"])
        fin_dir = np.asarray(rec["final_dir"])
        init_tiles = np.asarray(bench.init_tiles)

        for e in range(b):
            finished = bool(done[:, e].any())
            L = int(np.argmax(done[:, e])) + 1 if finished else base_params.max_steps
            success = bool(term[L - 1, e]) and float(rew[L - 1, e]) > 0

            # Состояния эпизода: grids[k] — состояние ПЕРЕД действием k, последнее
            # состояние берём из финального timestep. Значит g_full[k] — состояние
            # после k шагов, а терминальное лежит по индексу L.
            #
            # Рендерим ВСЕГДА одно и то же число кадров, а режем уже на хосте: если
            # звать jit-функцию на каждой новой длине эпизода, XLA перекомпилирует её
            # под каждую из 96 возможных длин, и сбор встаёт колом.
            g_full = np.concatenate([grids[:, e], fin_grid[e][None]], axis=0)
            p_full = np.concatenate([poss[:, e], fin_pos[e][None]], axis=0)
            d_full = np.concatenate([dirs[:, e], fin_dir[e][None]], axis=0)
            frames_all = dataset_render(jnp.asarray(g_full), _agent_batch(p_full, d_full))
            frames = np.asarray(frames_all[: L + 1], dtype=np.uint8)
            g, pp, dd = g_full[: L + 1], p_full[: L + 1], d_full[: L + 1]

            ep_id = written
            path = os.path.join(out_dir, f"ep_{ep_id:07d}.h5")
            with h5py.File(path, "w") as h:
                h.create_dataset("frames", data=frames, compression="gzip", compression_opts=args.gzip)
                h.create_dataset("actions", data=acts[:L, e].astype(np.uint8))
                h.create_dataset("rewards", data=rew[:L, e].astype(np.float32))
                h.create_dataset("grid", data=g.astype(np.uint8), compression="gzip", compression_opts=args.gzip)
                h.create_dataset("agent_pos", data=pp.astype(np.uint8))
                h.create_dataset("agent_dir", data=dd.astype(np.uint8))
                rid = int(ruleset_ids[e])
                h.attrs.update(
                    benchmark=args.benchmark,
                    ruleset_id=rid,
                    reset_index=int(idx[e]),
                    seed_base=xc.SEED_RANGES[args.benchmark],
                    steps=L,
                    num_frames=frames.shape[0],
                    outcome="success" if success else "timeout",
                    final_reward=float(rew[L - 1, e]),
                    square_color=int(init_tiles[rid, 0, 1]),
                    inert_1=f"{int(init_tiles[rid, 1, 0])}/{int(init_tiles[rid, 1, 1])}",
                    inert_2=f"{int(init_tiles[rid, 2, 0])}/{int(init_tiles[rid, 2, 1])}",
                    tile_size=xc.DATASET_TILE_SIZE,
                )
            size = os.path.getsize(path)
            total_bytes += size
            total_frames += frames.shape[0]
            n_success += int(success)
            index_file.write(
                json.dumps(
                    {
                        "id": ep_id,
                        "path": os.path.relpath(path, args.out_root),
                        "steps": L,
                        "num_frames": int(frames.shape[0]),
                        "outcome": "success" if success else "timeout",
                        "final_reward": float(rew[L - 1, e]),
                        "ruleset_id": rid,
                        "reset_index": int(idx[e]),
                        "square_color": int(init_tiles[rid, 0, 1]),
                    }
                )
                + "\n"
            )
            written += 1

        seed_offset += b
        if (written // args.batch) % 4 == 0 or written >= args.episodes:
            el = time.time() - t0
            print(
                f"  {written}/{args.episodes} эпизодов, {total_frames} кадров, "
                f"{total_bytes / 1e6:.0f} МБ ({total_bytes / max(total_frames, 1) / 1024:.1f} КБ/кадр), "
                f"успех {n_success / max(written, 1):.3f}, {el:.0f} с",
                flush=True,
            )

    index_file.close()
    report = {
        "benchmark": args.benchmark,
        "episodes": written,
        "frames": total_frames,
        "bytes": total_bytes,
        "kb_per_frame": total_bytes / max(total_frames, 1) / 1024,
        "success_rate": n_success / max(written, 1),
        "temperature": args.temperature,
        "seconds": time.time() - t0,
        "max_steps": int(base_params.max_steps),
        "tile_size": xc.DATASET_TILE_SIZE,
        "num_rulesets": int(bench.num_rulesets()),
    }
    with open(os.path.join(args.out_root, f"report_{args.benchmark}.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))
    print(f"COLLECT_{args.benchmark.upper()}_STATUS=OK")


def _agent_batch(pos, direction):
    """Собирает AgentState с батчевой размерностью для vmap-рендера."""
    from xminigrid.types import AgentState

    return AgentState(
        position=jnp.asarray(pos),
        direction=jnp.asarray(direction),
        pocket=jnp.zeros((pos.shape[0], 2), dtype=jnp.uint8),
    )


if __name__ == "__main__":
    main()
