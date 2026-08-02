"""Этап 1: раскатка эпизодов на GPU, сохранение СИМВОЛЬНЫХ траекторий в шарды.

Кадры тут не рисуются намеренно: рендер идёт на numpy и раскладывается по процессам
CPU, а среда на GPU выдаёт 98 тыс. полезных шагов/с и не является узким местом.
Шарды маленькие (символьное состояние — 133 байта на шаг), зато этап 2 становится
возобновляемым и параллельным.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import shape_common as sc

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, choices=list(sc.SETS))
    p.add_argument("--batch", type=int, default=1296)  # 324 * 4, круги по задачам не рвутся
    args = p.parse_args()

    cfg = sc.SETS[args.set]
    total = sc.episodes_in(args.set)
    env, base_params, net, policy_params, benches = sc.make_env_and_policy()
    bench = benches[cfg["tasks"]]
    max_steps = int(base_params.max_steps)
    out_dir = os.path.join(sc.SHARD_ROOT, args.set)
    os.makedirs(out_dir, exist_ok=True)

    print(f"набор {args.set}: {total} эпизодов, задачи {cfg['tasks']}, сиды от {cfg['seed_base']}")
    print(f"кругов по задачам {cfg['rounds']}, случайной политикой каждый {sc.RANDOM_EVERY}-й круг")

    def rollout(params, keys, use_random, rng):
        ts = jax.vmap(env.reset)(params, keys)

        def body(carry, _):
            rng, ts, done = carry
            rng, k_pol, k_rand = jax.random.split(rng, 3)
            dist, _ = net.apply(policy_params, ts.observation)
            trained = jax.random.categorical(k_pol, dist.logits)
            # случайная политика — только по ДОПУСТИМЫМ действиям, маска есть в наблюдении
            rnd = jax.random.categorical(k_rand, jnp.where(ts.observation["action_mask"], 0.0, -1e9))
            action = jnp.where(use_random, rnd, trained)

            def step_if_active(pp, tt, aa, dd):
                return jax.lax.cond(dd, lambda: tt, lambda: env.step(pp, tt, aa))

            nts = jax.vmap(step_if_active)(params, ts, action, done)
            rec = dict(
                grid=ts.state.grid.astype(jnp.uint8),
                pos=ts.state.agent.position.astype(jnp.uint8),
                dir=ts.state.agent.direction.astype(jnp.uint8),
                pocket=ts.state.agent.pocket.astype(jnp.uint8),
                action=action.astype(jnp.uint8),
                reward=nts.reward.astype(jnp.float32),
                done=nts.last(),
                active=~done,
            )
            return (rng, nts, done | nts.last()), rec

        (_, final_ts, _), rec = jax.lax.scan(body, (rng, ts, jnp.zeros(keys.shape[0], bool)), None, max_steps)
        rec["final_grid"] = final_ts.state.grid.astype(jnp.uint8)
        rec["final_pos"] = final_ts.state.agent.position.astype(jnp.uint8)
        rec["final_dir"] = final_ts.state.agent.direction.astype(jnp.uint8)
        rec["final_pocket"] = final_ts.state.agent.pocket.astype(jnp.uint8)
        return rec

    rollout_jit = jax.jit(rollout)

    t0 = time.time()
    n_success = n_random = 0
    frames_success = frames_fail = 0
    shard_id = 0
    for start in range(0, total, args.batch):
        n = min(args.batch, total - start)
        idx = np.arange(start, start + n)
        tasks = idx % sc.NUM_TASKS
        rounds = idx // sc.NUM_TASKS
        use_random = (rounds % sc.RANDOM_EVERY) == 0

        params = base_params.replace(ruleset=jax.vmap(bench.get_ruleset)(jnp.asarray(tasks)))
        keys = jax.vmap(jax.random.key)(jnp.asarray(cfg["seed_base"] + idx))
        rec = rollout_jit(params, keys, jnp.asarray(use_random), jax.random.key(cfg["seed_base"] + start))
        rec = {k: np.asarray(v) for k, v in rec.items()}

        done = rec["done"] & rec["active"]  # шаг, на котором эпизод реально закончился
        finished = done.any(axis=0)
        length = np.where(finished, done.argmax(axis=0) + 1, max_steps)
        rew = np.take_along_axis(rec["reward"], (length - 1)[None], axis=0)[0]
        success = finished & (rew >= 0.9)

        np.savez_compressed(
            os.path.join(out_dir, f"shard_{shard_id:05d}.npz"),
            episode_id=idx.astype(np.int32),
            task=tasks.astype(np.int32),
            round=rounds.astype(np.int32),
            use_random=use_random,
            length=length.astype(np.int32),
            success=success,
            final_reward=rew.astype(np.float32),
            grid=rec["grid"].transpose(1, 0, 2, 3, 4),
            pos=rec["pos"].transpose(1, 0, 2),
            dir=rec["dir"].transpose(1, 0),
            pocket=rec["pocket"].transpose(1, 0, 2),
            action=rec["action"].transpose(1, 0),
            reward=rec["reward"].transpose(1, 0),
            final_grid=rec["final_grid"],
            final_pos=rec["final_pos"],
            final_dir=rec["final_dir"],
            final_pocket=rec["final_pocket"],
        )
        n_success += int(success.sum())
        n_random += int(use_random.sum())
        frames_success += int((length[success] + 1).sum())
        frames_fail += int((length[~success] + 1).sum())
        shard_id += 1
        if shard_id % 5 == 0 or start + n >= total:
            el = time.time() - t0
            print(
                f"  {start + n}/{total} эпизодов, успех {n_success / (start + n):.3f}, "
                f"{el:.0f} с",
                flush=True,
            )

    total_frames = frames_success + frames_fail
    report = {
        "set": args.set,
        "episodes": total,
        "shards": shard_id,
        "success_rate": n_success / total,
        "random_share": n_random / total,
        "frames_total": total_frames,
        "frames_from_success": frames_success,
        "frames_from_fail": frames_fail,
        "frame_balance": frames_success / max(total_frames, 1),
        "seconds": time.time() - t0,
        "checkpoint": sc.CKPT,
        "seed_base": cfg["seed_base"],
        "tasks_split": cfg["tasks"],
    }
    with open(os.path.join(sc.SHARD_ROOT, f"rollout_{args.set}.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))
    print(f"ROLLOUT_{args.set.upper()}_STATUS=OK")


if __name__ == "__main__":
    main()
