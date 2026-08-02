"""Этап 1: набрать состояния val_ood с распределением ЭКСПЕРТА и отрендерить кадры.

Зачем именно так. Нам нужна контрольная группа, где истинная неопределённость известна
из кода: у 6 деревьев из 9 набор форм на поле восстанавливает правило однозначно, значит
действие жадного эксперта там — детерминированная функция кадра, и истинная энтропия
ровно ноль. У 3 деревьев (0-based 2, 4, 6) на поле по одному шару, кубу и пирамиде, и
какая пара крафтит — из кадра не видно.

Второе деление: до первого удачного крафта и после. Шестиугольник это продукт первого
этапа, на старте его на поле нет. Значит его появление в сетке = первый этап пройден,
и с этого момента неоднозначности нет даже у трёх спорных деревьев: оставшийся объект
и есть катализатор.

Логиты эксперта пишем прямо по ходу раскатки — так не надо восстанавливать TimeStep,
чтобы спросить политику задним числом.
"""

from __future__ import annotations

import argparse
import os
from itertools import product

import numpy as np

import shape_common as sc

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.tree_util as jtu  # noqa: E402
from flax import serialization  # noqa: E402

CKPT = f"{sc.REPO}/runs/privileged_rl_easy_20260730/shape_easy/checkpoints/best_val/train_state.msgpack"
STAGE1_ROLE_INDICES = ((0, 1), (0, 2), (1, 2))
TREES = list(product(range(3), repeat=2))
PALETTES = 36


def ambiguous(tree0):
    s1, s2 = TREES[tree0]
    return s2 not in STAGE1_ROLE_INDICES[s1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", default="val", choices=["train", "val"])
    p.add_argument("--rounds", type=int, default=2, help="эпизодов на задачу")
    p.add_argument("--per-episode", type=int, default=8, help="сколько шагов брать из эпизода")
    p.add_argument("--seed-base", type=int, default=13_000_000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    env, base_params, net, _, benches = sc.make_env_and_policy(load_weights=False)
    bench = benches[args.bank]
    max_steps = int(base_params.max_steps)
    init_params = base_params.replace(ruleset=bench.get_ruleset(0))
    init_ts = env.reset(init_params, jax.random.key(0))
    template = net.init(jax.random.key(0), jtu.tree_map(lambda x: x[None], init_ts.observation))
    with open(CKPT, "rb") as f:
        policy_params = serialization.from_state_dict(
            template, serialization.msgpack_restore(f.read())["params"]
        )

    from xminigrid.core.constants import Tiles

    HEX = int(Tiles.HEX)

    def rollout(params, keys):
        ts = jax.vmap(env.reset)(params, keys)

        def body(carry, _):
            ts, done = carry
            dist, _ = net.apply(policy_params, ts.observation)
            action = dist.mode()
            logits = dist.logits

            def step_if_active(pp, tt, aa, dd):
                return jax.lax.cond(dd, lambda: tt, lambda: env.step(pp, tt, aa))

            nts = jax.vmap(step_if_active)(params, ts, action, done)
            rec = dict(
                grid=ts.state.grid.astype(jnp.uint8),
                pos=ts.state.agent.position.astype(jnp.uint8),
                dir=ts.state.agent.direction.astype(jnp.uint8),
                pocket=ts.state.agent.pocket.astype(jnp.uint8),
                logits=logits.astype(jnp.float32),
                action=action.astype(jnp.uint8),
                active=~done,
            )
            return (nts, done | nts.last()), rec

        _, rec = jax.lax.scan(body, (ts, jnp.zeros(keys.shape[0], bool)), None, max_steps)
        return rec

    rollout_jit = jax.jit(rollout)
    render = sc.make_render_fn(base_params)

    def downsample_half(rgb):
        s = rgb.astype(np.uint16)
        summed = s[0::2, 0::2] + s[0::2, 1::2] + s[1::2, 0::2] + s[1::2, 1::2]
        return ((summed + 2) // 4).astype(np.uint8)

    rng = np.random.default_rng(0)
    frames, logits_all, trees, stage1, expert_a = [], [], [], [], []

    for rnd in range(args.rounds):
        tasks = np.arange(sc.NUM_TASKS)
        idx = rnd * sc.NUM_TASKS + tasks
        params = base_params.replace(ruleset=jax.vmap(bench.get_ruleset)(jnp.asarray(tasks)))
        keys = jax.vmap(jax.random.key)(jnp.asarray(args.seed_base + idx))
        rec = {k: np.asarray(v) for k, v in rollout_jit(params, keys).items()}

        for e in range(len(tasks)):
            alive = np.where(rec["active"][:, e])[0]
            if len(alive) == 0:
                continue
            take = rng.choice(alive, size=min(args.per_episode, len(alive)), replace=False)
            t0 = int(tasks[e]) // PALETTES
            for k in take:
                g = rec["grid"][k, e]
                frames.append(downsample_half(
                    render(g, rec["pos"][k, e], rec["dir"][k, e], rec["pocket"][k, e])
                ))
                logits_all.append(rec["logits"][k, e])
                expert_a.append(int(rec["action"][k, e]))
                trees.append(t0)
                # шестиугольник на поле = первый этап пройден
                stage1.append(int((g[..., 0] == HEX).any()))
        print(f"  круг {rnd}: набрано {len(frames)} кадров", flush=True)

    frames = np.stack(frames)
    out = dict(
        rgb=frames,
        expert_logits=np.stack(logits_all).astype(np.float32),
        expert_action=np.asarray(expert_a, dtype=np.int64),
        tree=np.asarray(trees, dtype=np.int64),
        stage1_done=np.asarray(stage1, dtype=np.int64),
        determined=np.asarray([0 if ambiguous(t) else 1 for t in trees], dtype=np.int64),
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"кадров {len(frames)}, из них правило видно: {int(out['determined'].sum())}, "
          f"после крафта: {int(out['stage1_done'].sum())}")
    print(f"ENTROPY_STATES_STATUS=OK -> {args.out}")


if __name__ == "__main__":
    main()
