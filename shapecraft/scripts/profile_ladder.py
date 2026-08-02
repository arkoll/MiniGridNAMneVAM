"""Лестница чекпоинтов: доля успеха и плотность событий крафта на каждом.

Зашумление не даёт опустить успех ниже ~0.87, поэтому единственная рабочая ручка для
равномерного распределения по успеху — смесь чекпоинтов разного качества. Коллега сохранил
их специально: train_sr_05/20/50/80/95, best_val, final.

Меряем на обоих сплитах одним и тем же прогоном, плюс считаем, сколько шагов в эпизоде
реально меняют сетку — это и есть события крафта, ради которых датасет собирается.
"""

from __future__ import annotations

import json
import os
import sys
import types

import numpy as np

REPO = "/home/User8/xland-minigrid-five-object"
sys.path.insert(0, os.path.join(REPO, "training"))
sys.path.insert(0, os.path.join(REPO, "src"))

if "tensorboardX" not in sys.modules:
    stub = types.ModuleType("tensorboardX")

    class _SW:
        def __init__(self, *a, **k):
            pass

        def add_scalar(self, *a, **k):
            pass

        def close(self):
            pass

    stub.SummaryWriter = _SW
    sys.modules["tensorboardX"] = stub

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.tree_util as jtu  # noqa: E402
from flax import serialization  # noqa: E402

import train_privileged_crafting_ppo as T  # noqa: E402

ENV_ID = "XLand-MiniGrid-ShapeCraft-Easy-8x8-v1"
CKPT_ROOT = f"{REPO}/runs/privileged_rl_easy_20260730/shape_easy/checkpoints"
LADDER = ("train_sr_05", "train_sr_20", "train_sr_50", "train_sr_80", "train_sr_95", "best_val", "final")
BATCH = 512


def main():
    train_bench, val_bench = T.task_banks(ENV_ID)
    env, base_params = T.make_env(ENV_ID, autoreset=False)
    _, dir_count = T.embodiment_spec(ENV_ID)
    net = T.PrivilegedActorCritic(
        num_actions=env.num_actions(base_params), agent_direction_classes=dir_count + 1
    )
    max_steps = int(base_params.max_steps)

    init_params = base_params.replace(ruleset=train_bench.get_ruleset(0))
    init_ts = env.reset(init_params, jax.random.key(0))
    template = net.init(jax.random.key(0), jtu.tree_map(lambda x: x[None], init_ts.observation))

    def load(name):
        path = os.path.join(CKPT_ROOT, name, "train_state.msgpack")
        with open(path, "rb") as f:
            return serialization.from_state_dict(template, serialization.msgpack_restore(f.read())["params"])

    def rollout(policy_params, params, keys, rng):
        ts = jax.vmap(env.reset)(params, keys)

        def body(carry, _):
            rng, ts, done, success, length, events = carry
            rng, k = jax.random.split(rng)
            dist, _ = net.apply(policy_params, ts.observation)
            action = jax.random.categorical(k, dist.logits)

            def step_if_active(p, t, a, d):
                return jax.lax.cond(d, lambda: t, lambda: env.step(p, t, a))

            nts = jax.vmap(step_if_active)(params, ts, action, done)
            events = events + (~done) * jnp.any(nts.state.grid != ts.state.grid, axis=(1, 2, 3))
            new_done = nts.last()
            success = success | ((~done) & new_done & (nts.reward >= 0.9))
            length = length + (~done)
            done = done | new_done
            return (rng, nts, done, success, length, events), None

        n = keys.shape[0]
        init = (rng, ts, jnp.zeros(n, bool), jnp.zeros(n, bool), jnp.zeros(n, jnp.int32), jnp.zeros(n, jnp.int32))
        (_, _, done, success, length, events), _ = jax.lax.scan(body, init, None, max_steps)
        return success, length, events

    rollout_jit = jax.jit(rollout)
    rows = []
    print(f"{'чекпоинт':>12} | {'сплит':>5} | успех | длина усп | длина пров | событий/эпизод | кадров/эпизод")
    for name in LADDER:
        path = os.path.join(CKPT_ROOT, name, "train_state.msgpack")
        if not os.path.exists(path):
            print(f"{name:>12} | нет такого чекпоинта")
            continue
        pp = load(name)
        row = {"checkpoint": name}
        for split, bench, base in (("train", train_bench, 300_000), ("val", val_bench, 400_000)):
            ids = jnp.arange(BATCH) % bench.num_rulesets()
            params = base_params.replace(ruleset=jax.vmap(bench.get_ruleset)(ids))
            keys = jax.vmap(jax.random.key)(jnp.arange(base, base + BATCH))
            s, ln, ev = rollout_jit(pp, params, keys, jax.random.key(base))
            s, ln, ev = np.asarray(s), np.asarray(ln), np.asarray(ev)
            row[split] = dict(
                success=float(s.mean()),
                len_success=float(ln[s].mean()) if s.any() else None,
                len_fail=float(ln[~s].mean()) if (~s).any() else None,
                events=float(ev.mean()),
                frames=float(ln.mean() + 1),
            )
            r = row[split]
            ls = f"{r['len_success']:.0f}" if r["len_success"] else "--"
            lf = f"{r['len_fail']:.0f}" if r["len_fail"] else "--"
            print(
                f"{name:>12} | {split:>5} | {r['success']:.3f} | {ls:>9} | {lf:>10} | "
                f"{r['events']:>14.1f} | {r['frames']:>13.1f}"
            )
        rows.append(row)

    out = "/home/user8_2/AIRI_WAM/reports/xland/ladder_profile.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nотчёт: {out}")
    print("LADDER_STATUS=OK")


if __name__ == "__main__":
    main()
