"""Случайная политика по допустимым действиям: нижняя граница доли успеха.

Нужна, чтобы понять, достижим ли вообще баланс успех/провал: вся лестница обученных
чекпоинтов при стохастической раскатке даёт 0.80-1.00, ниже не опускается.
"""

from __future__ import annotations

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

import train_privileged_crafting_ppo as T  # noqa: E402

ENV_ID = "XLand-MiniGrid-ShapeCraft-Easy-8x8-v1"
BATCH = 512


def main():
    train_bench, val_bench = T.task_banks(ENV_ID)
    env, base_params = T.make_env(ENV_ID, autoreset=False)
    max_steps = int(base_params.max_steps)

    def rollout(params, keys, rng, masked):
        ts = jax.vmap(env.reset)(params, keys)

        def body(carry, _):
            rng, ts, done, success, length, events = carry
            rng, k = jax.random.split(rng)
            if masked:
                logits = jnp.where(ts.observation["action_mask"], 0.0, -1e9)
            else:
                logits = jnp.zeros_like(ts.observation["action_mask"], dtype=jnp.float32)
            action = jax.random.categorical(k, logits)

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

    for masked in (True, False):
        fn = jax.jit(lambda p, k, r, m=masked: rollout(p, k, r, m))
        label = "по допустимым" if masked else "по всем шести"
        for split, bench, base in (("train", train_bench, 500_000), ("val", val_bench, 600_000)):
            ids = jnp.arange(BATCH) % bench.num_rulesets()
            params = base_params.replace(ruleset=jax.vmap(bench.get_ruleset)(ids))
            keys = jax.vmap(jax.random.key)(jnp.arange(base, base + BATCH))
            s, ln, ev = fn(params, keys, jax.random.key(base))
            s, ln, ev = np.asarray(s), np.asarray(ln), np.asarray(ev)
            ls = f"{ln[s].mean():.0f}" if s.any() else "--"
            print(
                f"случайная ({label}) | {split:>5}: успех {s.mean():.3f}, длина усп {ls}, "
                f"событий/эпизод {ev.mean():.1f}, кадров/эпизод {ln.mean() + 1:.1f}"
            )
    print("RANDOM_STATUS=OK")


if __name__ == "__main__":
    main()
