"""Эталон: доля побед ЭКСПЕРТНОЙ политики на тех же банках и тех же сидах.

Нужен, чтобы число WAM было с чем сравнить: сколько вообще берётся на этих раскладках.
Считается на GPU за секунды, сокет не нужен.

Температура. Раньше здесь стоял жёстко `dist.mode()`, то есть эталон мерился ЖАДНО — той
же ручкой, которая нашей модели стоила 0.33 доли побед. Сравнивать жадного эксперта с
сэмплирующей моделью нечестно, поэтому появился `--temperature`: 0 сохраняет прежнее
поведение и прежние числа, больше нуля — выборка из тех же логитов.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/user8_2/AIRI_WAM/scripts/xland")
import shape_common as sc  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.tree_util as jtu  # noqa: E402
from flax import serialization  # noqa: E402

CKPT = f"{sc.REPO}/runs/privileged_rl_easy_20260730/shape_easy/checkpoints/best_val/train_state.msgpack"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.0, help="0 — жадно, как раньше")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    temperature = float(args.temperature)

    env, base_params, net, _, benches = sc.make_env_and_policy(load_weights=False)
    max_steps = int(base_params.max_steps)
    init_params = base_params.replace(ruleset=benches["train"].get_ruleset(0))
    init_ts = env.reset(init_params, jax.random.key(0))
    template = net.init(jax.random.key(0), jtu.tree_map(lambda x: x[None], init_ts.observation))
    with open(CKPT, "rb") as f:
        params_policy = serialization.from_state_dict(template, serialization.msgpack_restore(f.read())["params"])

    def rollout(params, keys, rng):
        ts = jax.vmap(env.reset)(params, keys)

        def body(carry, _):
            rng, ts, done, success, length = carry
            rng, k = jax.random.split(rng)
            dist, _ = net.apply(params_policy, ts.observation)
            if temperature > 0.0:
                action = jax.random.categorical(k, dist.logits / temperature)
            else:
                action = dist.mode()

            def step_if_active(pp, tt, aa, dd):
                return jax.lax.cond(dd, lambda: tt, lambda: env.step(pp, tt, aa))

            nts = jax.vmap(step_if_active)(params, ts, action, done)
            success = success | ((~done) & nts.last() & (nts.reward >= 0.9))
            length = length + (~done)
            return (rng, nts, done | nts.last(), success, length), None

        n = keys.shape[0]
        init = (rng, ts, jnp.zeros(n, bool), jnp.zeros(n, bool), jnp.zeros(n, jnp.int32))
        (_, _, _, success, length), _ = jax.lax.scan(body, init, None, max_steps)
        return success, length

    rollout_jit = jax.jit(rollout)
    out = {"temperature": temperature, "policy": "greedy" if temperature == 0.0 else f"sample T={temperature}"}
    # те же банки и те же сиды, что в замкнутом контуре
    for bank, seed_base in (("train", 21_000_000), ("val", 22_000_000)):
        bench = benches[bank]
        idx = np.arange(args.episodes)
        tasks = idx % int(bench.num_rulesets())
        params = base_params.replace(ruleset=jax.vmap(bench.get_ruleset)(jnp.asarray(tasks)))
        keys = jax.vmap(jax.random.key)(jnp.asarray(seed_base + idx))
        s, ln = rollout_jit(params, keys, jax.random.key(args.seed))
        s, ln = np.asarray(s), np.asarray(ln)
        sr = float(s.mean())
        out[bank] = {
            "episodes": int(args.episodes),
            "success_rate": sr,
            "success_rate_stderr": float(np.sqrt(sr * (1 - sr) / args.episodes)),
            "mean_length": float(ln.mean()),
            "mean_length_on_success": float(ln[s].mean()) if s.any() else None,
            "seed_base": seed_base,
            # поэпизодно, чтобы считать исходную награду среды 1 - 0.9*шаг/лимит
            # и её разброс, а не только среднее
            "success": s.astype(int).tolist(),
            "length": ln.astype(int).tolist(),
            "max_steps": int(max_steps),
        }
        print(
            f"эксперт T={temperature}, банк {bank}: доля побед {sr:.4f} ± {out[bank]['success_rate_stderr']:.4f}, "
            f"средняя длина {ln.mean():.1f}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("EXPERT_REFERENCE_STATUS=OK")


if __name__ == "__main__":
    main()
