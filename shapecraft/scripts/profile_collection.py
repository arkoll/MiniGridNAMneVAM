"""Профилирование сбора: скорость раскатки, скорость рендера и ручка зашумления.

Три вопроса, без ответов на которые нельзя ни спланировать сбор, ни назвать время:

  1. сколько шагов среды в секунду даёт векторизованная раскатка с их политикой;
  2. сколько кадров в секунду даёт их рендер (он на numpy, под jit не идёт);
  3. какой ручкой и до какой степени можно опустить долю успеха.

Про ручку. Логиты их политики маскируются (-1e9 на недопустимых действиях), поэтому
температура маску не ломает. Но если логиты насыщены, температура ничего не даст —
как было с нашей политикой в Baba. Поэтому меряем обе ручки: температуру и
epsilon-жадность, ограниченную ДОПУСТИМЫМИ действиями.
"""

from __future__ import annotations

import json
import os
import sys
import time
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
CKPT = f"{REPO}/runs/privileged_rl_easy_20260730/shape_easy/checkpoints/best_val/train_state.msgpack"
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
    with open(CKPT, "rb") as f:
        policy_params = serialization.from_state_dict(template, serialization.msgpack_restore(f.read())["params"])
    print(f"среда {ENV_ID}, max_steps={max_steps}, задач train/val {train_bench.num_rulesets()}/{val_bench.num_rulesets()}")

    def rollout(params, keys, temperature, epsilon, rng):
        ts = jax.vmap(env.reset)(params, keys)

        def body(carry, _):
            rng, ts, done, success, length, changed = carry
            rng, k_pol, k_eps, k_rand = jax.random.split(rng, 4)
            dist, _ = net.apply(policy_params, ts.observation)
            logits = dist.logits / temperature
            action = jax.random.categorical(k_pol, logits)
            # epsilon-жадность ТОЛЬКО по допустимым действиям: маска в наблюдении
            mask = ts.observation["action_mask"]
            rand_action = jax.random.categorical(k_rand, jnp.where(mask, 0.0, -1e9))
            explore = jax.random.uniform(k_eps, shape=action.shape) < epsilon
            action = jnp.where(explore, rand_action, action)

            def step_if_active(p, t, a, d):
                return jax.lax.cond(d, lambda: t, lambda: env.step(p, t, a))

            nts = jax.vmap(step_if_active)(params, ts, action, done)
            # «кадр изменился» — прокси события в мире, нужен для оценки полезности данных
            changed = changed + (~done) * jnp.any(
                nts.state.grid != ts.state.grid, axis=(1, 2, 3)
            )
            new_done = nts.last()
            success = success | ((~done) & new_done & (nts.reward >= 0.9))
            length = length + (~done)
            done = done | new_done
            return (rng, nts, done, success, length, changed), None

        init = (
            rng,
            ts,
            jnp.zeros(keys.shape[0], bool),
            jnp.zeros(keys.shape[0], bool),
            jnp.zeros(keys.shape[0], jnp.int32),
            jnp.zeros(keys.shape[0], jnp.int32),
        )
        (_, _, done, success, length, changed), _ = jax.lax.scan(body, init, None, max_steps)
        return success, length, done, changed

    rollout_jit = jax.jit(rollout)

    def run(bench, temperature, epsilon, seed_base):
        ids = jnp.arange(BATCH) % bench.num_rulesets()
        params = base_params.replace(ruleset=jax.vmap(bench.get_ruleset)(ids))
        keys = jax.vmap(jax.random.key)(jnp.arange(seed_base, seed_base + BATCH))
        s, ln, dn, ch = rollout_jit(params, keys, temperature, epsilon, jax.random.key(seed_base))
        return np.asarray(s), np.asarray(ln), np.asarray(dn), np.asarray(ch)

    print("\n=== скорость раскатки ===")
    t0 = time.time()
    s, ln, dn, ch = run(train_bench, 1.0, 0.0, 1)
    jax.block_until_ready(s)
    compile_s = time.time() - t0
    t0 = time.time()
    s, ln, dn, ch = run(train_bench, 1.0, 0.0, 2)
    jax.block_until_ready(s)
    dt = time.time() - t0
    env_steps = BATCH * max_steps
    print(f"компиляция {compile_s:.1f} с; {BATCH} эпизодов за {dt:.2f} с")
    print(f"это {env_steps / dt / 1e3:.0f} тыс. шагов среды/с (с учётом холостых шагов после конца эпизода)")
    print(f"полезных шагов: {ln.sum()}, то есть {ln.sum() / dt / 1e3:.1f} тыс. полезных шагов/с")

    print("\n=== скорость рендера (numpy, их функция) ===")
    p1 = base_params.replace(ruleset=train_bench.get_ruleset(0))
    ts1 = env.reset(p1, jax.random.key(0))
    frame = np.asarray(env.render(p1, ts1))
    t0 = time.time()
    N = 200
    for _ in range(N):
        env.render(p1, ts1)
    rdt = time.time() - t0
    print(f"кадр {frame.shape}, {N / rdt:.0f} кадров/с в один поток")

    print("\n=== ручки зашумления ===")
    rows = []
    for name, temp, eps in [
        ("как есть", 1.0, 0.0),
        ("T=1.5", 1.5, 0.0),
        ("T=3", 3.0, 0.0),
        ("T=10", 10.0, 0.0),
        ("eps=0.05", 1.0, 0.05),
        ("eps=0.10", 1.0, 0.10),
        ("eps=0.15", 1.0, 0.15),
        ("eps=0.25", 1.0, 0.25),
        ("eps=0.40", 1.0, 0.40),
    ]:
        out = {}
        for split, bench, base in (("train", train_bench, 100_000), ("val", val_bench, 200_000)):
            s, ln, dn, ch = run(bench, temp, eps, base)
            out[split] = dict(
                success=float(s.mean()),
                mean_len=float(ln.mean()),
                len_success=float(ln[s].mean()) if s.any() else float("nan"),
                len_fail=float(ln[~s].mean()) if (~s).any() else float("nan"),
                changed=float(ch.mean()),
            )
        rows.append(dict(knob=name, temperature=temp, epsilon=eps, **out))
        tr, va = out["train"], out["val"]
        print(
            f"  {name:>9}: train успех {tr['success']:.3f} (длина усп/пров {tr['len_success']:.0f}/{tr['len_fail']:.0f}) | "
            f"val успех {va['success']:.3f} (длина усп/пров {va['len_success']:.0f}/{va['len_fail']:.0f})"
        )

    out_path = "/home/user8_2/AIRI_WAM/reports/xland/collection_profile.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "rollout_seconds_per_512_episodes": dt,
                "render_fps_single_thread": N / rdt,
                "frame_shape": list(frame.shape),
                "max_steps": max_steps,
                "rows": rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nотчёт: {out_path}")
    print("PROFILE_STATUS=OK")


if __name__ == "__main__":
    main()
