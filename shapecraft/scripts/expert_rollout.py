"""Экспертный сбор: ЖАДНАЯ политика, только успешные эпизоды, квота на каждую задачу.

Отличие от прошлого сбора. Тот датасет собирался под обусловленную действием world-модель,
и одна шестая эпизодов шла случайной политикой — там это ценность: покрытие состояний и
честная истина для вопроса «что делает вот это действие». В joint-модели действие не подаётся,
а ПРЕДСКАЗЫВАЕТСЯ, и случайные действия становятся шумом в разметке: по кадру их предсказать
нельзя в принципе.

Поэтому здесь: argmax вместо сэмплирования, случайной компоненты нет вовсе, провальные эпизоды
не сохраняются. Тогда цель головы действий — детерминированная функция состояния.

Квота на задачу, а не общее число эпизодов: жадная политика решает валидационные задачи реже,
чем обучающие (0.859 против 0.964 по метаданным чекпоинта), и при общей квоте наборы разъехались
бы по представленности задач. Досбор идёт новыми сидами, пока каждая из 324 задач не наберёт своё.

Остаточное смещение назвать честно: внутри задачи сохраняются те раскладки, которые политика
СМОГЛА решить. На val_ood таких меньше, значит набор чуть легче своей полной популяции.
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
import jax.tree_util as jtu  # noqa: E402
from flax import serialization  # noqa: E402

CKPT = f"{sc.REPO}/runs/privileged_rl_easy_20260730/shape_easy/checkpoints/best_val/train_state.msgpack"
DST = "/home/user8_2/AIRI_WAM/data/xland-expert"
SETS = {
    "train": {"tasks": "train", "seed_base": 11_000_000},
    "val_id": {"tasks": "train", "seed_base": 12_000_000},
    "val_ood": {"tasks": "val", "seed_base": 13_000_000},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, choices=list(SETS))
    p.add_argument("--per-task", type=int, required=True)
    p.add_argument("--batch", type=int, default=1296)
    p.add_argument("--max-rounds", type=int, default=40, help="защита от бесконечного добора")
    args = p.parse_args()

    cfg = SETS[args.set]
    target_total = args.per_task * sc.NUM_TASKS
    env, base_params, net, _, benches = sc.make_env_and_policy(load_weights=False)
    bench = benches[cfg["tasks"]]
    max_steps = int(base_params.max_steps)

    init_params = base_params.replace(ruleset=bench.get_ruleset(0))
    init_ts = env.reset(init_params, jax.random.key(0))
    template = net.init(jax.random.key(0), jtu.tree_map(lambda x: x[None], init_ts.observation))
    with open(CKPT, "rb") as f:
        policy_params = serialization.from_state_dict(template, serialization.msgpack_restore(f.read())["params"])

    out_dir = os.path.join(DST, "_shards", args.set)
    os.makedirs(out_dir, exist_ok=True)
    print(f"набор {args.set}: цель {args.per_task} успешных эпизодов на каждую из {sc.NUM_TASKS} задач")
    print(f"политика: {os.path.basename(os.path.dirname(CKPT))}, ЖАДНО (argmax), провалы не сохраняются")

    def rollout(params, keys):
        ts = jax.vmap(env.reset)(params, keys)

        def body(carry, _):
            ts, done = carry
            dist, _ = net.apply(policy_params, ts.observation)
            action = dist.mode()  # жадно: цель головы действий должна быть детерминированной

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
            return (nts, done | nts.last()), rec

        (final_ts, _), rec = jax.lax.scan(body, (ts, jnp.zeros(keys.shape[0], bool)), None, max_steps)
        rec["final_grid"] = final_ts.state.grid.astype(jnp.uint8)
        rec["final_pos"] = final_ts.state.agent.position.astype(jnp.uint8)
        rec["final_dir"] = final_ts.state.agent.direction.astype(jnp.uint8)
        rec["final_pocket"] = final_ts.state.agent.pocket.astype(jnp.uint8)
        return rec

    rollout_jit = jax.jit(rollout)

    quota = np.zeros(sc.NUM_TASKS, dtype=np.int32)
    seed_cursor = 0
    shard_id = 0
    kept = attempted = 0
    lengths = []
    t0 = time.time()

    for round_index in range(args.max_rounds):
        need = np.where(quota < args.per_task)[0]
        if need.size == 0:
            break
        # задачи, которым ещё не хватает, повторяются по кругу до размера батча
        tasks = np.resize(need, min(args.batch, max(len(need), 256)))
        idx = np.arange(seed_cursor, seed_cursor + len(tasks))
        seed_cursor += len(tasks)

        params = base_params.replace(ruleset=jax.vmap(bench.get_ruleset)(jnp.asarray(tasks)))
        keys = jax.vmap(jax.random.key)(jnp.asarray(cfg["seed_base"] + idx))
        rec = rollout_jit(params, keys)
        rec = {k: np.asarray(v) for k, v in rec.items()}

        done = rec["done"] & rec["active"]
        finished = done.any(axis=0)
        length = np.where(finished, done.argmax(axis=0) + 1, max_steps)
        rew = np.take_along_axis(rec["reward"], (length - 1)[None], axis=0)[0]
        success = finished & (rew >= 0.9)
        attempted += len(tasks)

        keep = []
        for e in range(len(tasks)):
            t = int(tasks[e])
            if success[e] and quota[t] < args.per_task:
                quota[t] += 1
                keep.append(e)
        if not keep:
            continue
        keep = np.asarray(keep)
        lengths += (length[keep] + 1).tolist()
        kept += len(keep)

        np.savez_compressed(
            os.path.join(out_dir, f"shard_{shard_id:05d}.npz"),
            task=tasks[keep].astype(np.int32),
            seed=(cfg["seed_base"] + idx[keep]).astype(np.int64),
            length=length[keep].astype(np.int32),
            final_reward=rew[keep].astype(np.float32),
            grid=rec["grid"][:, keep].transpose(1, 0, 2, 3, 4),
            pos=rec["pos"][:, keep].transpose(1, 0, 2),
            dir=rec["dir"][:, keep].transpose(1, 0),
            pocket=rec["pocket"][:, keep].transpose(1, 0, 2),
            action=rec["action"][:, keep].transpose(1, 0),
            final_grid=rec["final_grid"][keep],
            final_pos=rec["final_pos"][keep],
            final_dir=rec["final_dir"][keep],
            final_pocket=rec["final_pocket"][keep],
        )
        shard_id += 1
        print(
            f"  круг {round_index}: попыток {attempted}, сохранено {kept}/{target_total}, "
            f"задач недобрано {int((quota < args.per_task).sum())}, {time.time() - t0:.0f} с",
            flush=True,
        )

    report = {
        "set": args.set,
        "policy": "best_val greedy",
        "checkpoint": CKPT,
        "per_task": args.per_task,
        "episodes": int(kept),
        "target": int(target_total),
        "tasks_short": int((quota < args.per_task).sum()),
        "attempted": int(attempted),
        "greedy_success_rate": kept / max(attempted, 1),
        "frames_total": int(sum(lengths)),
        "frames_per_episode": float(np.mean(lengths)) if lengths else 0.0,
        "seed_base": cfg["seed_base"],
        "tasks_split": cfg["tasks"],
        "shards": shard_id,
        "seconds": time.time() - t0,
    }
    os.makedirs(DST, exist_ok=True)
    with open(os.path.join(DST, f"rollout_{args.set}.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))
    print(f"EXPERT_ROLLOUT_{args.set.upper()}_STATUS=OK")


if __name__ == "__main__":
    main()
