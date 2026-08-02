"""Смешанный сбор с ЭКСПЕРТНОЙ разметкой: половина эпизодов экспертом, половина случайно.

Зачем. Прошлый чисто экспертный датасет дал точность действий 0.87, но долю побед в
замкнутом контуре 0.34 — хуже, чем старый смешанный (0.56 / 0.40). Причина: обучаясь только
на успешных экспертных траекториях, модель не видит ни одного состояния вне них, и первая же
ошибка выводит её туда, где обучающего сигнала нет.

Что меняем. Состояния берём из двух источников поровну по ЭПИЗОДАМ, но целевое действие в
каждом кадре — всегда действие ЭКСПЕРТА, спрошенного в этом самом состоянии (жадно, argmax).
На экспертных эпизодах выполненное и целевое действия совпадают. На случайных выполняется
случайное допустимое действие, а метка остаётся экспертной: «вот куда ушёл, а вот что надо
было делать отсюда». Это и есть тот сигнал возврата на путь, которого не было.

Ограничение назвать честно. Эксперт привилегированный (видит правила) и обучался на своём
же распределении. Далеко от него его жадное действие ничем не гарантировано — размечает он
случайные состояния «как умеет», а не идеально.

Баланс по кадрам сильно смещён в сторону случайных: успешный экспертный эпизод это 28 кадров,
случайный всегда 193 (лимит шагов). При равенстве по эпизодам получается около 87% кадров из
случайных. Это осознанный выбор: ставка на покрытие состояний.
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

# тот же чекпоинт, что размечал прошлый экспертный датасет: сравнение должно
# отличаться только составом состояний, не разметчиком
CKPT = f"{sc.REPO}/runs/privileged_rl_easy_20260730/shape_easy/checkpoints/best_val/train_state.msgpack"
DST = os.environ.get("DAGGER_DST", "/home/user8_2/AIRI_WAM/data/xland-dagger")

# у экспертной половины сиды те же, что в прошлом наборе: её эпизоды буквально совпадают
# с прошлыми, и разница между прогонами — ровно добавленная случайная половина
SETS = {
    "train": {"tasks": "train", "seed_base": {"expert": 11_000_000, "random": 31_000_000}},
    "val_id": {"tasks": "train", "seed_base": {"expert": 12_000_000, "random": 32_000_000}},
    "val_ood": {"tasks": "val", "seed_base": {"expert": 13_000_000, "random": 33_000_000}},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", required=True, choices=list(SETS))
    p.add_argument("--policy", required=True, choices=["expert", "random"])
    p.add_argument("--per-task", type=int, required=True)
    # шард = один круг по задачам. Мелкие шарды нужны, чтобы 12 воркеров рендера
    # разобрали работу поровну: у случайных эпизодов кадров в 7 раз больше
    p.add_argument("--shard-rounds", type=int, default=1)
    p.add_argument("--batch", type=int, default=1296, help="размер батча экспертной ветки")
    p.add_argument("--max-rounds", type=int, default=90, help="защита от бесконечного добора")
    args = p.parse_args()

    cfg = SETS[args.set]
    seed_base = cfg["seed_base"][args.policy]
    is_random = args.policy == "random"
    target_total = args.per_task * sc.NUM_TASKS

    env, base_params, net, _, benches = sc.make_env_and_policy(load_weights=False)
    bench = benches[cfg["tasks"]]
    max_steps = int(base_params.max_steps)

    init_params = base_params.replace(ruleset=bench.get_ruleset(0))
    init_ts = env.reset(init_params, jax.random.key(0))
    template = net.init(jax.random.key(0), jtu.tree_map(lambda x: x[None], init_ts.observation))
    with open(CKPT, "rb") as f:
        policy_params = serialization.from_state_dict(
            template, serialization.msgpack_restore(f.read())["params"]
        )

    out_dir = os.path.join(DST, "_shards", f"{args.set}_{args.policy}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"набор {args.set}, половина {args.policy}: {args.per_task} эпизодов на каждую из {sc.NUM_TASKS} задач")
    print(f"метка действия: эксперт best_val жадно; выполняется: {'случайное допустимое' if is_random else 'то же экспертное'}")
    print(f"лимит шагов среды: {max_steps}", flush=True)

    def rollout(params, keys, rng):
        ts = jax.vmap(env.reset)(params, keys)

        def body(carry, _):
            rng, ts, done = carry
            rng, k_rand = jax.random.split(rng)
            dist, _ = net.apply(policy_params, ts.observation)
            expert = dist.mode()  # МЕТКА: детерминированная функция состояния
            if is_random:
                logits = jnp.where(ts.observation["action_mask"], 0.0, -1e9)
                executed = jax.random.categorical(k_rand, logits)
            else:
                executed = expert

            def step_if_active(pp, tt, aa, dd):
                return jax.lax.cond(dd, lambda: tt, lambda: env.step(pp, tt, aa))

            nts = jax.vmap(step_if_active)(params, ts, executed, done)
            rec = dict(
                grid=ts.state.grid.astype(jnp.uint8),
                pos=ts.state.agent.position.astype(jnp.uint8),
                dir=ts.state.agent.direction.astype(jnp.uint8),
                pocket=ts.state.agent.pocket.astype(jnp.uint8),
                action=expert.astype(jnp.uint8),  # <- в датасет уходит именно она
                executed=executed.astype(jnp.uint8),
                reward=nts.reward.astype(jnp.float32),
                done=nts.last(),
                active=~done,
            )
            return (rng, nts, done | nts.last()), rec

        (_, final_ts, _), rec = jax.lax.scan(
            body, (rng, ts, jnp.zeros(keys.shape[0], bool)), None, max_steps
        )
        rec["final_grid"] = final_ts.state.grid.astype(jnp.uint8)
        rec["final_pos"] = final_ts.state.agent.position.astype(jnp.uint8)
        rec["final_dir"] = final_ts.state.agent.direction.astype(jnp.uint8)
        rec["final_pocket"] = final_ts.state.agent.pocket.astype(jnp.uint8)
        return rec

    rollout_jit = jax.jit(rollout)

    quota = np.zeros(sc.NUM_TASKS, dtype=np.int32)
    seed_cursor = 0
    shard_id = 0
    kept = attempted = n_success = 0
    agree = agree_total = 0
    lengths = []
    t0 = time.time()

    for round_index in range(args.max_rounds if not is_random else args.per_task + 1):
        need = np.where(quota < args.per_task)[0]
        if need.size == 0:
            break
        if is_random:
            # случайные эпизоды не фильтруются: ровно один круг по всем недобравшим задачам
            tasks = np.repeat(need, args.shard_rounds)[: need.size * args.shard_rounds]
        else:
            tasks = np.resize(need, min(args.batch, max(len(need), 256)))
        idx = np.arange(seed_cursor, seed_cursor + len(tasks))
        seed_cursor += len(tasks)

        params = base_params.replace(ruleset=jax.vmap(bench.get_ruleset)(jnp.asarray(tasks)))
        keys = jax.vmap(jax.random.key)(jnp.asarray(seed_base + idx))
        rec = rollout_jit(params, keys, jax.random.key(seed_base + round_index))
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
            if quota[t] >= args.per_task:
                continue
            # экспертная половина — только успехи (цель должна быть достижимой траекторией),
            # случайная — всё подряд, в этом и смысл
            if is_random or success[e]:
                quota[t] += 1
                keep.append(e)
        if not keep:
            continue
        keep = np.asarray(keep)
        lengths += (length[keep] + 1).tolist()
        kept += len(keep)
        n_success += int(success[keep].sum())
        for e in keep:
            L = int(length[e])
            agree += int((rec["action"][:L, e] == rec["executed"][:L, e]).sum())
            agree_total += L

        np.savez_compressed(
            os.path.join(out_dir, f"shard_{shard_id:05d}.npz"),
            task=tasks[keep].astype(np.int32),
            seed=(seed_base + idx[keep]).astype(np.int64),
            length=length[keep].astype(np.int32),
            success=success[keep],
            final_reward=rew[keep].astype(np.float32),
            grid=rec["grid"][:, keep].transpose(1, 0, 2, 3, 4),
            pos=rec["pos"][:, keep].transpose(1, 0, 2),
            dir=rec["dir"][:, keep].transpose(1, 0),
            pocket=rec["pocket"][:, keep].transpose(1, 0, 2),
            action=rec["action"][:, keep].transpose(1, 0),
            executed=rec["executed"][:, keep].transpose(1, 0),
            final_grid=rec["final_grid"][keep],
            final_pos=rec["final_pos"][keep],
            final_dir=rec["final_dir"][keep],
            final_pocket=rec["final_pocket"][keep],
        )
        shard_id += 1
        if shard_id % 5 == 0 or int((quota < args.per_task).sum()) == 0:
            print(
                f"  круг {round_index}: попыток {attempted}, сохранено {kept}/{target_total}, "
                f"задач недобрано {int((quota < args.per_task).sum())}, {time.time() - t0:.0f} с",
                flush=True,
            )

    report = {
        "set": args.set,
        "policy": args.policy,
        "label": "expert_greedy(best_val)",
        "checkpoint": CKPT,
        "per_task": args.per_task,
        "episodes": int(kept),
        "target": int(target_total),
        "tasks_short": int((quota < args.per_task).sum()),
        "attempted": int(attempted),
        "success_rate": n_success / max(kept, 1),
        "executed_equals_label": agree / max(agree_total, 1),
        "frames_total": int(sum(lengths)),
        "frames_per_episode": float(np.mean(lengths)) if lengths else 0.0,
        "seed_base": seed_base,
        "tasks_split": cfg["tasks"],
        "max_steps": max_steps,
        "shards": shard_id,
        "seconds": time.time() - t0,
    }
    os.makedirs(DST, exist_ok=True)
    with open(os.path.join(DST, f"rollout_{args.set}_{args.policy}.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, ensure_ascii=False))
    print(f"DAGGER_ROLLOUT_{args.set.upper()}_{args.policy.upper()}_STATUS=OK")


if __name__ == "__main__":
    main()
