"""Замкнутый контур: WAM ведёт агента в НАСТОЯЩЕЙ среде, считаем долю побед.

Самая честная проверка из возможных: никаких окон, никакой истины под рукой — только кадр
на вход и действие на выход, шаг за шагом до победы или лимита.

Устройство. Среда на JAX, модель на torch, поэтому они разведены по процессам и общаются
через unix-сокет — механизм ментора, `xland_action_server.py` отдаёт по одному действию
на кадр. Из трёх предсказанных действий исполняется только ПЕРВОЕ, дальше перепланирование
с нового кадра.

Кадр среды рисуется в 256 и ужимается ровно вдвое целочисленным усреднением — тем же
выражением, каким собирались обучающие данные. Иначе модель встретила бы чужие пиксели.

Считаем на ОБОИХ банках задач: обучающем и отложенном. Разница между ними и есть эффект
переноса на невиданные привязки цвета, измеренный от начала до конца.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/home/user8_2/AIRI_WAM/scripts/xland")
import shape_common as sc  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def connect_with_retry(path, timeout=180.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(str(path))
            return client
        except OSError as error:
            last = error
            client.close()
            time.sleep(0.25)
    raise TimeoutError(f"не удалось подключиться к {path}: {last}")


def request_action(client, rgb):
    payload = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    client.sendall(struct.pack("!I", len(payload)) + payload)
    response = client.recv(1)
    if len(response) != 1:
        raise ConnectionError("сервер действий вернул неполный ответ")
    return int(struct.unpack("!B", response)[0])


def downsample_half(rgb, target=128):
    if rgb.shape[:2] == (target, target):
        return rgb
    s = rgb.astype(np.uint16)
    summed = s[0::2, 0::2] + s[0::2, 1::2] + s[1::2, 0::2] + s[1::2, 1::2]
    return ((summed + 2) // 4).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    p.add_argument("--bank", required=True, choices=["train", "val"])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed-base", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    env, base_params, _, _, benches = sc.make_env_and_policy(load_weights=False)
    bench = benches[args.bank]
    num_tasks = int(bench.num_rulesets())
    max_steps = int(base_params.max_steps)
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)

    # прогрев компиляции до подключения, чтобы сервер не ждал
    warm_params = base_params.replace(ruleset=bench.get_ruleset(0))
    warm = reset_env(warm_params, jax.random.key(args.seed_base))
    jax.block_until_ready(step_env(warm_params, warm, 0).reward)

    client = connect_with_retry(args.socket)
    results = []
    per_task = defaultdict(list)
    t0 = time.time()
    try:
        for i in range(args.episodes):
            task = i % num_tasks
            seed = args.seed_base + i
            params = base_params.replace(ruleset=bench.get_ruleset(task))
            host_params = jax.device_get(params)
            ts = reset_env(params, jax.random.key(seed))
            success, length = False, 0
            for step in range(max_steps):
                host_ts = jax.device_get(ts)
                rgb = np.asarray(env.render(host_params, host_ts), dtype=np.uint8)
                action = request_action(client, downsample_half(rgb))
                if not 0 <= action < 6:
                    raise ValueError(f"недопустимое действие {action}")
                ts = step_env(params, ts, action)
                host_next = jax.device_get(ts)
                length = step + 1
                if bool(host_next.last()):
                    success = float(host_next.reward) >= 0.9
                    break
            results.append({"episode": i, "task": task, "seed": seed, "success": bool(success), "length": int(length)})
            per_task[task].append(float(success))
            if (i + 1) % 25 == 0:
                sr = float(np.mean([r["success"] for r in results]))
                print(f"  {i + 1}/{args.episodes}: доля побед {sr:.3f}, {time.time() - t0:.0f} с", flush=True)
    finally:
        try:
            client.sendall(struct.pack("!I", 0))
        finally:
            client.close()

    sr = float(np.mean([r["success"] for r in results]))
    lengths = np.array([r["length"] for r in results])
    wins = np.array([r["success"] for r in results])
    n = len(results)
    summary = {
        "bank": args.bank,
        "episodes": n,
        "tasks_covered": len(per_task),
        "success_rate": sr,
        "success_rate_stderr": float(np.sqrt(sr * (1 - sr) / max(n, 1))),
        "mean_length": float(lengths.mean()),
        "mean_length_on_success": float(lengths[wins].mean()) if wins.any() else None,
        "max_steps": max_steps,
        "seed_base": args.seed_base,
        "seconds": time.time() - t0,
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(
        f"банк {args.bank}: доля побед {sr:.4f} ± {summary['success_rate_stderr']:.4f} "
        f"на {n} эпизодах, средняя длина {lengths.mean():.1f}"
    )
    print(f"CLOSED_LOOP_{args.bank.upper()}_STATUS=OK")


if __name__ == "__main__":
    main()
