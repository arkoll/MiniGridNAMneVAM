"""Запись пары эпизодов для гифки по гипотезе о привязке цвета к форме.

Одно и то же дерево правил и один и тот же сид раскладки, две палитры: знакомая (банк
train) и отложенная (банк val). Формы, позиции и правила совпадают, отличаются ТОЛЬКО
цвета. Модель ведёт агента сама, кадры сохраняются как есть.

Ворота: символьные сетки двух эпизодов на первом шаге обязаны совпадать по слою плиток
и различаться по слою цветов. Если это не так, панели сравнивать нельзя и скрипт падает.

Эпизод выбирается по объявленному заранее правилу: перебираем сиды подряд от --seed-base
и берём ПЕРВЫЙ, где обе панели решены. Сколько сидов просмотрено, пишется в меты.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time

import numpy as np

sys.path.insert(0, "/home/user8_2/AIRI_WAM/scripts/xland")
import shape_common as sc  # noqa: E402

import jax  # noqa: E402


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
    """Ровно то же ужатие, каким собирались обучающие кадры."""
    if rgb.shape[:2] == (target, target):
        return rgb
    s = rgb.astype(np.uint16)
    summed = s[0::2, 0::2] + s[0::2, 1::2] + s[1::2, 0::2] + s[1::2, 1::2]
    return ((summed + 2) // 4).astype(np.uint8)


def find_task(split, tree_id, palette_id, num_tasks):
    """Номер задачи с нужным деревом; палитра либо заданная, либо первая по порядку."""
    for i in range(num_tasks):
        meta = sc.task_meta(split, i)
        if meta["tree_id"] == tree_id and (palette_id is None or meta["palette_id"] == palette_id):
            return i, meta
    raise SystemExit(f"в сплите {split} нет дерева {tree_id} с палитрой {palette_id}")


def run_episode(client, env, base_params, bench, task, seed, reset_env, step_env):
    params = base_params.replace(ruleset=bench.get_ruleset(task))
    host_params = jax.device_get(params)
    ts = reset_env(params, jax.random.key(seed))

    frames, actions = [], []
    grid0 = np.asarray(jax.device_get(ts).observation["img"], dtype=np.int32)
    success, length = False, 0
    for step in range(int(base_params.max_steps)):
        host_ts = jax.device_get(ts)
        rgb = np.asarray(env.render(host_params, host_ts), dtype=np.uint8)
        frames.append(rgb)
        action = request_action(client, downsample_half(rgb))
        if not 0 <= action < 6:
            raise ValueError(f"недопустимое действие {action}")
        actions.append(action)
        ts = step_env(params, ts, action)
        host_next = jax.device_get(ts)
        length = step + 1
        if bool(host_next.last()):
            success = float(host_next.reward) >= 0.9
            break
    # последний кадр: состояние ПОСЛЕ финального действия, иначе победа не видна
    frames.append(np.asarray(env.render(host_params, jax.device_get(ts)), dtype=np.uint8))
    return {
        "frames": np.stack(frames),
        "actions": actions,
        "success": bool(success),
        "length": int(length),
        "grid0": grid0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", required=True)
    p.add_argument("--tree", type=int, default=1, help="tree_id, одинаковый в обоих сплитах")
    p.add_argument("--palette-train", type=int, default=None)
    p.add_argument("--palette-val", type=int, default=None)
    p.add_argument("--seed-base", type=int, default=41000000)
    p.add_argument("--candidates", type=int, default=12)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    env, base_params, _, _, benches = sc.make_env_and_policy(load_weights=False)
    num_tasks = int(benches["train"].num_rulesets())
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)

    task_tr, meta_tr = find_task("train", args.tree, args.palette_train, num_tasks)
    task_va, meta_va = find_task("val", args.tree, args.palette_val, num_tasks)
    print(f"знакомая палитра: {meta_tr['uid']} {meta_tr['colors']}", flush=True)
    print(f"отложенная палитра: {meta_va['uid']} {meta_va['colors']}", flush=True)
    if meta_tr["rules"] != meta_va["rules"]:
        raise SystemExit("правила у панелей разошлись, сравнивать нельзя")

    warm = base_params.replace(ruleset=benches["train"].get_ruleset(task_tr))
    jax.block_until_ready(step_env(warm, reset_env(warm, jax.random.key(0)), 0).reward)

    client = connect_with_retry(args.socket)
    chosen = None
    tried = 0
    try:
        for k in range(args.candidates):
            seed = args.seed_base + k
            tried = k + 1
            left = run_episode(client, env, base_params, benches["train"], task_tr, seed,
                               reset_env, step_env)
            right = run_episode(client, env, base_params, benches["val"], task_va, seed,
                                reset_env, step_env)
            ok = left["success"] and right["success"]
            print(f"сид {seed}: знакомая {left['success']} за {left['length']}, "
                  f"отложенная {right['success']} за {right['length']}"
                  f"{'  <- берём' if ok else ''}", flush=True)
            if ok:
                chosen = (seed, left, right)
                break
    finally:
        try:
            client.sendall(struct.pack("!I", 0))
        finally:
            client.close()

    if chosen is None:
        raise SystemExit(f"за {args.candidates} сидов не нашлось пары, где решены обе панели")

    seed, left, right = chosen

    # ворота: раскладка одна и та же, отличаются только цвета
    tiles_same = bool(np.array_equal(left["grid0"][:, :, 0], right["grid0"][:, :, 0]))
    agent_same = bool(np.array_equal(left["grid0"][:, :, 2], right["grid0"][:, :, 2]))
    colors_differ = bool(not np.array_equal(left["grid0"][:, :, 1], right["grid0"][:, :, 1]))
    print(f"ворота: плитки совпадают {tiles_same}, агент совпадает {agent_same}, "
          f"цвета различаются {colors_differ}", flush=True)
    if not (tiles_same and agent_same and colors_differ):
        raise SystemExit("ВОРОТА НЕ ПРОЙДЕНЫ: панели различаются не только цветом")

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "frames.npz"),
                        left=left["frames"], right=right["frames"])
    meta = {
        "seed": seed,
        "seeds_tried": tried,
        "candidates": args.candidates,
        "tree_id": args.tree,
        "rules": meta_tr["rules"],
        "left": {"uid": meta_tr["uid"], "colors": meta_tr["colors"],
                 "actions": left["actions"], "length": left["length"],
                 "success": left["success"], "frames": int(left["frames"].shape[0])},
        "right": {"uid": meta_va["uid"], "colors": meta_va["colors"],
                  "actions": right["actions"], "length": right["length"],
                  "success": right["success"], "frames": int(right["frames"].shape[0])},
        "gates": {"tiles_same": tiles_same, "agent_same": agent_same,
                  "colors_differ": colors_differ},
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"записал {args.out}: кадров {left['frames'].shape[0]} и {right['frames'].shape[0]}")
    print("RECORD_H1_STATUS=OK")


if __name__ == "__main__":
    main()
