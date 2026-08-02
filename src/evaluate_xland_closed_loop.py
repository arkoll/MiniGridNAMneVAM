#!/usr/bin/env python3
"""Evaluate the joint action head as a closed-loop RGB-only XLand policy."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import socket
import struct
import sys
import time

import jax
import numpy as np
from tensorboardX import SummaryWriter


ENV_ID = "XLand-MiniGrid-ExactCraft-Easy-8x8-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xland-repo", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tb-dir", type=Path, required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=3000000)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--connect-timeout", type=float, default=120.0)
    return parser.parse_args()


def connect_with_retry(path: Path, timeout: float) -> socket.socket:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(str(path))
            return client
        except OSError as error:
            last_error = error
            client.close()
            time.sleep(0.25)
    raise TimeoutError(f"Could not connect to {path}: {last_error}")


def request_action(client: socket.socket, rgb: np.ndarray) -> int:
    payload = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    client.sendall(struct.pack("!I", len(payload)) + payload)
    response = client.recv(1)
    if len(response) != 1:
        raise ConnectionError("Action server returned an incomplete response")
    return int(struct.unpack("!B", response)[0])


def resize_half_area(rgb: np.ndarray, target_size: int) -> np.ndarray:
    """Exact 2x area downsample without adding OpenCV to the JAX env."""
    if rgb.shape[:2] == (target_size, target_size):
        return rgb
    if rgb.shape[:2] != (2 * target_size, 2 * target_size):
        raise ValueError(
            f"Expected {target_size} or {2 * target_size}px RGB, got {rgb.shape}"
        )
    source = rgb.astype(np.uint16)
    summed = (
        source[0::2, 0::2]
        + source[0::2, 1::2]
        + source[1::2, 0::2]
        + source[1::2, 1::2]
    )
    return ((summed + 2) // 4).astype(np.uint8)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tb_dir.mkdir(parents=True, exist_ok=True)
    repo = args.xland_repo.expanduser().resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "training"))
    from train_privileged_crafting_ppo import make_env, task_banks

    env, base_params = make_env(ENV_ID, autoreset=False)
    train_bank, _ = task_banks(ENV_ID)
    num_tasks = int(train_bank.num_rulesets())
    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)

    warm_params = base_params.replace(ruleset=train_bank.get_ruleset(0))
    warm = reset_env(warm_params, jax.random.key(args.seed_start))
    warm = step_env(warm_params, warm, 0)
    jax.block_until_ready(warm.reward)

    client = connect_with_retry(args.socket.expanduser().resolve(), args.connect_timeout)
    results = []
    task_successes = defaultdict(list)
    started = time.time()
    try:
        for episode_index in range(args.episodes):
            task_index = episode_index % num_tasks
            seed = args.seed_start + episode_index
            params = base_params.replace(
                ruleset=train_bank.get_ruleset(task_index)
            )
            host_params = jax.device_get(params)
            timestep = reset_env(params, jax.random.key(seed))
            success = False
            length = 0

            for step in range(int(base_params.max_steps)):
                host_timestep = jax.device_get(timestep)
                rgb = np.asarray(
                    env.render(host_params, host_timestep),
                    dtype=np.uint8,
                )
                rgb = resize_half_area(rgb, args.image_size)
                action = request_action(client, rgb)
                if not 0 <= action < 6:
                    raise ValueError(f"Invalid predicted action {action}")

                timestep = step_env(params, timestep, action)
                host_next = jax.device_get(timestep)
                terminal = bool(host_next.last())
                reward = float(host_next.reward)
                length = step + 1
                success = terminal and reward >= 0.9
                if terminal:
                    break

            result = {
                "episode": episode_index,
                "task_index": task_index,
                "seed": seed,
                "success": bool(success),
                "length": int(length),
            }
            results.append(result)
            task_successes[task_index].append(float(success))
            print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        try:
            client.sendall(struct.pack("!I", 0))
        finally:
            client.close()

    success_rate = float(np.mean([row["success"] for row in results]))
    mean_length = float(np.mean([row["length"] for row in results]))
    per_task = {
        str(task): float(np.mean(values))
        for task, values in sorted(task_successes.items())
    }
    summary = {
        "env_id": ENV_ID,
        "split": "train_tasks",
        "global_step": args.global_step,
        "episodes": args.episodes,
        "success_rate": success_rate,
        "mean_length": mean_length,
        "per_task_success_rate": per_task,
        "elapsed_seconds": time.time() - started,
        "input": "rgb_only",
        "image_size": args.image_size,
        "results": results,
    }
    output_path = (
        args.output_dir
        / f"closed_loop_step_{args.global_step:07d}.json"
    )
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    writer = SummaryWriter(str(args.tb_dir))
    writer.add_scalar(
        "closed_loop/success_rate",
        success_rate,
        args.global_step,
    )
    writer.add_scalar(
        "closed_loop/mean_episode_length",
        mean_length,
        args.global_step,
    )
    for task, value in per_task.items():
        writer.add_scalar(
            f"closed_loop/task_{task}_success_rate",
            value,
            args.global_step,
        )
    writer.close()
    print(json.dumps({"event": "summary", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
