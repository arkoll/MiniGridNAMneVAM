#!/usr/bin/env python3
"""Collect exactly N successful RGB PPO trajectories for XLand ExactCraft Easy."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import jax
import numpy as np
from flax import serialization

ENV_ID = "XLand-MiniGrid-ExactCraft-Easy-8x8-v1"
ACTION_NAMES = ("move_forward", "turn_clockwise", "turn_counterclockwise", "pick_up", "put_down", "toggle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xland-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-successes", type=int, required=True)
    parser.add_argument("--episode-offset", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def downsample_2x(rgb: np.ndarray, size: int) -> np.ndarray:
    if rgb.shape[:2] == (size, size):
        return rgb
    if rgb.shape[:2] != (2 * size, 2 * size):
        raise ValueError(f"Expected {size} or {2 * size}px render, got {rgb.shape}")
    source = rgb.astype(np.uint16)
    summed = source[0::2, 0::2] + source[0::2, 1::2] + source[1::2, 0::2] + source[1::2, 1::2]
    return ((summed + 2) // 4).astype(np.uint8)


def save_episode(path: Path, frames: list[np.ndarray], actions: list[int], metadata: dict) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        rgb=np.stack(frames).astype(np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.target_successes <= 0:
        raise ValueError("--target-successes must be positive")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sys.path[:0] = [str(args.xland_repo.expanduser().resolve()), str(args.xland_repo.expanduser().resolve() / "training")]
    from train_privileged_crafting_ppo import PrivilegedActorCritic, make_env, task_banks

    env, base_params = make_env(ENV_ID, autoreset=False)
    train_bank, _ = task_banks(ENV_ID)
    num_tasks = int(train_bank.num_rulesets())

    network = PrivilegedActorCritic(num_actions=6, agent_direction_classes=5)
    restored = serialization.msgpack_restore(args.checkpoint.expanduser().read_bytes())
    variables = restored["params"]

    @jax.jit
    def choose_action(observation):
        distribution, _ = network.apply(variables, observation)
        return distribution.mode()

    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)
    warm_params = base_params.replace(ruleset=train_bank.get_ruleset(0))
    warm = reset_env(warm_params, jax.random.key(args.seed_start))
    warm = step_env(warm_params, warm, choose_action(warm.observation))
    jax.block_until_ready(warm.reward)

    successes = attempts = transitions = 0
    started = time.time()
    while successes < args.target_successes:
        episode_id = args.episode_offset + successes
        seed = args.seed_start + attempts
        task_index = episode_id % num_tasks
        params = base_params.replace(ruleset=train_bank.get_ruleset(task_index))
        timestep = reset_env(params, jax.random.key(seed))
        host_params = jax.device_get(params)
        host = jax.device_get(timestep)
        frames = [downsample_2x(np.asarray(env.render(host_params, host), dtype=np.uint8), args.image_size)]
        actions: list[int] = []
        reward = 0.0
        terminal = False

        for _ in range(int(base_params.max_steps)):
            action = int(jax.device_get(choose_action(timestep.observation)))
            timestep = step_env(params, timestep, action)
            host = jax.device_get(timestep)
            actions.append(action)
            transitions += 1
            frames.append(downsample_2x(np.asarray(env.render(host_params, host), dtype=np.uint8), args.image_size))
            terminal = bool(host.last())
            reward = float(host.reward)
            if terminal:
                break

        attempts += 1
        if terminal and reward >= 0.9:
            output = args.output_dir / f"episode_train_{episode_id:05d}.npz"
            if output.exists():
                raise FileExistsError(f"Refusing to overwrite {output}")
            metadata = {
                "schema_version": 2,
                "env_id": ENV_ID,
                "split": "train",
                "task_index": task_index,
                "seed": seed,
                "collector": "privileged_ppo_success_only",
                "policy_checkpoint": str(args.checkpoint),
                "policy_stochastic": False,
                "success": True,
                "length": len(actions),
                "terminal_reward": reward,
                "image_size": args.image_size,
                "action_mapping": dict(enumerate(ACTION_NAMES)),
                "frame_action_alignment": "rgb[t] maps to rgb[t+1] via actions[t]",
                "attempt": attempts,
            }
            save_episode(output, frames, actions, metadata)
            successes += 1

        if successes == 1 or successes % args.log_every == 0:
            print(json.dumps({
                "successes": successes,
                "target_successes": args.target_successes,
                "attempts": attempts,
                "acceptance_rate": successes / attempts,
                "transitions": transitions,
                "elapsed_seconds": time.time() - started,
            }, sort_keys=True), flush=True)

    print(json.dumps({
        "event": "finished",
        "successes": successes,
        "attempts": attempts,
        "success_rate": successes / attempts,
        "transitions": transitions,
        "output_dir": str(args.output_dir),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
