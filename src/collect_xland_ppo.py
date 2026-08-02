#!/usr/bin/env python3
"""Collect atomic ExactCraft episodes from the best privileged PPO policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import serialization
from tensorboardX import SummaryWriter


ENV_ID = "XLand-MiniGrid-ExactCraft-Easy-8x8-v1"
ACTION_NAMES = (
    "move_forward",
    "turn_clockwise",
    "turn_counterclockwise",
    "pick_up",
    "put_down",
    "toggle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xland-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tb-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def atomic_save_episode(
    path: Path,
    *,
    frames: list[np.ndarray],
    symbolic: list[np.ndarray],
    pockets: list[np.ndarray],
    actions: list[int],
    rewards: list[float],
    dones: list[bool],
    rule_encoding: np.ndarray,
    goal_encoding: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    tmp_path = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp_path,
        rgb=np.stack(frames).astype(np.uint8),
        symbolic=np.stack(symbolic).astype(np.uint8),
        pocket=np.stack(pockets).astype(np.uint8),
        actions=np.asarray(actions, dtype=np.int64),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        rule_encoding=np.asarray(rule_encoding, dtype=np.int16),
        goal_encoding=np.asarray(goal_encoding, dtype=np.int16),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(tmp_path, path)


def main() -> None:
    args = parse_args()
    args.xland_repo = args.xland_repo.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.tb_dir = args.tb_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tb_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.xland_repo))
    sys.path.insert(0, str(args.xland_repo / "training"))
    from train_privileged_crafting_ppo import (  # noqa: PLC0415
        PrivilegedActorCritic,
        make_env,
        task_banks,
    )

    env, base_params = make_env(ENV_ID, autoreset=False)
    train_bank, val_bank = task_banks(ENV_ID)
    bank = train_bank if args.split == "train" else val_bank
    num_tasks = int(bank.num_rulesets())

    network = PrivilegedActorCritic(
        num_actions=6,
        agent_direction_classes=5,
    )
    restored = serialization.msgpack_restore(args.checkpoint.read_bytes())
    variables = restored["params"]

    @jax.jit
    def deterministic_action(observation):
        distribution, _ = network.apply(variables, observation)
        return distribution.mode()

    @jax.jit
    def stochastic_action(observation, key):
        distribution, _ = network.apply(variables, observation)
        return distribution.sample(seed=key)

    reset_env = jax.jit(env.reset)
    step_env = jax.jit(env.step)

    # Compile once before measuring collection throughput.
    warm_params = base_params.replace(ruleset=bank.get_ruleset(0))
    warm_timestep = reset_env(warm_params, jax.random.key(args.seed_start))
    if args.stochastic:
        warm_action = stochastic_action(
            warm_timestep.observation,
            jax.random.key(args.seed_start + 1),
        )
    else:
        warm_action = deterministic_action(warm_timestep.observation)
    warm_timestep = step_env(warm_params, warm_timestep, warm_action)
    jax.block_until_ready(warm_timestep.reward)

    writer = SummaryWriter(str(args.tb_dir))
    rolling = deque(maxlen=100)
    action_counts: Counter[int] = Counter()
    task_counts: Counter[int] = Counter()
    successes = 0
    transitions = 0
    started = time.time()

    for local_index in range(args.episodes):
        episode_index = args.episode_offset + local_index
        seed = args.seed_start + local_index
        task_index = episode_index % num_tasks
        task_counts[task_index] += 1
        params = base_params.replace(ruleset=bank.get_ruleset(task_index))
        timestep = reset_env(params, jax.random.key(seed))
        host_params = jax.device_get(params)
        host_timestep = jax.device_get(timestep)

        rule_encoding = np.asarray(host_timestep.observation["rule_encoding"])
        goal_encoding = np.asarray(host_timestep.observation["goal_encoding"])
        frames = [
            np.asarray(env.render(host_params, host_timestep), dtype=np.uint8)
        ]
        symbolic = [
            np.asarray(host_timestep.observation["img"], dtype=np.uint8)
        ]
        pockets = [
            np.asarray(host_timestep.observation["pocket"], dtype=np.uint8)
        ]
        actions: list[int] = []
        rewards: list[float] = []
        dones: list[bool] = []
        episode_started = time.time()
        success = False

        for step in range(int(base_params.max_steps)):
            if args.stochastic:
                action_key = jax.random.key(seed * 257 + step + 1)
                action = int(
                    jax.device_get(
                        stochastic_action(timestep.observation, action_key)
                    )
                )
            else:
                action = int(
                    jax.device_get(deterministic_action(timestep.observation))
                )

            next_timestep = step_env(params, timestep, action)
            host_timestep = jax.device_get(next_timestep)
            terminal = bool(host_timestep.last())
            reward = float(host_timestep.reward)
            success = success or (terminal and reward >= 0.9)

            actions.append(action)
            rewards.append(reward)
            dones.append(terminal)
            action_counts[action] += 1
            transitions += 1

            timestep = next_timestep
            frames.append(
                np.asarray(
                    env.render(host_params, host_timestep),
                    dtype=np.uint8,
                )
            )
            symbolic.append(
                np.asarray(host_timestep.observation["img"], dtype=np.uint8)
            )
            pockets.append(
                np.asarray(host_timestep.observation["pocket"], dtype=np.uint8)
            )
            if terminal:
                break

        output_path = args.output_dir / f"episode_{args.split}_{seed:08d}.npz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} exists; use --overwrite or a new seed range"
            )

        metadata = {
            "schema_version": 1,
            "env_id": ENV_ID,
            "split": args.split,
            "task_index": task_index,
            "seed": seed,
            "collector": "privileged_ppo",
            "policy_checkpoint": str(args.checkpoint),
            "policy_stochastic": bool(args.stochastic),
            "success": bool(success),
            "length": len(actions),
            "image_size": 256,
            "action_mapping": dict(enumerate(ACTION_NAMES)),
            "frame_action_alignment": "actions[t] maps rgb[t] to rgb[t+1]",
            "rule_encoding": rule_encoding.tolist(),
            "goal_encoding": goal_encoding.tolist(),
        }
        atomic_save_episode(
            output_path,
            frames=frames,
            symbolic=symbolic,
            pockets=pockets,
            actions=actions,
            rewards=rewards,
            dones=dones,
            rule_encoding=rule_encoding,
            goal_encoding=goal_encoding,
            metadata=metadata,
        )

        successes += int(success)
        rolling.append(float(success))
        elapsed = time.time() - started
        writer.add_scalar("episode/success", float(success), episode_index)
        writer.add_scalar("episode/length", len(actions), episode_index)
        writer.add_scalar(
            "episode/wall_seconds",
            time.time() - episode_started,
            episode_index,
        )
        writer.add_scalar(
            "rolling_100/success_rate",
            float(np.mean(rolling)),
            episode_index,
        )
        writer.add_scalar(
            "collection/episodes_per_second",
            (local_index + 1) / max(elapsed, 1e-6),
            episode_index,
        )
        writer.add_scalar(
            "collection/transitions_per_second",
            transitions / max(elapsed, 1e-6),
            episode_index,
        )

        if (local_index + 1) % args.log_every == 0 or local_index == 0:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "split": args.split,
                        "completed": local_index + 1,
                        "requested": args.episodes,
                        "success_rate": successes / (local_index + 1),
                        "rolling_100_success_rate": float(np.mean(rolling)),
                        "transitions": transitions,
                        "episodes_per_second": (local_index + 1)
                        / max(elapsed, 1e-6),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    elapsed = time.time() - started
    summary = {
        "env_id": ENV_ID,
        "split": args.split,
        "episodes": args.episodes,
        "episode_offset": args.episode_offset,
        "seed_start": args.seed_start,
        "policy_checkpoint": str(args.checkpoint),
        "policy_stochastic": bool(args.stochastic),
        "successes": successes,
        "success_rate": successes / max(args.episodes, 1),
        "transitions": transitions,
        "elapsed_seconds": elapsed,
        "episodes_per_second": args.episodes / max(elapsed, 1e-6),
        "action_counts": dict(sorted(action_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
    }
    summary_path = (
        args.output_dir
        / f"summary_{args.split}_{args.seed_start:08d}_{args.episodes}.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    writer.close()
    print(json.dumps({"event": "finished", **summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
