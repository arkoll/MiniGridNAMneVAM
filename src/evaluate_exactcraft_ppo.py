#!/usr/bin/env python3
"""Evaluate the privileged ExactCraft PPO policy on train and OOD task banks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from flax import serialization
import imageio.v3 as iio
import jax
import numpy as np
from torch.utils.tensorboard import SummaryWriter

ENV_ID = "XLand-MiniGrid-ExactCraft-Easy-8x8-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xland-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tb-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--global-step", type=int, required=True)
    return parser.parse_args()


def run_split(env, base_params, bank, action_fn, episodes, seed_start):
    reset_env, step_env = jax.jit(env.reset), jax.jit(env.step)
    num_tasks = int(bank.num_rulesets())
    successes, lengths, first_success_video = [], [], None

    for episode in range(episodes):
        task_index = episode % num_tasks
        params = base_params.replace(ruleset=bank.get_ruleset(task_index))
        timestep = reset_env(params, jax.random.key(seed_start + episode))
        host_params = jax.device_get(params)
        frames = [np.asarray(env.render(host_params, jax.device_get(timestep)), dtype=np.uint8)]
        success = False

        for step in range(int(base_params.max_steps)):
            action = int(jax.device_get(action_fn(timestep.observation)))
            timestep = step_env(params, timestep, action)
            host_timestep = jax.device_get(timestep)
            frames.append(np.asarray(env.render(host_params, host_timestep), dtype=np.uint8))
            if bool(host_timestep.last()):
                success = float(host_timestep.reward) >= 0.9
                break

        successes.append(float(success))
        lengths.append(len(frames) - 1)
        if success and first_success_video is None:
            first_success_video = np.stack(frames)

    return {
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "episodes": episodes,
        "per_episode_success": successes,
        "per_episode_length": lengths,
        "video": first_success_video,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tb_dir.mkdir(parents=True, exist_ok=True)
    repo = args.xland_repo.resolve()
    sys.path[:0] = [str(repo), str(repo / "training")]
    if "tensorboardX" not in sys.modules:
        import types
        sys.modules["tensorboardX"] = types.SimpleNamespace(SummaryWriter=SummaryWriter)
    from train_privileged_crafting_ppo import PrivilegedActorCritic, make_env, task_banks

    env, base_params = make_env(ENV_ID, autoreset=False)
    train_bank, val_bank = task_banks(ENV_ID)
    network = PrivilegedActorCritic(num_actions=6, agent_direction_classes=5)
    restored = serialization.msgpack_restore(args.checkpoint.read_bytes())
    variables = restored["params"]

    @jax.jit
    def action_fn(observation):
        distribution, _ = network.apply(variables, observation)
        return distribution.mode()

    # Compile once before timing/evaluating episodes.
    warm_params = base_params.replace(ruleset=train_bank.get_ruleset(0))
    warm = jax.jit(env.reset)(warm_params, jax.random.key(0))
    jax.block_until_ready(action_fn(warm.observation))

    results = {
        "train": run_split(env, base_params, train_bank, action_fn, args.episodes, 5_000_000),
        "val_ood": run_split(env, base_params, val_bank, action_fn, args.episodes, 6_000_000),
    }

    writer = SummaryWriter(str(args.tb_dir))
    for split, result in results.items():
        writer.add_scalar(f"ppo/{split}/success_rate", result["success_rate"], args.global_step)
        writer.add_scalar(f"ppo/{split}/mean_episode_length", result["mean_length"], args.global_step)
        video = result.pop("video")
        if video is not None:
            video_path = args.output_dir / f"ppo_{split}_successful.mp4"
            iio.imwrite(video_path, video, fps=4)
            # MP4 is saved above. TensorBoard video encoding is disabled because
            # this environment uses NumPy 2, incompatible with its legacy video helper.

    writer.close()
    (args.output_dir / "ppo_eval_summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True)
    )
    print(json.dumps({split: {key: value for key, value in result.items() if not key.startswith("per_episode")} for split, result in results.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
