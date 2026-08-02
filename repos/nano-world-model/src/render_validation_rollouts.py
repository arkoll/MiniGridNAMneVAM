#!/usr/bin/env python3
"""Run real-environment Baba validation rollouts and record annotated MP4s."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch.utils.tensorboard import SummaryWriter

import baba
from diffusion.df_sample import dfot_sample
from experiments.train_experiment import NanoWMTrainingModule


ACTION_NAMES = ("idle", "up", "right", "down", "left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tb-dir", type=Path, required=True)
    parser.add_argument("--env-id", default="env/make_win")
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Logical per-episode timeout in real environment steps.",
    )
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--hold-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(config_dir: Path):
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(
            config_name="config",
            overrides=[
                "model=nanowm_baba_s2",
                "dataset=baba/make_win_random",
                "experiment=baba_joint_smoke",
                "logger.name=tensorboard",
                "wandb.enabled=false",
            ],
        )


def load_model(cfg, checkpoint_path: Path, device: torch.device):
    module = NanoWMTrainingModule(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = module.load_state_dict(
        checkpoint["state_dict"], strict=False
    )
    relevant_missing = [
        key
        for key in missing
        if key.startswith("model.") or key.startswith("action_head.")
    ]
    if relevant_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={relevant_missing}, "
            f"unexpected={unexpected}"
        )
    module.to(device)
    module.eval()
    return module, int(checkpoint.get("global_step", -1))


def make_env(env_id: str, seed: int, max_steps: int):
    # Baba samples layouts through NumPy's global RNG.
    np.random.seed(seed)
    env = baba.make(env_id, max_steps=max_steps)
    env.reset(seed=seed)
    return env


def render_square(env, image_size: int) -> np.ndarray:
    # Baba's rule-word glyphs require the native 32 px tile canvas.
    frame = np.asarray(
        env.render(mode="rgb_array", tile_size=32), dtype=np.uint8
    )
    height, width = frame.shape[:2]
    scale = min(image_size / height, image_size / width)
    resized = cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    top = (image_size - resized.shape[0]) // 2
    left = (image_size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def rgb_batch_to_model(frames: list[np.ndarray], device: torch.device):
    array = np.stack(frames)
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(0, 3, 1, 2).contiguous() / 127.5 - 1.0
    return tensor


def model_pixels_to_rgb(pixels: torch.Tensor) -> np.ndarray:
    pixels = ((pixels.float().clamp(-1, 1) + 1.0) * 127.5).round()
    pixels = pixels.to(torch.uint8).cpu().permute(0, 2, 3, 1)
    return pixels.numpy()


@torch.inference_mode()
def predict_action_and_next_frame(
    module: NanoWMTrainingModule,
    context_frames: list[np.ndarray],
    *,
    sampling_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    pixels = rgb_batch_to_model(context_frames, device)
    context_latent = module._vae_encode(pixels)
    latent_shape = context_latent[:, None].repeat(1, 4, 1, 1, 1)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        predicted_latents = dfot_sample(
            diffusion=module.diffusion,
            model=module.model,
            shape=latent_shape.shape,
            context=latent_shape[:, :1],
            n_context_frames=1,
            scheduling_mode="sequential",
            num_sampling_steps=sampling_steps,
            model_kwargs={"y": None},
            device=device,
            progress=False,
            history_stabilization_level=(
                module.args.experiment.diffusion.history_stabilization_level
            ),
        )
        feature_model = getattr(module.model, "_orig_mod", module.model)
        logits = module.action_head(feature_model.last_hidden_features)
        action_plan = logits.argmax(dim=-1)

    # The environment evaluation is receding-horizon: execute only the first
    # selected action, then observe the real environment and predict again.
    chosen_actions = action_plan[:, 0].cpu().numpy()
    predicted_next_latent = predicted_latents[:, 1].float()
    predicted_next_pixels = module._vae_decode(predicted_next_latent)
    predicted_next_rgb = model_pixels_to_rgb(predicted_next_pixels)
    return chosen_actions, predicted_next_rgb


def comparison_frame(
    actual_rgb: np.ndarray,
    imagined_rgb: np.ndarray | None,
    *,
    rollout_index: int,
    seed: int,
    env_id: str,
    step: int,
    max_steps: int,
    action: int | None,
    reward: float,
    success: bool,
    done: bool,
    timeout: bool,
    env_done: bool,
) -> np.ndarray:
    canvas = np.full((640, 960, 3), 22, dtype=np.uint8)
    panel_size = 360
    image_top = 115
    left_x = 60
    right_x = 540

    actual_panel = cv2.resize(
        cv2.cvtColor(actual_rgb, cv2.COLOR_RGB2BGR),
        (panel_size, panel_size),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas[
        image_top : image_top + panel_size,
        left_x : left_x + panel_size,
    ] = actual_panel
    if imagined_rgb is None:
        predicted_panel = np.full_like(actual_panel, 12)
        cv2.putText(
            predicted_panel,
            "NO PREDICTION YET",
            (42, panel_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
    else:
        predicted_panel = cv2.resize(
            cv2.cvtColor(imagined_rgb, cv2.COLOR_RGB2BGR),
            (panel_size, panel_size),
            interpolation=cv2.INTER_NEAREST,
        )
    canvas[
        image_top : image_top + panel_size,
        right_x : right_x + panel_size,
    ] = predicted_panel

    cv2.putText(
        canvas,
        "Baba Is AI - real-environment validation",
        (32, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"rollout={rollout_index:03d}   seed={seed}   env={env_id}",
        (32, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "REAL NEXT OBSERVATION",
        (left_x, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (90, 235, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "MODEL-PREDICTED NEXT OBSERVATION",
        (right_x, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 190, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        (left_x - 2, image_top - 2),
        (left_x + panel_size + 1, image_top + panel_size + 1),
        (90, 235, 120),
        2,
    )
    cv2.rectangle(
        canvas,
        (right_x - 2, image_top - 2),
        (right_x + panel_size + 1, image_top + panel_size + 1),
        (120, 190, 255),
        2,
    )

    action_name = "none" if action is None else ACTION_NAMES[action]
    if success:
        result = "SUCCESS - ENVIRONMENT REPORTED WIN"
        status_color = (60, 220, 60)
    elif timeout:
        result = "FAIL - LOGICAL TIMEOUT REACHED"
        status_color = (80, 80, 255)
    elif env_done:
        result = "FAIL - ENVIRONMENT TERMINATED WITHOUT WIN"
        status_color = (80, 80, 255)
    else:
        result = "RUNNING"
        status_color = (230, 230, 230)

    cv2.putText(
        canvas,
        f"Step: {step:03d}/{max_steps:03d}    "
        f"Executed action: {action_name.upper()}    Reward: {reward:.2f}",
        (32, 512),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Environment done: {int(env_done)}    Timeout: {int(timeout)}    "
        f"Task completed: {int(success)}    Rollout done: {int(done)}",
        (32, 550),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Result: {result}",
        (32, 590),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Left: observed after action | Right: prediction made before action",
        (32, 622),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (165, 165, 165),
        1,
        cv2.LINE_AA,
    )
    return canvas


def write_mp4(
    path: Path, frames: list[np.ndarray], fps: int, hold_frames: int
) -> None:
    height, width = frames[0].shape[:2]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        process = subprocess.Popen(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s:v",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        for frame in frames:
            for _ in range(hold_frames):
                process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr else b""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(
                f"ffmpeg failed for {path}: "
                f"{stderr.decode('utf-8', errors='replace')}"
            )
        return

    # Portable fallback for machines without ffmpeg. Re-encode these files to
    # H.264/yuv420p before viewing them in QuickTime.
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for {path}")
    try:
        for frame in frames:
            for _ in range(hold_frames):
                writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NanoWM rollout sampling")
    if args.num_rollouts <= 0 or args.max_steps <= 0:
        raise ValueError("--num-rollouts and --max-steps must be positive")

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tb_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    repo_src = Path(__file__).resolve().parent
    cfg = load_config(repo_src / "configs")
    module, checkpoint_step = load_model(cfg, args.checkpoint, device)

    rollouts = []
    for rollout_index in range(args.num_rollouts):
        seed = args.seed_start + rollout_index
        env = make_env(args.env_id, seed, args.max_steps)
        initial = render_square(env, args.image_size)
        rollouts.append(
            {
                "index": rollout_index,
                "seed": seed,
                "env": env,
                "last_rgb": initial,
                "frames": [
                    comparison_frame(
                        initial,
                        None,
                        rollout_index=rollout_index,
                        seed=seed,
                        env_id=args.env_id,
                        step=0,
                        max_steps=args.max_steps,
                        action=None,
                        reward=0.0,
                        success=False,
                        done=False,
                        timeout=False,
                        env_done=False,
                    )
                ],
                "actions": [],
                "rewards": [],
                "success": False,
                "done": False,
                "wall_seconds": 0.0,
            }
        )

    evaluation_started = time.perf_counter()
    for step in range(1, args.max_steps + 1):
        active = [rollout for rollout in rollouts if not rollout["done"]]
        if not active:
            break

        inference_started = time.perf_counter()
        chosen_actions, imagined_next_frames = predict_action_and_next_frame(
            module,
            [rollout["last_rgb"] for rollout in active],
            sampling_steps=args.sampling_steps,
            device=device,
        )
        inference_seconds = time.perf_counter() - inference_started

        for active_index, rollout in enumerate(active):
            action = int(chosen_actions[active_index])
            _, reward, env_done, _ = rollout["env"].step(action)
            actual_next = render_square(rollout["env"], args.image_size)
            task_success = bool(
                reward > 0
                or getattr(rollout["env"].unwrapped, "is_win", False)
            )
            timeout = step >= args.max_steps
            done = bool(env_done or task_success or timeout)

            rollout["actions"].append(action)
            rollout["rewards"].append(float(reward))
            rollout["success"] = task_success
            rollout["done"] = done
            rollout["last_rgb"] = actual_next
            rollout["wall_seconds"] += inference_seconds / len(active)
            rollout["frames"].append(
                comparison_frame(
                    actual_next,
                    imagined_next_frames[active_index],
                    rollout_index=rollout["index"],
                    seed=rollout["seed"],
                    env_id=args.env_id,
                    step=step,
                    max_steps=args.max_steps,
                    action=action,
                    reward=float(reward),
                    success=task_success,
                    done=done,
                    timeout=timeout,
                    env_done=bool(env_done),
                )
            )

        print(
            json.dumps(
                {
                    "step": step,
                    "active_before_step": len(active),
                    "finished": sum(rollout["done"] for rollout in rollouts),
                    "successes": sum(
                        rollout["success"] for rollout in rollouts
                    ),
                    "inference_seconds": inference_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    writer = SummaryWriter(str(args.tb_dir))
    records = []
    for rollout in rollouts:
        rollout["env"].close()
        filename = (
            f"rollout_{rollout['index']:03d}_seed_{rollout['seed']}"
            f"_steps_{len(rollout['actions']):03d}"
            f"_success_{int(rollout['success'])}.mp4"
        )
        video_path = args.output_dir / filename
        write_mp4(
            video_path,
            rollout["frames"],
            fps=args.fps,
            hold_frames=args.hold_frames,
        )
        record = {
            "rollout_index": rollout["index"],
            "seed": rollout["seed"],
            "env_id": args.env_id,
            "max_steps": args.max_steps,
            "steps_executed": len(rollout["actions"]),
            "success": rollout["success"],
            "success_definition": (
                "real_environment_task_completed_before_timeout"
            ),
            "episode_return": float(sum(rollout["rewards"])),
            "actions": rollout["actions"],
            "action_names": [
                ACTION_NAMES[action] for action in rollout["actions"]
            ],
            "rewards": rollout["rewards"],
            "model_inference_seconds": rollout["wall_seconds"],
            "video_path": str(video_path),
        }
        records.append(record)
        writer.add_scalar(
            "environment_rollout/success",
            float(record["success"]),
            rollout["index"],
        )
        writer.add_scalar(
            "environment_rollout/steps",
            record["steps_executed"],
            rollout["index"],
        )
        writer.add_scalar(
            "environment_rollout/return",
            record["episode_return"],
            rollout["index"],
        )
        writer.add_scalar(
            "environment_rollout/model_inference_seconds",
            record["model_inference_seconds"],
            rollout["index"],
        )
        if rollout["index"] < 4:
            video_tensor = torch.from_numpy(
                np.stack(rollout["frames"])[:, :, :, ::-1].copy()
            )
            video_tensor = (
                video_tensor.permute(0, 3, 1, 2).float() / 255.0
            )
            writer.add_video(
                f"environment_examples/rollout_{rollout['index']:03d}",
                video_tensor.unsqueeze(0),
                rollout["index"],
                fps=args.fps,
            )
        print(json.dumps(record, sort_keys=True), flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "env_id": args.env_id,
        "num_rollouts": len(records),
        "logical_timeout_steps": args.max_steps,
        "sampling_steps": args.sampling_steps,
        "success_definition": (
            "real_environment_task_completed_before_timeout"
        ),
        "successes": int(sum(record["success"] for record in records)),
        "success_rate": float(np.mean([record["success"] for record in records])),
        "mean_steps": float(
            np.mean([record["steps_executed"] for record in records])
        ),
        "mean_return": float(
            np.mean([record["episode_return"] for record in records])
        ),
        "wall_seconds": time.perf_counter() - evaluation_started,
        "output_dir": str(args.output_dir),
        "tb_dir": str(args.tb_dir),
    }
    (args.output_dir / "rollouts.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    writer.add_hparams(
        {
            "checkpoint_step": checkpoint_step,
            "logical_timeout_steps": args.max_steps,
            "sampling_steps": args.sampling_steps,
            "num_rollouts": len(records),
        },
        {
            "hparam/real_environment_success_rate": summary["success_rate"],
            "hparam/mean_steps": summary["mean_steps"],
            "hparam/mean_return": summary["mean_return"],
        },
    )
    writer.flush()
    writer.close()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
