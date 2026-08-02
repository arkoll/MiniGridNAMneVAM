#!/usr/bin/env python3
"""Render the same OOD XLand level under greedy and sampled NanoWM actions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time

import imageio.v2 as imageio
import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ENV_ID = "XLand-MiniGrid-ExactCraft-Easy-8x8-v1"
ACTION_NAMES = ("forward", "turn right", "turn left", "pick up", "put down", "toggle")
RESPONSE_SIZE = struct.calcsize("!B6f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xland-repo", type=Path, required=True)
    parser.add_argument("--nanowm-python", type=Path, required=True)
    parser.add_argument("--trace-server", type=Path, required=True)
    parser.add_argument("--nanowm-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4000000)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=4035000)
    parser.add_argument("--action-gpu", default="3")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--fps", type=int, default=1)
    return parser.parse_args()


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        chunk = connection.recv(size)
        if not chunk:
            raise ConnectionError("Trace server disconnected")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def connect_with_retry(path: Path, timeout: float = 120.0) -> socket.socket:
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


def request_action(client: socket.socket, rgb: np.ndarray) -> tuple[int, np.ndarray]:
    payload = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    client.sendall(struct.pack("!I", len(payload)) + payload)
    action, *probabilities = struct.unpack("!B6f", receive_exact(client, RESPONSE_SIZE))
    return int(action), np.asarray(probabilities, dtype=np.float32)


def resize_half_area(rgb: np.ndarray, target_size: int) -> np.ndarray:
    if rgb.shape[:2] == (target_size, target_size):
        return rgb
    if rgb.shape[:2] != (2 * target_size, 2 * target_size):
        raise ValueError(f"Unexpected RGB shape {rgb.shape}")
    source = rgb.astype(np.uint16)
    summed = source[0::2, 0::2] + source[0::2, 1::2] + source[1::2, 0::2] + source[1::2, 1::2]
    return ((summed + 2) // 4).astype(np.uint8)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_panel(canvas: Image.Image, origin_x: int, title: str, rgb: np.ndarray, action: int | None,
               probabilities: np.ndarray | None, step: int, terminal: bool) -> None:
    draw = ImageDraw.Draw(canvas)
    title_font, text_font, small_font = font(27), font(18), font(15)
    draw.text((origin_x, 18), title, fill="white", font=title_font)
    state_text = "terminal" if terminal else f"step {step + 1}"
    if action is not None:
        state_text += f"   chosen: {ACTION_NAMES[action]}"
    draw.text((origin_x, 53), state_text, fill=(230, 230, 230), font=text_font)

    image = Image.fromarray(rgb).resize((360, 360), Image.Resampling.NEAREST)
    canvas.paste(image, (origin_x, 85))
    draw.rectangle((origin_x, 85, origin_x + 359, 444), outline=(210, 210, 210), width=2)

    if probabilities is None:
        draw.text((origin_x, 475), "No action: episode finished", fill=(160, 160, 160), font=text_font)
        return

    draw.text((origin_x, 455), "softmax action distribution", fill=(230, 230, 230), font=text_font)
    for index, (name, probability) in enumerate(zip(ACTION_NAMES, probabilities)):
        y = 485 + index * 31
        selected = index == action
        color = (255, 193, 7) if selected else (74, 144, 226)
        draw.text((origin_x, y), name, fill="white", font=small_font)
        left, width = origin_x + 115, int(210 * float(probability))
        draw.rectangle((left, y + 2, left + 210, y + 21), fill=(54, 54, 60))
        draw.rectangle((left, y + 2, left + max(width, 1), y + 21), fill=color)
        draw.text((left + 216, y), f"{probability:.3f}", fill=color if selected else (220, 220, 220), font=small_font)


def compose_frame(greedy: dict, sample: dict, step: int, seed: int, task_index: int) -> np.ndarray:
    canvas = Image.new("RGB", (960, 700), (25, 25, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 658), f"OOD ExactCraft Easy | same seed={seed}, task graph={task_index} | yellow = selected action", fill=(180, 180, 180), font=font(16))
    draw_panel(canvas, 50, "GREEDY (argmax)", step=step, **greedy)
    draw_panel(canvas, 550, "SAMPLING (softmax, T=1)", step=step, **sample)
    return np.asarray(canvas)


def start_server(args: argparse.Namespace, strategy: str, socket_path: Path, sampling_seed: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.action_gpu
    command = [
        str(args.nanowm_python), str(args.trace_server),
        "--nanowm-repo", str(args.nanowm_repo),
        "--config", str(args.config),
        "--checkpoint", str(args.checkpoint),
        "--socket", str(socket_path),
        "--strategy", strategy,
        "--sampling-seed", str(sampling_seed),
        "--device", "cuda",
    ]
    return subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    repo = args.xland_repo.expanduser().resolve()
    sys.path[:0] = [str(repo), str(repo / "training")]
    from train_privileged_crafting_ppo import make_env, task_banks

    env, base_params = make_env(ENV_ID, autoreset=False)
    _, ood_bank = task_banks(ENV_ID)
    if not 0 <= args.task_index < int(ood_bank.num_rulesets()):
        raise ValueError("Invalid OOD task index")
    params = base_params.replace(ruleset=ood_bank.get_ruleset(args.task_index))
    reset_env, step_env = jax.jit(env.reset), jax.jit(env.step)

    prefix = f"/tmp/xland_trace_{os.getpid()}"
    greedy_socket, sample_socket = Path(prefix + "_greedy.sock"), Path(prefix + "_sample.sock")
    for path in (greedy_socket, sample_socket):
        path.unlink(missing_ok=True)

    greedy_process = start_server(args, "greedy", greedy_socket, args.sampling_seed)
    sample_process = start_server(args, "sample", sample_socket, args.sampling_seed)
    clients = []
    try:
        greedy_client = connect_with_retry(greedy_socket)
        sample_client = connect_with_retry(sample_socket)
        clients = [greedy_client, sample_client]

        states = [reset_env(params, jax.random.key(args.seed)), reset_env(params, jax.random.key(args.seed))]
        active = [True, True]
        last_rgb = [None, None]
        writer = imageio.get_writer(args.output, fps=args.fps, codec="libx264", quality=8)
        try:
            for step in range(args.max_steps):
                panels = []
                for index, (client, strategy) in enumerate(zip(clients, ("greedy", "sample"))):
                    if active[index]:
                        timestep = jax.device_get(states[index])
                        rgb = resize_half_area(np.asarray(env.render(jax.device_get(params), timestep), dtype=np.uint8), args.image_size)
                        action, probabilities = request_action(client, rgb)
                        states[index] = step_env(params, states[index], action)
                        next_state = jax.device_get(states[index])
                        active[index] = not bool(next_state.last())
                        last_rgb[index] = rgb
                        panels.append({"rgb": rgb, "action": action, "probabilities": probabilities, "terminal": not active[index]})
                    else:
                        panels.append({"rgb": last_rgb[index], "action": None, "probabilities": None, "terminal": True})
                writer.append_data(compose_frame(panels[0], panels[1], step, args.seed, args.task_index))
                if not any(active):
                    break
        finally:
            writer.close()
        print(f"saved={args.output}", flush=True)
    finally:
        for client in clients:
            try:
                client.sendall(struct.pack("!I", 0))
                client.close()
            except OSError:
                pass
        for process in (greedy_process, sample_process):
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        for path in (greedy_socket, sample_socket):
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
