#!/usr/bin/env python3
"""Serve a trained joint model's first discrete action over a Unix socket."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import struct
import sys

import numpy as np
from omegaconf import OmegaConf
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nanowm-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    # Разбиение предельного цикла. Жадная политика без памяти, попав в состояние, где
    # её догадка о правиле неверна, возвращается в него же и стоит до лимита шагов
    # (замерено: 664 провала из 664 — ровно на лимите). Два независимых способа выйти:
    #   --temperature  сэмплировать по собственной неуверенности модели (0 = argmax);
    #   --epsilon      доля принудительно равномерных действий, работает и при уверенной голове.
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("Client disconnected mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def main() -> None:
    args = parse_args()
    repo = args.nanowm_repo.expanduser().resolve()
    sys.path.insert(0, str(repo / "src"))
    from experiments.train_experiment import NanoWMTrainingModule

    config = OmegaConf.load(args.config.expanduser().resolve())
    module = NanoWMTrainingModule(config)
    checkpoint = torch.load(
        args.checkpoint.expanduser().resolve(),
        map_location="cpu",
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    critical_missing = [
        key
        for key in missing
        if key.startswith("model.") or key.startswith("action_head.")
    ]
    critical_unexpected = [
        key
        for key in unexpected
        if key.startswith("model.") or key.startswith("action_head.")
    ]
    if critical_missing or critical_unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={critical_missing}, "
            f"unexpected={critical_unexpected}"
        )

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    module.to(device)
    module.eval()
    image_size = int(config.model.image_size)
    expected_bytes = image_size * image_size * 3

    socket_path = args.socket.expanduser().resolve()
    if socket_path.exists():
        socket_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    mode = (
        f"epsilon={args.epsilon}" if args.epsilon > 0.0
        else (f"temperature={args.temperature}" if args.temperature > 0.0 else "argmax")
    )
    print(
        f"READY socket={socket_path} checkpoint={args.checkpoint} "
        f"image_size={image_size} policy={mode} seed={args.seed}",
        flush=True,
    )

    try:
        connection, _ = server.accept()
        with connection:
            while True:
                header = receive_exact(connection, 4)
                payload_size = struct.unpack("!I", header)[0]
                if payload_size == 0:
                    break
                if payload_size != expected_bytes:
                    raise ValueError(
                        f"Expected {expected_bytes} RGB bytes, got {payload_size}"
                    )
                payload = receive_exact(connection, payload_size)
                rgb = np.frombuffer(payload, dtype=np.uint8).reshape(
                    image_size,
                    image_size,
                    3,
                )
                video = (
                    torch.from_numpy(rgb.copy())
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device=device, dtype=torch.float32)
                    / 127.5
                    - 1.0
                )
                with torch.inference_mode():
                    logits = module.predict_action_logits(video)[0, 0]
                    if args.epsilon > 0.0 and rng.random() < args.epsilon:
                        action = int(rng.integers(int(logits.shape[-1])))
                    elif args.temperature > 0.0:
                        probs = torch.softmax(logits.float() / args.temperature, dim=-1)
                        action = int(torch.multinomial(probs, 1).item())
                    else:
                        action = int(logits.argmax().item())
                connection.sendall(struct.pack("!B", action))
    finally:
        server.close()
        if socket_path.exists():
            os.unlink(socket_path)


if __name__ == "__main__":
    main()
