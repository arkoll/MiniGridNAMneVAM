#!/usr/bin/env python3
"""Serve a NanoWM policy action and its probability distribution for tracing."""

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
    parser.add_argument("--strategy", choices=("greedy", "sample"), required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sampling-seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    while size:
        chunk = connection.recv(size)
        if not chunk:
            raise ConnectionError("Client disconnected mid-message")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def main() -> None:
    args = parse_args()
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    repo = args.nanowm_repo.expanduser().resolve()
    sys.path.insert(0, str(repo / "src"))
    from experiments.train_experiment import NanoWMTrainingModule

    config = OmegaConf.load(args.config.expanduser().resolve())
    module = NanoWMTrainingModule(config)
    state_dict = torch.load(args.checkpoint.expanduser().resolve(), map_location="cpu").get("state_dict")
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    critical = [key for key in [*missing, *unexpected] if key.startswith(("model.", "action_head."))]
    if critical:
        raise RuntimeError(f"Checkpoint mismatch: {critical}")

    device = torch.device(args.device)
    module.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(args.sampling_seed)
    image_size = int(config.model.image_size)
    expected_bytes = image_size * image_size * 3

    socket_path = args.socket.expanduser().resolve()
    if socket_path.exists():
        socket_path.unlink()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    print(f"READY strategy={args.strategy} socket={socket_path}", flush=True)

    try:
        connection, _ = server.accept()
        with connection:
            while True:
                payload_size = struct.unpack("!I", receive_exact(connection, 4))[0]
                if payload_size == 0:
                    break
                if payload_size != expected_bytes:
                    raise ValueError(f"Expected {expected_bytes} RGB bytes, got {payload_size}")
                rgb = np.frombuffer(receive_exact(connection, payload_size), dtype=np.uint8).reshape(image_size, image_size, 3)
                video = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32) / 127.5 - 1.0
                with torch.inference_mode():
                    logits = module.predict_action_logits(video)[0, 0]
                    probabilities = torch.softmax(logits / args.temperature, dim=-1)
                    if args.strategy == "greedy":
                        action = int(logits.argmax().item())
                    else:
                        action = int(torch.multinomial(probabilities, 1, generator=generator).item())
                response = struct.pack("!B6f", action, *probabilities.detach().float().cpu().tolist())
                connection.sendall(response)
    finally:
        server.close()
        if socket_path.exists():
            os.unlink(socket_path)


if __name__ == "__main__":
    main()
