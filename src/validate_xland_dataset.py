#!/usr/bin/env python3
"""Validate every XLand RGB trajectory before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    files = sorted(data_dir.glob("episode_train_*.npz"))
    if len(files) != args.expected_count:
        raise RuntimeError(f"Expected {args.expected_count} files, found {len(files)}")

    expected_ids = list(range(args.expected_count))
    ids = []
    total_transitions = 0
    lengths = []
    for index, path in enumerate(files, start=1):
        suffix = path.stem.removeprefix("episode_train_")
        ids.append(int(suffix))
        with np.load(path, allow_pickle=False) as episode:
            if set(episode.files) != {"rgb", "actions", "metadata"}:
                raise RuntimeError(f"{path}: unexpected keys {episode.files}")
            rgb = np.asarray(episode["rgb"])
            actions = np.asarray(episode["actions"])
            raw = episode["metadata"]
            metadata = json.loads(raw.item() if raw.ndim == 0 else str(raw))
        if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[1:] != (128, 128, 3):
            raise RuntimeError(f"{path}: invalid RGB shape/dtype {rgb.shape}/{rgb.dtype}")
        if actions.ndim != 1 or rgb.shape[0] != actions.shape[0] + 1:
            raise RuntimeError(f"{path}: broken frame/action alignment")
        if actions.size == 0 or actions.min() < 0 or actions.max() >= 6:
            raise RuntimeError(f"{path}: invalid action ids")
        if metadata.get("env_id") != "XLand-MiniGrid-ExactCraft-Easy-8x8-v1":
            raise RuntimeError(f"{path}: wrong environment")
        if metadata.get("success") is not True or float(metadata.get("terminal_reward", -1.0)) < 0.9:
            raise RuntimeError(f"{path}: not a verified successful trajectory")
        if int(metadata.get("length", -1)) != int(actions.shape[0]):
            raise RuntimeError(f"{path}: metadata length mismatch")
        total_transitions += int(actions.shape[0])
        lengths.append(int(actions.shape[0]))
        if index % 1000 == 0:
            print(json.dumps({"checked": index, "total": len(files)}), flush=True)

    if ids != expected_ids:
        raise RuntimeError("Trajectory IDs are not exactly 0..N-1")
    print(json.dumps({
        "event": "validated",
        "episodes": len(files),
        "success_rate": 1.0,
        "total_transitions": total_transitions,
        "mean_length": float(np.mean(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
