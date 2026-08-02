#!/usr/bin/env python3
"""Create an atomic 128x128 copy of the collected XLand RGB trajectories."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def resize_episode(job: tuple[str, str, int]) -> dict:
    source_raw, output_raw, size = job
    source = Path(source_raw)
    output = Path(output_raw)
    cv2.setNumThreads(1)

    with np.load(source, allow_pickle=False) as episode:
        payload = {key: np.asarray(episode[key]) for key in episode.files}

    rgb = np.asarray(payload["rgb"], dtype=np.uint8)
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3], got {rgb.shape} in {source}")
    if rgb.shape[1:3] != (256, 256):
        raise ValueError(f"Expected 256x256 source frames, got {rgb.shape} in {source}")

    payload["rgb"] = np.stack(
        [
            cv2.resize(
                frame,
                (size, size),
                interpolation=cv2.INTER_AREA,
            )
            for frame in rgb
        ]
    ).astype(np.uint8)

    if "metadata" in payload:
        raw = payload["metadata"]
        raw = raw.item() if raw.ndim == 0 else str(raw)
        metadata = json.loads(raw)
        metadata["source_image_size"] = int(rgb.shape[1])
        metadata["image_size"] = int(size)
        metadata["resize_method"] = "opencv_inter_area"
        payload["metadata"] = np.asarray(
            json.dumps(metadata, sort_keys=True)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, output)
    return {
        "frames": int(payload["rgb"].shape[0]),
        "transitions": int(np.asarray(payload["actions"]).shape[0]),
    }


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if source_dir == output_dir:
        raise ValueError("Refusing to overwrite the source dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(source_dir.rglob("episode_train_*.npz"))
    if not sources:
        raise FileNotFoundError(f"No train episodes below {source_dir}")

    jobs = []
    for source in sources:
        relative = source.relative_to(source_dir)
        output = output_dir / relative
        if output.exists():
            raise FileExistsError(
                f"{output} already exists; use a fresh output directory"
            )
        jobs.append((str(source), str(output), int(args.size)))

    total_frames = 0
    total_transitions = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(resize_episode, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            total_frames += result["frames"]
            total_transitions += result["transitions"]
            if completed == 1 or completed % 250 == 0:
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "episodes": len(jobs),
                            "frames": total_frames,
                            "transitions": total_transitions,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    print(
        json.dumps(
            {
                "event": "finished",
                "episodes": completed,
                "frames": total_frames,
                "transitions": total_transitions,
                "size": int(args.size),
                "method": "opencv_inter_area",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
