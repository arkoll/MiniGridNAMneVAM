#!/usr/bin/env python3
"""Render three successful PPO XLand trajectories as one clean GIF."""
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image

ROOT = Path("/home/user8_3/xland/xland_joint_wam")
DATA = ROOT / "data/exactcraft_easy_100k_128/train"
OUT = ROOT / "runs/ppo_visualizations/ppo_exactcraft_three_rooms.gif"
EPISODES = ("episode_train_00000.npz", "episode_train_00001.npz", "episode_train_00002.npz")
SIZE = 512

frames = []
for filename in EPISODES:
    archive = np.load(DATA / filename)
    frames.extend(
        np.asarray(Image.fromarray(frame).resize((SIZE, SIZE), Image.Resampling.NEAREST))
        for frame in archive["rgb"]
    )
OUT.parent.mkdir(parents=True, exist_ok=True)
imageio.mimsave(OUT, frames, duration=0.30, loop=0)
print(OUT)
