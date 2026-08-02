"""Baba Is AI episode files as a NanoWM DataSource."""

from functools import lru_cache
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from .base import DataSource, TrajectoryData


class BabaDataSource(DataSource):
    """Read atomic episode ``.npz`` files produced by ``collect_baba.py``."""

    NUM_ACTIONS = 5

    def __init__(self, data_path: str, n_rollout=None, **_kwargs):
        self.data_path = Path(data_path).expanduser().resolve()
        if not self.data_path.is_dir():
            raise FileNotFoundError(f"Baba dataset directory not found: {self.data_path}")

        self.files = sorted(self.data_path.glob("episode_*.npz"))
        if n_rollout is not None:
            self.files = self.files[: int(n_rollout)]
        if not self.files:
            raise FileNotFoundError(
                f"No episode_*.npz files found in {self.data_path}"
            )

    @lru_cache(maxsize=8)
    def _load_episode(self, index: int) -> Dict:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as episode:
            rgb = np.asarray(episode["rgb"], dtype=np.uint8)
            actions = np.asarray(episode["actions"], dtype=np.int64)
            metadata = {}
            metadata_key = (
                "metadata_json" if "metadata_json" in episode else "metadata"
            )
            if metadata_key in episode:
                raw = episode[metadata_key]
                raw = raw.item() if raw.ndim == 0 else str(raw)
                metadata = json.loads(raw)

        if rgb.shape[0] != actions.shape[0] + 1:
            raise ValueError(
                f"Broken transition alignment in {path}: "
                f"{rgb.shape[0]} frames for {actions.shape[0]} actions"
            )
        if actions.size and (actions.min() < 0 or actions.max() >= self.NUM_ACTIONS):
            raise ValueError(f"Out-of-range action in {path}")
        return {"rgb": rgb, "actions": actions, "metadata": metadata}

    def load_trajectory(self, index: int) -> TrajectoryData:
        episode = self._load_episode(index)
        actions = torch.from_numpy(episode["actions"]).long()
        actions = F.one_hot(actions, num_classes=self.NUM_ACTIONS).float()
        # NanoWM indexes one action row per frame. Real actions occupy rows
        # 0..T-1; the final frame receives an explicit idle padding row.
        idle_padding = torch.zeros(1, self.NUM_ACTIONS, dtype=torch.float32)
        idle_padding[0, 0] = 1.0
        actions = torch.cat([actions, idle_padding], dim=0)
        states = torch.empty(actions.shape[0], 0, dtype=torch.float32)
        return TrajectoryData(
            states=states,
            actions=actions,
            seq_length=actions.shape[0],
            meta=episode["metadata"],
        )

    def get_seq_length(self, index: int) -> int:
        return int(self._load_episode(index)["rgb"].shape[0])

    def load_visual_frames(
        self, index: int, start: int, end: int, step: int = 1
    ) -> torch.Tensor:
        rgb = self._load_episode(index)["rgb"][start:end:step]
        frames = torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0
        return frames.permute(0, 3, 1, 2).contiguous()

    def get_num_trajectories(self) -> int:
        return len(self.files)

    @property
    def action_dim(self) -> int:
        return self.NUM_ACTIONS

    @property
    def state_dim(self) -> int:
        return 0
