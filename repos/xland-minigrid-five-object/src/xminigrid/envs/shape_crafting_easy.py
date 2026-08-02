"""Depth-2 color-invariant ShapeCraft Easy tasks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import jax.numpy as jnp

from ..core.constants import TILES_REGISTRY, Tiles
from ..core.goals import AgentNearGoal
from ..core.rules import ShapeTileNearRule
from ..types import RuleSet
from .five_object_crafting import BLUE_KEY, YELLOW_STAR, FiveObjectCraftingEnv
from .shape_crafting import COLOR_IDS, COLOR_NAMES_BY_ID
from .xland import XLandEnvParams


ROLE_NAMES = ("A", "B", "C", "D")
ROLE_SHAPES = (
    Tiles.BALL,
    Tiles.SQUARE,
    Tiles.PYRAMID,
    Tiles.HEX,
)
SHAPE_NAMES = ("ball", "square", "pyramid", "hexagon")
RULE_IDS = tuple(f"R{index}" for index in range(1, 7))
RULE_TEXT = {
    "R1": "ball+square->hexagon",
    "R2": "ball+pyramid->hexagon",
    "R3": "square+pyramid->hexagon",
    "R4": "hexagon+ball->blue key",
    "R5": "hexagon+square->blue key",
    "R6": "hexagon+pyramid->blue key",
}
STAGE1_SHAPES = (
    (Tiles.BALL, Tiles.SQUARE),
    (Tiles.BALL, Tiles.PYRAMID),
    (Tiles.SQUARE, Tiles.PYRAMID),
)
STAGE1_ROLE_INDICES = ((0, 1), (0, 2), (1, 2))
STAGE2_CATALYST_SHAPES = (Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID)


def _make_color_index_palettes(split: str) -> tuple[tuple[int, ...], ...]:
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    parity = 0 if split == "train" else 1
    palettes = tuple(
        assignment
        for assignment in product(range(len(COLOR_IDS)), repeat=len(ROLE_SHAPES))
        if len(set(assignment)) == len(assignment)
        and all(
            (color_index - role_index) % 2 == parity
            for role_index, color_index in enumerate(assignment)
        )
    )
    assert len(palettes) == 36
    return palettes


TRAIN_COLOR_INDEX_PALETTES = _make_color_index_palettes("train")
VAL_COLOR_INDEX_PALETTES = _make_color_index_palettes("val")


def _tile(role_index: int, palette: tuple[int, ...]):
    return TILES_REGISTRY[
        ROLE_SHAPES[role_index],
        COLOR_IDS[palette[role_index]],
    ]


@dataclass(frozen=True)
class ShapeCraftEasyTask:
    task_index: int
    uid: str
    split: str
    tree_id: int
    palette_id: int
    stage1_idx: int
    stage2_idx: int
    rule_ids: tuple[str, str]
    initial_role_indices: tuple[int, int, int]
    palette_color_ids: tuple[int, int, int, int]
    palette_color_names: tuple[str, str, str, str]
    role_tiles: tuple[tuple[int, int], ...]
    ruleset: RuleSet

    @property
    def rule_text(self) -> tuple[str, str]:
        return tuple(RULE_TEXT[rule_id] for rule_id in self.rule_ids)


def _make_task(
    split: str,
    task_index: int,
    tree_id: int,
    palette_id: int,
    palette: tuple[int, ...],
    stage1_idx: int,
    stage2_idx: int,
) -> ShapeCraftEasyTask:
    role_tiles = tuple(_tile(index, palette) for index in range(4))
    color_ids = tuple(COLOR_IDS[index] for index in palette)
    color_names = tuple(COLOR_NAMES_BY_ID[color_id] for color_id in color_ids)
    first_indices = STAGE1_ROLE_INDICES[stage1_idx]
    initial_role_indices = (
        first_indices[0],
        first_indices[1],
        stage2_idx,
    )
    stage1_rule = ShapeTileNearRule(
        shape_a=STAGE1_SHAPES[stage1_idx][0],
        shape_b=STAGE1_SHAPES[stage1_idx][1],
        prod_tile=role_tiles[3],
    )
    stage2_rule = ShapeTileNearRule(
        shape_a=Tiles.HEX,
        shape_b=STAGE2_CATALYST_SHAPES[stage2_idx],
        prod_tile=BLUE_KEY,
    )
    ruleset = RuleSet(
        goal=AgentNearGoal(tile=YELLOW_STAR).encode(),
        rules=jnp.stack((stage2_rule.encode(), stage1_rule.encode())),
        init_tiles=jnp.stack(
            tuple(role_tiles[index] for index in initial_role_indices)
        ),
    )
    return ShapeCraftEasyTask(
        task_index=task_index,
        uid=f"{split}-T{tree_id:02d}-P{palette_id:02d}",
        split=split,
        tree_id=tree_id,
        palette_id=palette_id,
        stage1_idx=stage1_idx,
        stage2_idx=stage2_idx,
        rule_ids=(
            f"R{stage1_idx + 1}",
            f"R{stage2_idx + 4}",
        ),
        initial_role_indices=initial_role_indices,
        palette_color_ids=color_ids,
        palette_color_names=color_names,
        role_tiles=tuple(
            tuple(int(value) for value in tile)
            for tile in role_tiles
        ),
        ruleset=ruleset,
    )


def _make_tasks(
    split: str,
    palettes: tuple[tuple[int, ...], ...],
) -> tuple[ShapeCraftEasyTask, ...]:
    tasks = []
    task_index = 1
    for tree_id, (stage1_idx, stage2_idx) in enumerate(
        product(range(3), repeat=2),
        start=1,
    ):
        for palette_id, palette in enumerate(palettes, start=1):
            tasks.append(
                _make_task(
                    split=split,
                    task_index=task_index,
                    tree_id=tree_id,
                    palette_id=palette_id,
                    palette=palette,
                    stage1_idx=stage1_idx,
                    stage2_idx=stage2_idx,
                )
            )
            task_index += 1
    assert len(tasks) == 9 * 36
    return tuple(tasks)


TRAIN_TASKS = _make_tasks("train", TRAIN_COLOR_INDEX_PALETTES)
VAL_TASKS = _make_tasks("val", VAL_COLOR_INDEX_PALETTES)


def get_shape_easy_task(split: str, task_index: int) -> ShapeCraftEasyTask:
    if split == "train":
        tasks = TRAIN_TASKS
    elif split == "val":
        tasks = VAL_TASKS
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if not 1 <= task_index <= len(tasks):
        raise ValueError(f"task_index must be in [1, {len(tasks)}], got {task_index}")
    return tasks[task_index - 1]


class ShapeCraftEasyEnv(FiveObjectCraftingEnv):
    """ShapeCraft color OOD benchmark with depth-2 Easy dynamics."""

    def default_params(self, **kwargs) -> XLandEnvParams:
        params = super().default_params().replace(
            max_steps=192,
            grid_type="SHAPE_CRAFT_EASY_V1",
        )
        return params.replace(**kwargs)


def make_shape_easy_env_and_params(
    split: str,
    task_index: int,
    **param_overrides,
) -> tuple[ShapeCraftEasyEnv, XLandEnvParams, ShapeCraftEasyTask]:
    task = get_shape_easy_task(split, task_index)
    env = ShapeCraftEasyEnv()
    params = env.default_params(ruleset=task.ruleset, **param_overrides)
    return env, params, task


__all__ = [
    "ROLE_NAMES",
    "ROLE_SHAPES",
    "SHAPE_NAMES",
    "RULE_IDS",
    "RULE_TEXT",
    "TRAIN_COLOR_INDEX_PALETTES",
    "VAL_COLOR_INDEX_PALETTES",
    "TRAIN_TASKS",
    "VAL_TASKS",
    "ShapeCraftEasyTask",
    "ShapeCraftEasyEnv",
    "get_shape_easy_task",
    "make_shape_easy_env_and_params",
]
