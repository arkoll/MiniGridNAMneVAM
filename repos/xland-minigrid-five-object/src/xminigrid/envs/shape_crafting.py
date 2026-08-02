"""Color-invariant shape crafting tasks for XLand-MiniGrid.

The abstract dynamics depend only on input shapes.  Object colors are sampled
from the same vocabulary in train and validation, while every shape has
disjoint train/validation color assignments.  Consequently no concrete
(rule, shape, color) binding leaks across the split.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import jax.numpy as jnp

from ..core.constants import TILES_REGISTRY, Colors, Tiles
from ..core.goals import AgentNearGoal
from ..core.rules import ShapeTileNearRule
from ..types import RuleSet
from .five_object_crafting import (
    BLUE_KEY,
    YELLOW_STAR,
    FiveObjectCraftingEnv,
)
from .xland import XLandEnvParams


ROLE_NAMES = ("A", "B", "C", "D", "E")
ROLE_SHAPES = (
    Tiles.BALL,
    Tiles.SQUARE,
    Tiles.PYRAMID,
    Tiles.HEX,
    Tiles.DIAMOND,
)
SHAPE_NAMES = ("ball", "square", "pyramid", "hexagon", "diamond")

COLOR_IDS = (
    Colors.RED,
    Colors.GREEN,
    Colors.ORANGE,
    Colors.PURPLE,
    Colors.PINK,
    Colors.WHITE,
)
COLOR_NAMES_BY_ID = {
    Colors.RED: "red",
    Colors.GREEN: "green",
    Colors.ORANGE: "orange",
    Colors.PURPLE: "purple",
    Colors.PINK: "pink",
    Colors.WHITE: "white",
}

RULE_IDS = tuple(f"R{index}" for index in range(1, 10))
RULE_TEXT = {
    "R1": "ball+square->hexagon",
    "R2": "ball+pyramid->hexagon",
    "R3": "square+pyramid->hexagon",
    "R4": "hexagon+ball->diamond",
    "R5": "hexagon+square->diamond",
    "R6": "hexagon+pyramid->diamond",
    "R7": "diamond+ball->blue key",
    "R8": "diamond+square->blue key",
    "R9": "diamond+pyramid->blue key",
}

STAGE1_SHAPES = (
    (Tiles.BALL, Tiles.SQUARE),
    (Tiles.BALL, Tiles.PYRAMID),
    (Tiles.SQUARE, Tiles.PYRAMID),
)
STAGE1_ROLE_INDICES = ((0, 1), (0, 2), (1, 2))
STAGE2_CATALYST_SHAPES = (Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID)
STAGE2_CATALYST_ROLE_INDICES = (0, 1, 2)
STAGE3_CATALYST_SHAPES = (Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID)
STAGE3_CATALYST_ROLE_INDICES = (0, 1, 2)


def _make_color_index_palettes(split: str) -> tuple[tuple[int, ...], ...]:
    """Create 36 balanced, distinct-color palettes for one split.

    For every role/shape, train gets three colors and validation gets the
    complementary three.  Each individual scene uses five distinct colors.
    """

    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    parity = 0 if split == "train" else 1
    palettes = tuple(
        assignment
        for assignment in product(range(len(COLOR_IDS)), repeat=len(ROLE_SHAPES))
        if len(set(assignment)) == len(assignment)
        and all((color_index - role_index) % 2 == parity for role_index, color_index in enumerate(assignment))
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
class ShapeCraftTask:
    task_index: int
    uid: str
    split: str
    tree_id: int
    palette_id: int
    stage1_idx: int
    stage2_idx: int
    stage3_idx: int
    rule_ids: tuple[str, str, str]
    initial_role_indices: tuple[int, int, int, int]
    palette_color_ids: tuple[int, int, int, int, int]
    palette_color_names: tuple[str, str, str, str, str]
    role_tiles: tuple[tuple[int, int], ...]
    ruleset: RuleSet

    @property
    def rule_text(self) -> tuple[str, str, str]:
        return tuple(RULE_TEXT[rule_id] for rule_id in self.rule_ids)


def _make_task(
    split: str,
    task_index: int,
    tree_id: int,
    palette_id: int,
    palette: tuple[int, ...],
    stage1_idx: int,
    stage2_idx: int,
    stage3_idx: int,
) -> ShapeCraftTask:
    role_tiles = tuple(_tile(role_index, palette) for role_index in range(5))
    role_color_ids = tuple(COLOR_IDS[color_index] for color_index in palette)
    role_color_names = tuple(COLOR_NAMES_BY_ID[color_id] for color_id in role_color_ids)

    stage1_inputs = STAGE1_ROLE_INDICES[stage1_idx]
    stage2_role = STAGE2_CATALYST_ROLE_INDICES[stage2_idx]
    stage3_role = STAGE3_CATALYST_ROLE_INDICES[stage3_idx]
    initial_role_indices = (
        stage1_inputs[0],
        stage1_inputs[1],
        stage2_role,
        stage3_role,
    )

    stage1_rule = ShapeTileNearRule(
        shape_a=STAGE1_SHAPES[stage1_idx][0],
        shape_b=STAGE1_SHAPES[stage1_idx][1],
        prod_tile=role_tiles[3],
    )
    stage2_rule = ShapeTileNearRule(
        shape_a=Tiles.HEX,
        shape_b=STAGE2_CATALYST_SHAPES[stage2_idx],
        prod_tile=role_tiles[4],
    )
    stage3_rule = ShapeTileNearRule(
        shape_a=Tiles.DIAMOND,
        shape_b=STAGE3_CATALYST_SHAPES[stage3_idx],
        prod_tile=BLUE_KEY,
    )

    # Reverse-topological order prevents two stages firing on one action.
    encoded_rules = jnp.stack(
        (
            stage3_rule.encode(),
            stage2_rule.encode(),
            stage1_rule.encode(),
        )
    )
    initial_tiles = jnp.stack(tuple(role_tiles[index] for index in initial_role_indices))
    ruleset = RuleSet(
        goal=AgentNearGoal(tile=YELLOW_STAR).encode(),
        rules=encoded_rules,
        init_tiles=initial_tiles,
    )
    return ShapeCraftTask(
        task_index=task_index,
        uid=f"{split}-T{tree_id:02d}-P{palette_id:02d}",
        split=split,
        tree_id=tree_id,
        palette_id=palette_id,
        stage1_idx=stage1_idx,
        stage2_idx=stage2_idx,
        stage3_idx=stage3_idx,
        rule_ids=(
            f"R{stage1_idx + 1}",
            f"R{stage2_idx + 4}",
            f"R{stage3_idx + 7}",
        ),
        initial_role_indices=initial_role_indices,
        palette_color_ids=role_color_ids,
        palette_color_names=role_color_names,
        role_tiles=tuple(tuple(int(value) for value in tile) for tile in role_tiles),
        ruleset=ruleset,
    )


def _make_tasks(
    split: str,
    palettes: tuple[tuple[int, ...], ...],
) -> tuple[ShapeCraftTask, ...]:
    tasks = []
    task_index = 1
    for tree_id, (stage1_idx, stage2_idx, stage3_idx) in enumerate(
        product(range(3), repeat=3),
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
                    stage3_idx=stage3_idx,
                )
            )
            task_index += 1
    assert len(tasks) == 27 * 36
    return tuple(tasks)


TRAIN_TASKS = _make_tasks("train", TRAIN_COLOR_INDEX_PALETTES)
VAL_TASKS = _make_tasks("val", VAL_COLOR_INDEX_PALETTES)


def get_shape_task(split: str, task_index: int) -> ShapeCraftTask:
    if split == "train":
        tasks = TRAIN_TASKS
    elif split == "val":
        tasks = VAL_TASKS
    else:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")
    if not 1 <= task_index <= len(tasks):
        raise ValueError(f"task_index must be in [1, {len(tasks)}], got {task_index}")
    return tasks[task_index - 1]


class ShapeCraftEnv(FiveObjectCraftingEnv):
    """ExactCraft layout with color-invariant input-shape dynamics."""

    def default_params(self, **kwargs) -> XLandEnvParams:
        params = super().default_params().replace(grid_type="SHAPE_CRAFT_V1")
        return params.replace(**kwargs)


def make_shape_env_and_params(
    split: str,
    task_index: int,
    **param_overrides,
) -> tuple[ShapeCraftEnv, XLandEnvParams, ShapeCraftTask]:
    task = get_shape_task(split, task_index)
    env = ShapeCraftEnv()
    params = env.default_params(ruleset=task.ruleset, **param_overrides)
    return env, params, task


__all__ = [
    "ROLE_NAMES",
    "ROLE_SHAPES",
    "SHAPE_NAMES",
    "COLOR_IDS",
    "COLOR_NAMES_BY_ID",
    "RULE_IDS",
    "RULE_TEXT",
    "TRAIN_COLOR_INDEX_PALETTES",
    "VAL_COLOR_INDEX_PALETTES",
    "TRAIN_TASKS",
    "VAL_TASKS",
    "ShapeCraftTask",
    "ShapeCraftEnv",
    "get_shape_task",
    "make_shape_env_and_params",
]
