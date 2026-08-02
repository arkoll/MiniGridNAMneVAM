"""Depth-2 ExactCraft Easy tasks.

Easy keeps the complete two-room key-door-goal pipeline while reducing the
world dynamics from nine rules and three stages to six rules and two stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import jax.numpy as jnp

from ..core.goals import AgentNearGoal
from ..core.rules import TileNearRule
from ..types import RuleSet
from .five_object_crafting import (
    A,
    B,
    BLUE_KEY,
    C,
    D,
    YELLOW_STAR,
    FiveObjectCraftingEnv,
)
from .xland import XLandEnvParams


RULE_IDS = tuple(f"R{index}" for index in range(1, 7))
RULE_TEXT = {
    "R1": "A+B->D",
    "R2": "A+C->D",
    "R3": "B+C->D",
    "R4": "D+A->K",
    "R5": "D+B->K",
    "R6": "D+C->K",
}

STAGE1_RULES = (
    TileNearRule(tile_a=A, tile_b=B, prod_tile=D),
    TileNearRule(tile_a=A, tile_b=C, prod_tile=D),
    TileNearRule(tile_a=B, tile_b=C, prod_tile=D),
)
STAGE2_RULES = (
    TileNearRule(tile_a=D, tile_b=A, prod_tile=BLUE_KEY),
    TileNearRule(tile_a=D, tile_b=B, prod_tile=BLUE_KEY),
    TileNearRule(tile_a=D, tile_b=C, prod_tile=BLUE_KEY),
)
STAGE1_INPUTS = ((A, B), (A, C), (B, C))
STAGE1_SYMBOLS = (("A", "B"), ("A", "C"), ("B", "C"))
STAGE2_CATALYSTS = (A, B, C)
STAGE2_SYMBOLS = ("A", "B", "C")


@dataclass(frozen=True)
class ExactCraftEasyTask:
    task_id: int
    split: str
    stage1_idx: int
    stage2_idx: int
    rule_ids: tuple[str, str]
    initial_symbols: tuple[str, str, str]
    ruleset: RuleSet

    @property
    def rule_text(self) -> tuple[str, str]:
        return tuple(RULE_TEXT[rule_id] for rule_id in self.rule_ids)


def _make_tasks() -> tuple[ExactCraftEasyTask, ...]:
    tasks = []
    for task_id, (stage1_idx, stage2_idx) in enumerate(
        product(range(3), repeat=2),
        start=1,
    ):
        stage1_rule = STAGE1_RULES[stage1_idx]
        stage2_rule = STAGE2_RULES[stage2_idx]
        split = "val" if stage1_idx == stage2_idx else "train"
        ruleset = RuleSet(
            goal=AgentNearGoal(tile=YELLOW_STAR).encode(),
            # Reverse-topological order prevents both stages from firing after
            # a single put-down action.
            rules=jnp.stack((stage2_rule.encode(), stage1_rule.encode())),
            init_tiles=jnp.stack(
                (
                    STAGE1_INPUTS[stage1_idx][0],
                    STAGE1_INPUTS[stage1_idx][1],
                    STAGE2_CATALYSTS[stage2_idx],
                )
            ),
        )
        tasks.append(
            ExactCraftEasyTask(
                task_id=task_id,
                split=split,
                stage1_idx=stage1_idx,
                stage2_idx=stage2_idx,
                rule_ids=(
                    f"R{stage1_idx + 1}",
                    f"R{stage2_idx + 4}",
                ),
                initial_symbols=(
                    STAGE1_SYMBOLS[stage1_idx][0],
                    STAGE1_SYMBOLS[stage1_idx][1],
                    STAGE2_SYMBOLS[stage2_idx],
                ),
                ruleset=ruleset,
            )
        )

    result = tuple(tasks)
    train_tasks = tuple(task for task in result if task.split == "train")
    val_tasks = tuple(task for task in result if task.split == "val")
    assert len(result) == 9
    assert len(train_tasks) == 6
    assert len(val_tasks) == 3
    assert {rule for task in train_tasks for rule in task.rule_ids} == set(RULE_IDS)
    assert {rule for task in val_tasks for rule in task.rule_ids} == set(RULE_IDS)
    return result


TASKS = _make_tasks()
TRAIN_TASKS = tuple(task for task in TASKS if task.split == "train")
VAL_TASKS = tuple(task for task in TASKS if task.split == "val")


def get_easy_task(task_id: int) -> ExactCraftEasyTask:
    if not 1 <= task_id <= len(TASKS):
        raise ValueError(f"task_id must be in [1, {len(TASKS)}], got {task_id}")
    return TASKS[task_id - 1]


class ExactCraftEasyEnv(FiveObjectCraftingEnv):
    """ExactCraft layout with the depth-2 Easy ruleset."""

    def default_params(self, **kwargs) -> XLandEnvParams:
        params = super().default_params().replace(
            max_steps=192,
            grid_type="EXACT_CRAFT_EASY_V1",
        )
        return params.replace(**kwargs)


def make_easy_env_and_params(
    task_id: int,
    **param_overrides,
) -> tuple[ExactCraftEasyEnv, XLandEnvParams, ExactCraftEasyTask]:
    task = get_easy_task(task_id)
    env = ExactCraftEasyEnv()
    params = env.default_params(ruleset=task.ruleset, **param_overrides)
    return env, params, task


__all__ = [
    "RULE_IDS",
    "RULE_TEXT",
    "TASKS",
    "TRAIN_TASKS",
    "VAL_TASKS",
    "ExactCraftEasyTask",
    "ExactCraftEasyEnv",
    "get_easy_task",
    "make_easy_env_and_params",
]
