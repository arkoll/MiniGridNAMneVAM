#!/usr/bin/env python3
"""Checks for ExactCraft, ShapeCraft, and EmbodiedExactCraft Easy levels."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

import xminigrid
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.grid import room
from xminigrid.core.rules import check_rule
from xminigrid.envs.easy_crafting import (
    RULE_IDS as EXACT_EASY_RULE_IDS,
    TASKS as EXACT_EASY_TASKS,
    TRAIN_TASKS as EXACT_EASY_TRAIN_TASKS,
    VAL_TASKS as EXACT_EASY_VAL_TASKS,
    make_easy_env_and_params,
)
from xminigrid.envs.embodied_crafting import (
    EMBODIMENTS,
    make_embodied_env_and_params,
)
from xminigrid.envs.five_object_crafting import (
    BLUE_KEY,
    BLUE_LOCKED_DOOR,
    D,
    TASKS as EXACT_MEDIUM_TASKS,
    YELLOW_STAR,
)
from xminigrid.envs.shape_crafting import (
    TRAIN_TASKS as SHAPE_MEDIUM_TRAIN_TASKS,
    VAL_TASKS as SHAPE_MEDIUM_VAL_TASKS,
)
from xminigrid.envs.shape_crafting_easy import (
    ROLE_SHAPES,
    RULE_IDS as SHAPE_EASY_RULE_IDS,
    TRAIN_COLOR_INDEX_PALETTES,
    TRAIN_TASKS as SHAPE_EASY_TRAIN_TASKS,
    VAL_COLOR_INDEX_PALETTES,
    VAL_TASKS as SHAPE_EASY_VAL_TASKS,
    make_shape_easy_env_and_params,
)
from xminigrid.envs.shape_crafting import COLOR_IDS
from xminigrid.types import AgentState


EASY_REGISTRY_IDS = (
    "XLand-MiniGrid-ExactCraft-Easy-8x8-v1",
    "XLand-MiniGrid-ShapeCraft-Easy-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Standard-Easy-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Stride2-Easy-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Omni8-Easy-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Crab-Easy-8x8-v1",
)
MEDIUM_REGISTRY_IDS = (
    "XLand-MiniGrid-ExactCraft-Medium-8x8-v1",
    "XLand-MiniGrid-ShapeCraft-Medium-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Standard-Medium-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Stride2-Medium-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Omni8-Medium-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Crab-Medium-8x8-v1",
)
LEGACY_IDS = (
    "XLand-MiniGrid-FiveObjectCrafting-8x8",
    "XLand-MiniGrid-ExactCraft-8x8-v1",
    "XLand-MiniGrid-ShapeCraft-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Standard-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Stride2-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Omni8-8x8-v1",
    "XLand-MiniGrid-EmbodiedExactCraft-Crab-8x8-v1",
)


def count_exact_tiles(grid: np.ndarray, tile) -> int:
    return int(np.all(grid == np.asarray(tile), axis=-1).sum())


def check_exact_easy_split_and_chains() -> None:
    assert len(EXACT_EASY_TASKS) == 9
    assert len(EXACT_EASY_TRAIN_TASKS) == 6
    assert len(EXACT_EASY_VAL_TASKS) == 3
    train_rules = {
        rule
        for task in EXACT_EASY_TRAIN_TASKS
        for rule in task.rule_ids
    }
    val_rules = {
        rule
        for task in EXACT_EASY_VAL_TASKS
        for rule in task.rule_ids
    }
    assert train_rules == val_rules == set(EXACT_EASY_RULE_IDS)

    for task in EXACT_EASY_TASKS:
        assert task.split == (
            "val" if task.stage1_idx == task.stage2_idx else "train"
        )
        assert task.ruleset.rules.shape == (2, 7)
        assert task.ruleset.init_tiles.shape == (3, 2)
        assert np.all(np.asarray(task.ruleset.rules)[:, 0] == 3)

        grid = room(7, 7)
        agent = AgentState(
            position=jnp.asarray((4, 4)),
            direction=jnp.asarray(0),
        )
        position = jnp.asarray((3, 2))
        initial = np.asarray(task.ruleset.init_tiles)
        grid = grid.at[3, 2].set(initial[0])
        grid = grid.at[3, 3].set(initial[1])
        grid, agent = check_rule(
            task.ruleset.rules,
            grid,
            agent,
            jnp.asarray(4),
            position,
        )
        assert np.array_equal(np.asarray(grid[3, 3]), np.asarray(D))

        grid = grid.at[3, 2].set(initial[2])
        grid, agent = check_rule(
            task.ruleset.rules,
            grid,
            agent,
            jnp.asarray(4),
            position,
        )
        assert np.array_equal(np.asarray(grid[3, 3]), np.asarray(BLUE_KEY))


def check_exact_easy_resets() -> None:
    for task in EXACT_EASY_TASKS:
        env, params, _ = make_easy_env_and_params(task.task_id)
        assert params.max_steps == 192
        for seed in (0, 17):
            timestep = env.reset(params, jax.random.PRNGKey(seed))
            grid = np.asarray(timestep.state.grid)
            assert grid.shape == (8, 8, 2)
            assert timestep.observation["img"].shape == (8, 8, 3)
            assert count_exact_tiles(grid, BLUE_LOCKED_DOOR) == 1
            assert count_exact_tiles(grid, YELLOW_STAR) == 1
            assert count_exact_tiles(grid, BLUE_KEY) == 0
            assert count_exact_tiles(grid, D) == 0
            assert int(
                np.isin(
                    grid[..., 0],
                    (Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID),
                ).sum()
            ) == 3

    env, params, _ = make_easy_env_and_params(2)
    timestep = jax.jit(env.reset)(params, jax.random.PRNGKey(123))
    timestep = jax.jit(env.step)(params, timestep, jnp.asarray(1))
    assert int(timestep.state.step_num) == 1
    assert env.render(params, timestep).shape == (256, 256, 3)


def check_shape_easy_split_and_dynamics() -> None:
    assert len(TRAIN_COLOR_INDEX_PALETTES) == 36
    assert len(VAL_COLOR_INDEX_PALETTES) == 36
    assert len(SHAPE_EASY_TRAIN_TASKS) == 324
    assert len(SHAPE_EASY_VAL_TASKS) == 324

    train_trees = {
        (task.stage1_idx, task.stage2_idx)
        for task in SHAPE_EASY_TRAIN_TASKS
    }
    val_trees = {
        (task.stage1_idx, task.stage2_idx)
        for task in SHAPE_EASY_VAL_TASKS
    }
    assert train_trees == val_trees
    assert len(train_trees) == 9
    assert {
        rule
        for task in SHAPE_EASY_TRAIN_TASKS
        for rule in task.rule_ids
    } == set(SHAPE_EASY_RULE_IDS)
    assert {
        rule
        for task in SHAPE_EASY_VAL_TASKS
        for rule in task.rule_ids
    } == set(SHAPE_EASY_RULE_IDS)

    global_train_colors = set()
    global_val_colors = set()
    for role_index in range(4):
        train_colors = {
            task.palette_color_ids[role_index]
            for task in SHAPE_EASY_TRAIN_TASKS
        }
        val_colors = {
            task.palette_color_ids[role_index]
            for task in SHAPE_EASY_VAL_TASKS
        }
        assert len(train_colors) == len(val_colors) == 3
        assert train_colors.isdisjoint(val_colors)
        global_train_colors.update(train_colors)
        global_val_colors.update(val_colors)
    assert global_train_colors == global_val_colors == set(COLOR_IDS)
    assert all(
        len(set(task.palette_color_ids)) == 4
        for task in SHAPE_EASY_TRAIN_TASKS + SHAPE_EASY_VAL_TASKS
    )

    train_shape_colors = {
        (ROLE_SHAPES[index], task.palette_color_ids[index])
        for task in SHAPE_EASY_TRAIN_TASKS
        for index in range(4)
    }
    val_shape_colors = {
        (ROLE_SHAPES[index], task.palette_color_ids[index])
        for task in SHAPE_EASY_VAL_TASKS
        for index in range(4)
    }
    assert train_shape_colors.isdisjoint(val_shape_colors)

    for task in (
        SHAPE_EASY_TRAIN_TASKS[0],
        SHAPE_EASY_TRAIN_TASKS[-1],
        SHAPE_EASY_VAL_TASKS[0],
        SHAPE_EASY_VAL_TASKS[-1],
    ):
        assert task.ruleset.rules.shape == (2, 7)
        assert task.ruleset.init_tiles.shape == (3, 2)
        assert np.all(np.asarray(task.ruleset.rules)[:, 0] == 12)
        grid = room(7, 7)
        agent = AgentState(
            position=jnp.asarray((4, 4)),
            direction=jnp.asarray(0),
        )
        position = jnp.asarray((3, 2))
        stage1_shapes = (
            (Tiles.BALL, Tiles.SQUARE),
            (Tiles.BALL, Tiles.PYRAMID),
            (Tiles.SQUARE, Tiles.PYRAMID),
        )[task.stage1_idx]
        grid = grid.at[3, 2].set(
            TILES_REGISTRY[stage1_shapes[0], Colors.BROWN]
        )
        grid = grid.at[3, 3].set(
            TILES_REGISTRY[stage1_shapes[1], Colors.YELLOW]
        )
        grid, agent = check_rule(
            task.ruleset.rules,
            grid,
            agent,
            jnp.asarray(4),
            position,
        )
        assert np.array_equal(
            np.asarray(grid[3, 3]),
            np.asarray(task.role_tiles[3]),
        )

        catalyst_shape = (
            Tiles.BALL,
            Tiles.SQUARE,
            Tiles.PYRAMID,
        )[task.stage2_idx]
        grid = grid.at[3, 2].set(
            TILES_REGISTRY[catalyst_shape, Colors.BROWN]
        )
        grid, agent = check_rule(
            task.ruleset.rules,
            grid,
            agent,
            jnp.asarray(4),
            position,
        )
        assert np.array_equal(np.asarray(grid[3, 3]), np.asarray(BLUE_KEY))


def check_shape_easy_resets() -> None:
    for split in ("train", "val"):
        for task_index in (1, 162, 324):
            env, params, _ = make_shape_easy_env_and_params(split, task_index)
            assert params.max_steps == 192
            timestep = env.reset(params, jax.random.PRNGKey(task_index))
            grid = np.asarray(timestep.state.grid)
            assert timestep.observation["img"].shape == (8, 8, 3)
            assert count_exact_tiles(grid, BLUE_LOCKED_DOOR) == 1
            assert count_exact_tiles(grid, YELLOW_STAR) == 1
            assert count_exact_tiles(grid, BLUE_KEY) == 0
            assert int((grid[..., 0] == Tiles.HEX).sum()) == 0
            assert int(
                np.isin(
                    grid[..., 0],
                    (Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID),
                ).sum()
            ) == 3

    env, params, _ = make_shape_easy_env_and_params("train", 1)
    timestep = jax.jit(env.reset)(params, jax.random.PRNGKey(3))
    timestep = jax.jit(env.step)(params, timestep, jnp.asarray(1))
    assert env.render(params, timestep).shape == (256, 256, 3)


def check_embodied_easy() -> None:
    for task_id in (1, 5, 9):
        for seed in (0, 19):
            snapshots = []
            for embodiment in EMBODIMENTS:
                env, params, task = make_embodied_env_and_params(
                    embodiment,
                    task_id,
                    difficulty="easy",
                )
                assert params.max_steps == 192
                assert "EASY" in params.grid_type
                timestep = env.reset(params, jax.random.PRNGKey(seed))
                snapshots.append(
                    (
                        np.asarray(timestep.state.grid),
                        np.asarray(timestep.state.agent.position),
                        np.asarray(timestep.state.rule_encoding),
                        task.split,
                    )
                )
            for snapshot in snapshots[1:]:
                assert np.array_equal(snapshot[0], snapshots[0][0])
                assert np.array_equal(snapshot[1], snapshots[0][1])
                assert np.array_equal(snapshot[2], snapshots[0][2])
                assert snapshot[3] == snapshots[0][3]

    for embodiment in EMBODIMENTS:
        easy_env, easy_params, _ = make_embodied_env_and_params(
            embodiment,
            2,
            difficulty="easy",
        )
        easy_timestep = jax.jit(easy_env.reset)(
            easy_params,
            jax.random.PRNGKey(4),
        )
        easy_timestep = jax.jit(easy_env.step)(
            easy_params,
            easy_timestep,
            jnp.asarray(1),
        )
        assert easy_env.render(easy_params, easy_timestep).shape == (256, 256, 3)

        _, medium_params, medium_task = make_embodied_env_and_params(
            embodiment,
            27,
        )
        assert medium_params.max_steps == 256
        assert "MEDIUM" in medium_params.grid_type
        assert medium_task.task_id == 27


def check_registry_and_medium_unchanged() -> None:
    registry = set(xminigrid.registered_environments())
    for registry_id in EASY_REGISTRY_IDS + MEDIUM_REGISTRY_IDS + LEGACY_IDS:
        assert registry_id in registry

    assert len(EXACT_MEDIUM_TASKS) == 27
    assert len(SHAPE_MEDIUM_TRAIN_TASKS) == 972
    assert len(SHAPE_MEDIUM_VAL_TASKS) == 972

    exact_task = EXACT_EASY_TASKS[0]
    env, params = xminigrid.make(
        "XLand-MiniGrid-ExactCraft-Easy-8x8-v1",
        ruleset=exact_task.ruleset,
    )
    timestep = env.reset(params, jax.random.PRNGKey(0))
    assert timestep.observation["img"].shape == (8, 8, 3)
    assert params.max_steps == 192

    shape_task = SHAPE_EASY_TRAIN_TASKS[0]
    env, params = xminigrid.make(
        "XLand-MiniGrid-ShapeCraft-Easy-8x8-v1",
        ruleset=shape_task.ruleset,
    )
    timestep = env.reset(params, jax.random.PRNGKey(0))
    assert timestep.observation["img"].shape == (8, 8, 3)


def main() -> None:
    check_exact_easy_split_and_chains()
    check_exact_easy_resets()
    check_shape_easy_split_and_dynamics()
    check_shape_easy_resets()
    check_embodied_easy()
    check_registry_and_medium_unchanged()
    print(
        json.dumps(
            {
                "status": "ok",
                "exact_easy": {
                    "rules": 6,
                    "trees": 9,
                    "train": 6,
                    "val": 3,
                    "initial_objects": 3,
                    "active_rules": 2,
                },
                "shape_easy": {
                    "rules": 6,
                    "trees_per_split": 9,
                    "palettes_per_split": 36,
                    "train_tasks": 324,
                    "val_tasks": 324,
                    "shape_color_overlap": 0,
                },
                "embodied_easy": list(EMBODIMENTS),
                "canonical_medium_ids": len(MEDIUM_REGISTRY_IDS),
                "legacy_ids_preserved": len(LEGACY_IDS),
                "jit_and_rendering": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
