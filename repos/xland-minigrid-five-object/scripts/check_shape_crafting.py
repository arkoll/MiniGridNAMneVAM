#!/usr/bin/env python3
"""Regression and split checks for ShapeCraft-8x8-v1."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

import xminigrid
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.grid import room
from xminigrid.core.rules import ShapeTileNearRule, check_rule
from xminigrid.envs.five_object_crafting import (
    BLUE_KEY,
    BLUE_LOCKED_DOOR,
    YELLOW_STAR,
)
from xminigrid.envs.shape_crafting import (
    COLOR_IDS,
    ROLE_SHAPES,
    RULE_IDS,
    TRAIN_COLOR_INDEX_PALETTES,
    TRAIN_TASKS,
    VAL_COLOR_INDEX_PALETTES,
    VAL_TASKS,
    make_shape_env_and_params,
)
from xminigrid.types import AgentState


def count_exact_tiles(grid: np.ndarray, tile: jax.Array) -> int:
    return int(np.all(grid == np.asarray(tile), axis=-1).sum())


def check_split() -> None:
    assert len(TRAIN_COLOR_INDEX_PALETTES) == 36
    assert len(VAL_COLOR_INDEX_PALETTES) == 36
    assert len(TRAIN_TASKS) == 972
    assert len(VAL_TASKS) == 972
    assert len({task.uid for task in TRAIN_TASKS + VAL_TASKS}) == 1944

    train_trees = {
        (task.stage1_idx, task.stage2_idx, task.stage3_idx)
        for task in TRAIN_TASKS
    }
    val_trees = {
        (task.stage1_idx, task.stage2_idx, task.stage3_idx)
        for task in VAL_TASKS
    }
    assert train_trees == val_trees
    assert len(train_trees) == 27
    assert {rule for task in TRAIN_TASKS for rule in task.rule_ids} == set(RULE_IDS)
    assert {rule for task in VAL_TASKS for rule in task.rule_ids} == set(RULE_IDS)

    global_train_colors = set()
    global_val_colors = set()
    for role_index, _shape in enumerate(ROLE_SHAPES):
        train_colors = {
            task.palette_color_ids[role_index]
            for task in TRAIN_TASKS
        }
        val_colors = {
            task.palette_color_ids[role_index]
            for task in VAL_TASKS
        }
        assert len(train_colors) == len(val_colors) == 3
        assert train_colors.isdisjoint(val_colors)
        global_train_colors.update(train_colors)
        global_val_colors.update(val_colors)

    assert global_train_colors == global_val_colors == set(COLOR_IDS)
    assert all(len(set(task.palette_color_ids)) == 5 for task in TRAIN_TASKS)
    assert all(len(set(task.palette_color_ids)) == 5 for task in VAL_TASKS)

    # Stronger than the requested no-overlap condition: a shape-color binding
    # never crosses the split, irrespective of which rule uses the shape.
    train_shape_colors = {
        (ROLE_SHAPES[role_index], task.palette_color_ids[role_index])
        for task in TRAIN_TASKS
        for role_index in range(len(ROLE_SHAPES))
    }
    val_shape_colors = {
        (ROLE_SHAPES[role_index], task.palette_color_ids[role_index])
        for task in VAL_TASKS
        for role_index in range(len(ROLE_SHAPES))
    }
    assert train_shape_colors.isdisjoint(val_shape_colors)
    assert all(
        np.all(np.asarray(task.ruleset.rules)[:, 0] == 12)
        for task in TRAIN_TASKS + VAL_TASKS
    )


def _apply_pair(
    rules: jax.Array,
    shape_a: int,
    color_a: int,
    shape_b: int,
    color_b: int,
) -> np.ndarray:
    grid = room(7, 7)
    grid = grid.at[3, 2].set(TILES_REGISTRY[shape_a, color_a])
    grid = grid.at[3, 3].set(TILES_REGISTRY[shape_b, color_b])
    agent = AgentState(
        position=jnp.asarray((4, 4)),
        direction=jnp.asarray(0),
    )
    grid, _ = check_rule(
        rules,
        grid,
        agent,
        jnp.asarray(4),
        jnp.asarray((3, 2)),
    )
    return np.asarray(grid[3, 3])


def check_color_invariant_dynamics() -> None:
    # Deliberately use input colors outside each task's concrete palette.
    for task in (TRAIN_TASKS[0], TRAIN_TASKS[517], VAL_TASKS[0], VAL_TASKS[-1]):
        arbitrary = (Colors.BROWN, Colors.YELLOW)

        stage1_shapes = (
            (Tiles.BALL, Tiles.SQUARE),
            (Tiles.BALL, Tiles.PYRAMID),
            (Tiles.SQUARE, Tiles.PYRAMID),
        )[task.stage1_idx]
        produced_d = _apply_pair(
            task.ruleset.rules,
            stage1_shapes[0],
            arbitrary[0],
            stage1_shapes[1],
            arbitrary[1],
        )
        assert np.array_equal(produced_d, np.asarray(task.role_tiles[3]))

        catalyst_shape = (Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID)[task.stage2_idx]
        produced_e = _apply_pair(
            task.ruleset.rules,
            Tiles.HEX,
            arbitrary[1],
            catalyst_shape,
            arbitrary[0],
        )
        assert np.array_equal(produced_e, np.asarray(task.role_tiles[4]))

        catalyst_shape = (Tiles.BALL, Tiles.SQUARE, Tiles.PYRAMID)[task.stage3_idx]
        produced_key = _apply_pair(
            task.ruleset.rules,
            Tiles.DIAMOND,
            arbitrary[0],
            catalyst_shape,
            arbitrary[1],
        )
        assert np.array_equal(produced_key, np.asarray(BLUE_KEY))

        # Input order is symmetric.
        produced_key_swapped = _apply_pair(
            task.ruleset.rules,
            catalyst_shape,
            Colors.GREEN,
            Tiles.DIAMOND,
            Colors.RED,
        )
        assert np.array_equal(produced_key_swapped, np.asarray(BLUE_KEY))

    # The same shapes with different colors encode to exactly one abstract
    # rule; product color remains task-controlled.
    rule = ShapeTileNearRule(
        shape_a=Tiles.BALL,
        shape_b=Tiles.SQUARE,
        prod_tile=TILES_REGISTRY[Tiles.HEX, Colors.PINK],
    )
    encoding = np.asarray(rule.encode())
    assert encoding.tolist() == [
        12,
        Tiles.BALL,
        Tiles.SQUARE,
        Tiles.HEX,
        Colors.PINK,
        0,
        0,
    ]


def check_resets_and_registry() -> None:
    saw_agent_left = False
    saw_agent_right = False
    for split, task_indices in (("train", (1, 486, 972)), ("val", (1, 486, 972))):
        for task_index in task_indices:
            env, params, task = make_shape_env_and_params(split, task_index)
            for seed in (0, 17):
                timestep = env.reset(params, jax.random.PRNGKey(seed))
                grid = np.asarray(timestep.state.grid)
                agent_y, agent_x = map(int, timestep.state.agent.position)

                assert grid.shape == (8, 8, 2)
                assert timestep.observation["img"].shape == (8, 8, 3)
                assert timestep.observation["pocket"].shape == (2,)
                assert count_exact_tiles(grid, BLUE_LOCKED_DOOR) == 1
                assert count_exact_tiles(grid, YELLOW_STAR) == 1
                assert count_exact_tiles(grid, BLUE_KEY) == 0
                assert int((grid[..., 0] == Tiles.HEX).sum()) == 0
                assert int((grid[..., 0] == Tiles.DIAMOND).sum()) == 0
                assert int(np.isin(grid[..., 0], ROLE_SHAPES[:3]).sum()) == 4
                assert task.uid.startswith(f"{split}-")

                door_y, divider_x = np.argwhere(
                    np.all(grid == np.asarray(BLUE_LOCKED_DOOR), axis=-1)
                )[0]
                if agent_x < divider_x:
                    saw_agent_left = True
                    assert divider_x == 4
                else:
                    saw_agent_right = True
                    assert divider_x == 3
                assert 0 < door_y < params.height - 1
                assert 0 < agent_y < params.height - 1

    assert saw_agent_left and saw_agent_right

    env, params, task = make_shape_env_and_params("train", 1)
    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)
    timestep = reset_jit(params, jax.random.PRNGKey(123))
    timestep = step_jit(params, timestep, jnp.asarray(1))
    assert int(timestep.state.step_num) == 1
    assert env.render(params, timestep).shape == (256, 256, 3)

    registry = set(xminigrid.registered_environments())
    assert "XLand-MiniGrid-FiveObjectCrafting-8x8" in registry
    assert "XLand-MiniGrid-ExactCraft-8x8-v1" in registry
    assert "XLand-MiniGrid-ShapeCraft-8x8-v1" in registry

    registered_env, registered_params = xminigrid.make(
        "XLand-MiniGrid-ShapeCraft-8x8-v1",
        ruleset=task.ruleset,
    )
    registered_timestep = registered_env.reset(
        registered_params,
        jax.random.PRNGKey(99),
    )
    assert registered_timestep.observation["img"].shape == (8, 8, 3)


def main() -> None:
    check_split()
    check_color_invariant_dynamics()
    check_resets_and_registry()
    print(
        json.dumps(
            {
                "status": "ok",
                "environment": "XLand-MiniGrid-ShapeCraft-8x8-v1",
                "train_tasks": len(TRAIN_TASKS),
                "val_tasks": len(VAL_TASKS),
                "trees_per_split": 27,
                "palettes_per_split": 36,
                "colors_per_shape_per_split": 3,
                "shared_global_colors": len(COLOR_IDS),
                "shape_color_overlap": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
