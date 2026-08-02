#!/usr/bin/env python3
"""Smoke and invariant checks for the five-object crafting task family."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from xminigrid.core.actions import take_action
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.grid import room, vertical_line
from xminigrid.core.rules import check_rule
from xminigrid.types import AgentState
from xminigrid.envs.five_object_crafting import (
    BLUE_KEY,
    BLUE_LOCKED_DOOR,
    B,
    C,
    D,
    E,
    RULE_IDS,
    TASKS,
    TRAIN_TASKS,
    VAL_TASKS,
    YELLOW_STAR,
    make_env_and_params,
)


def count_exact_tiles(grid: np.ndarray, tile: jax.Array) -> int:
    return int(np.all(grid == np.asarray(tile), axis=-1).sum())


def check_split() -> None:
    assert len(TASKS) == 27
    assert len(TRAIN_TASKS) == 18
    assert len(VAL_TASKS) == 9
    train_rules = {rule for task in TRAIN_TASKS for rule in task.rule_ids}
    val_rules = {rule for task in VAL_TASKS for rule in task.rule_ids}
    assert train_rules == val_rules == set(RULE_IDS)


def check_resets() -> None:
    saw_agent_left = False
    saw_agent_right = False
    for task in TASKS:
        for seed in (0, 1, 17):
            env, params, _ = make_env_and_params(task.task_id)
            timestep = env.reset(params, jax.random.PRNGKey(seed))
            grid = np.asarray(timestep.state.grid)
            agent_y, agent_x = map(int, timestep.state.agent.position)

            assert count_exact_tiles(grid, BLUE_LOCKED_DOOR) == 1
            assert count_exact_tiles(grid, YELLOW_STAR) == 1
            assert count_exact_tiles(grid, BLUE_KEY) == 0
            assert count_exact_tiles(grid, D) == 0
            assert count_exact_tiles(grid, E) == 0
            assert np.all(np.asarray(timestep.state.rule_encoding)[:, 0] == 3)

            star_y, star_x = np.argwhere(
                np.all(grid == np.asarray(YELLOW_STAR), axis=-1)
            )[0]
            assert (params.height, params.width) == (8, 8)
            _, divider_x = np.argwhere(
                np.all(grid == np.asarray(BLUE_LOCKED_DOOR), axis=-1)
            )[0]
            if agent_x < divider_x:
                saw_agent_left = True
                assert divider_x == 4
                assert star_x > divider_x
                agent_room_columns = divider_x - 1
                goal_room_columns = params.width - divider_x - 2
            else:
                saw_agent_right = True
                assert divider_x == 3
                assert star_x < divider_x
                agent_room_columns = params.width - divider_x - 2
                goal_room_columns = divider_x - 1
            assert agent_room_columns == 3
            assert goal_room_columns == 2
            assert 0 < agent_y < params.height - 1
            assert 0 < star_y < params.height - 1

            observation = timestep.observation
            assert observation["img"].shape == (8, 8, 3)
            assert observation["pocket"].shape == (2,)
            agent_layer = np.asarray(observation["img"][..., 2])
            assert np.count_nonzero(agent_layer) == 1
            assert int(agent_layer[agent_y, agent_x]) == int(
                timestep.state.agent.direction
            ) + 1
            assert np.array_equal(
                np.asarray(observation["img"][..., :2]),
                grid,
            )

    assert saw_agent_left and saw_agent_right

    env, params, _ = make_env_and_params(2)
    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)
    timestep = reset_jit(params, jax.random.PRNGKey(123))
    timestep = step_jit(params, timestep, jnp.asarray(1))
    assert int(timestep.state.step_num) == 1

    # Changing the agent direction changes only the agent's own rendered tile,
    # proving that the map itself is not camera-rotated or FOV-shaded.
    render_a = env.render(params, timestep)
    rotated_agent = timestep.state.agent.replace(
        direction=(timestep.state.agent.direction + 1) % 4
    )
    rotated_timestep = timestep.replace(
        state=timestep.state.replace(agent=rotated_agent)
    )
    render_b = env.render(params, rotated_timestep)
    agent_y, agent_x = map(int, timestep.state.agent.position)
    tile_size = render_a.shape[0] // params.height
    outside_agent = np.ones(render_a.shape[:2], dtype=np.bool_)
    outside_agent[
        agent_y * tile_size : (agent_y + 1) * tile_size,
        agent_x * tile_size : (agent_x + 1) * tile_size,
    ] = False
    assert np.array_equal(render_a[outside_agent], render_b[outside_agent])
    assert render_a.shape == (256, 256, 3)


def check_crafting_chains() -> None:
    for task in TASKS:
        grid = room(7, 7)
        agent = AgentState(position=jnp.asarray((4, 4)), direction=jnp.asarray(0))
        position = jnp.asarray((3, 2))

        # Stage 1: base pair -> D.
        stage1_inputs = np.asarray(task.ruleset.init_tiles[:2])
        grid = grid.at[3, 2].set(stage1_inputs[0])
        grid = grid.at[3, 3].set(stage1_inputs[1])
        grid, agent = check_rule(task.ruleset.rules, grid, agent, 4, position)
        assert np.array_equal(np.asarray(grid[3, 3]), np.asarray(D))

        # Stage 2: D + selected catalyst -> E.
        grid = grid.at[3, 2].set(task.ruleset.init_tiles[2])
        grid, agent = check_rule(task.ruleset.rules, grid, agent, 4, position)
        assert np.array_equal(np.asarray(grid[3, 3]), np.asarray(E))

        # Stage 3: E + selected catalyst -> blue key.
        grid = grid.at[3, 2].set(task.ruleset.init_tiles[3])
        grid, agent = check_rule(task.ruleset.rules, grid, agent, 4, position)
        assert np.array_equal(np.asarray(grid[3, 3]), np.asarray(BLUE_KEY))


def check_blue_key_opens_only_blue_locked_door() -> None:
    grid = room(5, 5).at[2, 2].set(BLUE_LOCKED_DOOR)
    blue_key_agent = AgentState(
        position=jnp.asarray((2, 1)),
        direction=jnp.asarray(1),
        pocket=BLUE_KEY,
    )
    opened_grid, _, _ = take_action(grid, blue_key_agent, 5)
    assert int(opened_grid[2, 2, 0]) == Tiles.DOOR_OPEN
    assert int(opened_grid[2, 2, 1]) == Colors.BLUE

    red_key_agent = blue_key_agent.replace(
        pocket=TILES_REGISTRY[Tiles.KEY, Colors.RED]
    )
    still_locked_grid, _, _ = take_action(grid, red_key_agent, 5)
    assert np.array_equal(
        np.asarray(still_locked_grid[2, 2]),
        np.asarray(BLUE_LOCKED_DOOR),
    )


def check_visible_carry_mechanics() -> None:
    env, params, _ = make_env_and_params(2)
    timestep = env.reset(params, jax.random.PRNGKey(0))

    wall = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
    floor = TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK]
    empty = TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY]
    grid = room(8, 8)
    grid = vertical_line(grid, 4, 0, 8, wall)
    grid = grid.at[2, 4].set(BLUE_LOCKED_DOOR)
    grid = grid.at[3, 6].set(YELLOW_STAR)
    grid = grid.at[3, 2].set(C)
    grid = grid.at[5, 3].set(B)
    agent = AgentState(
        position=jnp.asarray((4, 2)),
        direction=jnp.asarray(0),
        pocket=empty,
    )
    timestep = timestep.replace(
        state=timestep.state.replace(grid=grid, agent=agent)
    )
    timestep = timestep.replace(observation=env._full_map_observation(timestep))

    # Pick up C directly in front: it leaves the grid but remains explicit in
    # both the symbolic pocket observation and the RGB frame.
    picked = env.step(params, timestep, jnp.asarray(3))
    assert np.array_equal(np.asarray(picked.state.agent.pocket), np.asarray(C))
    assert np.array_equal(np.asarray(picked.observation["pocket"]), np.asarray(C))
    assert np.array_equal(np.asarray(picked.state.grid[3, 2]), np.asarray(floor))

    carried_render = env.render(params, picked)
    hidden_agent = picked.state.agent.replace(pocket=empty)
    hidden_timestep = picked.replace(
        state=picked.state.replace(agent=hidden_agent)
    )
    hidden_render = env.render(params, hidden_timestep)
    agent_y, agent_x = map(int, picked.state.agent.position)
    tile_size = carried_render.shape[0] // params.height
    outside_agent = np.ones(carried_render.shape[:2], dtype=np.bool_)
    outside_agent[
        agent_y * tile_size : (agent_y + 1) * tile_size,
        agent_x * tile_size : (agent_x + 1) * tile_size,
    ] = False
    assert np.array_equal(
        carried_render[outside_agent],
        hidden_render[outside_agent],
    )
    assert not np.array_equal(
        carried_render[~outside_agent],
        hidden_render[~outside_agent],
    )

    # The white C pyramid follows the triangle tip when direction changes.
    agent_tile_up = carried_render[
        agent_y * tile_size : (agent_y + 1) * tile_size,
        agent_x * tile_size : (agent_x + 1) * tile_size,
    ]
    white_up = np.all(agent_tile_up > 220, axis=-1)
    assert np.where(white_up)[0].mean() < tile_size / 2

    turned = env.step(params, picked, jnp.asarray(1))
    turned_render = env.render(params, turned)
    agent_tile_right = turned_render[
        agent_y * tile_size : (agent_y + 1) * tile_size,
        agent_x * tile_size : (agent_x + 1) * tile_size,
    ]
    white_right = np.all(agent_tile_right > 220, axis=-1)
    assert np.where(white_right)[1].mean() > tile_size / 2

    # Move once, then verify the wall blocks the agent while carrying.
    moved = env.step(params, turned, jnp.asarray(0))
    assert tuple(map(int, moved.state.agent.position)) == (4, 3)
    blocked = env.step(params, moved, jnp.asarray(0))
    assert tuple(map(int, blocked.state.agent.position)) == (4, 3)
    assert np.array_equal(np.asarray(blocked.state.agent.pocket), np.asarray(C))

    # Facing occupied B: put_down must fail and C remains attached.
    facing_b = env.step(params, blocked, jnp.asarray(1))
    invalid_drop = env.step(params, facing_b, jnp.asarray(4))
    assert np.array_equal(
        np.asarray(invalid_drop.state.agent.pocket),
        np.asarray(C),
    )
    assert np.array_equal(np.asarray(invalid_drop.state.grid[5, 3]), np.asarray(B))

    # Turn toward a free adjacent floor tile and put C down successfully.
    facing_floor = env.step(params, invalid_drop, jnp.asarray(1))
    valid_drop = env.step(params, facing_floor, jnp.asarray(4))
    assert int(valid_drop.state.agent.pocket[0]) == Tiles.EMPTY
    assert np.array_equal(np.asarray(valid_drop.state.grid[4, 2]), np.asarray(C))


def main() -> None:
    check_split()
    check_resets()
    check_crafting_chains()
    check_blue_key_opens_only_blue_locked_door()
    check_visible_carry_mechanics()
    print(
        json.dumps(
            {
                "status": "ok",
                "tasks": len(TASKS),
                "train": len(TRAIN_TASKS),
                "val": len(VAL_TASKS),
                "primitive_rules_in_both_splits": list(RULE_IDS),
                "reset_seeds_checked_per_task": 3,
                "crafting_chains_checked": len(TASKS),
                "jit_reset_and_step": True,
                "map_shape": [8, 8],
                "observation_shape": {"img": [8, 8, 3], "pocket": [2]},
                "layout_mirroring": "random",
                "agent_room": "always_larger_3_columns",
                "goal_room": "always_smaller_2_columns",
                "static_full_map_render": True,
                "visible_attached_pocket_render": True,
                "carried_object_rotates_with_agent": True,
                "carrying_wall_collision": True,
                "occupied_drop_rejected": True,
                "free_adjacent_drop_succeeds": True,
                "blue_door_lock_check": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
