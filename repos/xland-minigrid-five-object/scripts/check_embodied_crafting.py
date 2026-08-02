#!/usr/bin/env python3
"""Transition, layout, rendering, and JIT checks for four embodiments."""

from __future__ import annotations

import json
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

import xminigrid
from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.grid import room
from xminigrid.envs.embodied_crafting import (
    EMBODIMENTS,
    make_embodied_env_and_params,
)
from xminigrid.envs.five_object_crafting import A, BLUE_LOCKED_DOOR
from xminigrid.types import AgentState


REGISTRY_IDS = {
    "standard": "XLand-MiniGrid-EmbodiedExactCraft-Standard-8x8-v1",
    "stride2": "XLand-MiniGrid-EmbodiedExactCraft-Stride2-8x8-v1",
    "omni8": "XLand-MiniGrid-EmbodiedExactCraft-Omni8-8x8-v1",
    "crab": "XLand-MiniGrid-EmbodiedExactCraft-Crab-8x8-v1",
}


def controlled_timestep(
    embodiment: str,
    grid,
    position=(4, 2),
    direction=0,
    pocket=None,
):
    env, params, task = make_embodied_env_and_params(embodiment, 2)
    timestep = env.reset(params, jax.random.PRNGKey(0))
    if pocket is None:
        pocket = TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY]
    agent = AgentState(
        position=jnp.asarray(position),
        direction=jnp.asarray(direction),
        pocket=pocket,
    )
    timestep = timestep.replace(
        state=timestep.state.replace(grid=grid, agent=agent)
    )
    timestep = timestep.replace(observation=env._full_map_observation(timestep))
    return env, params, task, timestep


def step_position(embodiment, grid, position, direction, action=0):
    env, params, _, timestep = controlled_timestep(
        embodiment,
        grid,
        position=position,
        direction=direction,
    )
    result = env.step(params, timestep, jnp.asarray(action))
    return tuple(map(int, result.state.agent.position)), result


def check_identical_layouts_and_splits() -> None:
    for task_id in (1, 2, 14, 27):
        for seed in (0, 9, 81):
            snapshots = []
            for embodiment in EMBODIMENTS:
                env, params, task = make_embodied_env_and_params(
                    embodiment,
                    task_id,
                )
                timestep = env.reset(params, jax.random.PRNGKey(seed))
                snapshots.append(
                    (
                        np.asarray(timestep.state.grid),
                        np.asarray(timestep.state.agent.position),
                        np.asarray(timestep.state.rule_encoding),
                        task.split,
                    )
                )
            reference = snapshots[0]
            for snapshot in snapshots[1:]:
                assert np.array_equal(snapshot[0], reference[0])
                assert np.array_equal(snapshot[1], reference[1])
                assert np.array_equal(snapshot[2], reference[2])
                assert snapshot[3] == reference[3]


def check_standard() -> None:
    grid = room(8, 8)
    position, _ = step_position("standard", grid, (4, 2), 0)
    assert position == (3, 2)


def check_stride2_adaptive() -> None:
    wall = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
    floor = TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK]

    clear_grid = room(8, 8)
    position, _ = step_position("stride2", clear_grid, (4, 2), 0)
    assert position == (2, 2)

    # Second microstep blocked: retain the valid first microstep.
    second_blocked = clear_grid.at[2, 2].set(wall)
    position, _ = step_position("stride2", second_blocked, (4, 2), 0)
    assert position == (3, 2)

    # First microstep blocked: no movement.
    first_blocked = clear_grid.at[3, 2].set(wall)
    position, _ = step_position("stride2", first_blocked, (4, 2), 0)
    assert position == (4, 2)

    # A locked door cannot be crossed; an open door is traversed as the first
    # microstep and the agent lands one cell inside the goal room.
    door_grid = room(8, 8)
    door_grid = door_grid.at[:, 4].set(wall)
    door_grid = door_grid.at[3, 4].set(BLUE_LOCKED_DOOR)
    position, _ = step_position("stride2", door_grid, (3, 3), 1)
    assert position == (3, 3)
    open_door = TILES_REGISTRY[Tiles.DOOR_OPEN, Colors.BLUE]
    door_grid = door_grid.at[3, 4].set(open_door).at[3, 5].set(floor)
    position, _ = step_position("stride2", door_grid, (3, 3), 1)
    assert position == (3, 5)


def check_omni8() -> None:
    wall = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
    clear_grid = room(8, 8)

    # Direction 1 is north-east.
    position, _ = step_position("omni8", clear_grid, (4, 2), 1)
    assert position == (3, 3)

    # A diagonal endpoint is not enough: blocked orthogonal clearance forbids
    # corner cutting.
    corner_blocked = clear_grid.at[3, 2].set(wall)
    position, _ = step_position("omni8", corner_blocked, (4, 2), 1)
    assert position == (4, 2)

    env, params, _, timestep = controlled_timestep(
        "omni8",
        clear_grid,
        direction=0,
    )
    right = env.step(params, timestep, jnp.asarray(1))
    assert int(right.state.agent.direction) == 1
    left = env.step(params, timestep, jnp.asarray(2))
    assert int(left.state.agent.direction) == 7


def check_crab() -> None:
    wall = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
    clear_grid = room(8, 8)

    # Facing north, the crab moves east and keeps facing north.
    position, result = step_position("crab", clear_grid, (4, 2), 0)
    assert position == (4, 3)
    assert int(result.state.agent.direction) == 0

    blocked = clear_grid.at[4, 3].set(wall)
    position, _ = step_position("crab", blocked, (4, 2), 0)
    assert position == (4, 2)


def check_interactions_are_preserved() -> None:
    floor = TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK]
    empty = TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY]
    for embodiment in EMBODIMENTS:
        grid = room(8, 8).at[3, 2].set(A)
        direction = 0
        env, params, _, timestep = controlled_timestep(
            embodiment,
            grid,
            position=(4, 2),
            direction=direction,
            pocket=empty,
        )
        picked = env.step(params, timestep, jnp.asarray(3))
        assert np.array_equal(np.asarray(picked.state.agent.pocket), np.asarray(A))
        assert np.array_equal(np.asarray(picked.state.grid[3, 2]), np.asarray(floor))

        put_grid = picked.state.grid.at[3, 2].set(floor)
        put_state = picked.state.replace(
            grid=put_grid,
            agent=picked.state.agent.replace(
                position=jnp.asarray((4, 2)),
                direction=jnp.asarray(direction),
            ),
        )
        put_timestep = picked.replace(state=put_state)
        put_timestep = put_timestep.replace(
            observation=env._full_map_observation(put_timestep)
        )
        placed = env.step(params, put_timestep, jnp.asarray(4))
        assert np.array_equal(np.asarray(placed.state.grid[3, 2]), np.asarray(A))
        assert int(placed.state.agent.pocket[0]) == Tiles.EMPTY


def reachable_positions(embodiment: str) -> set[tuple[int, int]]:
    """Exercise the real transition kernel on an empty agent-room-sized map."""

    grid = room(8, 8)
    env, _, _ = make_embodied_env_and_params(embodiment, 2)
    transition = jax.jit(env._take_action)
    start = (3, 2, 0)
    queue = deque([start])
    visited = {start}
    while queue:
        y, x, direction = queue.popleft()
        for action in (0, 1, 2):
            agent = AgentState(
                position=jnp.asarray((y, x)),
                direction=jnp.asarray(direction),
            )
            _, new_agent, _ = transition(
                grid,
                agent,
                jnp.asarray(action),
            )
            next_state = (
                int(new_agent.position[0]),
                int(new_agent.position[1]),
                int(new_agent.direction),
            )
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)
    return {(y, x) for y, x, _ in visited}


def check_empty_room_connectivity() -> None:
    expected = {
        (y, x)
        for y in range(1, 7)
        for x in range(1, 7)
    }
    for embodiment in EMBODIMENTS:
        assert reachable_positions(embodiment) == expected


def check_jit_registry_and_rendering() -> None:
    registry = set(xminigrid.registered_environments())
    rendered = {}
    for embodiment, registry_id in REGISTRY_IDS.items():
        assert registry_id in registry
        env, params, task = make_embodied_env_and_params(embodiment, 2)
        reset_jit = jax.jit(env.reset)
        step_jit = jax.jit(env.step)
        timestep = reset_jit(params, jax.random.PRNGKey(123))
        timestep = step_jit(params, timestep, jnp.asarray(1))
        frame = env.render(params, timestep)
        assert frame.shape == (256, 256, 3)
        rendered[embodiment] = frame

        registry_env, registry_params = xminigrid.make(
            registry_id,
            ruleset=task.ruleset,
        )
        registry_timestep = registry_env.reset(
            registry_params,
            jax.random.PRNGKey(7),
        )
        assert registry_timestep.observation["img"].shape == (8, 8, 3)

    assert not np.array_equal(rendered["standard"], rendered["stride2"])
    assert not np.array_equal(rendered["standard"], rendered["omni8"])
    assert not np.array_equal(rendered["standard"], rendered["crab"])


def main() -> None:
    check_identical_layouts_and_splits()
    check_standard()
    check_stride2_adaptive()
    check_omni8()
    check_crab()
    check_interactions_are_preserved()
    check_empty_room_connectivity()
    check_jit_registry_and_rendering()
    print(
        json.dumps(
            {
                "status": "ok",
                "embodiments": list(EMBODIMENTS),
                "layout_and_task_split_identical": True,
                "stride2_progressive_fallback": [2, 1, 0],
                "omni8_corner_cutting": False,
                "crab_motion": "one cell right of heading",
                "empty_room_position_graph_connected": True,
                "interaction_actions_preserved": True,
                "jit_reset_and_step": True,
                "render_shape": [256, 256, 3],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
