"""Four embodiment variants over the unchanged ExactCraft task family."""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from ..core.actions import (
    move_forward,
    pick_up,
    put_down,
    take_action,
    toggle,
    turn_clockwise,
    turn_counterclockwise,
)
from ..core.constants import DIRECTIONS, TILES_REGISTRY, Colors, Tiles
from ..core.goals import check_goal
from ..core.grid import check_can_put, check_pickable, check_walkable
from ..core.rules import check_rule
from ..rendering.rgb_render import render as rgb_render
from ..rendering.rgb_render import render_tile
from ..types import AgentState, EnvCarry, State, StepType, TimeStep
from .five_object_crafting import (
    CraftingTask,
    FiveObjectCraftingEnv,
    _dilate_mask,
    _resize_sprite_nearest,
    get_task,
)
from .easy_crafting import ExactCraftEasyTask, get_easy_task
from .xland import XLandEnvParams


EMBODIMENTS = ("standard", "stride2", "omni8", "crab")
EMBODIMENT_LABELS = {
    "standard": "Standard",
    "stride2": "Stride2-Adaptive",
    "omni8": "Omni8",
    "crab": "Crab",
}

OMNI_DIRECTIONS = jnp.asarray(
    (
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ),
    dtype=jnp.int32,
)


def _clipped_position(grid, position, delta):
    return jnp.clip(
        position + delta,
        min=jnp.asarray((0, 0)),
        max=jnp.asarray((grid.shape[0] - 1, grid.shape[1] - 1)),
    )


def _move_stride2_adaptive(grid, agent):
    """Try two unit microsteps and stop at the last valid cell."""

    delta = jax.lax.dynamic_index_in_dim(
        DIRECTIONS,
        agent.direction,
        keepdims=False,
    )
    position = agent.position
    for _ in range(2):
        candidate = _clipped_position(grid, position, delta)
        position = jax.lax.select(
            check_walkable(grid, candidate),
            candidate,
            position,
        )
    new_agent = agent.replace(position=position)
    return grid, new_agent, position


def _move_crab(grid, agent):
    """Move one cell to the agent's right without changing its heading."""

    sideways_direction = (agent.direction + 1) % 4
    delta = jax.lax.dynamic_index_in_dim(
        DIRECTIONS,
        sideways_direction,
        keepdims=False,
    )
    candidate = _clipped_position(grid, agent.position, delta)
    position = jax.lax.select(
        check_walkable(grid, candidate),
        candidate,
        agent.position,
    )
    new_agent = agent.replace(position=position)
    return grid, new_agent, position


def _omni_delta(direction):
    return jax.lax.dynamic_index_in_dim(
        OMNI_DIRECTIONS,
        direction,
        keepdims=False,
    )


def _move_omni8(grid, agent):
    delta = _omni_delta(agent.direction)
    candidate = _clipped_position(grid, agent.position, delta)
    endpoint_walkable = check_walkable(grid, candidate)

    # Diagonal motion may not cut through a blocked corner.
    diagonal = jnp.not_equal(delta[0], 0) & jnp.not_equal(delta[1], 0)
    orthogonal_y = _clipped_position(
        grid,
        agent.position,
        jnp.asarray((delta[0], 0)),
    )
    orthogonal_x = _clipped_position(
        grid,
        agent.position,
        jnp.asarray((0, delta[1])),
    )
    corner_clear = check_walkable(grid, orthogonal_y) & check_walkable(
        grid,
        orthogonal_x,
    )
    can_move = endpoint_walkable & jnp.where(diagonal, corner_clear, True)
    position = jax.lax.select(can_move, candidate, agent.position)
    new_agent = agent.replace(position=position)
    return grid, new_agent, position


def _turn_omni8(grid, agent, increment):
    new_agent = agent.replace(direction=(agent.direction + increment) % 8)
    return grid, new_agent, agent.position


def _omni_target(grid, agent):
    return _clipped_position(grid, agent.position, _omni_delta(agent.direction))


def _pick_up_omni8(grid, agent):
    target = _omni_target(grid, agent)
    is_pickable = check_pickable(grid, target)
    pocket_empty = jnp.equal(agent.pocket[0], Tiles.EMPTY)
    new_grid, new_agent = jax.lax.cond(
        is_pickable & pocket_empty,
        lambda: (
            grid.at[target[0], target[1]].set(
                TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK]
            ),
            agent.replace(pocket=grid[target[0], target[1]]),
        ),
        lambda: (grid, agent),
    )
    return new_grid, new_agent, target


def _put_down_omni8(grid, agent):
    target = _omni_target(grid, agent)
    can_put = check_can_put(grid, target)
    has_object = jnp.not_equal(agent.pocket[0], Tiles.EMPTY)
    new_grid, new_agent = jax.lax.cond(
        can_put & has_object,
        lambda: (
            grid.at[target[0], target[1]].set(agent.pocket),
            agent.replace(
                pocket=TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY]
            ),
        ),
        lambda: (grid, agent),
    )
    return new_grid, new_agent, target


def _toggle_omni8(grid, agent):
    target = _omni_target(grid, agent)
    target_tile = grid[target[0], target[1]]
    new_grid = jax.lax.select(
        jnp.equal(target_tile[0], Tiles.DOOR_LOCKED)
        & jnp.all(
            jnp.equal(
                agent.pocket,
                TILES_REGISTRY[Tiles.KEY, target_tile[1]],
            )
        ),
        grid.at[target[0], target[1]].set(
            TILES_REGISTRY[Tiles.DOOR_OPEN, target_tile[1]]
        ),
        grid,
    )
    new_grid = jax.lax.select(
        jnp.equal(target_tile[0], Tiles.DOOR_CLOSED),
        grid.at[target[0], target[1]].set(
            TILES_REGISTRY[Tiles.DOOR_OPEN, target_tile[1]]
        ),
        new_grid,
    )
    new_grid = jax.lax.select(
        jnp.equal(target_tile[0], Tiles.DOOR_OPEN),
        grid.at[target[0], target[1]].set(
            TILES_REGISTRY[Tiles.DOOR_CLOSED, target_tile[1]]
        ),
        new_grid,
    )
    return new_grid, agent, target


def _take_omni8_action(grid, agent, action):
    actions: tuple[Callable, ...] = (
        lambda: _move_omni8(grid, agent),
        lambda: _turn_omni8(grid, agent, 1),
        lambda: _turn_omni8(grid, agent, -1),
        lambda: _pick_up_omni8(grid, agent),
        lambda: _put_down_omni8(grid, agent),
        lambda: _toggle_omni8(grid, agent),
    )
    return jax.lax.switch(action, actions)


def _take_stride_action(grid, agent, action):
    actions: tuple[Callable, ...] = (
        lambda: _move_stride2_adaptive(grid, agent),
        lambda: turn_clockwise(grid, agent),
        lambda: turn_counterclockwise(grid, agent),
        lambda: pick_up(grid, agent),
        lambda: put_down(grid, agent),
        lambda: toggle(grid, agent),
    )
    return jax.lax.switch(action, actions)


def _take_crab_action(grid, agent, action):
    actions: tuple[Callable, ...] = (
        lambda: _move_crab(grid, agent),
        lambda: turn_clockwise(grid, agent),
        lambda: turn_counterclockwise(grid, agent),
        lambda: pick_up(grid, agent),
        lambda: put_down(grid, agent),
        lambda: toggle(grid, agent),
    )
    return jax.lax.switch(action, actions)


def _object_sprite(pocket: np.ndarray, tile_size: int, angle: float):
    floor = np.asarray(TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK])
    floor_image = np.asarray(render_tile(
        tile=tuple(floor.tolist()),
        agent_direction=None,
        highlight=False,
        tile_size=tile_size,
    ), dtype=np.uint8)
    object_image = np.asarray(render_tile(
        tile=tuple(pocket.tolist()),
        agent_direction=None,
        highlight=False,
        tile_size=tile_size,
    ), dtype=np.uint8)
    object_mask = np.any(object_image != floor_image, axis=-1)
    rows, cols = np.where(object_mask)
    if len(rows) == 0:
        return None, None
    object_image = object_image[
        rows.min() : rows.max() + 1,
        cols.min() : cols.max() + 1,
    ]
    object_mask = object_mask[
        rows.min() : rows.max() + 1,
        cols.min() : cols.max() + 1,
    ]
    size = max(8, int(round(tile_size * 0.38)))
    object_image, object_mask = _resize_sprite_nearest(
        object_image,
        object_mask,
        size,
    )
    rgba = Image.fromarray(object_image).convert("RGBA")
    alpha = Image.fromarray((object_mask * 255).astype(np.uint8))
    rgba.putalpha(alpha)
    rgba = rgba.rotate(-angle, resample=Image.Resampling.NEAREST, expand=True)
    return rgba, np.asarray(rgba.getchannel("A")) > 0


def _draw_embodiment_sprite(
    grid_tile: np.ndarray,
    pocket: np.ndarray,
    direction: int,
    direction_count: int,
    embodiment: str,
    tile_size: int,
) -> np.ndarray:
    base = np.asarray(render_tile(
        tile=tuple(grid_tile.tolist()),
        agent_direction=None,
        highlight=False,
        tile_size=tile_size,
    ), dtype=np.uint8).copy()
    image = Image.fromarray(base).convert("RGBA")
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    scale = tile_size / 32.0
    center = tile_size / 2.0

    if embodiment == "stride2":
        color = (0, 190, 235, 255)
        outline = (5, 40, 55, 255)
        points = (
            (center, 3 * scale),
            (27 * scale, 18 * scale),
            (20 * scale, 18 * scale),
            (20 * scale, 28 * scale),
            (12 * scale, 28 * scale),
            (12 * scale, 18 * scale),
            (5 * scale, 18 * scale),
        )
        draw.polygon(points, fill=color, outline=outline, width=max(1, int(scale)))
        draw.line(
            (center, 6 * scale, center, 25 * scale),
            fill=(255, 255, 255, 255),
            width=max(1, int(2 * scale)),
        )
    elif embodiment == "omni8":
        color = (142, 68, 220, 255)
        draw.ellipse(
            (5 * scale, 5 * scale, 27 * scale, 27 * scale),
            fill=color,
            outline=(35, 12, 60, 255),
            width=max(1, int(2 * scale)),
        )
        draw.polygon(
            (
                (center, 3 * scale),
                (11 * scale, 14 * scale),
                (21 * scale, 14 * scale),
            ),
            fill=(255, 255, 255, 255),
        )
    elif embodiment == "crab":
        color = (15, 175, 125, 255)
        draw.ellipse(
            (6 * scale, 9 * scale, 26 * scale, 25 * scale),
            fill=color,
            outline=(5, 55, 40, 255),
            width=max(1, int(2 * scale)),
        )
        draw.ellipse((2 * scale, 7 * scale, 10 * scale, 15 * scale), fill=color)
        draw.ellipse((22 * scale, 7 * scale, 30 * scale, 15 * scale), fill=color)
        draw.polygon(
            (
                (center, 3 * scale),
                (12 * scale, 11 * scale),
                (20 * scale, 11 * scale),
            ),
            fill=(255, 225, 70, 255),
        )
    else:
        raise ValueError(f"Unknown custom embodiment: {embodiment}")

    angle = 360.0 * direction / direction_count
    layer = layer.rotate(
        -angle,
        resample=Image.Resampling.BICUBIC,
        center=(center, center),
    )
    image.alpha_composite(layer)

    if int(pocket[0]) != Tiles.EMPTY:
        object_rgba, object_mask = _object_sprite(pocket, tile_size, angle)
        if object_rgba is not None:
            theta = np.deg2rad(angle - 90.0)
            offset = tile_size * 0.27
            center_x = center + np.cos(theta) * offset
            center_y = center + np.sin(theta) * offset
            left = int(round(center_x - object_rgba.width / 2))
            top = int(round(center_y - object_rgba.height / 2))
            left = int(np.clip(left, 0, tile_size - object_rgba.width))
            top = int(np.clip(top, 0, tile_size - object_rgba.height))
            object_alpha = object_mask
            outline_mask = _dilate_mask(object_alpha) & ~object_alpha
            outline = Image.new("RGBA", object_rgba.size, (0, 0, 0, 0))
            outline.putalpha(
                Image.fromarray((outline_mask * 255).astype(np.uint8))
            )
            image.alpha_composite(outline, (left, top))
            image.alpha_composite(object_rgba, (left, top))
    return np.asarray(image.convert("RGB"))


class EmbodiedExactCraftEnv(FiveObjectCraftingEnv):
    """ExactCraft with an embodiment-specific action transition function."""

    embodiment: str = "standard"
    direction_count: int = 4
    difficulty: str = "medium"

    def default_params(self, **kwargs) -> XLandEnvParams:
        grid_type = (
            f"EMBODIED_EXACT_CRAFT_{self.embodiment.upper()}_"
            f"{self.difficulty.upper()}_V1"
        )
        params = super().default_params().replace(grid_type=grid_type)
        if self.difficulty == "easy":
            params = params.replace(max_steps=192)
        return params.replace(**kwargs)

    def _generate_problem(
        self,
        params: XLandEnvParams,
        key: jax.Array,
    ) -> State[EnvCarry]:
        state = super()._generate_problem(params, key)
        if self.direction_count == 8:
            direction = jax.random.randint(
                jax.random.fold_in(key, 818),
                shape=(),
                minval=0,
                maxval=8,
            )
            state = state.replace(
                agent=state.agent.replace(direction=direction)
            )
        return state

    def _take_action(self, grid, agent, action):
        if self.embodiment == "standard":
            return take_action(grid, agent, action)
        if self.embodiment == "stride2":
            return _take_stride_action(grid, agent, action)
        if self.embodiment == "omni8":
            return _take_omni8_action(grid, agent, action)
        if self.embodiment == "crab":
            return _take_crab_action(grid, agent, action)
        raise ValueError(f"Unknown embodiment: {self.embodiment}")

    def step(
        self,
        params: XLandEnvParams,
        timestep: TimeStep,
        action,
    ) -> TimeStep:
        new_grid, new_agent, changed_position = self._take_action(
            timestep.state.grid,
            timestep.state.agent,
            action,
        )
        new_grid, new_agent = check_rule(
            timestep.state.rule_encoding,
            new_grid,
            new_agent,
            action,
            changed_position,
        )
        new_state = timestep.state.replace(
            grid=new_grid,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
        )
        terminated = check_goal(
            new_state.goal_encoding,
            new_state.grid,
            new_state.agent,
            action,
            changed_position,
        )
        assert params.max_steps is not None
        truncated = jnp.equal(new_state.step_num, params.max_steps)
        reward = jax.lax.select(
            terminated,
            1.0 - 0.9 * (new_state.step_num / params.max_steps),
            0.0,
        )
        step_type = jax.lax.select(
            terminated | truncated,
            StepType.LAST,
            StepType.MID,
        )
        discount = jax.lax.select(
            terminated,
            jnp.asarray(0.0),
            jnp.asarray(1.0),
        )
        result = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation={},
        )
        return result.replace(observation=self._full_map_observation(result))

    def render(self, params: XLandEnvParams, timestep: TimeStep):
        if self.embodiment == "standard":
            return super().render(params, timestep)
        if params.render_mode != "rgb_array":
            return super().render(params, timestep)

        state = timestep.state
        grid = np.asarray(state.grid)
        image = rgb_render(grid, agent=None, view_size=params.view_size)
        agent_y, agent_x = map(int, state.agent.position)
        tile_size = int(image.shape[0] // params.height)
        agent_tile = _draw_embodiment_sprite(
            grid_tile=grid[agent_y, agent_x],
            pocket=np.asarray(state.agent.pocket),
            direction=int(state.agent.direction),
            direction_count=self.direction_count,
            embodiment=self.embodiment,
            tile_size=tile_size,
        )
        image[
            agent_y * tile_size : (agent_y + 1) * tile_size,
            agent_x * tile_size : (agent_x + 1) * tile_size,
        ] = agent_tile
        return image


class StandardEmbodiedCraftEnv(EmbodiedExactCraftEnv):
    embodiment = "standard"
    direction_count = 4
    difficulty = "medium"


class Stride2EmbodiedCraftEnv(EmbodiedExactCraftEnv):
    embodiment = "stride2"
    direction_count = 4
    difficulty = "medium"


class Omni8EmbodiedCraftEnv(EmbodiedExactCraftEnv):
    embodiment = "omni8"
    direction_count = 8
    difficulty = "medium"


class CrabEmbodiedCraftEnv(EmbodiedExactCraftEnv):
    embodiment = "crab"
    direction_count = 4
    difficulty = "medium"


class StandardEmbodiedCraftEasyEnv(StandardEmbodiedCraftEnv):
    difficulty = "easy"


class Stride2EmbodiedCraftEasyEnv(Stride2EmbodiedCraftEnv):
    difficulty = "easy"


class Omni8EmbodiedCraftEasyEnv(Omni8EmbodiedCraftEnv):
    difficulty = "easy"


class CrabEmbodiedCraftEasyEnv(CrabEmbodiedCraftEnv):
    difficulty = "easy"


ENV_CLASSES = {
    "standard": StandardEmbodiedCraftEnv,
    "stride2": Stride2EmbodiedCraftEnv,
    "omni8": Omni8EmbodiedCraftEnv,
    "crab": CrabEmbodiedCraftEnv,
}
EASY_ENV_CLASSES = {
    "standard": StandardEmbodiedCraftEasyEnv,
    "stride2": Stride2EmbodiedCraftEasyEnv,
    "omni8": Omni8EmbodiedCraftEasyEnv,
    "crab": CrabEmbodiedCraftEasyEnv,
}


def make_embodied_env_and_params(
    embodiment: str,
    task_id: int,
    difficulty: str = "medium",
    **param_overrides,
) -> tuple[
    EmbodiedExactCraftEnv,
    XLandEnvParams,
    CraftingTask | ExactCraftEasyTask,
]:
    if embodiment not in ENV_CLASSES:
        raise ValueError(
            f"embodiment must be one of {EMBODIMENTS}, got {embodiment!r}"
        )
    if difficulty == "medium":
        task = get_task(task_id)
        env = ENV_CLASSES[embodiment]()
    elif difficulty == "easy":
        task = get_easy_task(task_id)
        env = EASY_ENV_CLASSES[embodiment]()
    else:
        raise ValueError(
            f"difficulty must be 'easy' or 'medium', got {difficulty!r}"
        )
    params = env.default_params(ruleset=task.ruleset, **param_overrides)
    return env, params, task


__all__ = [
    "EMBODIMENTS",
    "EMBODIMENT_LABELS",
    "EmbodiedExactCraftEnv",
    "StandardEmbodiedCraftEnv",
    "Stride2EmbodiedCraftEnv",
    "Omni8EmbodiedCraftEnv",
    "CrabEmbodiedCraftEnv",
    "StandardEmbodiedCraftEasyEnv",
    "Stride2EmbodiedCraftEasyEnv",
    "Omni8EmbodiedCraftEasyEnv",
    "CrabEmbodiedCraftEasyEnv",
    "make_embodied_env_and_params",
]
