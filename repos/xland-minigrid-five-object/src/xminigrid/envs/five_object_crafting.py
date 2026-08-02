"""Two-room five-object TileNear crafting tasks for XLand-MiniGrid.

The task family has nine shared primitive rules and 27 three-stage task trees.
All primitive rules occur in both splits; validation only holds out complete
rule combinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

from ..core.constants import TILES_REGISTRY, Colors, Tiles
from ..core.goals import AgentNearGoal
from ..core.grid import room, sample_coordinates, sample_direction, vertical_line
from ..core.rules import TileNearRule
from ..rendering.rgb_render import render as rgb_render
from ..rendering.rgb_render import render_tile
from ..types import AgentState, EnvCarry, RuleSet, State, TimeStep
from .xland import XLandEnvParams, XLandMiniGrid


# Five ordinary object identities.
A = TILES_REGISTRY[Tiles.BALL, Colors.RED]
B = TILES_REGISTRY[Tiles.SQUARE, Colors.GREEN]
C = TILES_REGISTRY[Tiles.PYRAMID, Colors.WHITE]
D = TILES_REGISTRY[Tiles.HEX, Colors.PURPLE]
E = TILES_REGISTRY[Tiles.BALL, Colors.ORANGE]

# Special objects, excluded from the five-object vocabulary.
BLUE_KEY = TILES_REGISTRY[Tiles.KEY, Colors.BLUE]
YELLOW_STAR = TILES_REGISTRY[Tiles.STAR, Colors.YELLOW]
BLUE_LOCKED_DOOR = TILES_REGISTRY[Tiles.DOOR_LOCKED, Colors.BLUE]

OBJECT_SYMBOLS = ("A", "B", "C", "D", "E")
RULE_IDS = tuple(f"R{idx}" for idx in range(1, 10))


def _resize_sprite_nearest(
    image: np.ndarray,
    mask: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_indices = np.linspace(0, image.shape[0] - 1, size).astype(np.int32)
    x_indices = np.linspace(0, image.shape[1] - 1, size).astype(np.int32)
    return (
        image[np.ix_(y_indices, x_indices)],
        mask[np.ix_(y_indices, x_indices)],
    )


def _dilate_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask)
    for y_offset in range(3):
        for x_offset in range(3):
            dilated |= padded[
                y_offset : y_offset + mask.shape[0],
                x_offset : x_offset + mask.shape[1],
            ]
    return dilated


def _render_agent_with_pocket(
    grid_tile: np.ndarray,
    pocket: np.ndarray,
    direction: int,
    tile_size: int,
) -> np.ndarray:
    """Render a carried object at the triangle tip, inside the agent cell."""

    agent_tile = render_tile(
        tile=tuple(grid_tile.tolist()),
        agent_direction=direction,
        highlight=False,
        tile_size=tile_size,
    ).copy()
    if int(pocket[0]) == Tiles.EMPTY:
        return agent_tile

    floor = np.asarray(TILES_REGISTRY[Tiles.FLOOR, Colors.BLACK])
    floor_image = render_tile(
        tile=tuple(floor.tolist()),
        agent_direction=None,
        highlight=False,
        tile_size=tile_size,
    )
    object_image = render_tile(
        tile=tuple(pocket.tolist()),
        agent_direction=None,
        highlight=False,
        tile_size=tile_size,
    )
    object_mask = np.any(object_image != floor_image, axis=-1)

    # Base sprites face up. Rotate clockwise with agent direction.
    rotation = (-direction) % 4
    object_image = np.rot90(object_image, k=rotation)
    object_mask = np.rot90(object_mask, k=rotation)
    object_rows, object_cols = np.where(object_mask)
    if len(object_rows) == 0:
        return agent_tile

    object_image = object_image[
        object_rows.min() : object_rows.max() + 1,
        object_cols.min() : object_cols.max() + 1,
    ]
    object_mask = object_mask[
        object_rows.min() : object_rows.max() + 1,
        object_cols.min() : object_cols.max() + 1,
    ]
    sprite_size = max(8, int(round(tile_size * 0.40)))
    object_image, object_mask = _resize_sprite_nearest(
        object_image,
        object_mask,
        sprite_size,
    )

    center = tile_size // 2
    front_offset = int(round(tile_size * 0.27))
    centers = (
        (center - front_offset, center),
        (center, center + front_offset),
        (center + front_offset, center),
        (center, center - front_offset),
    )
    center_y, center_x = centers[direction]
    top = int(np.clip(center_y - sprite_size // 2, 0, tile_size - sprite_size))
    left = int(np.clip(center_x - sprite_size // 2, 0, tile_size - sprite_size))
    target = agent_tile[top : top + sprite_size, left : left + sprite_size]

    outline = _dilate_mask(object_mask) & ~object_mask
    target[outline] = np.asarray((10, 10, 10), dtype=np.uint8)
    target[object_mask] = object_image[object_mask]
    return agent_tile


STAGE1_RULES = (
    TileNearRule(tile_a=A, tile_b=B, prod_tile=D),  # R1
    TileNearRule(tile_a=A, tile_b=C, prod_tile=D),  # R2
    TileNearRule(tile_a=B, tile_b=C, prod_tile=D),  # R3
)
STAGE2_RULES = (
    TileNearRule(tile_a=D, tile_b=A, prod_tile=E),  # R4
    TileNearRule(tile_a=D, tile_b=B, prod_tile=E),  # R5
    TileNearRule(tile_a=D, tile_b=C, prod_tile=E),  # R6
)
STAGE3_RULES = (
    TileNearRule(tile_a=E, tile_b=A, prod_tile=BLUE_KEY),  # R7
    TileNearRule(tile_a=E, tile_b=B, prod_tile=BLUE_KEY),  # R8
    TileNearRule(tile_a=E, tile_b=C, prod_tile=BLUE_KEY),  # R9
)

STAGE1_INPUTS = ((A, B), (A, C), (B, C))
STAGE2_CATALYSTS = (A, B, C)
STAGE3_CATALYSTS = (A, B, C)

RULE_TEXT = {
    "R1": "A+B->D",
    "R2": "A+C->D",
    "R3": "B+C->D",
    "R4": "D+A->E",
    "R5": "D+B->E",
    "R6": "D+C->E",
    "R7": "E+A->K",
    "R8": "E+B->K",
    "R9": "E+C->K",
}


@dataclass(frozen=True)
class CraftingTask:
    task_id: int
    split: str
    stage1_idx: int
    stage2_idx: int
    stage3_idx: int
    rule_ids: tuple[str, str, str]
    initial_symbols: tuple[str, str, str, str]
    ruleset: RuleSet

    @property
    def rule_text(self) -> tuple[str, str, str]:
        return tuple(RULE_TEXT[rule_id] for rule_id in self.rule_ids)


def _symbol_for_base_index(index: int) -> str:
    return ("A", "B", "C")[index]


def _make_tasks() -> tuple[CraftingTask, ...]:
    tasks = []
    for task_id, (stage1_idx, stage2_idx, stage3_idx) in enumerate(
        product(range(3), repeat=3),
        start=1,
    ):
        stage1_rule = STAGE1_RULES[stage1_idx]
        stage2_rule = STAGE2_RULES[stage2_idx]
        stage3_rule = STAGE3_RULES[stage3_idx]

        # Reverse-topological evaluation prevents two crafting stages from
        # firing during the same put-down action.
        encoded_rules = jnp.stack(
            (
                stage3_rule.encode(),
                stage2_rule.encode(),
                stage1_rule.encode(),
            )
        )
        initial_tiles = jnp.stack(
            (
                STAGE1_INPUTS[stage1_idx][0],
                STAGE1_INPUTS[stage1_idx][1],
                STAGE2_CATALYSTS[stage2_idx],
                STAGE3_CATALYSTS[stage3_idx],
            )
        )
        split = "val" if stage3_idx == (stage1_idx + stage2_idx) % 3 else "train"
        ruleset = RuleSet(
            goal=AgentNearGoal(tile=YELLOW_STAR).encode(),
            rules=encoded_rules,
            init_tiles=initial_tiles,
        )
        stage1_symbols = (("A", "B"), ("A", "C"), ("B", "C"))[stage1_idx]
        tasks.append(
            CraftingTask(
                task_id=task_id,
                split=split,
                stage1_idx=stage1_idx,
                stage2_idx=stage2_idx,
                stage3_idx=stage3_idx,
                rule_ids=(
                    f"R{stage1_idx + 1}",
                    f"R{stage2_idx + 4}",
                    f"R{stage3_idx + 7}",
                ),
                initial_symbols=(
                    stage1_symbols[0],
                    stage1_symbols[1],
                    _symbol_for_base_index(stage2_idx),
                    _symbol_for_base_index(stage3_idx),
                ),
                ruleset=ruleset,
            )
        )

    task_tuple = tuple(tasks)
    train_tasks = tuple(task for task in task_tuple if task.split == "train")
    val_tasks = tuple(task for task in task_tuple if task.split == "val")
    train_rules = {rule for task in train_tasks for rule in task.rule_ids}
    val_rules = {rule for task in val_tasks for rule in task.rule_ids}
    assert len(task_tuple) == 27
    assert len(train_tasks) == 18
    assert len(val_tasks) == 9
    assert train_rules == val_rules == set(RULE_IDS)
    return task_tuple


TASKS = _make_tasks()
TRAIN_TASKS = tuple(task for task in TASKS if task.split == "train")
VAL_TASKS = tuple(task for task in TASKS if task.split == "val")


def get_task(task_id: int) -> CraftingTask:
    if not 1 <= task_id <= len(TASKS):
        raise ValueError(f"task_id must be in [1, {len(TASKS)}], got {task_id}")
    return TASKS[task_id - 1]


class FiveObjectCraftingEnv(XLandMiniGrid):
    """Two rooms separated by one blue locked door.

    Agent and four ingredient instances are sampled in one room. A yellow star
    is sampled in the other. The active three-rule task must produce the blue
    key before the agent can open the door and reach the star.
    """

    def default_params(self, **kwargs) -> XLandEnvParams:
        params = XLandEnvParams(
            height=8,
            width=8,
            view_size=8,
            max_steps=256,
            grid_type="FIVE_OBJECT_CRAFTING",
        )
        return params.replace(**kwargs)

    def observation_shape(self, params: XLandEnvParams) -> dict[str, tuple[int, ...]]:
        return {
            "img": (params.height, params.width, 3),
            "pocket": (2,),
        }

    @staticmethod
    def _full_map_observation(timestep: TimeStep) -> dict[str, jax.Array]:
        """Return the complete map in fixed world coordinates.

        Channels 0–1 contain tile and color IDs. Channel 2 is zero except at
        the agent position, where it stores direction + 1. Pocket state remains
        separate so no task-relevant information is dropped.
        """

        state = timestep.state
        agent_layer = jnp.zeros(state.grid.shape[:2], dtype=jnp.uint8)
        agent_layer = agent_layer.at[
            state.agent.position[0],
            state.agent.position[1],
        ].set(state.agent.direction.astype(jnp.uint8) + 1)
        return {
            "img": jnp.concatenate((state.grid, agent_layer[..., None]), axis=-1),
            "pocket": state.agent.pocket,
        }

    def reset(self, params: XLandEnvParams, key: jax.Array) -> TimeStep:
        timestep = super().reset(params, key)
        return timestep.replace(observation=self._full_map_observation(timestep))

    def step(self, params: XLandEnvParams, timestep: TimeStep, action) -> TimeStep:
        timestep = super().step(params, timestep, action)
        return timestep.replace(observation=self._full_map_observation(timestep))

    def render(self, params: XLandEnvParams, timestep: TimeStep):
        """Render a fixed-orientation full map without field-of-view shading."""

        if params.render_mode != "rgb_array":
            return super().render(params, timestep)

        state = timestep.state
        grid = np.asarray(state.grid)
        image = rgb_render(grid, agent=None, view_size=params.view_size)
        agent_y, agent_x = map(int, state.agent.position)
        tile_size = int(image.shape[0] // params.height)
        agent_tile = _render_agent_with_pocket(
            grid_tile=grid[agent_y, agent_x],
            pocket=np.asarray(state.agent.pocket),
            direction=int(state.agent.direction),
            tile_size=tile_size,
        )
        image[
            agent_y * tile_size : (agent_y + 1) * tile_size,
            agent_x * tile_size : (agent_x + 1) * tile_size,
        ] = agent_tile
        return image

    def _generate_problem(self, params: XLandEnvParams, key: jax.Array) -> State[EnvCarry]:
        if params.width < 8 or params.height < 8:
            raise ValueError("Use width and height >= 8.")

        (
            next_key,
            door_key,
            side_key,
            objects_key,
            star_key,
            direction_key,
        ) = jax.random.split(key, num=6)

        agent_on_left = jax.random.bernoulli(side_key)
        # On an even-width map a fixed one-tile divider makes the rooms
        # asymmetric. Shift the divider during mirroring so the agent always
        # gets three interior columns and the star room gets two.
        divider_x = jnp.where(
            agent_on_left,
            params.width // 2,
            params.width // 2 - 1,
        )
        door_y = jax.random.randint(door_key, shape=(), minval=1, maxval=params.height - 1)
        grid = room(params.height, params.width)
        grid = vertical_line(
            grid,
            divider_x,
            0,
            params.height,
            TILES_REGISTRY[Tiles.WALL, Colors.GREY],
        )
        grid = grid.at[door_y, divider_x].set(BLUE_LOCKED_DOOR)

        x_coords = jnp.broadcast_to(
            jnp.arange(params.width)[None, :],
            (params.height, params.width),
        )
        left_mask = x_coords < divider_x
        right_mask = x_coords > divider_x
        agent_room_mask = jnp.where(agent_on_left, left_mask, right_mask)
        goal_room_mask = jnp.where(agent_on_left, right_mask, left_mask)

        num_objects = len(params.ruleset.init_tiles)
        object_and_agent_positions = sample_coordinates(
            objects_key,
            grid,
            num=num_objects + 1,
            mask=agent_room_mask,
        )
        star_position = sample_coordinates(
            star_key,
            grid,
            num=1,
            mask=goal_room_mask,
        )[0]

        for idx in range(num_objects):
            position = object_and_agent_positions[idx]
            grid = grid.at[position[0], position[1]].set(params.ruleset.init_tiles[idx])
        grid = grid.at[star_position[0], star_position[1]].set(YELLOW_STAR)

        agent = AgentState(
            position=object_and_agent_positions[-1],
            direction=sample_direction(direction_key),
        )
        return State(
            key=next_key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=params.ruleset.goal,
            rule_encoding=params.ruleset.rules,
            carry=EnvCarry(),
        )


def make_env_and_params(
    task_id: int,
    **param_overrides,
) -> tuple[FiveObjectCraftingEnv, XLandEnvParams, CraftingTask]:
    task = get_task(task_id)
    env = FiveObjectCraftingEnv()
    params = env.default_params(ruleset=task.ruleset, **param_overrides)
    return env, params, task


# Stable name for the exact color+shape dynamics.  Keep the original class
# name as a backwards-compatible alias for existing scripts/checkpoints.
ExactCraftEnv = FiveObjectCraftingEnv


__all__ = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "BLUE_KEY",
    "YELLOW_STAR",
    "BLUE_LOCKED_DOOR",
    "RULE_TEXT",
    "TASKS",
    "TRAIN_TASKS",
    "VAL_TASKS",
    "CraftingTask",
    "FiveObjectCraftingEnv",
    "ExactCraftEnv",
    "get_task",
    "make_env_and_params",
]
