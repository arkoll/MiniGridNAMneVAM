#!/usr/bin/env python3
"""Run and render one deterministic successful T02 crafting episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.grid import room, vertical_line
from xminigrid.types import AgentState
from xminigrid.envs.five_object_crafting import (
    A,
    B,
    D,
    E,
    BLUE_KEY,
    BLUE_LOCKED_DOOR,
    YELLOW_STAR,
    make_env_and_params,
)


ACTION_NAMES = {
    0: "FORWARD",
    1: "TURN RIGHT",
    2: "TURN LEFT",
    3: "PICK_UP",
    4: "PUT_DOWN",
    5: "TOGGLE",
}
DIRECTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")

# T02: R1 A+B->D, R4 D+A->E, R8 E+B->K.
SCRIPTED_ACTIONS = (
    3,  # pick A1
    1,
    0,
    2,
    4,  # craft D
    0,
    3,  # pick D
    2,
    0,
    1,
    0,
    4,  # craft E
    0,
    3,  # pick E
    1,
    0,
    2,
    4,  # craft blue key
    0,
    1,
    3,  # pick blue key
    0,
    1,
    0,
    2,
    5,  # unlock blue door
    0,
    0,
    0,  # try to enter star tile -> AgentNearGoal success
)

EVENTS = {
    0: ("EPISODE START", "#52606d"),
    1: ("PICK A", "#7c3aed"),
    5: ("CRAFT D", "#0f766e"),
    7: ("PICK D", "#7c3aed"),
    12: ("CRAFT E", "#0f766e"),
    14: ("PICK E", "#7c3aed"),
    18: ("CRAFT BLUE KEY", "#0f766e"),
    21: ("PICK BLUE KEY", "#7c3aed"),
    26: ("OPEN BLUE DOOR", "#2563eb"),
    29: ("SUCCESS = 1", "#15803d"),
}

COLS = 6
ROWS = 5
PANEL_WIDTH = 365
PANEL_HEIGHT = 465
HEADER_HEIGHT = 250
MARGIN = 45


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def exact_count(grid: np.ndarray, tile) -> int:
    return int(np.all(grid == np.asarray(tile), axis=-1).sum())


def pocket_name(pocket: np.ndarray) -> str:
    for tile, name in (
        (A, "A"),
        (B, "B"),
        (D, "D"),
        (E, "E"),
        (BLUE_KEY, "BLUE KEY"),
    ):
        if np.array_equal(pocket, np.asarray(tile)):
            return name
    if int(pocket[0]) == Tiles.EMPTY:
        return "EMPTY"
    return f"T{int(pocket[0])}/C{int(pocket[1])}"


def deterministic_initial_state():
    env, params, task = make_env_and_params(2)
    timestep = env.reset(params, jax.random.PRNGKey(2026))
    wall = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
    empty = TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY]

    grid = room(8, 8)
    grid = vertical_line(grid, 4, 0, 8, wall)
    grid = grid.at[3, 4].set(BLUE_LOCKED_DOOR)
    grid = grid.at[3, 6].set(YELLOW_STAR)
    grid = grid.at[5, 1].set(A)  # first-stage A
    grid = grid.at[4, 2].set(B)  # first-stage B
    grid = grid.at[2, 1].set(A)  # second-stage catalyst
    grid = grid.at[2, 3].set(B)  # third-stage catalyst
    agent = AgentState(
        position=jnp.asarray((6, 1)),
        direction=jnp.asarray(0),
        pocket=empty,
    )
    timestep = timestep.replace(
        state=timestep.state.replace(grid=grid, agent=agent)
    )
    timestep = timestep.replace(observation=env._full_map_observation(timestep))
    return env, params, task, timestep


def run_episode():
    env, params, task, timestep = deterministic_initial_state()
    frames = [("RESET", timestep)]

    for action in SCRIPTED_ACTIONS:
        timestep = env.step(params, timestep, jnp.asarray(action))
        frames.append((ACTION_NAMES[action], timestep))

    assert len(frames) == 30
    assert task.rule_ids == ("R1", "R4", "R8")
    assert exact_count(np.asarray(frames[5][1].state.grid), D) == 1
    assert exact_count(np.asarray(frames[12][1].state.grid), E) == 1
    assert exact_count(np.asarray(frames[18][1].state.grid), BLUE_KEY) == 1
    assert int(frames[26][1].state.grid[3, 4, 0]) == Tiles.DOOR_OPEN
    assert bool(frames[-1][1].last())
    assert float(frames[-1][1].reward) > 0.0
    assert int(frames[-1][1].state.step_num) == len(SCRIPTED_ACTIONS)
    return env, params, task, frames


def render_panel(env, params, step: int, action_name: str, timestep) -> Image.Image:
    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(panel)
    title_font = load_font(22, bold=True)
    small_font = load_font(14)
    badge_font = load_font(16, bold=True)

    event, event_color = EVENTS.get(step, ("NAVIGATION", "#94a3b8"))
    draw.text(
        (16, 17),
        f"S{step:02d} · {action_name}",
        fill="#263238",
        font=title_font,
    )
    draw.rounded_rectangle((16, 51, 215, 84), radius=16, fill=event_color)
    draw.text(
        (115, 67),
        event,
        fill="#ffffff",
        font=badge_font,
        anchor="mm",
    )

    pocket = pocket_name(np.asarray(timestep.state.agent.pocket))
    direction = DIRECTION_NAMES[int(timestep.state.agent.direction)]
    position = tuple(map(int, timestep.state.agent.position))
    draw.text(
        (16, 94),
        f"pocket={pocket} · pos={position} · dir={direction}",
        fill="#52606d",
        font=small_font,
    )

    frame = Image.fromarray(env.render(params, timestep))
    frame = frame.resize((325, 325), Image.Resampling.NEAREST)
    border_color = event_color if step in EVENTS else "#263238"
    frame = ImageOps.expand(frame, border=3, fill=border_color)
    panel.paste(frame, (17, 126))
    return panel


def make_storyboard():
    env, params, task, frames = run_episode()
    width = 2 * MARGIN + COLS * PANEL_WIDTH
    height = HEADER_HEIGHT + ROWS * PANEL_HEIGHT + MARGIN
    canvas = Image.new("RGB", (width, height), "#f5f7f8")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(47, bold=True)
    subtitle_font = load_font(24)

    draw.text(
        (MARGIN, 35),
        "One complete scripted episode · T02",
        fill="#1f2933",
        font=title_font,
    )
    draw.text(
        (MARGIN, 103),
        "R1 A+B→D  ·  R4 D+A→E  ·  R8 E+B→BLUE KEY  ·  unlock door  ·  AgentNear(yellow star)",
        fill="#52606d",
        font=subtitle_font,
    )
    draw.text(
        (MARGIN, 151),
        "Read left-to-right, then top-to-bottom. Every panel is the real state after the labelled environment action.",
        fill="#52606d",
        font=subtitle_font,
    )
    draw.text(
        (MARGIN, 199),
        f"29 actions · final reward={float(frames[-1][1].reward):.3f} · final SUCCESS=1",
        fill="#15803d",
        font=subtitle_font,
    )

    for index, (action_name, timestep) in enumerate(frames):
        row, col = divmod(index, COLS)
        panel = render_panel(env, params, index, action_name, timestep)
        canvas.paste(
            panel,
            (MARGIN + col * PANEL_WIDTH, HEADER_HEIGHT + row * PANEL_HEIGHT),
        )
    return canvas, task, frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/five_object_scripted_episode_T02.png"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/five_object_scripted_episode_T02.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    storyboard, task, frames = make_storyboard()
    storyboard.save(args.output, format="PNG", optimize=True)
    manifest = {
        "task_id": task.task_id,
        "split": task.split,
        "rule_ids": list(task.rule_ids),
        "rule_text": list(task.rule_text),
        "actions": [ACTION_NAMES[action] for action in SCRIPTED_ACTIONS],
        "action_ids": list(SCRIPTED_ACTIONS),
        "num_actions": len(SCRIPTED_ACTIONS),
        "final_step": int(frames[-1][1].state.step_num),
        "final_reward": float(frames[-1][1].reward),
        "success": int(bool(frames[-1][1].last()) and float(frames[-1][1].reward) > 0),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output.resolve())
    print(args.manifest.resolve())
    print(
        f"shape={storyboard.width}x{storyboard.height} "
        f"frames={len(frames)} success={manifest['success']}"
    )


if __name__ == "__main__":
    main()
