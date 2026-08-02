#!/usr/bin/env python3
"""Render an action-by-action proof of visible carried-object mechanics."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw, ImageFont, ImageOps

from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.grid import room, vertical_line
from xminigrid.types import AgentState
from xminigrid.envs.five_object_crafting import (
    B,
    C,
    BLUE_LOCKED_DOOR,
    YELLOW_STAR,
    make_env_and_params,
)


COLS = 4
ROWS = 2
PANEL_WIDTH = 550
PANEL_HEIGHT = 620
HEADER_HEIGHT = 220
MARGIN = 50


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


def make_initial_timestep():
    env, params, _ = make_env_and_params(2)
    timestep = env.reset(params, jax.random.PRNGKey(0))
    wall = TILES_REGISTRY[Tiles.WALL, Colors.GREY]
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
    return env, params, timestep


def build_sequence():
    env, params, initial = make_initial_timestep()
    picked = env.step(params, initial, jnp.asarray(3))
    turned = env.step(params, picked, jnp.asarray(1))
    moved = env.step(params, turned, jnp.asarray(0))
    wall_blocked = env.step(params, moved, jnp.asarray(0))
    facing_b = env.step(params, wall_blocked, jnp.asarray(1))
    occupied_drop = env.step(params, facing_b, jnp.asarray(4))
    facing_floor = env.step(params, occupied_drop, jnp.asarray(1))
    valid_drop = env.step(params, facing_floor, jnp.asarray(4))

    sequence = (
        ("1 · FACE C", "C is one tile ahead", initial),
        ("2 · PICK_UP", "C is attached and visible", picked),
        ("3 · TURN RIGHT", "C rotates with the agent", turned),
        ("4 · MOVE", "C follows the agent", moved),
        ("5 · MOVE BLOCKED", "the wall prevents movement", wall_blocked),
        ("6 · DROP BLOCKED", "B occupies target; C stays attached", occupied_drop),
        ("7 · VALID DROP", "free adjacent floor; C is released", valid_drop),
    )
    return env, params, sequence


def render_panel(env, params, title: str, subtitle: str, timestep) -> Image.Image:
    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(panel)
    title_font = load_font(28, bold=True)
    body_font = load_font(20)
    pocket_font = load_font(19, bold=True)

    pocket_tile = int(timestep.state.agent.pocket[0])
    pocket_label = "C attached" if pocket_tile == Tiles.PYRAMID else "empty pocket"
    badge_color = "#7c3aed" if pocket_tile == Tiles.PYRAMID else "#52606d"

    draw.text((20, 20), title, fill="#263238", font=title_font)
    draw.text((20, 61), subtitle, fill="#52606d", font=body_font)
    draw.rounded_rectangle((20, 98, 210, 138), radius=20, fill=badge_color)
    draw.text(
        (115, 118),
        pocket_label,
        fill="#ffffff",
        font=pocket_font,
        anchor="mm",
    )

    frame = Image.fromarray(env.render(params, timestep))
    frame = frame.resize((470, 470), Image.Resampling.NEAREST)
    frame = ImageOps.expand(frame, border=3, fill="#263238")
    panel.paste(frame, (38, 148))
    return panel


def make_grid() -> Image.Image:
    env, params, sequence = build_sequence()
    width = 2 * MARGIN + COLS * PANEL_WIDTH
    height = HEADER_HEIGHT + ROWS * PANEL_HEIGHT + MARGIN
    canvas = Image.new("RGB", (width, height), "#f5f7f8")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(47, bold=True)
    subtitle_font = load_font(25)

    draw.text(
        (MARGIN, 35),
        "Visible carried-object mechanics",
        fill="#1f2933",
        font=title_font,
    )
    draw.text(
        (MARGIN, 104),
        "The pocket object stays in every RGB frame, follows the triangle tip, and rotates with direction.",
        fill="#52606d",
        font=subtitle_font,
    )
    draw.text(
        (MARGIN, 150),
        "Walls block movement; occupied targets reject put_down; only an adjacent free floor accepts the object.",
        fill="#52606d",
        font=subtitle_font,
    )

    for index, (title, subtitle, timestep) in enumerate(sequence):
        row, col = divmod(index, COLS)
        panel = render_panel(env, params, title, subtitle, timestep)
        canvas.paste(
            panel,
            (MARGIN + col * PANEL_WIDTH, HEADER_HEIGHT + row * PANEL_HEIGHT),
        )

    legend_x = MARGIN + 3 * PANEL_WIDTH
    legend_y = HEADER_HEIGHT + PANEL_HEIGHT
    legend = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "#ffffff")
    legend_draw = ImageDraw.Draw(legend)
    legend_draw.text(
        (30, 35),
        "MECHANICS",
        fill="#263238",
        font=load_font(30, bold=True),
    )
    lines = (
        "PICK_UP: object → pocket",
        "RGB: pocket sprite stays visible",
        "MOVE/TURN: sprite follows agent",
        "WALL/OBJECT: movement blocked",
        "PUT_DOWN occupied: no change",
        "PUT_DOWN floor: object released",
    )
    for line_index, line in enumerate(lines):
        legend_draw.text(
            (30, 105 + line_index * 58),
            line,
            fill="#52606d",
            font=load_font(22),
        )
    canvas.paste(legend, (legend_x, legend_y))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/five_object_carry_mechanics_grid.png"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid()
    grid.save(args.output, format="PNG", optimize=True)
    print(args.output.resolve())
    print(f"shape={grid.width}x{grid.height} action_frames=7")


if __name__ == "__main__":
    main()
