#!/usr/bin/env python3
"""Render the four ExactCraft embodiments on shared and controlled layouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from PIL import Image, ImageDraw, ImageFont, ImageOps

from xminigrid.core.constants import TILES_REGISTRY, Colors, Tiles
from xminigrid.core.grid import room
from xminigrid.envs.embodied_crafting import (
    EMBODIMENT_LABELS,
    make_embodied_env_and_params,
)
from xminigrid.types import AgentState


EMBODIMENTS = ("standard", "stride2", "omni8", "crab")
DESCRIPTIONS = {
    "standard": "forward = 1 cardinal cell",
    "stride2": "forward = 2 cells; fallback to 1/0",
    "omni8": "8 headings; forward can be diagonal",
    "crab": "forward action = 1 cell right of heading",
}
SPRITE_COLORS = {
    "standard": "#e03131",
    "stride2": "#00a9d6",
    "omni8": "#8e44dc",
    "crab": "#0faf7d",
}
CONTROL_DIRECTIONS = {
    "standard": 0,
    "stride2": 0,
    "omni8": 1,
    "crab": 0,
}
EXPECTED = {
    "standard": "(5,2) → (4,2)",
    "stride2": "(5,2) → (3,2)",
    "omni8": "(5,2) → (4,3)",
    "crab": "(5,2) → (5,3)",
}


def load_font(size: int, bold: bool = False):
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


def labelled_frame(frame, label: str, size: int) -> Image.Image:
    image = Image.fromarray(frame).resize(
        (size, size),
        Image.Resampling.NEAREST,
    )
    image = ImageOps.expand(image, border=3, fill="#263238")
    canvas = Image.new("RGB", (image.width, image.height + 42), "#ffffff")
    canvas.paste(image, (0, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (image.width // 2, 21),
        label,
        fill="#37474f",
        font=load_font(21, bold=True),
        anchor="mm",
    )
    return canvas


def initial_frame(embodiment: str):
    env, params, _ = make_embodied_env_and_params(embodiment, 2)
    timestep = env.reset(params, jax.random.PRNGKey(19))
    return env.render(params, timestep)


def controlled_pair(embodiment: str):
    env, params, _ = make_embodied_env_and_params(embodiment, 2)
    timestep = env.reset(params, jax.random.PRNGKey(0))
    grid = room(8, 8)
    agent = AgentState(
        position=jnp.asarray((5, 2)),
        direction=jnp.asarray(CONTROL_DIRECTIONS[embodiment]),
        pocket=TILES_REGISTRY[Tiles.EMPTY, Colors.EMPTY],
    )
    timestep = timestep.replace(
        state=timestep.state.replace(grid=grid, agent=agent)
    )
    timestep = timestep.replace(observation=env._full_map_observation(timestep))
    before = env.render(params, timestep)
    after_timestep = env.step(params, timestep, jnp.asarray(0))
    after = env.render(params, after_timestep)
    return before, after


def render_panel(embodiment: str) -> Image.Image:
    width = 610
    height = 1030
    panel = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(panel)
    title_font = load_font(34, bold=True)
    body_font = load_font(22)
    small_font = load_font(20)

    draw.rounded_rectangle(
        (18, 17, width - 18, 76),
        radius=28,
        fill=SPRITE_COLORS[embodiment],
    )
    draw.text(
        (width // 2, 46),
        EMBODIMENT_LABELS[embodiment],
        fill="#ffffff",
        font=title_font,
        anchor="mm",
    )
    draw.text(
        (width // 2, 108),
        DESCRIPTIONS[embodiment],
        fill="#455a64",
        font=body_font,
        anchor="mm",
    )

    shared = labelled_frame(initial_frame(embodiment), "SAME T02 · SAME SEED 19", 450)
    panel.paste(shared, ((width - shared.width) // 2, 140))

    before, after = controlled_pair(embodiment)
    before_image = labelled_frame(before, "before action=0", 250)
    after_image = labelled_frame(after, "after action=0", 250)
    panel.paste(before_image, (42, 662))
    panel.paste(after_image, (318, 662))
    draw.text(
        (width // 2, 994),
        EXPECTED[embodiment],
        fill="#263238",
        font=small_font,
        anchor="mm",
    )
    return panel


def make_figure() -> Image.Image:
    width = 2600
    height = 1280
    margin = 55
    panel_gap = 18
    canvas = Image.new("RGB", (width, height), "#f5f7f8")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 35),
        "EmbodiedExactCraft-8x8-v1 · four action dynamics",
        fill="#1f2933",
        font=load_font(51, bold=True),
    )
    draw.text(
        (margin, 105),
        "Identical ExactCraft rules, 18/9 task split and two-room layout; only agent appearance and locomotion change.",
        fill="#52606d",
        font=load_font(27),
    )

    top = 170
    for index, embodiment in enumerate(EMBODIMENTS):
        panel = render_panel(embodiment)
        x = margin + index * (panel.width + panel_gap)
        canvas.paste(panel, (x, top))

    draw.text(
        (margin, 1225),
        "All four retain pickup · putdown · toggle, visible carried objects, full 8×8 observation and collision-safe movement.",
        fill="#37474f",
        font=load_font(24),
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/embodied_exactcraft_four_scenarios.png"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    figure.save(args.output, format="PNG", optimize=True)
    print(args.output.resolve())
    print(f"shape={figure.width}x{figure.height} embodiments={len(EMBODIMENTS)}")


if __name__ == "__main__":
    main()
