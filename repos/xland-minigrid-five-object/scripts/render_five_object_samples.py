#!/usr/bin/env python3
"""Render a labelled grid of initial states from the custom task family."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from xminigrid.envs.five_object_crafting import (
    TASKS,
    YELLOW_STAR,
    make_env_and_params,
)


SAMPLE_TASK_IDS = (2, 3, 4, 6, 7, 8, 10, 12, 1, 5, 11, 26)
COLS = 4
ROWS = 3
PANEL_WIDTH = 660
PANEL_HEIGHT = 790
HEADER_HEIGHT = 310
FOOTER_HEIGHT = 180
MARGIN = 55


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


def render_panel(task_id: int, seed: int) -> Image.Image:
    env, params, task = make_env_and_params(task_id)
    timestep = env.reset(params, jax.random.PRNGKey(seed))
    env_image = Image.fromarray(env.render(params, timestep))
    env_image = env_image.resize((600, 600), Image.Resampling.NEAREST)
    env_image = ImageOps.expand(env_image, border=3, fill="#263238")

    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(panel)
    title_font = load_font(31, bold=True)
    label_font = load_font(23, bold=True)
    body_font = load_font(21)
    small_font = load_font(19)

    split_color = "#0f766e" if task.split == "train" else "#7c3aed"
    draw.rounded_rectangle((18, 15, 212, 66), radius=23, fill=split_color)
    draw.text(
        (115, 40),
        f"T{task.task_id:02d}  {task.split.upper()}",
        fill="#ffffff",
        font=label_font,
        anchor="mm",
    )
    draw.text(
        (235, 40),
        " → ".join(task.rule_ids),
        fill="#263238",
        font=title_font,
        anchor="lm",
    )

    grid = np.asarray(timestep.state.grid)
    agent_x = int(timestep.state.agent.position[1])
    star_x = int(
        np.argwhere(np.all(grid == np.asarray(YELLOW_STAR), axis=-1))[0][1]
    )
    side = (
        "LARGE LEFT → SMALL RIGHT"
        if agent_x < star_x
        else "LARGE RIGHT → SMALL LEFT"
    )
    draw.text(
        (20, 82),
        "  ·  ".join(task.rule_text),
        fill="#37474f",
        font=body_font,
    )
    draw.text(
        (20, 113),
        f"initial: {' '.join(task.initial_symbols)}    path: {side}",
        fill="#607d8b",
        font=small_font,
    )
    panel.paste(env_image, (28, 153))
    return panel


def make_grid() -> Image.Image:
    canvas_width = 2 * MARGIN + COLS * PANEL_WIDTH
    canvas_height = HEADER_HEIGHT + ROWS * PANEL_HEIGHT + FOOTER_HEIGHT
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#f5f7f8")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(52, bold=True)
    subtitle_font = load_font(27)
    footer_font = load_font(23)

    draw.text(
        (MARGIN, 42),
        "XLand-MiniGrid · Five-object TileNear crafting samples",
        fill="#1f2933",
        font=title_font,
    )
    draw.text(
        (MARGIN, 118),
        "8×8 full map · larger agent room · smaller star room · blue locked door · no initial key",
        fill="#52606d",
        font=subtitle_font,
    )
    draw.text(
        (MARGIN, 170),
        "Green badges = train combinations   ·   Purple badges = held-out validation combinations",
        fill="#52606d",
        font=subtitle_font,
    )
    draw.text(
        (MARGIN, 222),
        "Fixed world coordinates · no 5×5 crop · no camera rotation or field-of-view shading.",
        fill="#52606d",
        font=subtitle_font,
    )

    for index, task_id in enumerate(SAMPLE_TASK_IDS):
        row, col = divmod(index, COLS)
        panel = render_panel(task_id, seed=1000 + task_id)
        x = MARGIN + col * PANEL_WIDTH
        y = HEADER_HEIGHT + row * PANEL_HEIGHT
        canvas.paste(panel, (x, y))

    footer_y = HEADER_HEIGHT + ROWS * PANEL_HEIGHT + 32
    draw.text(
        (MARGIN, footer_y),
        "A=red ball · B=green square · C=white pyramid · D=purple hex · E=orange ball · K=blue key",
        fill="#37474f",
        font=footer_font,
    )
    draw.text(
        (MARGIN, footer_y + 45),
        "Task success: craft K, unlock the blue door, enter the other room, move next to the yellow star.",
        fill="#37474f",
        font=footer_font,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/five_object_env_samples_grid.png"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid()
    grid.save(args.output, format="PNG", optimize=True)
    print(args.output.resolve())
    print(f"shape={grid.width}x{grid.height} samples={len(SAMPLE_TASK_IDS)}")


if __name__ == "__main__":
    main()
