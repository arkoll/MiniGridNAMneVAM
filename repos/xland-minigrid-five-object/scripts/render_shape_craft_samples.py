#!/usr/bin/env python3
"""Render ShapeCraft's strict color split and representative 8x8 maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
from PIL import Image, ImageDraw, ImageFont, ImageOps

from xminigrid.envs.shape_crafting import (
    COLOR_NAMES_BY_ID,
    ROLE_NAMES,
    SHAPE_NAMES,
    TRAIN_TASKS,
    VAL_TASKS,
    make_shape_env_and_params,
)
from xminigrid.rendering.rgb_render import COLORS_MAP


WIDTH = 2520
MARGIN = 70
PANEL_WIDTH = 760
PANEL_HEIGHT = 660
MAP_SIZE = 520
SAMPLES = (
    ("train", 1, 1, 101),
    ("train", 14, 13, 202),
    ("train", 27, 36, 303),
    ("val", 1, 1, 404),
    ("val", 14, 13, 505),
    ("val", 27, 36, 606),
)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    filenames = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for filename in filenames:
        if Path(filename).exists():
            return ImageFont.truetype(filename, size=size)
    return ImageFont.load_default()


def rgb(color_id: int) -> tuple[int, int, int]:
    values = COLORS_MAP[color_id]
    return tuple(int(value) for value in values)


def role_color_sets(tasks, role_index: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                task.palette_color_ids[role_index]
                for task in tasks
            }
        )
    )


def draw_palette_table(canvas: Image.Image, top: int) -> int:
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(33, bold=True)
    row_font = load_font(29, bold=True)
    color_font = load_font(23)
    note_font = load_font(25)

    x_role = MARGIN
    x_train = 490
    x_val = 1480
    draw.text((x_role, top), "ROLE / SHAPE", fill="#263238", font=header_font)
    draw.text((x_train, top), "TRAIN: 3 colors per shape", fill="#087f5b", font=header_font)
    draw.text((x_val, top), "VAL: complementary 3 colors", fill="#7048e8", font=header_font)

    row_top = top + 62
    swatch_width = 270
    swatch_height = 54
    swatch_gap = 22
    for role_index, (role_name, shape_name) in enumerate(zip(ROLE_NAMES, SHAPE_NAMES)):
        y = row_top + role_index * 78
        draw.text(
            (x_role, y + swatch_height // 2),
            f"{role_name}  {shape_name}",
            fill="#263238",
            font=row_font,
            anchor="lm",
        )
        for group_x, tasks in ((x_train, TRAIN_TASKS), (x_val, VAL_TASKS)):
            for color_index, color_id in enumerate(role_color_sets(tasks, role_index)):
                x = group_x + color_index * (swatch_width + swatch_gap)
                draw.rounded_rectangle(
                    (x, y, x + swatch_width, y + swatch_height),
                    radius=14,
                    fill=rgb(color_id),
                    outline="#263238",
                    width=2,
                )
                text_color = "#111111" if color_id in (2, 8, 9, 11) else "#ffffff"
                draw.text(
                    (x + swatch_width // 2, y + swatch_height // 2),
                    COLOR_NAMES_BY_ID[color_id].upper(),
                    fill=text_color,
                    font=color_font,
                    anchor="mm",
                )

    note_y = row_top + 5 * 78 + 20
    draw.text(
        (MARGIN, note_y),
        "Same global vocabulary in both splits: RED · GREEN · ORANGE · PURPLE · PINK · WHITE",
        fill="#455a64",
        font=note_font,
    )
    draw.text(
        (MARGIN, note_y + 42),
        "For every shape: train colors ∩ val colors = ∅   ·   Exact (shape, color) overlap = 0",
        fill="#455a64",
        font=note_font,
    )
    return note_y + 105


def render_panel(split: str, tree_id: int, palette_id: int, seed: int) -> Image.Image:
    task_index = (tree_id - 1) * 36 + palette_id
    env, params, task = make_shape_env_and_params(split, task_index)
    timestep = env.reset(params, jax.random.PRNGKey(seed))
    map_image = Image.fromarray(env.render(params, timestep))
    map_image = map_image.resize((MAP_SIZE, MAP_SIZE), Image.Resampling.NEAREST)
    map_image = ImageOps.expand(map_image, border=3, fill="#263238")

    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), "#ffffff")
    draw = ImageDraw.Draw(panel)
    badge_font = load_font(26, bold=True)
    title_font = load_font(28, bold=True)
    body_font = load_font(21)
    split_fill = "#087f5b" if split == "train" else "#7048e8"
    draw.rounded_rectangle((18, 16, 250, 68), radius=24, fill=split_fill)
    draw.text(
        (134, 42),
        f"{split.upper()} · T{tree_id:02d}",
        fill="#ffffff",
        font=badge_font,
        anchor="mm",
    )
    draw.text(
        (274, 42),
        " → ".join(task.rule_ids),
        fill="#263238",
        font=title_font,
        anchor="lm",
    )
    palette = "  ".join(
        f"{role}={color}"
        for role, color in zip(ROLE_NAMES, task.palette_color_names)
    )
    draw.text((18, 83), palette, fill="#455a64", font=body_font)
    panel.paste(map_image, ((PANEL_WIDTH - map_image.width) // 2, 122))
    return panel


def make_figure() -> Image.Image:
    height = 2250
    canvas = Image.new("RGB", (WIDTH, height), "#f5f7f8")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(52, bold=True)
    subtitle_font = load_font(29)
    footer_font = load_font(23)

    draw.text(
        (MARGIN, 42),
        "ShapeCraft-8x8-v1 · color-invariant TileNear dynamics",
        fill="#1f2933",
        font=title_font,
    )
    draw.text(
        (MARGIN, 116),
        "Rules match input shapes only; colors vary without changing the learned world dynamics.",
        fill="#52606d",
        font=subtitle_font,
    )
    samples_top = draw_palette_table(canvas, 190)

    for index, (split, tree_id, palette_id, seed) in enumerate(SAMPLES):
        row, col = divmod(index, 3)
        panel = render_panel(split, tree_id, palette_id, seed)
        x = MARGIN + col * (PANEL_WIDTH + 50)
        y = samples_top + row * (PANEL_HEIGHT + 36)
        canvas.paste(panel, (x, y))

    footer_y = samples_top + 2 * (PANEL_HEIGHT + 36) + 10
    draw.text(
        (MARGIN, footer_y),
        "All 27 abstract trees occur in both splits · 36 palettes per split · 972 tasks per split",
        fill="#37474f",
        font=footer_font,
    )
    draw.text(
        (MARGIN, footer_y + 40),
        "8×8 fixed full map · random smart mirroring · larger agent room · blue door · yellow-star goal",
        fill="#37474f",
        font=footer_font,
    )
    return canvas.crop((0, 0, WIDTH, footer_y + 105))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/shape_craft_split_and_samples.png"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    figure.save(args.output, format="PNG", optimize=True)
    print(args.output.resolve())
    print(f"shape={figure.width}x{figure.height} samples={len(SAMPLES)}")


if __name__ == "__main__":
    main()
