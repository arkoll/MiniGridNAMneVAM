#!/usr/bin/env python3
"""Render Easy/Medium comparisons for all three crafting families."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
from PIL import Image, ImageDraw, ImageFont, ImageOps

from xminigrid.envs.easy_crafting import make_easy_env_and_params
from xminigrid.envs.embodied_crafting import (
    EMBODIMENT_LABELS,
    make_embodied_env_and_params,
)
from xminigrid.envs.five_object_crafting import make_env_and_params
from xminigrid.envs.shape_crafting import make_shape_env_and_params
from xminigrid.envs.shape_crafting_easy import (
    make_shape_easy_env_and_params,
)


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


def frame(env, params, seed: int):
    timestep = env.reset(params, jax.random.PRNGKey(seed))
    return env.render(params, timestep)


def map_panel(
    image,
    title: str,
    subtitle: str,
    badge: str,
    badge_color: str,
    size: int = 500,
) -> Image.Image:
    width = size + 54
    height = size + 152
    panel = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(panel)
    compact = size <= 350
    badge_right = 136 if compact else 176
    badge_center = (18 + badge_right) // 2
    draw.rounded_rectangle(
        (18, 14, badge_right, 62),
        radius=22,
        fill=badge_color,
    )
    draw.text(
        (badge_center, 38),
        badge,
        fill="#ffffff",
        font=load_font(21 if compact else 24, bold=True),
        anchor="mm",
    )
    draw.text(
        (150 if compact else 195, 38),
        title,
        fill="#263238",
        font=load_font(22 if compact else 29, bold=True),
        anchor="lm",
    )
    draw.text(
        (width // 2, 88),
        subtitle,
        fill="#52606d",
        font=load_font(17 if compact else 21),
        anchor="mm",
    )
    rendered = Image.fromarray(image).resize(
        (size, size),
        Image.Resampling.NEAREST,
    )
    rendered = ImageOps.expand(rendered, border=3, fill="#263238")
    panel.paste(rendered, ((width - rendered.width) // 2, 120))
    return panel


def exact_panels():
    easy_env, easy_params, easy_task = make_easy_env_and_params(2)
    medium_env, medium_params, medium_task = make_env_and_params(2)
    return (
        map_panel(
            frame(easy_env, easy_params, 19),
            "ExactCraft",
            "6 rules · depth 2 · 3 initial objects · 6/3 split",
            "EASY",
            "#16865c",
        ),
        map_panel(
            frame(medium_env, medium_params, 19),
            "ExactCraft",
            "9 rules · depth 3 · 4 initial objects · 18/9 split",
            "MEDIUM",
            "#6f42c1",
        ),
        easy_task,
        medium_task,
    )


def shape_panels():
    easy_env, easy_params, _ = make_shape_easy_env_and_params("train", 1)
    medium_env, medium_params, _ = make_shape_env_and_params("train", 1)
    return (
        map_panel(
            frame(easy_env, easy_params, 27),
            "ShapeCraft",
            "4 shapes · 9 trees · 324 tasks per split",
            "EASY",
            "#16865c",
        ),
        map_panel(
            frame(medium_env, medium_params, 27),
            "ShapeCraft",
            "5 shapes · 27 trees · 972 tasks per split",
            "MEDIUM",
            "#6f42c1",
        ),
    )


def embodied_panel(embodiment: str, seed: int) -> Image.Image:
    env, params, _ = make_embodied_env_and_params(
        embodiment,
        2,
        difficulty="easy",
    )
    return map_panel(
        frame(env, params, seed),
        {
            "standard": "Standard",
            "stride2": "Stride2",
            "omni8": "Omni8",
            "crab": "Crab",
        }[embodiment],
        "same 6-rule Easy task bank",
        "EASY",
        "#16865c",
        size=350,
    )


def make_figure() -> Image.Image:
    width = 2500
    height = 2360
    margin = 60
    canvas = Image.new("RGB", (width, height), "#f5f7f8")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 38),
        "Crafting benchmark difficulties · Easy and Medium",
        fill="#1f2933",
        font=load_font(51, bold=True),
    )
    draw.text(
        (margin, 108),
        "The 8×8 two-room world and final objective stay fixed; Easy shortens only the learned crafting dynamics.",
        fill="#52606d",
        font=load_font(27),
    )

    exact_easy, exact_medium, easy_task, medium_task = exact_panels()
    shape_easy, shape_medium = shape_panels()
    pair_gap = 60
    pair_width = exact_easy.width * 2 + pair_gap
    left = (width - pair_width) // 2

    draw.text(
        (margin, 175),
        "Exact matching: shape + color",
        fill="#263238",
        font=load_font(34, bold=True),
    )
    canvas.paste(exact_easy, (left, 220))
    canvas.paste(exact_medium, (left + exact_easy.width + pair_gap, 220))
    draw.text(
        (width // 2, 905),
        f"Example paths: Easy {' → '.join(easy_task.rule_ids)}   ·   Medium {' → '.join(medium_task.rule_ids)}",
        fill="#455a64",
        font=load_font(23),
        anchor="mm",
    )

    draw.text(
        (margin, 955),
        "Shape-only matching with complementary color bindings",
        fill="#263238",
        font=load_font(34, bold=True),
    )
    canvas.paste(shape_easy, (left, 1000))
    canvas.paste(shape_medium, (left + shape_easy.width + pair_gap, 1000))

    draw.text(
        (margin, 1690),
        "EmbodiedExactCraft Easy · all four bodies use the same 6-rule task bank",
        fill="#263238",
        font=load_font(34, bold=True),
    )
    bodies = ("standard", "stride2", "omni8", "crab")
    small_gap = 18
    body_panels = [
        embodied_panel(body, seed=101 + index)
        for index, body in enumerate(bodies)
    ]
    body_width = body_panels[0].width
    body_left = (
        width - (4 * body_width + 3 * small_gap)
    ) // 2
    for index, panel in enumerate(body_panels):
        canvas.paste(
            panel,
            (body_left + index * (body_width + small_gap), 1740),
        )

    draw.text(
        (margin, 2290),
        "Easy: 6 rules · 2 active stages · 3 ingredients · horizon 192    |    Medium: 9 · 3 · 4 · 256",
        fill="#37474f",
        font=load_font(24),
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/easy_medium_crafting_comparison.png"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    figure.save(args.output, format="PNG", optimize=True)
    print(args.output.resolve())
    print(f"shape={figure.width}x{figure.height}")


if __name__ == "__main__":
    main()
