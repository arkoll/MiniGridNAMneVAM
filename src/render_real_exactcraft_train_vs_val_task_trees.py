#!/usr/bin/env python3
"""Render actual ExactCraft-Easy task trees: task 2 train; tasks 1 and 5 validation."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, RegularPolygon, FancyBboxPatch

OUT = Path("/home/user8_3/xland/xland_joint_wam/runs/xland_model_size_final_200eval/plots")
TEXT = "#111111"
COLORS = {"A": "#e95d5d", "B": "#61a86c", "C": "#e7e7e7", "D": "#8758af",
          "K": "#4d8fc7", "goal": "#fbd826", "door": "#445469"}
# From xminigrid/envs/easy_crafting.py:
# task 2=(stage1_idx=0, stage2_idx=1), task 1=(0,0), task 5=(1,1).
TASKS = [
    ("Train tree · task 2", "seen during training", "#2cbeac",
     "R1  A + B → D", "R5  D + B → K", ("A", "B"), "B"),
    ("Held-out tree · task 1", "never seen during training", "#515eea",
     "R1  A + B → D", "R4  D + A → K", ("A", "B"), "A"),
    ("Held-out tree · task 5", "never seen during training", "#515eea",
     "R2  A + C → D", "R5  D + B → K", ("A", "C"), "B"),
]

def node(ax, x, y, token, note=""):
    if token == "A":
        patch = Circle((x, y), .058, facecolor=COLORS[token], edgecolor=TEXT, lw=1.2)
    elif token == "B":
        patch = Rectangle((x-.055, y-.055), .11, .11, facecolor=COLORS[token], edgecolor=TEXT, lw=1.2)
    elif token == "C":
        patch = RegularPolygon((x, y), 3, radius=.07, orientation=0, facecolor=COLORS[token], edgecolor=TEXT, lw=1.2)
    elif token == "D":
        patch = RegularPolygon((x, y), 6, radius=.068, orientation=.2, facecolor=COLORS[token], edgecolor=TEXT, lw=1.2)
    elif token == "K":
        patch = FancyBboxPatch((x-.07, y-.037), .12, .074, boxstyle="round,pad=0.012,rounding_size=0.02",
                               facecolor=COLORS[token], edgecolor=TEXT, lw=1.2)
    else:
        patch = RegularPolygon((x, y), 5, radius=.075, facecolor=COLORS["goal"], edgecolor=TEXT, lw=1.2)
    ax.add_patch(patch)
    ax.text(x, y, token if token != "goal" else "★", ha="center", va="center", fontsize=11, color=TEXT, weight="bold")
    if note:
        ax.text(x, y-.11, note, ha="center", va="top", fontsize=8, color=TEXT)

def arrow(ax, start, end, dashed=False):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color=TEXT, lw=1.4,
                                                           linestyle="--" if dashed else "-", shrinkA=4, shrinkB=5))

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), facecolor="#ffffff")
for ax, (title, badge, badge_color, rule1, rule2, inputs, catalyst) in zip(axes, TASKS):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.set_facecolor("#ffffff")
    ax.set_title(title, color=TEXT, weight="bold", fontsize=12, pad=10)
    ax.text(.5, .91, badge, ha="center", va="center", color="#ffffff", fontsize=8.8,
            bbox=dict(boxstyle="round,pad=.38", facecolor=badge_color, edgecolor="none"))

    # Stage 1: two initial objects produce D.
    node(ax, .13, .70, inputs[0]); node(ax, .13, .48, inputs[1]); node(ax, .37, .59, "D", "stage 1")
    arrow(ax, (.20,.68), (.31,.61)); arrow(ax, (.20,.50), (.31,.57))
    ax.text(.25, .35, rule1, ha="center", fontsize=9, color=TEXT)

    # Stage 2: D from stage 1 plus its catalyst produces the key.
    node(ax, .57, .59, "D"); node(ax, .57, .37, catalyst); node(ax, .81, .48, "K", "blue key")
    arrow(ax, (.44,.59), (.50,.59), dashed=True)
    arrow(ax, (.64,.57), (.75,.50)); arrow(ax, (.64,.39), (.75,.46))
    ax.text(.68, .24, rule2, ha="center", fontsize=9, color=TEXT)

    # The key opens the door; reaching the star finishes the same task pipeline.
    arrow(ax, (.81,.40), (.81,.30))
    door = Rectangle((.75,.17), .12, .12, facecolor=COLORS["door"], edgecolor=TEXT, lw=1.2)
    ax.add_patch(door); ax.text(.81,.23,"door",ha="center",va="center",color="#ffffff",fontsize=8,weight="bold")
    arrow(ax, (.81,.16), (.81,.075))
    node(ax,.81,.035,"goal")

fig.text(.5, .005, "ExactCraft-Easy: A = red ball, B = green square, C = white pyramid, D = purple hexagon, K = blue key. "
         "All six primitive rules occur in both splits; only complete two-rule trees are held out.",
         ha="center", va="bottom", color=TEXT, fontsize=8.7)
fig.tight_layout(rect=(0, .07, 1, 1))
for suffix in (".png", ".pdf"):
    fig.savefig(OUT / f"real_exactcraft_easy_train_vs_val_task_trees{suffix}",
                dpi=220 if suffix == ".png" else None, facecolor=fig.get_facecolor())
