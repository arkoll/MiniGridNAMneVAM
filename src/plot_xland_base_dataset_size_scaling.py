#!/usr/bin/env python3
"""OOD data-scaling bars for the Base model."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/user8_3/xland/xland_joint_wam/runs/xland_scaling_base_15k_dual_loss/plots")
OUT.mkdir(parents=True, exist_ok=True)
sizes = ["1k", "3k", "10k", "30k", "100k"]
# OOD columns: T=0, T=1. The 30k point reuses the existing n=200 evaluation.
video = np.array([[.33, .80], [.30, .87], [.40, .83], [.47, .87], [.50, .97]])
action = np.array([[.23, .67], [.20, .80], [.40, .90], [.48, .84], [.37, .90]])
TEXT, GRID = "#111111", "#b0b0b0"
def draw(column, temperature, ymin, ymax):
    fig, ax = plt.subplots(figsize=(7.5, 5.0), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    trajectory_count = np.array([1_000, 3_000, 10_000, 30_000, 100_000])
    ax.plot(trajectory_count, video[:, column], color="#445469", marker="o", markersize=5.5,
            linewidth=2.15, label="Video + action")
    ax.plot(trajectory_count, action[:, column], color="#2cbeac", marker="o", markersize=5.5,
            linewidth=2.15, label="Action-only")
    ax.set_xscale("log")
    ax.set_xticks(trajectory_count, ["1k", "3k", "10k", "30k", "100k"])
    ax.set_xlabel("Training trajectories", color=TEXT)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(ymin, ymax + .001, .05))
    ax.set_title(f"OOD success rate, T={temperature}", color=TEXT, fontsize=14)
    ax.grid(which="major", axis="both", color=GRID, alpha=.8, linewidth=.8)
    ax.minorticks_off()
    ax.tick_params(colors=TEXT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT)
    ax.spines["bottom"].set_color(TEXT)
    legend = ax.legend(loc="upper right", frameon=True, fontsize=9)
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_edgecolor(GRID)
    legend.get_frame().set_linewidth(.7)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"base_dataset_size_scaling_ood_t{temperature}{suffix}",
                    dpi=220 if suffix == ".png" else None, facecolor=fig.get_facecolor())
    plt.close(fig)

draw(0, 0, .10, .60)
draw(1, 1, .60, 1.00)
