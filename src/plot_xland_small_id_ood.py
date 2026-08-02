#!/usr/bin/env python3
"""Slide chart: Small model success rate on ID and OOD task graphs (n=200)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/user8_3/xland/xland_joint_wam/runs/xland_model_size_final_200eval")
RESULTS, OUT = ROOT / "results", ROOT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
TEXT, GRID = "#111111", "#b0b0b0"
SPLITS = [("ID", "train"), ("OOD", "val_ood")]
SERIES = [
    ("Video + action, T=0", "video", 0, "#445469"),
    ("Video + action, T=1", "video", 1, "#2cbeac"),
    ("Action-only, T=0", "action_only", 0, "#515eea"),
    ("Action-only, T=1", "action_only", 1, "#fbd826"),
]

def load(mode, split, temperature):
    path = RESULTS / f"small_{mode}" / f"{split}_temperature_{temperature}_step_0060000.json"
    with path.open() as f:
        values = np.asarray([r["success"] for r in json.load(f)["results"]], dtype=float)
    return values.mean(), 1.96 * values.std(ddof=1) / np.sqrt(values.size)

fig, ax = plt.subplots(figsize=(8.4, 5.1), facecolor="#ffffff")
ax.set_facecolor("#ffffff")
x, width = np.arange(2), 0.18
for offset, (label, mode, temperature, color) in zip((-1.5*width, -.5*width, .5*width, 1.5*width), SERIES):
    values, errors = zip(*(load(mode, split, temperature) for _, split in SPLITS))
    bars = ax.bar(x + offset, values, width, yerr=errors, capsize=3.5,
                  color=color, edgecolor=TEXT, linewidth=.6, label=label,
                  error_kw={"ecolor": TEXT, "elinewidth": .8})
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=3, fontsize=9, color=TEXT)

ax.set_xticks(x, [name for name, _ in SPLITS], color=TEXT)
ax.set_ylim(.35, 1.0)
ax.set_yticks(np.arange(.35, 1.001, .05))
ax.minorticks_off()
ax.tick_params(colors=TEXT)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(TEXT)
ax.spines["bottom"].set_color(TEXT)
legend = ax.legend(loc="upper right", frameon=True, ncol=2, fontsize=8.8)
legend.get_frame().set_facecolor("#ffffff")
legend.get_frame().set_edgecolor(GRID)
legend.get_frame().set_linewidth(.7)
fig.tight_layout()
for suffix in (".png", ".pdf"):
    fig.savefig(OUT / f"small_id_vs_ood_success_rate{suffix}", dpi=220 if suffix == ".png" else None,
                facecolor=fig.get_facecolor())
