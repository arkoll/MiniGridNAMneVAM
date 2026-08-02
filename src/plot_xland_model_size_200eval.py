#!/usr/bin/env python3
"""Create model-size closed-loop SR bar charts and a Markdown table (n=200)."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--results-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)

COLORS = {
    "text": "#111111",
    "background": "#f6f2ec",
    "grid": "#d0a583",
    "video_t0": "#445469",
    "video_t1": "#2cbeac",
    "action_t0": "#515eea",
    "action_t1": "#fbd826",
}
models = [("Small", "small"), ("Base", "base"), ("Large", "large")]
series = [
    ("Video + action, T=0", "video", 0, COLORS["video_t0"]),
    ("Video + action, T=1", "video", 1, COLORS["video_t1"]),
    ("Action-only, T=0", "action_only", 0, COLORS["action_t0"]),
    ("Action-only, T=1", "action_only", 1, COLORS["action_t1"]),
]

def load(slug, variant, split, temperature):
    path = args.results_root / f"{slug}_{variant}" / f"{split}_temperature_{temperature}_step_0060000.json"
    with path.open() as handle:
        data = json.load(handle)
    successes = np.asarray([row["success"] for row in data["results"]], dtype=float)
    sr = float(successes.mean())
    se = float(successes.std(ddof=1) / np.sqrt(successes.size))
    return sr, 1.96 * se

for split, title, filename in [
    ("train", "Train task graphs", "train_success_rate.png"),
    ("val_ood", "OOD task graphs", "ood_success_rate.png"),
]:
    x = np.arange(len(models))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.5, 6), facecolor=COLORS["background"])
    ax.set_facecolor(COLORS["background"])
    for index, (label, variant, temperature, color) in enumerate(series):
        values, errors = zip(*(load(slug, variant, split, temperature) for _, slug in models))
        bars = ax.bar(
            x + (index - 1.5) * width, values, width,
            yerr=errors, capsize=4, color=color, edgecolor=COLORS["text"],
            linewidth=0.6, error_kw={"ecolor": COLORS["text"], "linewidth": 1},
            label=label,
        )
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=9, color=COLORS["text"])
    ax.set_xticks(x, [name for name, _ in models], color=COLORS["text"])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Success rate (95% CI)", color=COLORS["text"])
    ax.set_title(f"Closed-loop success rate, n=200 — {title}", color=COLORS["text"])
    ax.tick_params(colors=COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["text"])
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.75, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, frameon=False, labelcolor=COLORS["text"])
    fig.tight_layout()
    fig.savefig(args.output_dir / filename, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)

lines = [
    "# Closed-loop success rate (n=200)",
    "",
    "Values are **SR ± 95% CI** across the 200 evaluation episodes.",
    "",
    "| Model | Mode | Train, T=0 | OOD, T=0 | Train, T=1 | OOD, T=1 |",
    "|---|---|---:|---:|---:|---:|",
]
for name, slug in models:
    for mode, variant in [("Video + action", "video"), ("Action-only", "action_only")]:
        values = []
        for split, temperature in [("train", 0), ("val_ood", 0), ("train", 1), ("val_ood", 1)]:
            sr, ci = load(slug, variant, split, temperature)
            values.append(f"{sr:.3f} ± {ci:.3f}")
        lines.append(f"| {name} | {mode} | " + " | ".join(values) + " |")
(args.output_dir / "success_rate_summary.md").write_text("\n".join(lines) + "\n")


# Slide-ready model-size scaling curve. This is a scaling comparison, not a fitted power law.
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True, facecolor=COLORS["background"])
x = np.arange(len(models))
for ax, (split, panel_title) in zip(axes, [("train", "Train task graphs"), ("val_ood", "OOD task graphs")]):
    ax.set_facecolor(COLORS["background"])
    for label, variant, temperature, color in series:
        values, errors = zip(*(load(slug, variant, split, temperature) for _, slug in models))
        ax.errorbar(x, values, yerr=errors, color=color, marker="o", markersize=7,
                    linewidth=2, capsize=4, label=label, markeredgecolor=COLORS["text"],
                    markeredgewidth=0.6)
    ax.set_title(panel_title, color=COLORS["text"])
    ax.set_xticks(x, [name for name, _ in models], color=COLORS["text"])
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.75, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["text"])
axes[0].set_ylabel("Success rate (95% CI)", color=COLORS["text"])
fig.suptitle("Model-size scaling at fixed training compute (15k steps)", color=COLORS["text"], y=0.98)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, labelcolor=COLORS["text"], bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.10, 1, 0.94))
fig.savefig(args.output_dir / "model_size_scaling.png", dpi=220, facecolor=fig.get_facecolor())
plt.close(fig)


# Combined OOD chart: temperatures side by side for each training mode.
STYLE = {"background": "#ffffff", "text": "#111111", "grid": "#b0b0b0", "series": ["#445469", "#2cbeac", "#515eea"]}
fig, ax = plt.subplots(figsize=(10.2, 5.3), facecolor=STYLE["background"])
ax.set_facecolor(STYLE["background"])

# For each model: Video T=0/T=1, then Action-only T=0/T=1.
width = 0.18
offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
ood_series = [
    ("Video + action, T=0", "video", 0, STYLE["series"][0]),
    ("Video + action, T=1", "video", 1, STYLE["series"][1]),
    ("Action-only, T=0", "action_only", 0, STYLE["series"][2]),
    ("Action-only, T=1", "action_only", 1, "#fbd826"),
]
for offset, (label, variant, temperature, color) in zip(offsets, ood_series):
    values, errors = zip(*(load(slug, variant, "val_ood", temperature) for _, slug in models))
    bars = ax.bar(
        x + offset, values, width,
        yerr=errors, capsize=3.5, color=color,
        edgecolor=STYLE["text"], linewidth=0.6, label=label,
        error_kw={"ecolor": STYLE["text"], "elinewidth": 0.8},
    )
    ax.bar_label(bars, labels=[f"{value:.2f}" for value in values],
                 padding=3, fontsize=8.5, color=STYLE["text"])

ax.set_xticks(x, [name for name, _ in models], color=STYLE["text"])
ax.set_xlabel("Model size", color=STYLE["text"])
ax.set_ylim(0.35, 1.00)
ax.set_yticks(np.arange(0.35, 1.001, 0.05))
ax.set_ylabel("")
ax.set_title("OOD success rate by model size and temperature", color=STYLE["text"], loc="center")
ax.minorticks_off()
ax.tick_params(colors=STYLE["text"])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(STYLE["text"])
ax.spines["bottom"].set_color(STYLE["text"])
legend = ax.legend(loc="lower right", frameon=True, ncol=2, fontsize=9)
legend.get_frame().set_facecolor("#ffffff")
legend.get_frame().set_edgecolor("#b0b0b0")
legend.get_frame().set_linewidth(0.7)
fig.tight_layout()
png = args.output_dir / "model_size_scaling_ood_temperatures.png"
fig.savefig(png, dpi=220, facecolor=fig.get_facecolor())
fig.savefig(png.with_suffix(".pdf"), facecolor=fig.get_facecolor())
plt.close(fig)


# Separate OOD bar charts by temperature.
for temperature in (0, 1):
    fig, ax = plt.subplots(figsize=(7.4, 5.0), facecolor=STYLE["background"])
    ax.set_facecolor(STYLE["background"])
    width = 0.34
    for index, (label, variant) in enumerate([("Video + action", "video"), ("Action-only", "action_only")]):
        values, errors = zip(*(load(slug, variant, "val_ood", temperature) for _, slug in models))
        bars = ax.bar(
            x + (index - 0.5) * width, values, width,
            yerr=errors, capsize=3.5, color=STYLE["series"][index],
            edgecolor=STYLE["text"], linewidth=0.6, label=label,
            error_kw={"ecolor": STYLE["text"], "elinewidth": 0.8},
        )
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values],
                     padding=3, fontsize=9, color=STYLE["text"])
    ax.set_xticks(x, [name for name, _ in models], color=STYLE["text"])
    ax.set_xlabel("Model size", color=STYLE["text"])
    ymin, ymax = ((.35, .65) if temperature == 0 else (.65, 1.00))
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.arange(ymin, ymax + .001, .05))
    ax.set_title(f"OOD success rate, T={temperature}", color=STYLE["text"], fontsize=14)
    ax.minorticks_off()
    ax.tick_params(colors=STYLE["text"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(STYLE["text"])
    ax.spines["bottom"].set_color(STYLE["text"])
    legend = ax.legend(loc="upper right", frameon=True, fontsize=9)
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_edgecolor("#b0b0b0")
    legend.get_frame().set_linewidth(.7)
    fig.tight_layout()
    png = args.output_dir / f"model_size_scaling_ood_bars_t{temperature}.png"
    fig.savefig(png, dpi=220, facecolor=fig.get_facecolor())
    fig.savefig(png.with_suffix(".pdf"), facecolor=fig.get_facecolor())
    plt.close(fig)
