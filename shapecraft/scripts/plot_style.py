"""Единый стиль графиков проекта.

Правила про ось X (Training step, шаг сетки 2500), 2x2 dashboard и сглаживание скользящим
средним относятся к графикам обучения; здесь по оси X температура или категория, поэтому
из стиля берётся всё остальное: фон, цвета, сетка, рамки, легенда, диапазон Y, dpi.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#111111"
BG = "#ffffff"
GRID = "#b0b0b0"
SERIES = ["#445469", "#2cbeac", "#515eea"]
PALETTE = ["#111111", "#445469", "#f6f2ec", "#2cbeac", "#515eea",
           "#fbd826", "#b9d0fe", "#d0a583", "#e6d5c3"]
DPI = 220
LW = 2.15
MS = 4.5


def new_figure(figsize=(9.0, 5.4)):
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


def finish(fig, ax, title, xlabel=None, legend_loc="lower right", legend_ncol=1):
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.grid(True, which="major", color=GRID, alpha=0.8, linewidth=0.8)
    ax.minorticks_off()
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.0)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=INK)
    ax.set_ylabel("")  # подписи оси Y по стилю нет
    ax.tick_params(colors=INK, labelsize=10, which="both")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # legend_loc — либо строка, либо пара (loc, bbox_to_anchor) для выноса за область
        kw = {}
        if isinstance(legend_loc, tuple):
            legend_loc, kw["bbox_to_anchor"] = legend_loc
        leg = ax.legend(loc=legend_loc, frameon=True, fontsize=9.5, facecolor=BG,
                        edgecolor=GRID, framealpha=1.0, handlelength=3.0,
                        borderpad=0.7, labelspacing=0.5, ncol=legend_ncol,
                        columnspacing=1.8, **kw)
        leg.get_frame().set_linewidth(0.8)
        for t in leg.get_texts():
            t.set_color(INK)

    fig.suptitle(title, fontsize=13, color=INK, y=0.975)
    fig.tight_layout(rect=(0, 0, 1, 0.94))


def save(fig, path_png):
    fig.savefig(path_png, facecolor=BG, dpi=DPI)
    fig.savefig(path_png.replace(".png", ".pdf"), facecolor=BG)
    plt.close(fig)
    print("записал", path_png, "и pdf")
