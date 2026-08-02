"""Два графика итогового сравнения: по одному на температуру.

Ось x — четыре подхода к составу датасета. Ось y — доля побед. По два столбца на подход:
знакомые привязки цвета к форме (val_id) и отложенные (val_ood).
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = "/home/user8_2/AIRI_WAM/reports/xland/final"
ID, OOD = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#14150f", "#55574e", "#dcded6"

MODELS = [
    ("final_v1", "1. только\nэкспертные\nсостояния"),
    ("final_v2", "2. 50/50 состояний,\nдействия\nэкспертные"),
    ("final_v3", "3. 50/50 состояний,\nдействия\nфактические"),
    ("final_v4b", "4. двухфазно:\nсначала мир,\nпотом действия"),
]
TEMPS = [("1.0", "T = 1  (сэмплирование из распределения модели)", "final_T1.png"),
         ("0", "T = 0  (argmax, жадная раскатка)", "final_T0.png")]


def load(model, temp, bank):
    f = os.path.join(OUT, f"{model}_T{temp}", f"closed_loop_{bank}.json")
    if not os.path.exists(f):
        return None, None
    d = json.load(open(f))
    sr = d["success_rate"]
    return sr, np.sqrt(sr * (1 - sr) / d["episodes"])


for temp, title, fname in TEMPS:
    rows = []
    for key, label in MODELS:
        a, ae = load(key, temp, "train")   # знакомые привязки
        b, be = load(key, temp, "val")     # отложенные
        rows.append((label, a, ae, b, be))
    if all(r[1] is None and r[3] is None for r in rows):
        print(f"нет данных для T={temp}, пропускаю")
        continue

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x = np.arange(len(rows))
    w = 0.36

    for off, ci, color, lab in ((-w / 2, 1, ID, "знакомые привязки (val_id)"),
                                (w / 2, 3, OOD, "отложенные привязки (val_ood)")):
        vals = [r[ci] if r[ci] is not None else 0 for r in rows]
        errs = [r[ci + 1] if r[ci + 1] is not None else 0 for r in rows]
        bars = ax.bar(x + off, vals, w, color=color, label=lab, zorder=3,
                      edgecolor="white", linewidth=1.4)
        ax.errorbar(x + off, vals, yerr=errs, fmt="none", ecolor=INK2,
                    elinewidth=1.2, capsize=3, zorder=4)
        for xi, v in zip(x + off, vals):
            if v > 0:
                ax.text(xi, v + 0.022, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=9.5, color=INK, fontweight="bold")

    ax.axhline(0.985, color=INK2, lw=1, ls=(0, (4, 4)), zorder=2)
    ax.text(len(rows) - 0.42, 0.995, "привилегированный эксперт, 0.985",
            ha="right", va="bottom", fontsize=8.5, color=INK2)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylabel("Доля решённых задач", fontsize=11, color=INK2)
    ax.set_title(title, fontsize=13, color=INK, pad=14, loc="left")
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=10)
    # подписи столбцов многострочные, поэтому легенду отодвигаем заметно ниже
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=2,
                    frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.tight_layout()
    path = os.path.join(OUT, fname)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("записал", path)

print("PLOT_FINAL_STATUS=OK")
