"""Развёртка по температуре для модели на полностью экспертном датасете. Только доля побед.

Оформление намеренно повторяет temp_success_rate.png, чтобы два графика читались рядом:
цвет — модель, штрих — банк задач, те же температуры, те же 200 эпизодов на точку,
те же сиды. Эталон привилегированного эксперта берётся из уже посчитанной сетки.
"""

from __future__ import annotations

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

V1 = "/home/user8_2/AIRI_WAM/reports/xland/temp_v1"
EXP = "/home/user8_2/AIRI_WAM/reports/xland/stochastic"
DST = os.path.join(V1, "temp_success_rate_expert_dataset.png")
WAM, EXPC = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#14150f", "#55574e", "#dcded6"


def load_model():
    out = {"val": [], "train": []}
    for d in sorted(glob.glob(f"{V1}/T*/")):
        t = float(re.search(r"/T([0-9.]+)/$", d).group(1))
        for bank in ("val", "train"):
            f = os.path.join(d, f"closed_loop_{bank}.json")
            if os.path.exists(f):
                j = json.load(open(f))
                out[bank].append((t, j["success_rate"], j["success_rate_stderr"]))
    for k in out:
        out[k].sort()
    return out


def load_expert():
    out = {"val": [], "train": []}
    for f in sorted(glob.glob(f"{EXP}/expert_T*.json")):
        t = float(re.search(r"expert_T([0-9.]+)\.json", f).group(1))
        j = json.load(open(f))
        for bank in ("val", "train"):
            if bank in j:
                out[bank].append((t, j[bank]["success_rate"], j[bank]["success_rate_stderr"]))
    for k in out:
        out[k].sort()
    return out


model, expert = load_model(), load_expert()
if not model["val"]:
    print("нет ни одной точки модели")
    raise SystemExit

SERIES = [
    (model["val"], WAM, "-", "экспертный датасет · отложенные (val_ood)"),
    (model["train"], WAM, "--", "экспертный датасет · знакомые (val_id)"),
    (expert["val"], EXPC, "-", "привилегированный эксперт · отложенные"),
    (expert["train"], EXPC, "--", "привилегированный эксперт · знакомые"),
]

fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=170)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for rows, color, style, label in SERIES:
    if not rows:
        continue
    x = [r[0] for r in rows]
    y = [r[1] for r in rows]
    e = [r[2] for r in rows]
    ax.errorbar(x, y, yerr=e, color=color, ls=style, lw=2, marker="o", ms=5.5,
                markeredgecolor="white", markeredgewidth=1.1,
                elinewidth=1, capsize=0, ecolor=color, alpha=0.95, label=label, zorder=3)

ax.set_xlabel("температура выбора действия   (0 — жадно, argmax)", fontsize=10.5, color=INK2)
ax.set_ylabel("Доля решённых задач", fontsize=10.5, color=INK2)
ax.set_title("Доля побед по температуре: модель на полностью экспертном датасете",
             fontsize=12.5, color=INK, pad=14, loc="left")
ax.set_xlim(-0.25, 6.35)
ax.set_ylim(0.25, 1.03)
ax.set_xticks([0, 1, 2, 3, 4, 6])
ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9.5)
leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                frameon=False, fontsize=9.5, handlelength=3.4,
                columnspacing=2.4, handletextpad=0.8)
for t in leg.get_texts():
    t.set_color(INK2)

fig.tight_layout()
fig.savefig(DST, facecolor="white")
print("записал", DST)
print("точек модели:", len(model["val"]), "отложенные,", len(model["train"]), "знакомые")
