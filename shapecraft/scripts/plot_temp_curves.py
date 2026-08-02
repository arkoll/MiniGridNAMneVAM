"""Два графика развёртки по температуре: доля побед и исходная награда среды.

Цвет — модель, штрих — банк задач. val_id это банк обучающих задач с невиданными сидами
(знакомая привязка цвета к форме, новая раскладка), val_ood — валидационные задачи
(комплементарные привязки).
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = "/home/user8_2/AIRI_WAM/reports/xland/stochastic/chart_data.json"
DST = "/home/user8_2/AIRI_WAM/reports/xland/stochastic"
WAM, EXP = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#14150f", "#55574e", "#dcded6"

SERIES = [
    ("wam", "val_ood", WAM, "-", "WAM · отложенные (val_ood)"),
    ("wam", "val_id", WAM, "--", "WAM · знакомые (val_id)"),
    ("expert", "val_ood", EXP, "-", "эксперт · отложенные"),
    ("expert", "val_id", EXP, "--", "эксперт · знакомые"),
]

PANELS = [
    ("sr", "sr_err", "Доля решённых задач",
     "Доля побед в замкнутом контуре по температуре раскатки", "temp_success_rate.png", (0.25, 1.03)),
    ("reward", "reward_err", "Средняя награда за эпизод",
     "Награда среды по температуре раскатки (учитывает скорость)", "temp_reward.png", (0.05, 0.93)),
]

data = json.load(open(SRC))

for key, errkey, ylab, title, fname, ylim in PANELS:
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=170)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.axvline(2, color=INK2, lw=1, ls=(0, (1, 4)), alpha=0.6, zorder=1)
    ax.annotate("рабочая точка WAM", xy=(2.12, ylim[0] + 0.03 * (ylim[1] - ylim[0])),
                ha="left", va="bottom", fontsize=8.5, color=INK2)

    for who, bank, color, style, label in SERIES:
        rows = data[who][bank]
        x = [r["t"] for r in rows]
        y = [r[key] for r in rows]
        e = [r[errkey] for r in rows]
        ax.errorbar(x, y, yerr=e, color=color, ls=style, lw=2, marker="o", ms=5.5,
                    markeredgecolor="white", markeredgewidth=1.1,
                    elinewidth=1, capsize=0, ecolor=color, alpha=0.95,
                    label=label, zorder=3)

    ax.set_xlabel("температура выбора действия   (0 — жадно, argmax)", fontsize=10.5, color=INK2)
    ax.set_ylabel(ylab, fontsize=10.5, color=INK2)
    ax.set_title(title, fontsize=12.5, color=INK, pad=14, loc="left")
    ax.set_xlim(-0.25, 6.35)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 1, 2, 3, 4, 6])
    ax.grid(True, color=GRID, lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5)
    # легенда под осями: четыре кривые в поле перекрывают данные при любом угле
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                    frameon=False, fontsize=9.5, handlelength=3.4,
                    columnspacing=2.4, handletextpad=0.8)
    for text in leg.get_texts():
        text.set_color(INK2)

    fig.text(0.012, 0.012,
             "чекпоинт 120 000 шагов · веса не менялись, менялось только правило выбора действия · "
             "200 эпизодов на точку · усы — стандартная ошибка",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = os.path.join(DST, fname)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("записал", out)

print("PLOT_TEMP_STATUS=OK")
