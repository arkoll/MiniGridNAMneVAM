"""Три графика в едином стиле: две развёртки по температуре и сравнение датасетов.

Цвета серий назначаются по порядку из стиля: первая серия — #445469, вторая — #2cbeac.
В развёртках цвет отвечает за политику (наша модель против PPO), штрих — за банк задач.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, "/home/user8_2/AIRI_WAM/scripts/xland")
from plot_style import LW, MS, SERIES, finish, new_figure, save  # noqa: E402

OUT = "/home/user8_2/AIRI_WAM/reports/xland/styled"
os.makedirs(OUT, exist_ok=True)
MAX_STEPS = 192


def sr_err(sr, n):
    return float(np.sqrt(sr * (1 - sr) / n))


# --------------------------------------------------------------- источники данных
def load_dagger_curve():
    """Прежняя развёртка: модель на смешанном датасете с экспертной разметкой."""
    d = json.load(open("/home/user8_2/AIRI_WAM/reports/xland/stochastic/chart_data.json"))
    out = {}
    for who in ("wam", "expert"):
        for bank in ("val_id", "val_ood"):
            out[(who, bank)] = [(r["t"], r["sr"], r["sr_err"]) for r in d[who][bank]]
    return out


def load_v1_curve():
    """Новая развёртка: модель на полностью экспертном датасете."""
    out = {("wam", "val_ood"): [], ("wam", "val_id"): []}
    for dr in sorted(glob.glob("/home/user8_2/AIRI_WAM/reports/xland/temp_v1/T*/")):
        t = float(re.search(r"/T([0-9.]+)/$", dr).group(1))
        for bank, key in (("val", "val_ood"), ("train", "val_id")):
            f = os.path.join(dr, f"closed_loop_{bank}.json")
            if os.path.exists(f):
                j = json.load(open(f))
                out[("wam", key)].append((t, j["success_rate"], j["success_rate_stderr"]))
    for f in sorted(glob.glob("/home/user8_2/AIRI_WAM/reports/xland/stochastic/expert_T*.json")):
        t = float(re.search(r"expert_T([0-9.]+)\.json", f).group(1))
        j = json.load(open(f))
        for bank, key in (("val", "val_ood"), ("train", "val_id")):
            out.setdefault(("expert", key), []).append(
                (t, j[bank]["success_rate"], j[bank]["success_rate_stderr"]))
    for k in out:
        out[k].sort()
    return out


def load_final_bars():
    base = "/home/user8_2/AIRI_WAM/reports/xland/final"
    models = [
        ("final_v1", "Только экспертные\nсостояния и действия"),
        ("final_v2", "50/50 состояний,\nдействия экспертные"),
        ("final_v3", "50/50 состояний,\nдействия фактические"),
        ("final_v4b", "Двухфазно: сначала мир,\nпотом действия"),
    ]
    rows = []
    for key, label in models:
        cell = {}
        for bank, name in (("train", "val_id"), ("val", "val_ood")):
            f = f"{base}/{key}_T1.0/closed_loop_{bank}.json"
            if os.path.exists(f):
                j = json.load(open(f))
                cell[name] = (j["success_rate"], j["success_rate_stderr"])
        rows.append((label, cell))
    return rows


# ------------------------------------------------------------------- построение
def draw_curve(data, title, dst, ppo=True):
    fig, ax = new_figure()
    plan = [
        (("wam", "val_ood"), SERIES[0], "-", "Модель WAM · отложенные задачи (val_ood)"),
        (("wam", "val_id"), SERIES[0], "--", "Модель WAM · знакомые задачи (val_id)"),
        (("expert", "val_ood"), SERIES[1], "-", "Политика PPO · отложенные задачи (val_ood)"),
        (("expert", "val_id"), SERIES[1], "--", "Политика PPO · знакомые задачи (val_id)"),
    ]
    for key, color, style, label in plan:
        rows = data.get(key) or []
        if not rows:
            continue
        x = [r[0] for r in rows]
        y = [r[1] for r in rows]
        e = [r[2] for r in rows]
        ax.errorbar(x, y, yerr=e, color=color, ls=style, lw=LW, marker="o", ms=MS,
                    markeredgecolor="#ffffff", markeredgewidth=0.9,
                    elinewidth=0.9, capsize=0, ecolor=color, label=label, zorder=3)
    ax.set_xlim(-0.25, 6.35)
    ax.set_xticks([0, 1, 2, 3, 4, 6])
    finish(fig, ax, title, xlabel="Температура выбора действия  (0 — argmax)")
    save(fig, dst)


def draw_bars(rows, title, dst):
    fig, ax = new_figure(figsize=(9.6, 5.6))
    x = np.arange(len(rows))
    w = 0.36
    for off, name, color, label in ((-w / 2, "val_id", SERIES[0], "Знакомые задачи (val_id)"),
                                    (w / 2, "val_ood", SERIES[1], "Отложенные задачи (val_ood)")):
        vals = [r[1].get(name, (0, 0))[0] for r in rows]
        errs = [r[1].get(name, (0, 0))[1] for r in rows]
        ax.bar(x + off, vals, w, color=color, label=label, zorder=3,
               edgecolor="#ffffff", linewidth=1.2)
        ax.errorbar(x + off, vals, yerr=errs, fmt="none", ecolor="#111111",
                    elinewidth=1.1, capsize=3, zorder=4)
        for xi, v in zip(x + off, vals):
            if v > 0:
                ax.text(xi, v + 0.018, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=9.5, color="#111111", fontweight="bold", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    # столбцы высокие, внутри области графика легенде места нет — выносим под ось справа
    finish(fig, ax, title, xlabel=None, legend_loc=("upper right", (1.0, -0.11)))
    save(fig, dst)


def draw_combined(dst):
    """Обе развёртки на одном поле, чтобы сравнивать величины напрямую.

    Цвета PPO и смешанного датасета сохранены теми же, что на отдельных графиках, поэтому
    полностью экспертный датасет получает третий слот палитры (#515eea).
    """
    dag, v1 = load_dagger_curve(), load_v1_curve()
    fig, ax = new_figure(figsize=(9.4, 6.0))
    plan = [
        (dag.get(("wam", "val_ood")), SERIES[0], "-", "Смешанный датасет · отложенные (val_ood)"),
        (dag.get(("wam", "val_id")), SERIES[0], "--", "Смешанный датасет · знакомые (val_id)"),
        (v1.get(("wam", "val_ood")), SERIES[2], "-", "Экспертный датасет · отложенные (val_ood)"),
        (v1.get(("wam", "val_id")), SERIES[2], "--", "Экспертный датасет · знакомые (val_id)"),
        (dag.get(("expert", "val_ood")), SERIES[1], "-", "Политика PPO · отложенные (val_ood)"),
        (dag.get(("expert", "val_id")), SERIES[1], "--", "Политика PPO · знакомые (val_id)"),
    ]
    for rows, color, style, label in plan:
        if not rows:
            continue
        x = [r[0] for r in rows]
        y = [r[1] for r in rows]
        e = [r[2] for r in rows]
        ax.errorbar(x, y, yerr=e, color=color, ls=style, lw=LW, marker="o", ms=MS,
                    markeredgecolor="#ffffff", markeredgewidth=0.9,
                    elinewidth=0.9, capsize=0, ecolor=color, label=label, zorder=3)
    ax.set_xlim(-0.25, 6.35)
    ax.set_xticks([0, 1, 2, 3, 4, 6])
    # шесть подписей внутри поля не помещаются — выносим под ось в две колонки
    finish(fig, ax, "Доля решённых задач по температуре: два способа собрать датасет",
           xlabel="Температура выбора действия  (0 — argmax)",
           legend_loc=("upper right", (1.0, -0.13)), legend_ncol=2)
    save(fig, dst)


draw_combined(os.path.join(OUT, "temp_success_rate_both.png"))

draw_curve(load_dagger_curve(),
           "Доля решённых задач по температуре: смешанный датасет с экспертной разметкой",
           os.path.join(OUT, "temp_success_rate.png"))

draw_curve(load_v1_curve(),
           "Доля решённых задач по температуре: полностью экспертный датасет",
           os.path.join(OUT, "temp_success_rate_expert_dataset.png"))

draw_bars(load_final_bars(),
          "Доля решённых задач при T = 1: четыре способа собрать обучающие данные",
          os.path.join(OUT, "final_T1.png"))
print("STYLED_PLOTS_OK")
