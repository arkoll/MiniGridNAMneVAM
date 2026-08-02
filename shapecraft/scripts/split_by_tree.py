"""Разбор побед по классу задачи: определяется правило из кадра или нет.

На поле ставится РОВНО три объекта (five_object_crafting.py: num_objects = len(init_tiles)),
и это в точности пара первого этапа плюс катализатор второго. Отвлекающих объектов нет.

  STAGE1_ROLE_INDICES = ((0,1), (0,2), (1,2))     пара первого этапа
  initial_role_indices = (a, b, stage2_idx)       что лежит на поле

Значит у 6 деревьев из 9 одна форма повторяется дважды — по набору форм дерево
восстанавливается однозначно. У 3 деревьев на поле по одному шару, кубу и пирамиде,
и они между собой неразличимы: какая пара из трёх крафтит, из кадра не видно.
Цвет тут не помогает — он приписан форме палитрой, а палитра от дерева не зависит.

Задачи нумеруются деревом-старшим разрядом: 36 палитр на дерево, дерево = task // 36.
Порядок деревьев — product(range(3), repeat=2), то есть (stage1_idx, stage2_idx).
"""

from __future__ import annotations

import glob
import json
import os
import re
from itertools import product

import numpy as np

STAGE1_ROLE_INDICES = ((0, 1), (0, 2), (1, 2))
TREES = list(product(range(3), repeat=2))  # 0-based индекс дерева -> (stage1, stage2)
PALETTES = 36
MAX_STEPS = 192


def tree_of(task0):
    return task0 // PALETTES


def is_ambiguous(tree0):
    s1, s2 = TREES[tree0]
    return s2 not in STAGE1_ROLE_INDICES[s1]


def load(path):
    d = json.load(open(path))
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "success" in v[0]:
            return v
    return []


def report(name, eps):
    groups = {"правило видно": [], "3 варианта": []}
    for e in eps:
        t = tree_of(e["task"])
        if t >= len(TREES):
            continue
        groups["3 варианта" if is_ambiguous(t) else "правило видно"].append(e)
    print(f"\n{name}")
    print(f"  {'класс задачи':<16}{'эпизодов':>10}{'победы':>9}{'ошибка':>9}{'шагов на победу':>18}")
    for k, g in groups.items():
        if not g:
            continue
        s = np.array([e["success"] for e in g], dtype=bool)
        ln = np.array([e["length"] for e in g], dtype=float)
        sr = s.mean()
        err = np.sqrt(sr * (1 - sr) / len(s))
        wl = ln[s].mean() if s.any() else float("nan")
        print(f"  {k:<16}{len(g):>10}{sr:>9.3f}{err:>9.3f}{wl:>18.1f}")
    return groups


BASE = "/home/user8_2/AIRI_WAM/reports/xland/stochastic"
print("Деревья, где правило НЕ восстанавливается из кадра (0-based):",
      [t for t in range(9) if is_ambiguous(t)])
print("Доля таких задач:", sum(is_ambiguous(t) for t in range(9)), "из 9")

for path, name in (
    (f"{BASE}/temp_2.0/closed_loop_val.json", "WAM, T=2, отложенные задачи"),
    (f"{BASE}/train_temp_2.0/closed_loop_train.json", "WAM, T=2, знакомые задачи"),
    ("/home/user8_2/AIRI_WAM/reports/xland/closed_loop_dagger/final/closed_loop_val.json",
     "WAM, жадно, отложенные задачи"),
):
    if os.path.exists(path):
        report(name, load(path))
