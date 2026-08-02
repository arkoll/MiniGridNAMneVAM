"""Сводка: три датасета жадно и под сэмплированием, чекпоинты 30 000 шагов."""

import json
import os

import numpy as np

BASE = "/home/user8_2/AIRI_WAM/reports/xland/datasets_T2"
MAX_STEPS = 192

NAMES = [
    ("mixed_random_labels", "смешанный, метки случайные"),
    ("expert_only", "чисто экспертный"),
    ("mixed_expert_labels", "смешанный + метки эксперта"),
]
# прежние жадные числа на тех же банках и сидах (чекпоинты 30 000 шагов)
GREEDY = {
    "mixed_random_labels": {"val": 0.380, "train": 0.430},
    "expert_only": {"val": 0.335, "train": 0.345},
    "mixed_expert_labels": {"val": None, "train": None},
}


def load(path):
    d = json.load(open(path))
    eps = None
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "success" in v[0]:
            eps = v
            break
    s = np.array([e["success"] for e in eps], dtype=bool)
    ln = np.array([e["length"] for e in eps], dtype=float)
    r = np.where(s, 1.0 - 0.9 * ln / MAX_STEPS, 0.0)
    return {
        "sr": s.mean(),
        "err": np.sqrt(s.mean() * (1 - s.mean()) / len(s)),
        "reward": r.mean(),
        "win_len": ln[s].mean() if s.any() else float("nan"),
        "n": len(s),
    }


print("Все под температурой 2.0, по 200 эпизодов, чекпоинты 30 000 шагов, сиды 21/22 млн.\n")
print(f"{'датасет':<30}{'банк':<8}{'жадно':>8}{'T=2':>9}{'ошибка':>9}{'награда':>10}{'шагов/победа':>15}")
for key, label in NAMES:
    for bank, rus in (("val", "отлож."), ("train", "знаком.")):
        f = os.path.join(BASE, key, f"closed_loop_{bank}.json")
        if not os.path.exists(f):
            continue
        st = load(f)
        g = GREEDY[key][bank]
        gs = f"{g:.3f}" if g is not None else "—"
        print(f"{label:<30}{rus:<8}{gs:>8}{st['sr']:>9.3f}{st['err']:>9.3f}"
              f"{st['reward']:>10.3f}{st['win_len']:>15.1f}")

print("\nДля сравнения: тот же смешанный+эксперт на 120 000 шагов даёт 0.905 отлож. / 0.925 знаком.")
print("Привилегированный эксперт при T=2: 0.985 отлож. / 1.000 знаком.")
