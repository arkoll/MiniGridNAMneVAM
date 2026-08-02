"""Сводка по развёртке: WAM и эксперт на одной сетке температур, банк val."""

import glob
import json
import os
import re

BASE = "/home/user8_2/AIRI_WAM/reports/xland/stochastic"


def rows_wam():
    out = []
    for d in sorted(glob.glob(f"{BASE}/*/")):
        name = os.path.basename(d.rstrip("/"))
        f = os.path.join(d, "closed_loop_val.json")
        if not os.path.exists(f):
            continue
        j = json.load(open(f))
        if name.startswith("temp_"):
            key, label = float(name[5:]), f"T={float(name[5:]):g}"
        elif name.startswith("eps_"):
            key, label = 1e9 + float(name[4:]), f"шум {float(name[4:]):g}"
        else:
            key, label = 2e9, name
        out.append((key, label, j["success_rate"], j["success_rate_stderr"],
                    j["mean_length"], j.get("mean_length_on_success")))
    return sorted(out)


def rows_expert():
    out = []
    for f in sorted(glob.glob(f"{BASE}/expert_T*.json")):
        t = float(re.search(r"expert_T([0-9.]+)\.json", f).group(1))
        j = json.load(open(f))["val"]
        out.append((t, "жадно" if t == 0 else f"T={t:g}", j["success_rate"],
                    j["success_rate_stderr"], j["mean_length"], j.get("mean_length_on_success")))
    return sorted(out)


def table(title, rows):
    print(f"\n{title}")
    print(f"  {'политика':<14}{'победы':>9}{'ошибка':>9}{'длина':>9}{'длина побед':>13}")
    for _, label, sr, se, ml, mls in rows:
        mls_s = f"{mls:.1f}" if mls is not None else "—"
        print(f"  {label:<14}{sr:>9.3f}{se:>9.3f}{ml:>9.1f}{mls_s:>13}")


w, e = rows_wam(), rows_expert()
print("Банк val (комплементарные привязки), 200 эпизодов, сиды от 22 000 000.")
print("WAM: чекпоинт 120 000 шагов, веса не менялись — меняется только правило выбора действия.")
if w:
    table("WAM", w)
if e:
    table("Привилегированный эксперт (видит правила)", e)
if w and e:
    bw = max(w, key=lambda r: r[2])
    be = max(e, key=lambda r: r[2])
    d = bw[2] - be[2]
    sd = (bw[3] ** 2 + be[3] ** 2) ** 0.5
    print(f"\nЛучший WAM: {bw[1]} → {bw[2]:.3f}   |   лучший эксперт: {be[1]} → {be[2]:.3f}")
    print(f"Разница {d:+.3f} при ошибке {sd:.3f} ({abs(d) / sd:.1f} стандартных отклонения)")
