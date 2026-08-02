"""Данные для графика «температура против качества»: два банка, две метрики, две политики.

Метрики:
  победы  — доля эпизодов, где среда завершилась достижением цели;
  награда — ИСХОДНАЯ награда XLand: 1 - 0.9*шаг/лимит в момент достижения цели, иначе 0.
            Она сама учитывает скорость: дошёл быстро — ближе к 1, на последнем шаге — 0.1,
            не дошёл — 0. Считается точно из (успех, длина), лимит 192.

Названия банков. Банк train с невиданными сидами (21 млн) — это val_id: знакомая привязка
цвета к форме, новая раскладка. Банк val (22 млн) — val_ood: комплементарные привязки.
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

BASE = "/home/user8_2/AIRI_WAM/reports/xland/stochastic"
DAGGER = "/home/user8_2/AIRI_WAM/reports/xland/closed_loop_dagger/final"
MAX_STEPS = 192


def reward_of(success, length):
    s = np.asarray(success, dtype=bool)
    ln = np.asarray(length, dtype=float)
    return np.where(s, 1.0 - 0.9 * ln / MAX_STEPS, 0.0)


def stats(success, length):
    s = np.asarray(success, dtype=bool)
    n = len(s)
    r = reward_of(s, length)
    sr = float(s.mean())
    return {
        "n": n,
        "sr": round(sr, 4),
        "sr_err": round(float(np.sqrt(sr * (1 - sr) / n)), 4),
        "reward": round(float(r.mean()), 4),
        "reward_err": round(float(r.std(ddof=1) / np.sqrt(n)), 4),
        "mean_length": round(float(np.mean(length)), 1),
    }


def from_episode_file(path):
    d = json.load(open(path))
    eps = None
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "success" in v[0]:
            eps = v
            break
    if eps is None:
        return None
    return stats([e["success"] for e in eps], [e["length"] for e in eps])


out = {"wam": {"val_id": [], "val_ood": []}, "expert": {"val_id": [], "val_ood": []}}

# --- WAM, точка T=0 это прежние жадные замеры
for bank, fname in (("val_id", "closed_loop_train.json"), ("val_ood", "closed_loop_val.json")):
    st = from_episode_file(os.path.join(DAGGER, fname))
    if st:
        out["wam"][bank].append({"t": 0.0, **st})

# --- WAM, банк val_ood
for d in sorted(glob.glob(f"{BASE}/temp_*/closed_loop_val.json")):
    t = float(re.search(r"temp_([0-9.]+)/", d).group(1))
    st = from_episode_file(d)
    if st:
        out["wam"]["val_ood"].append({"t": t, **st})

# --- WAM, банк val_id
for d in sorted(glob.glob(f"{BASE}/train_temp_*/closed_loop_train.json")):
    t = float(re.search(r"train_temp_([0-9.]+)/", d).group(1))
    st = from_episode_file(d)
    if st:
        out["wam"]["val_id"].append({"t": t, **st})

# --- эксперт, обе банки из одного файла на температуру
for f in sorted(glob.glob(f"{BASE}/expert_T*.json")):
    t = float(re.search(r"expert_T([0-9.]+)\.json", f).group(1))
    d = json.load(open(f))
    for bank, key in (("val_id", "train"), ("val_ood", "val")):
        b = d.get(key)
        if not b or "success" not in b:
            continue
        out["expert"][bank].append({"t": t, **stats(b["success"], b["length"])})

for who in out:
    for bank in out[who]:
        out[who][bank].sort(key=lambda r: r["t"])

dst = "/home/user8_2/AIRI_WAM/reports/xland/stochastic/chart_data.json"
with open(dst, "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

for who in ("wam", "expert"):
    for bank in ("val_id", "val_ood"):
        rows = out[who][bank]
        print(f"\n{who} / {bank}: {len(rows)} точек")
        for r in rows:
            print(f"   T={r['t']:<4} победы {r['sr']:.3f}±{r['sr_err']:.3f}  "
                  f"награда {r['reward']:.3f}±{r['reward_err']:.3f}  длина {r['mean_length']}")
print(f"\n-> {dst}")
