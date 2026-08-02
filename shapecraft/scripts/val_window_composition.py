"""Состав зафиксированных окон валидации: чем они собраны и чем закончился эпизод.

Нужно для потолка: 1/6 эпизодов собрана СЛУЧАЙНОЙ политикой, и на её окнах действие
предсказать нельзя в принципе. Если доля таких окон одинакова в val_id и val_ood, потолок
опущен одинаково и сравнение остаётся честным.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

RUN = "/home/user8_2/AIRI_WAM/runs/shapecraft_joint_128"
VAL_DIR = "/home/user8_2/AIRI_WAM/data/xland-shape-joint128/val_both"

files = sorted(os.path.join(VAL_DIR, f) for f in os.listdir(VAL_DIR) if f.startswith("episode_train"))
with open(os.path.join(RUN, "validation_subset.json")) as f:
    slices = json.load(f)["slices"]

meta_cache = {}
stats = {"val_id": Counter(), "val_ood": Counter()}
for s in slices:
    idx = int(s["traj_idx"])
    if idx >= len(files):
        continue
    path = files[idx]
    if idx not in meta_cache:
        with np.load(path, allow_pickle=False) as ep:
            meta_cache[idx] = json.loads(str(ep["metadata"]))
    m = meta_cache[idx]
    which = "val_ood" if "_ood_" in os.path.basename(path) else "val_id"
    stats[which][m["policy"]] += 1
    stats[which][m["outcome"]] += 1
    stats[which]["всего"] += 1

out = {}
for name, c in stats.items():
    total = c["всего"]
    if not total:
        continue
    out[name] = {
        "windows": total,
        "random_policy_share": c["random"] / total,
        "train_sr_50_share": c["train_sr_50"] / total,
        "success_share": c["success"] / total,
        "timeout_share": c["timeout"] / total,
    }
    print(
        f"{name}: окон {total}, из них собрано случайной политикой {c['random'] / total:.3f}, "
        f"обученной {c['train_sr_50'] / total:.3f}; успешных эпизодов {c['success'] / total:.3f}"
    )

with open("/home/user8_2/AIRI_WAM/reports/xland/joint_run/val_composition.json", "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("COMPOSITION_STATUS=OK")
