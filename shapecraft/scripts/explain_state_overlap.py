"""Почему 6% состояний валидации встречаются в трейне, хотя палитры не пересекаются.

Гипотеза: после того как обе стадии крафта отработали, ингредиенты съедены, и на поле
остаются только стены, пол, дверь, синий ключ и жёлтая звезда — плитки, не зависящие от
палитры. Такие состояния обязаны совпадать между сплитами.

Если гипотеза верна, почти все пересекающиеся состояния не содержат ни одного объекта
палитры. Это важно для замера: в таких окнах сигнала об OOD нет вообще.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, "/home/user8_2/AIRI_WAM/scripts/xland")
import shape_common as sc  # noqa: E402

PALETTE_SHAPES = {3, 4, 5, 11}  # ball, square, pyramid, hex


def hsh(g, p, d, k):
    return hashlib.blake2b(
        np.concatenate([g.reshape(-1), p, [d], k]).tobytes(), digest_size=16
    ).hexdigest()


def scan(set_name, limit):
    files = sorted(glob.glob(os.path.join(sc.DATA_ROOT, set_name, "ep_*.h5")))[:limit]
    states = {}
    for path in files:
        with h5py.File(path, "r") as h:
            g, pp, dd, pk = h["grid"][:], h["agent_pos"][:], h["agent_dir"][:], h["agent_pocket"][:]
        for k in range(g.shape[0]):
            has_obj = bool(np.isin(g[k][:, :, 0], list(PALETTE_SHAPES)).any())
            states[hsh(g[k], pp[k], dd[k], pk[k])] = has_obj
    return states


LIM = 1500
train = scan("train", LIM)
print(f"train: {len(train)} уникальных состояний из {LIM} эпизодов")

report = {}
for s in ("val_id", "val_ood"):
    st = scan(s, LIM)
    shared = [h for h in st if h in train]
    shared_with_obj = sum(1 for h in shared if st[h])
    with_obj_total = sum(1 for v in st.values() if v)
    report[s] = {
        "states": len(st),
        "shared_with_train": len(shared),
        "shared_share": len(shared) / len(st),
        "shared_that_contain_palette_object": shared_with_obj,
        "states_with_palette_object": with_obj_total,
        "overlap_among_states_with_object": shared_with_obj / max(with_obj_total, 1),
    }
    print(
        f"{s}: состояний {len(st)}, совпало с train {len(shared)} ({len(shared) / len(st):.4f});\n"
        f"      из них содержат объект палитры: {shared_with_obj};\n"
        f"      состояний с объектом палитры всего {with_obj_total}, "
        f"их доля совпадения с train {shared_with_obj / max(with_obj_total, 1):.4f}"
    )

with open("/home/user8_2/AIRI_WAM/reports/xland/state_overlap_explained.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("OVERLAP_STATUS=OK")
