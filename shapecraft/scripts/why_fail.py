"""Как именно проигрывают эпизоды: упираются в лимит шагов или заканчиваются иначе."""

import glob
import json
from collections import Counter

FILES = sorted(glob.glob("/home/user8_2/AIRI_WAM/live/eval/step_*/closed_loop_val.json")) + [
    "/home/user8_2/AIRI_WAM/reports/xland/closed_loop_dagger/final/closed_loop_val.json",
]

print(f"{'источник':<26}{'побед':>7}{'провалов':>10}{'провалы на лимите':>20}{'длина побед':>14}")
for f in FILES:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    eps = d.get("episodes_detail") or d.get("episodes_list") or d.get("detail") or []
    if not eps:
        for k, v in d.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "success" in v[0]:
                eps = v
                break
    if not eps:
        print(f"{f.split('/')[-2]:<26} нет подробностей по эпизодам")
        continue
    win = [e for e in eps if e["success"]]
    fail = [e for e in eps if not e["success"]]
    at_limit = sum(1 for e in fail if e["length"] >= 192)
    name = f.split("/")[-2]
    wl = sum(e["length"] for e in win) / max(len(win), 1)
    print(f"{name:<26}{len(win):>7}{len(fail):>10}{at_limit:>15} из {len(fail):<4}{wl:>14.1f}")
    if fail and at_limit < len(fail):
        print("   длины провалов не на лимите:", Counter(e["length"] for e in fail if e["length"] < 192).most_common(6))
