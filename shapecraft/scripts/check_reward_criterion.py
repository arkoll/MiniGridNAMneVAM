"""Что на самом деле означает наш критерий успеха: сверяем награду и длину эпизода."""

import glob
import json

import numpy as np

# экспертные шарды: там лежат и длина, и финальная награда
shards = sorted(glob.glob("/home/user8_2/AIRI_WAM/data/xland-expert/_shards/val_ood/shard_*.npz"))[:4]
L, R = [], []
for s in shards:
    z = np.load(s)
    L.append(np.asarray(z["length"]))
    R.append(np.asarray(z["final_reward"]))
L, R = np.concatenate(L), np.concatenate(R)

print(f"эпизодов: {len(L)}")
print(f"длина: min {L.min()}, медиана {np.median(L):.0f}, max {L.max()}")
print(f"награда: min {R.min():.4f}, медиана {np.median(R):.4f}, max {R.max():.4f}")
print(f"наград >= 0.9: {(R >= 0.9).mean():.4f}")

# если награда = 1 - 0.9*t/max_steps, то по награде восстанавливается max_steps
implied = 0.9 * L / (1.0 - R)
print(f"\nвосстановленный max_steps из формулы 1 - 0.9*t/max_steps: "
      f"медиана {np.median(implied):.1f}, min {implied.min():.1f}, max {implied.max():.1f}")

# порог по награде в шагах
for thr in (0.9, 0.8, 0.5):
    steps = (1 - thr) * np.median(implied) / 0.9
    print(f"порог награды {thr}: это решение не длиннее {steps:.0f} шагов")

print("\nпримеры (длина -> награда):")
for i in np.argsort(L)[:: max(1, len(L) // 8)][:8]:
    print(f"  {L[i]:>3} шагов -> {R[i]:.4f}")

# в замкнутом контуре
with open("/home/user8_2/AIRI_WAM/reports/xland/closed_loop/closed_loop_val.json") as f:
    cl = json.load(f)
lens = np.array([r["length"] for r in cl["results"]])
wins = np.array([r["success"] for r in cl["results"]])
print(f"\nзамкнутый контур: победных {wins.sum()} из {len(wins)}")
if wins.any():
    print(f"  длина победных: min {lens[wins].min()}, медиана {np.median(lens[wins]):.0f}, max {lens[wins].max()}")
print(f"  длина непобедных: медиана {np.median(lens[~wins]):.0f}, доля упёршихся в лимит "
      f"{(lens[~wins] >= 192).mean():.3f}")
print("CRITERION_STATUS=OK")
