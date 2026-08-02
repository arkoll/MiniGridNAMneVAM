import glob
import json
import sys

import numpy as np

root = sys.argv[1]
fs = sorted(glob.glob(f"{root}/episode_train*.npz"))
starts, shapes = [], set()
for f in fs[:200]:
    z = np.load(f)
    m = json.loads(str(z["metadata"]))
    starts.append(m["slice_start"])
    shapes.add((z["rgb"].shape[0], z["actions"].shape[0], z["executed"].shape[0]))
print("файлов:", len(fs))
print("формы (кадров, действий, выполненных):", shapes)
print("начало отрезка: мин", min(starts), "макс", max(starts), "среднее", round(float(np.mean(starts)), 1))
print("разных начал:", len(set(starts)), "из", len(starts))
z = np.load(fs[0])
assert z["rgb"].shape[0] == z["actions"].shape[0] + 1, "рассинхрон кадров и действий"
print("метка/выполнено первые 8:", z["actions"][:8].tolist(), z["executed"][:8].tolist())
print("SLICE_OK")
