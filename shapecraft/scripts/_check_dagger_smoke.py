import glob
import json
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "/home/user8_2/AIRI_WAM/data/xland-dagger-smoke/val_ood"
f = sorted(glob.glob(f"{root}/episode_train*.npz"))
print("файлов:", len(f), "| пример:", f[0].split("/")[-1])
z = np.load(f[0])
print({k: z[k].shape for k in z.files if k != "metadata"})
print(json.loads(str(z["metadata"])))
print("метка (эксперт):", z["actions"][:14].tolist())
print("выполнено      :", z["executed"][:14].tolist())
print("rgb:", z["rgb"].dtype, z["rgb"].min(), z["rgb"].max())
assert z["rgb"].shape[0] == z["actions"].shape[0] + 1, "рассинхрон кадров и действий"

sys.path.insert(0, "/home/user8_2/AIRI_WAM/joint_wam/repos/nano-world-model")
from src.wm_datasets.data_source.xland import XLandDataSource  # noqa: E402

ds = XLandDataSource(root)
print("источник данных ментора видит эпизодов:", ds.get_num_trajectories())
tr = ds.load_trajectory(0)
print("actions one-hot:", tuple(tr.actions.shape), "| seq_length:", tr.seq_length)
print("кадры окна:", tuple(ds.load_visual_frames(0, 0, 4).shape))
print("SMOKE_OK")
