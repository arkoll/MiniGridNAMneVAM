"""Выгрузка рядов TensorBoard в json: полные валидационные ряды + прореженный трейн."""

import glob
import json
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
out = sys.argv[2]
paths = sorted(glob.glob(f"{run}/tb/**/events.out.tfevents.*", recursive=True))
ea = EventAccumulator(paths[-1], size_guidance={"scalars": 0})
ea.Reload()

VAL = [
    "val/action_accuracy",
    "val/action_loss",
    "val/action_sequence_accuracy",
    "val/video_loss",
    "val_loss",
    "val_split/ood_gap_action_accuracy",
    "val_split/val_id/action_accuracy",
    "val_split/val_id/action_accuracy_t0",
    "val_split/val_id/action_accuracy_t1",
    "val_split/val_id/action_accuracy_t2",
    "val_split/val_ood/action_accuracy",
    "val_split/val_ood/action_accuracy_t0",
    "val_split/val_ood/action_accuracy_t1",
    "val_split/val_ood/action_accuracy_t2",
]
TRAIN = ["train_loss", "train/action_loss", "train/video_loss", "train/action_accuracy"]

data = {}
for t in VAL:
    if t in ea.Tags()["scalars"]:
        ev = ea.Scalars(t)
        data[t] = {"step": [e.step for e in ev], "value": [round(e.value, 5) for e in ev]}

# трейн прореживаем скользящим средним по 50 точкам: 6119 сырых точек не нужны
for t in TRAIN:
    if t not in ea.Tags()["scalars"]:
        continue
    ev = ea.Scalars(t)
    s = np.array([e.step for e in ev])
    v = np.array([e.value for e in ev])
    k = 50
    n = len(v) // k * k
    data[t] = {
        "step": s[:n].reshape(-1, k).mean(1).round(0).astype(int).tolist(),
        "value": v[:n].reshape(-1, k).mean(1).round(5).tolist(),
    }

with open(out, "w") as f:
    json.dump(data, f)
print("рядов:", len(data), "-> ", out)
for t in VAL[:6]:
    if t in data:
        print(f"  {t}: {data[t]['value'][0]:.4f} -> {data[t]['value'][-1]:.4f}")
