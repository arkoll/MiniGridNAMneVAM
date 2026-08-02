"""Сводка по прогону shapecraft_joint_128: скаляры TensorBoard, базовые линии, графики.

Базовые линии обязательны: точность 0.56 сама по себе ничего не значит, пока не известно,
сколько даёт «всегда самое частое действие» и сколько даёт случайный выбор из шести.
Считаем их на ТЕХ ЖЕ окнах, на которых меряется модель, — по файлу зафиксированной выборки.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = "/home/user8_2/AIRI_WAM/runs/shapecraft_joint_128"
VAL_DIR = "/home/user8_2/AIRI_WAM/data/xland-shape-joint128/val_both"
OUT = "/home/user8_2/AIRI_WAM/reports/xland/joint_run"
os.makedirs(OUT, exist_ok=True)
ACTION_NAMES = ("forward", "turn_cw", "turn_ccw", "pickup", "put_down", "toggle")


def tb_scalars():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    path = None
    for root, _, files in os.walk(os.path.join(RUN, "tb")):
        for f in files:
            if f.startswith("events.out.tfevents"):
                path = os.path.join(root, f)
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags()["scalars"]:
        events = acc.Scalars(tag)
        out[tag] = {"step": [e.step for e in events], "value": [float(e.value) for e in events]}
    return out


def baselines():
    """Что дают тривиальные предсказатели на тех же окнах валидации."""
    files = sorted(
        os.path.join(VAL_DIR, f) for f in os.listdir(VAL_DIR) if f.startswith("episode_train")
    )
    with open(os.path.join(RUN, "validation_subset.json")) as f:
        slices = json.load(f)["slices"]

    cache = {}
    per_set = {"val_id": [], "val_ood": []}
    for s in slices:
        idx, start = int(s["traj_idx"]), int(s["start_frame"])
        if idx >= len(files):
            continue
        path = files[idx]
        if idx not in cache:
            with np.load(path, allow_pickle=False) as ep:
                cache[idx] = np.asarray(ep["actions"], dtype=np.int64)
        actions = cache[idx][start : start + 3]
        if actions.shape[0] < 3:
            continue
        which = "val_ood" if "_ood_" in os.path.basename(path) else "val_id"
        per_set[which].append(actions)

    result = {}
    for name, rows in per_set.items():
        if not rows:
            continue
        arr = np.stack(rows)  # [окон, 3]
        counts = Counter(arr.reshape(-1).tolist())
        majority_action, majority_count = counts.most_common(1)[0]
        # «всегда самое частое действие» — по каждой позиции своё самое частое
        per_position_major = []
        for p in range(3):
            c = Counter(arr[:, p].tolist())
            per_position_major.append(c.most_common(1)[0][1] / arr.shape[0])
        result[name] = {
            "windows": int(arr.shape[0]),
            "random_guess": 1 / 6,
            "majority_overall": majority_count / arr.size,
            "majority_action": ACTION_NAMES[majority_action],
            "majority_per_position": per_position_major,
            "majority_per_position_mean": float(np.mean(per_position_major)),
            "action_distribution": {
                ACTION_NAMES[a]: counts.get(a, 0) / arr.size for a in range(6)
            },
        }
    return result


def running_mean(values, window):
    """Скользящее среднее без завала на краях: np.convolve с mode='same' их портит."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return v
    window = max(1, min(window, len(v)))
    cumulative = np.cumsum(np.insert(v, 0, 0.0))
    out = np.empty_like(v)
    for i in range(len(v)):
        lo = max(0, i - window + 1)
        out[i] = (cumulative[i + 1] - cumulative[lo]) / (i + 1 - lo)
    return out


def plots(scalars, split_rows, base):
    steps = [r["step"] for r in split_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    for tag, label in (("train/video_loss", "видео"), ("train/action_loss", "действия")):
        if tag in scalars:
            s, v = scalars[tag]["step"], scalars[tag]["value"]
            k = max(1, len(v) // 300)
            ax.plot(s[::k], running_mean(v, 20)[::k], label=label)
    ax.set_title("Функции потерь на обучении")
    ax.set_xlabel("шаг")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    if "train/action_accuracy" in scalars:
        s, v = scalars["train/action_accuracy"]["step"], scalars["train/action_accuracy"]["value"]
        k = max(1, len(v) // 300)
        ax.plot(s[::k], running_mean(v, 50)[::k], color="grey", label="обучение", alpha=0.7)
    ax.plot(steps, [r["val_id"] for r in split_rows], "o-", label="val_id (знакомые привязки)")
    ax.plot(steps, [r["val_ood"] for r in split_rows], "s-", label="val_ood (невиданные привязки)")
    major = np.mean([b["majority_per_position_mean"] for b in base.values()]) if base else None
    if major:
        ax.axhline(major, ls="--", color="darkgreen",
                   label=f"всегда «вперёд» ({major:.3f})")
    ax.axhline(1 / 6, ls=":", color="black", label="случайный выбор из шести (0.167)")
    ax.set_title("Точность предсказания действия")
    ax.set_xlabel("шаг")
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    gaps = [r["val_id"] - r["val_ood"] for r in split_rows]
    errs = [r["se"] for r in split_rows]
    ax.errorbar(steps, gaps, yerr=errs, fmt="o-", capsize=4, color="crimson")
    ax.axhline(0, color="black", lw=1)
    ax.fill_between(steps, [-e for e in errs], errs, color="grey", alpha=0.2,
                    label="одна стандартная ошибка")
    ax.set_title("Разрыв val_id минус val_ood")
    ax.set_xlabel("шаг")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("ShapeCraft Easy: joint-модель, 30 000 шагов", fontsize=13)
    fig.tight_layout()
    path = os.path.join(OUT, "curves.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    scalars = tb_scalars()
    base = baselines()

    split_rows = []
    id_tag, ood_tag = "val_split/val_id/action_accuracy", "val_split/val_ood/action_accuracy"
    if id_tag in scalars:
        for i, step in enumerate(scalars[id_tag]["step"]):
            a = scalars[id_tag]["value"][i]
            b = scalars[ood_tag]["value"][i]
            n_id = base.get("val_id", {}).get("windows", 500)
            n_ood = base.get("val_ood", {}).get("windows", 500)
            se = float(np.sqrt(a * (1 - a) / n_id + b * (1 - b) / n_ood))
            split_rows.append({"step": int(step), "val_id": a, "val_ood": b, "gap": a - b, "se": se})

    fig_path = plots(scalars, split_rows, base)

    summary = {
        "run": RUN,
        "final": {
            tag: scalars[tag]["value"][-1]
            for tag in scalars
            if tag.startswith(("val", "train/action", "train/video"))
        },
        "split_by_step": split_rows,
        "baselines": base,
        "figure": fig_path,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== базовые линии на тех же окнах ===")
    for name, b in base.items():
        print(
            f"  {name}: окон {b['windows']}, случайно {b['random_guess']:.3f}, "
            f"самое частое действие «{b['majority_action']}» {b['majority_overall']:.3f}, "
            f"по позициям {b['majority_per_position_mean']:.3f}"
        )
        print("    распределение: " + ", ".join(f"{k}={v:.3f}" for k, v in b["action_distribution"].items()))
    print("\n=== раздельные метрики по шагам ===")
    for r in split_rows:
        print(
            f"  шаг {r['step']:>6}: val_id {r['val_id']:.4f}, val_ood {r['val_ood']:.4f}, "
            f"разрыв {r['gap']:+.4f} при стандартной ошибке {r['se']:.4f}"
        )
    print(f"\nграфик: {fig_path}")
    print("RESULTS_STATUS=OK")


if __name__ == "__main__":
    main()
