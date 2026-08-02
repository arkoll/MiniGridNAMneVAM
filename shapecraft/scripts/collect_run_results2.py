"""Сводка по прогону: скаляры TensorBoard, базовые линии, состав окон, графики.

Версия с аргументами: годится и для экспертного прогона, и для прошлого.
Базовые линии считаются на ТЕХ ЖЕ окнах, на которых меряется модель.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACTION_NAMES = ("forward", "turn_cw", "turn_ccw", "pickup", "put_down", "toggle")


def tb_scalars(run):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    path = None
    for root, _, files in os.walk(os.path.join(run, "tb")):
        for f in files:
            if f.startswith("events.out.tfevents"):
                path = os.path.join(root, f)
    acc = EventAccumulator(path, size_guidance={"scalars": 0})
    acc.Reload()
    return {
        tag: {"step": [e.step for e in acc.Scalars(tag)], "value": [float(e.value) for e in acc.Scalars(tag)]}
        for tag in acc.Tags()["scalars"]
    }


def window_stats(run, val_dir):
    """Базовые линии и состав окон валидации."""
    files = sorted(os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.startswith("episode_train"))
    with open(os.path.join(run, "validation_subset.json")) as f:
        slices = json.load(f)["slices"]

    cache = {}
    per_set = {"val_id": {"actions": [], "meta": Counter()}, "val_ood": {"actions": [], "meta": Counter()}}
    for s in slices:
        idx, start = int(s["traj_idx"]), int(s["start_frame"])
        if idx >= len(files):
            continue
        path = files[idx]
        if idx not in cache:
            with np.load(path, allow_pickle=False) as ep:
                meta = json.loads(str(ep["metadata"])) if "metadata" in ep else {}
                cache[idx] = (np.asarray(ep["actions"], dtype=np.int64), meta)
        actions, meta = cache[idx]
        window = actions[start : start + 3]
        if window.shape[0] < 3:
            continue
        which = "val_ood" if "_ood_" in os.path.basename(path) else "val_id"
        per_set[which]["actions"].append(window)
        per_set[which]["meta"][meta.get("policy", "?")] += 1
        per_set[which]["meta"][meta.get("outcome", "?")] += 1

    out = {}
    for name, d in per_set.items():
        if not d["actions"]:
            continue
        arr = np.stack(d["actions"])
        counts = Counter(arr.reshape(-1).tolist())
        major_action, _ = counts.most_common(1)[0]
        per_pos = [Counter(arr[:, p].tolist()).most_common(1)[0][1] / arr.shape[0] for p in range(3)]
        n = arr.shape[0]
        out[name] = {
            "windows": int(n),
            "random_guess": 1 / 6,
            "majority_action": ACTION_NAMES[major_action],
            "majority_per_position_mean": float(np.mean(per_pos)),
            "action_distribution": {ACTION_NAMES[a]: counts.get(a, 0) / arr.size for a in range(6)},
            "composition": {k: v / n for k, v in d["meta"].items()},
        }
    return out


def running_mean(values, window):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return v
    window = max(1, min(window, len(v)))
    cum = np.cumsum(np.insert(v, 0, 0.0))
    return np.array([(cum[i + 1] - cum[max(0, i - window + 1)]) / (i + 1 - max(0, i - window + 1)) for i in range(len(v))])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    scalars = tb_scalars(args.run)
    stats = window_stats(args.run, args.val_dir)

    rows = []
    tags = {
        "id_acc": "val_split/val_id/action_accuracy",
        "ood_acc": "val_split/val_ood/action_accuracy",
        "id_seq": "val_split/val_id/action_sequence_accuracy",
        "ood_seq": "val_split/val_ood/action_sequence_accuracy",
    }
    n_id = stats.get("val_id", {}).get("windows", 500)
    n_ood = stats.get("val_ood", {}).get("windows", 500)
    for i, step in enumerate(scalars[tags["id_acc"]]["step"]):
        a, b = scalars[tags["id_acc"]]["value"][i], scalars[tags["ood_acc"]]["value"][i]
        sa, sb = scalars[tags["id_seq"]]["value"][i], scalars[tags["ood_seq"]]["value"][i]
        rows.append(
            {
                "step": int(step),
                "id_acc": a,
                "ood_acc": b,
                "gap_acc": a - b,
                "se_acc": float(np.sqrt(a * (1 - a) / n_id + b * (1 - b) / n_ood)),
                "id_seq": sa,
                "ood_seq": sb,
                "gap_seq": sa - sb,
                "se_seq": float(np.sqrt(sa * (1 - sa) / n_id + sb * (1 - sb) / n_ood)),
            }
        )

    steps = [r["step"] for r in rows]
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
    ax.plot(steps, [r["id_acc"] for r in rows], "o-", label="val_id (знакомые привязки)")
    ax.plot(steps, [r["ood_acc"] for r in rows], "s-", label="val_ood (невиданные привязки)")
    major = np.mean([b["majority_per_position_mean"] for b in stats.values()]) if stats else None
    if major:
        ax.axhline(major, ls="--", color="darkgreen", label=f"всегда самое частое ({major:.3f})")
    ax.axhline(1 / 6, ls=":", color="black", label="случайно из шести (0.167)")
    ax.set_title("Точность предсказания действия")
    ax.set_xlabel("шаг")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.errorbar(steps, [r["gap_acc"] for r in rows], yerr=[r["se_acc"] for r in rows],
                fmt="o-", capsize=4, color="crimson", label="по отдельному действию")
    ax.errorbar(steps, [r["gap_seq"] for r in rows], yerr=[r["se_seq"] for r in rows],
                fmt="s--", capsize=4, color="darkorange", label="по всей тройке")
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Разрыв val_id минус val_ood")
    ax.set_xlabel("шаг")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(args.title or os.path.basename(args.run), fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(args.out, "curves.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)

    summary = {
        "run": args.run,
        "final": {t: scalars[t]["value"][-1] for t in scalars if t.startswith(("val", "train/action", "train/video"))},
        "split_by_step": rows,
        "window_stats": stats,
        "figure": fig_path,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== окна валидации ===")
    for name, b in stats.items():
        print(f"  {name}: окон {b['windows']}, самое частое «{b['majority_action']}» {b['majority_per_position_mean']:.3f}")
        print("    состав: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(b["composition"].items())))
        print("    действия: " + ", ".join(f"{k}={v:.3f}" for k, v in b["action_distribution"].items()))
    print("\n=== по шагам ===")
    for r in rows:
        print(
            f"  шаг {r['step']:>6}: действие {r['id_acc']:.4f} / {r['ood_acc']:.4f}, "
            f"разрыв {r['gap_acc']:+.4f} ± {r['se_acc']:.4f} | "
            f"тройка {r['id_seq']:.4f} / {r['ood_seq']:.4f}, разрыв {r['gap_seq']:+.4f} ± {r['se_seq']:.4f}"
        )
    print(f"\nграфик: {fig_path}")
    print("RESULTS2_STATUS=OK")


if __name__ == "__main__":
    main()
