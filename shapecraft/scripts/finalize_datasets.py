"""Этап 3: ворота по датасету, согласованные манифесты и отчёт по утечкам.

Ворота:
  1. индекс сходится с файлами на диске;
  2. длины согласованы: кадров = действий + 1 = состояний;
  3. кадры ПОБАЙТОВО совпадают с рендером сохранённого символьного состояния;
  4. каждая из 324 задач представлена ровно одинаковым числом эпизодов;
  5. доля случайной политики ровно 1/6 в каждом наборе;
  6. привязки цвета к форме не пересекаются между train и val_ood.

Манифесты: val_id и val_ood подрезаются так, чтобы по КАЖДОЙ задаче совпадало и число
успешных, и число провальных эпизодов. Тогда два валидационных набора отличаются ровно
привязкой цвета, а не составом.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys

import h5py
import numpy as np

import shape_common as sc

FAILS = []


def gate(name, ok, detail=""):
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def load_index(set_name):
    parts = sorted(glob.glob(os.path.join(sc.SHARD_ROOT, set_name, "index_part_*.jsonl")))
    rows = []
    for p in parts:
        with open(p) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    rows.sort(key=lambda r: r["id"])
    out = os.path.join(sc.DATA_ROOT, f"index_{set_name}.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def colors_of(set_name, rows, limit=400):
    """Множество пар (форма, цвет), реально встречающихся в стартовых кадрах."""
    pairs = set()
    for r in rows[:limit]:
        with h5py.File(os.path.join(sc.DATA_ROOT, r["path"]), "r") as h:
            g = h["grid"][0]
        for shape, color in g.reshape(-1, 2):
            if int(shape) in (3, 4, 5, 11):  # ball, square, pyramid, hex
                pairs.add((int(shape), int(color)))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=20)
    args = p.parse_args()

    env, base_params, *_ = sc.make_env_and_policy(load_weights=False)
    render = sc.make_render_fn(base_params)

    indices = {}
    for s in sc.SETS:
        rows = load_index(s)
        indices[s] = rows
        files = glob.glob(os.path.join(sc.DATA_ROOT, s, "ep_*.h5"))
        expected = sc.episodes_in(s)
        gate(
            f"{s}: индекс, файлы и план сходятся",
            len(rows) == len(files) == expected,
            f"индекс {len(rows)}, файлов {len(files)}, план {expected}",
        )

        # баланс по задачам и по политике
        per_task = collections.Counter(r["task_index"] for r in rows)
        gate(
            f"{s}: каждая задача представлена поровну",
            len(per_task) == sc.NUM_TASKS and len(set(per_task.values())) == 1,
            f"{len(per_task)} задач, по {set(per_task.values())} эпизодов",
        )
        n_rand = sum(1 for r in rows if r["policy"] == "random")
        gate(
            f"{s}: доля случайной политики ровно 1/{sc.RANDOM_EVERY}",
            abs(n_rand / len(rows) - 1 / sc.RANDOM_EVERY) < 1e-9,
            f"{n_rand}/{len(rows)}",
        )

        n_succ = sum(1 for r in rows if r["outcome"] == "success")
        fr_succ = sum(r["num_frames"] for r in rows if r["outcome"] == "success")
        fr_all = sum(r["num_frames"] for r in rows)
        print(
            f"       {s}: эпизодов {len(rows)}, успех {n_succ / len(rows):.3f}, "
            f"кадров {fr_all}, из них успешных {fr_succ / fr_all:.3f}"
        )

        # побайтовая сверка кадров с рендером состояния
        rng = np.random.default_rng(0)
        picks = rng.choice(len(rows), size=min(args.sample, len(rows)), replace=False)
        ok_bytes = ok_len = True
        checked = 0
        for i in picks:
            with h5py.File(os.path.join(sc.DATA_ROOT, rows[i]["path"]), "r") as h:
                frames, acts = h["frames"][:], h["actions"][:]
                g, pp, dd, pk = h["grid"][:], h["agent_pos"][:], h["agent_dir"][:], h["agent_pocket"][:]
            if not (frames.shape[0] == acts.shape[0] + 1 == g.shape[0]):
                ok_len = False
            ref = np.stack([render(g[k], pp[k], dd[k], pk[k]) for k in range(g.shape[0])])
            if not np.array_equal(ref, frames):
                ok_bytes = False
            checked += frames.shape[0]
        gate(f"{s}: длины согласованы (кадров = действий + 1)", ok_len)
        gate(f"{s}: кадры побайтово совпадают с рендером состояния", ok_bytes, f"проверено {checked} кадров")

    # непересечение привязок между обучающим и OOD-набором
    train_pairs = colors_of("train", indices["train"])
    ood_pairs = colors_of("val_ood", indices["val_ood"])
    gate(
        "привязки (форма, цвет) train и val_ood не пересекаются",
        not (train_pairs & ood_pairs),
        f"train {len(train_pairs)} пар, val_ood {len(ood_pairs)} пар, общих {len(train_pairs & ood_pairs)}",
    )
    id_pairs = colors_of("val_id", indices["val_id"])
    gate(
        "привязки val_id совпадают с train (это контроль, а не OOD)",
        id_pairs <= train_pairs,
        f"val_id {len(id_pairs)} пар",
    )

    # ---------------------------------------------------------- манифесты ---
    man_dir = os.path.join(sc.DATA_ROOT, "manifests")
    os.makedirs(man_dir, exist_ok=True)

    def by_task_outcome(rows):
        d = collections.defaultdict(lambda: {"success": [], "timeout": []})
        for r in rows:
            d[r["task_index"]][r["outcome"]].append(r)
        for v in d.values():
            for k in v:
                v[k].sort(key=lambda r: r["id"])
        return d

    a, b = by_task_outcome(indices["val_id"]), by_task_outcome(indices["val_ood"])
    kept = {"val_id": [], "val_ood": []}
    for task in range(sc.NUM_TASKS):
        for outcome in ("success", "timeout"):
            k = min(len(a[task][outcome]), len(b[task][outcome]))
            kept["val_id"] += a[task][outcome][:k]
            kept["val_ood"] += b[task][outcome][:k]

    summary = {}
    for s in sc.SETS:
        rows = kept[s] if s in kept else indices[s]
        rows = sorted(rows, key=lambda r: r["id"])
        with open(os.path.join(man_dir, f"{s}.txt"), "w") as f:
            for r in rows:
                f.write(r["path"] + "\n")
        n_succ = sum(1 for r in rows if r["outcome"] == "success")
        frames = sum(r["num_frames"] for r in rows)
        fr_succ = sum(r["num_frames"] for r in rows if r["outcome"] == "success")
        windows = {
            str(w): sum(max(0, r["num_frames"] - w + 1) for r in rows) for w in (3, 4, 8)
        }
        summary[s] = {
            "episodes": len(rows),
            "success": n_succ,
            "timeout": len(rows) - n_succ,
            "success_share": n_succ / len(rows),
            "frames": frames,
            "frames_from_success": fr_succ,
            "frame_balance": fr_succ / frames,
            "windows_by_length": windows,
        }
        print(
            f"манифест {s}: {len(rows)} эпизодов ({n_succ} успех / {len(rows) - n_succ} провал), "
            f"кадров {frames}, баланс по кадрам {fr_succ / frames:.3f}"
        )

    gate(
        "val_id и val_ood совпадают по числу успешных и провальных эпизодов",
        summary["val_id"]["success"] == summary["val_ood"]["success"]
        and summary["val_id"]["timeout"] == summary["val_ood"]["timeout"],
        f"{summary['val_id']['success']}/{summary['val_id']['timeout']} против "
        f"{summary['val_ood']['success']}/{summary['val_ood']['timeout']}",
    )
    rel = abs(summary["val_id"]["frames"] - summary["val_ood"]["frames"]) / summary["val_id"]["frames"]
    gate("val_id и val_ood совпадают по числу кадров с точностью до 2%", rel < 0.02, f"расхождение {rel:.4f}")

    # ------------------------------------------------------------ утечки ---
    def state_hashes(rows, limit=None):
        starts, states = [], set()
        for r in (rows if limit is None else rows[:limit]):
            with h5py.File(os.path.join(sc.DATA_ROOT, r["path"]), "r") as h:
                g, pp, dd, pk = h["grid"][:], h["agent_pos"][:], h["agent_dir"][:], h["agent_pocket"][:]
            def hsh(k):
                return hashlib.blake2b(
                    np.concatenate([g[k].reshape(-1), pp[k], [dd[k]], pk[k]]).tobytes(), digest_size=16
                ).hexdigest()
            starts.append(hsh(0))
            for k in range(g.shape[0]):
                states.add(hsh(k))
        return starts, states

    LIM = 1500
    tr_starts, tr_states = state_hashes(indices["train"], LIM)
    leak = {"sampled_episodes_per_set": LIM}
    dup = len(tr_starts) - len(set(tr_starts))
    leak["train_repeated_starts"] = dup
    print(f"\nутечки: повторов стартовой раскладки внутри train — {dup} из {len(tr_starts)}")
    for s in ("val_id", "val_ood"):
        st, ss = state_hashes(indices[s], LIM)
        ov_start = sum(1 for h in st if h in set(tr_starts))
        ov_state = len(ss & tr_states)
        leak[s] = {
            "start_overlap_with_train": ov_start,
            "start_overlap_share": ov_start / len(st),
            "state_overlap_with_train": ov_state,
            "state_overlap_share": ov_state / max(len(ss), 1),
        }
        print(
            f"утечки: {s} — стартов, встречающихся в train: {ov_start}/{len(st)} "
            f"({ov_start / len(st):.4f}); состояний: {ov_state}/{len(ss)} ({ov_state / max(len(ss), 1):.4f})"
        )

    with open(os.path.join(sc.DATA_ROOT, "datasets_summary.json"), "w") as f:
        json.dump({"manifests": summary, "leaks": leak, "checkpoint": sc.CKPT}, f, indent=2, ensure_ascii=False)

    if FAILS:
        print("\nFINALIZE_STATUS=FAIL: " + ", ".join(FAILS))
        sys.exit(1)
    print("\nFINALIZE_STATUS=OK")


if __name__ == "__main__":
    main()
