"""Сборка представлений датасета символьными ссылками — без дублирования кадров.

Кадры рендерятся один раз. Варианты сравнения — это разные ПОДМНОЖЕСТВА одних и тех же
файлов плюс, для варианта 3, другое поле действия внутри npz (переключается переменной
окружения XLAND_ACTION_FIELD, файлы не трогаются).

Подмножество экспертных эпизодов для смешанных вариантов берётся ПОРОВНУ ПО ЗАДАЧАМ, а не
префиксом по сиду: иначе состав задач разъехался бы между вариантом 1 и вариантами 2-4,
и разница в результате могла бы объясняться этим, а не составом данных.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict


def read_index(d):
    rows = []
    with open(os.path.join(d, "_index.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def link(files, src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(dst_dir):
        p = os.path.join(dst_dir, name)
        if os.path.islink(p):
            os.unlink(p)
    for name in files:
        os.symlink(os.path.join(src_dir, name), os.path.join(dst_dir, name))
    return len(files)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/user8_2/AIRI_WAM/data/final")
    p.add_argument("--expert-per-task", type=int, required=True,
                   help="сколько экспертных эпизодов на задачу берут смешанные варианты")
    args = p.parse_args()

    train = os.path.join(args.root, "train")
    rows = read_index(train)
    expert = [r for r in rows if r["policy"] == "expert_greedy"]
    random_ = [r for r in rows if r["policy"] == "random_masked"]
    print(f"в train: экспертных {len(expert)}, случайных {len(random_)}")

    def windows(rs):
        return sum(r["num_frames"] - 3 for r in rs)

    # экспертное подмножество — поровну по задачам, в порядке сида внутри задачи
    by_task = defaultdict(list)
    for r in sorted(expert, key=lambda r: r["seed"]):
        by_task[r["task_index"]].append(r)
    subset = []
    short = 0
    for t, rs in by_task.items():
        if len(rs) < args.expert_per_task:
            short += 1
        subset += rs[: args.expert_per_task]
    if short:
        print(f"ВНИМАНИЕ: {short} задач недобрали до {args.expert_per_task}")

    views = {
        "v1_expert": expert,          # вариант 1: все экспертные
        "v2_mixed": subset + random_,  # варианты 2 и 3: половина окон экспертных, половина случайных
        "v4_random": random_,          # вариант 4, фаза A
        "v4_expert": subset,           # вариант 4, фаза B
    }
    for name, rs in views.items():
        n = link([r["file"] for r in rs], train, os.path.join(args.root, name))
        print(f"{name:<12} эпизодов {n:>7}  окон {windows(rs):>9}  "
              f"задач {len(Counter(r['task_index'] for r in rs)):>4}  "
              f"мин/макс на задачу {min(Counter(r['task_index'] for r in rs).values())}/"
              f"{max(Counter(r['task_index'] for r in rs).values())}")

    # общая валидация: одна на все четыре прогона, иначе метрики несравнимы
    vb = os.path.join(args.root, "val_both")
    os.makedirs(vb, exist_ok=True)
    for name in os.listdir(vb):
        pth = os.path.join(vb, name)
        if os.path.islink(pth):
            os.unlink(pth)
    total = 0
    for s in ("val_id", "val_ood"):
        d = os.path.join(args.root, s)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.startswith("episode_train"):
                os.symlink(os.path.join(d, name), os.path.join(vb, name))
                total += 1
    print(f"val_both     ссылок {total}")
    print("BUILD_VIEWS_STATUS=OK")


if __name__ == "__main__":
    main()
