"""Гифки в три панели: истина, раскатка WAM, раскатка с подгонкой под настоящие плитки.

Третья панель — символьное чтение кадра, показанное глазами: каждая клетка 16x16
предсказанного кадра заменяется БЛИЖАЙШЕЙ настоящей плиткой из атласа, собранного по
реальным кадрам среды. Диффузионное размытие уходит, и остаётся ровно один вопрос:
угадала модель содержимое клетки или нет. Несовпавшие клетки обводятся красным.

Про метрику. Доля совпавших клеток по ВСЕМ 64 вводит в заблуждение: 55-60 из них — пустой
пол и стены, они совпадают всегда. Поэтому считаем отдельно ЗНАЧИМЫЕ клетки — те, где в
истине стоит не пол и не стена: объекты, агент, дверь, ключ, звезда. Плюс отдельно
считаем, сколько объектов модель придумала на пустом месте.

Окна выбираются там, где эксперт кладёт объект (`put_down`): на этом действии срабатывает
правило крафта, ради которого весь эксперимент.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

REPO = "/home/user8_2/AIRI_WAM/joint_wam/repos/nano-world-model"
sys.path.insert(0, os.path.join(REPO, "src"))

import imageio.v3 as iio  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

RUN = "/home/user8_2/AIRI_WAM/runs/shapecraft_expert_128"
TILE = 16
SIDE = 8
SCALE = 2
PUT_DOWN = 4


def load_module(device):
    from experiments.train_experiment import NanoWMTrainingModule

    cfg = OmegaConf.load(os.path.join(RUN, "config.yaml"))
    module = NanoWMTrainingModule(cfg)
    ckpt_dir = os.path.join(RUN, "checkpoints", "latest")
    ckpt = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt"))[-1]
    state = torch.load(os.path.join(ckpt_dir, ckpt), map_location="cpu")
    state = state.get("state_dict", state)
    missing, unexpected = module.load_state_dict(state, strict=False)
    critical = [k for k in list(missing) + list(unexpected) if k.startswith(("model.", "action_head."))]
    if critical:
        raise RuntimeError(f"чекпоинт не подошёл: {critical[:5]}")
    module.to(device).eval()
    print(f"чекпоинт: {ckpt}")
    return module, cfg


def pick_windows(val_dir, num_frames, count, seed=0):
    files = sorted(f for f in os.listdir(val_dir) if f.startswith("episode_train"))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    chosen, seen = [], set()
    for name in files:
        path = os.path.join(val_dir, name)
        with np.load(path, allow_pickle=False) as ep:
            actions = np.asarray(ep["actions"])
            meta = json.loads(str(ep["metadata"])) if "metadata" in ep else {}
            n_frames = int(ep["rgb"].shape[0])
        task = meta.get("task_index")
        if task in seen:
            continue
        good = [s for s in range(0, n_frames - num_frames + 1) if PUT_DOWN in actions[s : s + num_frames - 1]]
        if not good:
            continue
        chosen.append({"path": path, "start": int(good[len(good) // 2]), "meta": meta})
        seen.add(task)
        if len(chosen) >= count:
            break
    return chosen


def build_atlas(val_dir, sample_episodes=120):
    """Настоящие плитки среды плюс метка «фон»: пол и стена — две самые частые."""
    files = sorted(f for f in os.listdir(val_dir) if f.startswith("episode_train"))[:sample_episodes]
    seen, counts = {}, Counter()
    for name in files:
        with np.load(os.path.join(val_dir, name), allow_pickle=False) as ep:
            rgb = np.asarray(ep["rgb"])
        for k in range(0, rgb.shape[0], 3):
            f = rgb[k]
            for y in range(SIDE):
                for x in range(SIDE):
                    t = f[y * TILE : (y + 1) * TILE, x * TILE : (x + 1) * TILE]
                    key = t.tobytes()
                    seen.setdefault(key, t)
                    counts[key] += 1
    keys = list(seen)
    atlas = np.stack([seen[k] for k in keys]).astype(np.float32)
    order = [k for k, _ in counts.most_common()]
    background = {keys.index(order[0]), keys.index(order[1])}  # пол и стена
    print(f"атлас настоящих плиток: {atlas.shape[0]} штук, фоновых (пол и стена): {len(background)}")
    return atlas, background


def tile_ids(frame_u8, atlas):
    ids = np.empty((SIDE, SIDE), dtype=np.int32)
    for y in range(SIDE):
        for x in range(SIDE):
            t = frame_u8[y * TILE : (y + 1) * TILE, x * TILE : (x + 1) * TILE].astype(np.float32)
            ids[y, x] = int(np.abs(atlas - t).mean(axis=(1, 2, 3)).argmin())
    return ids


def draw_from_ids(ids, atlas, mismatch=None):
    """Рисуем кадр из плиток; несовпавшие клетки обводим красным."""
    out = np.zeros((SIDE * TILE, SIDE * TILE, 3), dtype=np.uint8)
    for y in range(SIDE):
        for x in range(SIDE):
            tile = atlas[ids[y, x]].astype(np.uint8).copy()
            if mismatch is not None and mismatch[y, x]:
                tile[0, :] = tile[-1, :] = tile[:, 0] = tile[:, -1] = (255, 0, 0)
            out[y * TILE : (y + 1) * TILE, x * TILE : (x + 1) * TILE] = tile
    return out


def upscale(img, k=SCALE):
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def caption(lines, width, height=76):
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    try:
        import matplotlib
        from PIL import Image, ImageDraw, ImageFont

        font_path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
        font = ImageFont.truetype(font_path, 13)
        img = Image.fromarray(strip)
        draw = ImageDraw.Draw(img)
        y = 5
        for line in lines:
            draw.text((10, y), line, fill=(235, 235, 235), font=font)
            y += 20
        return np.asarray(img)
    except Exception:
        return strip


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-dir", default="/home/user8_2/AIRI_WAM/data/xland-expert/val_ood")
    p.add_argument("--out", default="/home/user8_2/AIRI_WAM/reports/xland/triptych")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--ddim-steps", type=int, default=25)
    p.add_argument("--fps", type=float, default=1.2)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module, cfg = load_module(device)
    num_frames = int(cfg.model.num_frames)
    n_context = int(cfg.model.n_context_frames)

    windows = pick_windows(args.val_dir, num_frames, args.count)
    print(f"выбрано окон: {len(windows)}")
    atlas, background = build_atlas(args.val_dir)

    videos, metas = [], []
    for w in windows:
        with np.load(w["path"], allow_pickle=False) as ep:
            rgb = np.asarray(ep["rgb"])[w["start"] : w["start"] + num_frames]
            actions = np.asarray(ep["actions"])[w["start"] : w["start"] + num_frames - 1]
        videos.append(rgb)
        metas.append({**w["meta"], "start": w["start"], "actions": actions.tolist()})

    stack = np.stack(videos)
    video = torch.from_numpy(stack).permute(0, 1, 4, 2, 3).float().to(device) / 127.5 - 1.0
    batch = {"video": video, "action": torch.zeros(len(videos), num_frames, 6, device=device)}

    with torch.no_grad():
        logs = module.log_images(batch, split="val", ddim_steps=args.ddim_steps)
        logits = module.predict_action_logits(video)
    predicted_actions = logits.argmax(dim=-1).cpu().numpy()

    def to_u8(t):
        return ((t.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 4, 1).cpu().numpy().astype(np.uint8)

    samples = to_u8(logs["samples"])
    truth = to_u8(logs["gt"])

    names = ("ИСТИНА", "РАСКАТКА WAM", "ПОДГОНКА ПОД ПЛИТКИ")
    action_names = ("forward", "turn_cw", "turn_ccw", "pickup", "put_down", "toggle")
    written = []
    for b in range(len(videos)):
        frames_out, stats = [], []
        for t in range(num_frames):
            gt = truth[b, t]
            ids_true = tile_ids(gt, atlas)
            meaningful = np.array([[int(i) not in background for i in row] for row in ids_true])
            if t < n_context:
                pred = gt.copy()
                ids_pred = ids_true.copy()
                tag = "контекст (дан модели)"
            else:
                pred = samples[b, t]
                ids_pred = tile_ids(pred, atlas)
                tag = f"предсказание, шаг +{t - n_context + 1}"
            mismatch = ids_pred != ids_true
            fitted = draw_from_ids(ids_pred, atlas, mismatch)

            n_meaning = int(meaningful.sum())
            hit_meaning = int((~mismatch & meaningful).sum())
            invented = int(
                sum(
                    1
                    for y in range(SIDE)
                    for x in range(SIDE)
                    if not meaningful[y, x] and int(ids_pred[y, x]) not in background
                )
            )
            stats.append(
                {
                    "step": t - n_context + 1 if t >= n_context else 0,
                    "all_cells": float((~mismatch).mean()),
                    "meaningful_cells": hit_meaning / max(n_meaning, 1),
                    "meaningful_total": n_meaning,
                    "invented_objects": invented,
                }
            )

            row = np.concatenate([upscale(gt), upscale(pred), upscale(fitted)], axis=1)
            head = caption(
                [
                    "  |  ".join(names) + "     (красная рамка — клетка не совпала)",
                    f"{tag}",
                    f"все клетки {stats[-1]['all_cells'] * 100:.0f}%   |   "
                    f"ЗНАЧИМЫЕ {hit_meaning}/{n_meaning}   |   придумано лишних объектов: {invented}",
                ],
                row.shape[1],
                height=76,
            )
            frames_out.append(np.concatenate([head, row], axis=0))
        frames_out += [frames_out[-1]] * 3

        m = metas[b]
        name = f"triptych_task{m.get('task_index', b):03d}_seed{m.get('seed', 0)}.gif"
        path = os.path.join(args.out, name)
        iio.imwrite(path, np.stack(frames_out), extension=".gif", duration=int(1000 / args.fps), loop=0)
        info = {
            "file": name,
            "dataset": os.path.basename(args.val_dir),
            "task_index": m.get("task_index"),
            "seed": m.get("seed"),
            "window_start": m["start"],
            "expert_actions": [action_names[a] for a in m["actions"]],
            "predicted_actions": [action_names[a] for a in predicted_actions[b]],
            "per_step": stats,
            "ddim_steps": args.ddim_steps,
        }
        with open(path.replace(".gif", ".json"), "w") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        written.append(info)
        pred_steps = [s for s in stats if s["step"] > 0]
        print(
            f"  {name}: все клетки "
            + ", ".join(f"{s['all_cells'] * 100:.0f}%" for s in pred_steps)
            + " | значимые "
            + ", ".join(f"{s['meaningful_cells'] * 100:.0f}% ({s['meaningful_total']})" for s in pred_steps)
            + " | лишних "
            + ", ".join(str(s["invented_objects"]) for s in pred_steps)
        )
        print(f"      эксперт {info['expert_actions']} | модель {info['predicted_actions']}")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(written, f, indent=2, ensure_ascii=False)
    print(f"\nготово: {len(written)} гифок в {args.out}")
    print("TRIPTYCH_STATUS=OK")


if __name__ == "__main__":
    main()
