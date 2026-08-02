"""Этап 2: энтропия головы действий WAM на набранных состояниях, разложенная по классам.

Читаем так:
  «правило видно» + «до крафта»   — истинная энтропия РОВНО НОЛЬ (следует из кода среды).
                                     Всё, что здесь показывает модель, — её собственное незнание.
  «три варианта» + «до крафта»     — истинная энтропия больше нуля: какая из трёх пар крафтит,
                                     из кадра не выводится.
  любой класс + «после крафта»     — истинная энтропия снова ноль: шестиугольник на поле,
                                     оставшийся объект и есть катализатор.

Энтропия эксперта в тех же ячейках — контроль. Она отражает ТАКТИЧЕСКУЮ неуверенность
(два равнохороших маршрута), а не незнание правила: эксперт правила видит.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from omegaconf import OmegaConf


def entropy(p, axis=-1):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=axis)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--states", required=True)
    p.add_argument("--run-dir", default="/home/user8_2/AIRI_WAM/runs/shapecraft_dagger_128")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--repo", default="/home/user8_2/AIRI_WAM/joint_wam/repos/nano-world-model")
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    ckpt = args.checkpoint or (
        args.run_dir + "/checkpoints/across_timesteps/xland-joint-epoch=31-step=120000.ckpt"
    )
    sys.path.insert(0, args.repo + "/src")
    from experiments.train_experiment import NanoWMTrainingModule

    z = np.load(args.states)
    rgb = z["rgb"]
    n = len(rgb)
    print(f"состояний {n}, кадр {rgb.shape[1]}x{rgb.shape[2]}")

    config = OmegaConf.load(args.run_dir + "/config.yaml")
    module = NanoWMTrainingModule(config)
    state = torch.load(ckpt, map_location="cpu")
    module.load_state_dict(state.get("state_dict", state), strict=False)
    device = torch.device("cuda")
    module.to(device).eval()

    model_logits = np.zeros((n, 6), dtype=np.float32)
    with torch.inference_mode():
        for i in range(0, n, args.batch):
            chunk = rgb[i : i + args.batch]
            v = torch.from_numpy(np.ascontiguousarray(chunk)).permute(0, 3, 1, 2)
            v = v.unsqueeze(1).to(device=device, dtype=torch.float32) / 127.5 - 1.0
            out = module.predict_action_logits(v)
            model_logits[i : i + args.batch] = out[:, 0].float().cpu().numpy()
            if (i // args.batch) % 20 == 0:
                print(f"  {i + len(chunk)}/{n}", flush=True)

    pm = softmax(model_logits)
    pe = softmax(z["expert_logits"])
    hm, he = entropy(pm), entropy(pe)
    agree = pm.argmax(1) == z["expert_action"]

    det, s1 = z["determined"].astype(bool), z["stage1_done"].astype(bool)
    cells = [
        ("правило видно, до крафта", det & ~s1, "0 (ровно)"),
        ("правило видно, после",     det & s1,  "0 (ровно)"),
        ("три варианта, до крафта",  ~det & ~s1, "> 0"),
        ("три варианта, после",      ~det & s1, "0 (ровно)"),
    ]
    print(f"\n{'ячейка':<26}{'кадров':>8}{'H модели':>11}{'H эксперта':>13}"
          f"{'совпало с экспертом':>22}{'истинная H':>13}")
    for name, m, truth in cells:
        if m.sum() == 0:
            continue
        print(f"{name:<26}{int(m.sum()):>8}{hm[m].mean():>11.3f}{he[m].mean():>13.3f}"
              f"{agree[m].mean():>22.3f}{truth:>13}")

    print(f"\nВся выборка: H модели {hm.mean():.3f}, H эксперта {he.mean():.3f}, "
          f"совпадение {agree.mean():.3f}")
    print(f"Максимум возможной энтропии при 6 действиях: {np.log(6):.3f}")
    print("\nДоля кадров, где модель практически уверена (H < 0.35):")
    for name, m, truth in cells:
        if m.sum() == 0:
            continue
        print(f"  {name:<26}{(hm[m] < 0.35).mean():>7.3f}   (истинная H {truth})")
    print("ENTROPY_MEASURE_STATUS=OK")


if __name__ == "__main__":
    main()
