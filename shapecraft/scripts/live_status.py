"""Собирает LIVE.md — человекочитаемый статус продолжающегося прогона.

Читает три источника: хвост лога обучения (текущий шаг), события TensorBoard
(метрики валидации) и sr_curve.jsonl (доля побед по чекпоинтам). Никогда не падает:
любая нехватка данных превращается в прочерк, потому что этот файл переписывается
раз в минуту из сторожевого цикла и уронить цикл не должен.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time

LIVE = "/home/user8_2/AIRI_WAM/live"
STEP_RE = re.compile(r"step=(\d+)/epoch=(\d+)")


def read_meta():
    try:
        with open(os.path.join(LIVE, "meta.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def current_step(log_path):
    """Последний шаг из лога. Лог большой, читаем только хвост."""
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            f.seek(max(0, size - 200000))
            tail = f.read().decode("utf-8", "replace")
        hits = STEP_RE.findall(tail)
        if hits:
            return int(hits[-1][0]), int(hits[-1][1])
    except Exception:
        pass
    return None, None


def alive(pattern):
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        return out.returncode == 0
    except Exception:
        return False


def tb_val(run_dirs):
    """Валидационные ряды из TensorBoard, склеенные по всем каталогам прогона."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}
    tags = [
        "val_loss",
        "val/action_loss",
        "val/video_loss",
        "val/action_accuracy",
        "val_split/val_id/action_accuracy_t0",
        "val_split/val_ood/action_accuracy_t0",
        "val_split/ood_gap_action_accuracy",
    ]
    merged = {}
    for rd in run_dirs:
        paths = sorted(glob.glob(f"{rd}/tb/**/events.out.tfevents.*", recursive=True))
        for p in paths:
            try:
                ea = EventAccumulator(p, size_guidance={"scalars": 0})
                ea.Reload()
            except Exception:
                continue
            have = ea.Tags()["scalars"]
            for t in tags:
                if t in have:
                    for e in ea.Scalars(t):
                        merged.setdefault(e.step, {})[t] = e.value
    return merged


def hms(sec):
    sec = int(max(sec, 0))
    h, m = sec // 3600, (sec % 3600) // 60
    return (f"{h} ч {m} мин" if h else f"{m} мин")


def main():
    meta = read_meta()
    run1 = meta.get("run1", "")
    run2 = meta.get("run2", "")
    log = meta.get("train_log", os.path.join(LIVE, "train.log"))
    max_steps = int(meta.get("max_steps", 0) or 0)
    t0 = float(meta.get("t0", 0) or 0)
    step0 = int(meta.get("step0", 0) or 0)

    step, epoch = current_step(log)
    now = time.time()
    train_alive = alive("hydra.run.dir=" + run2) if run2 else False
    watch_alive = alive("sr_watcher.sh")

    L = []
    L.append("# Живой статус — продолжение прогона WAM")
    L.append("")
    L.append(f"Обновлено {time.strftime('%Y-%m-%d %H:%M:%S')} · файл переписывается раз в минуту.")
    L.append("")

    # ---- обучение
    if step is None:
        L.append("**Обучение:** ещё разогревается, первых шагов в логе нет.")
    else:
        done = step - step0
        el = now - t0 if t0 else 0
        rate = done / el if el > 60 and done > 0 else 0
        left = (max_steps - step) / rate if rate > 0 and max_steps else 0
        pct = 100.0 * done / max(max_steps - step0, 1)
        state = "ИДЁТ" if train_alive else "ОСТАНОВЛЕНО"
        L.append(f"**Обучение: {state}** — шаг {step:,} из {max_steps:,} (эпоха {epoch}).".replace(",", " "))
        L.append("")
        L.append(f"- пройдено с рестарта: {done:,} шагов за {hms(el)}".replace(",", " "))
        if rate:
            L.append(f"- скорость {rate:.2f} шага/с, до конца примерно {hms(left)} ({pct:.0f}% готово)")
    L.append("")

    ws = ""
    try:
        with open(os.path.join(LIVE, "watcher_state.txt")) as f:
            ws = f.read().strip()
    except Exception:
        pass
    L.append(f"**Замеры побед: {'ИДУТ' if watch_alive else 'ОСТАНОВЛЕНЫ'}** — {ws or 'нет состояния'}")
    L.append("")

    # ---- кривая побед
    rows = []
    try:
        with open(os.path.join(LIVE, "sr_curve.jsonl")) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:
        pass
    rows.sort(key=lambda r: r["step"])
    L.append("## Доля побед в замкнутом контуре, отложенные задачи")
    L.append("")
    if not rows:
        L.append("Пока ни одного замера.")
    else:
        L.append("| шаг | победы | ошибка | эпизодов | источник |")
        L.append("|---|---|---|---|---|")
        for r in rows:
            L.append(
                f"| {r['step']:,} | **{r['success_rate']:.3f}** | ±{r.get('stderr', 0):.3f} "
                f"| {r.get('episodes', '—')} | {r.get('origin', '')} |".replace(",", " ")
            )
        L.append("")
        L.append("Для сравнения: смешанный датасет со случайными метками 0.380, чисто экспертный 0.335, "
                 "привилегированный эксперт 0.845.")
    L.append("")

    # ---- метрики валидации
    L.append("## Метрики валидации, каждые 10 000 шагов")
    L.append("")
    merged = tb_val([d for d in (run1, run2) if d])
    if not merged:
        L.append("Событий TensorBoard пока нет.")
    else:
        L.append("| шаг | потери | действий | видео | точность | 1-е действие id | ood | разрыв |")
        L.append("|---|---|---|---|---|---|---|---|")
        for st in sorted(merged)[-14:]:
            m = merged[st]

            def g(k, d=4):
                v = m.get(k)
                return f"{v:.{d}f}" if v is not None else "—"

            L.append(
                f"| {st + 1:,} | {g('val_loss')} | {g('val/action_loss')} | {g('val/video_loss')} "
                f"| {g('val/action_accuracy')} | {g('val_split/val_id/action_accuracy_t0')} "
                f"| {g('val_split/val_ood/action_accuracy_t0')} "
                f"| {g('val_split/ood_gap_action_accuracy')} |".replace(",", " ")
            )
    L.append("")

    L.append("## Как остановить")
    L.append("")
    L.append("```")
    L.append(f"touch {LIVE}/STOP")
    L.append("```")
    L.append("")
    L.append("Обучение получит мягкий сигнал и сохранит чекпоинт, замеры остановятся после текущего. "
             "Чекпоинты и так пишутся каждые 2 000 шагов, так что потерять можно максимум их.")
    L.append("")
    L.append("Лог обучения: `" + log + "`  ")
    L.append("Лог замеров: `" + os.path.join(LIVE, "watcher.log") + "`")

    tmp = os.path.join(LIVE, "LIVE.md.tmp")
    with open(tmp, "w") as f:
        f.write("\n".join(L) + "\n")
    os.replace(tmp, os.path.join(LIVE, "LIVE.md"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # статус не должен ронять сторожевой цикл
        sys.stderr.write(f"live_status: {e}\n")
