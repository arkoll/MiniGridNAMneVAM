import glob
import json
import os


BASE = "/home/user8_2/AIRI_WAM/reports/xland/final"
NAMES = {
    "final_v1": "1. только экспертные состояния",
    "final_v2": "2. 50/50, метка экспертная",
    "final_v3": "3. 50/50, метка фактическая",
    "final_v4b": "4. двухфазно (мир -> действия)",
}
BANKS = [("train", "знакомые"), ("val", "отложенные")]

print(f"{'вариант':<32}{'T':>5}{'знакомые':>12}{'отложенные':>13}{'шагов/победа':>15}")
for m, label in NAMES.items():
    for t in ("0", "1.0", "2.0"):
        row, extra = {}, ""
        for bank, _ in BANKS:
            f = f"{BASE}/{m}_T{t}/closed_loop_{bank}.json"
            if os.path.exists(f):
                d = json.load(open(f))
                row[bank] = d["success_rate"]
                if bank == "val":
                    extra = f"{d.get('mean_length_on_success') or float('nan'):.1f}"
        if not row:
            continue
        f1 = f"{row['train']:.3f}" if "train" in row else "—"
        f2 = f"{row['val']:.3f}" if "val" in row else "—"
        print(f"{label:<32}{t:>5}{f1:>12}{f2:>13}{extra:>15}")
print("\nэпизодов на ячейку: 150, сиды 21/22 млн, одинаковые задачи во всех ячейках")
