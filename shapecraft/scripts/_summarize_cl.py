import glob
import json

rows = []
for f in sorted(glob.glob("/home/user8_2/AIRI_WAM/reports/xland/closed_loop*/**/closed_loop_*.json", recursive=True)):
    d = json.load(open(f))
    part = f.split("/reports/xland/")[1].replace("/closed_loop_", " | ").replace(".json", "")
    rows.append((part, d))
for name, d in rows:
    keys = [k for k in ("success_rate", "success_rate_stderr", "mean_length", "mean_length_on_success", "episodes") if k in d]
    print(name.ljust(42), " ".join(f"{k}={d[k]:.4f}" if isinstance(d[k], float) else f"{k}={d[k]}" for k in keys))
