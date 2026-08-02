import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
paths = glob.glob(f"{run}/tb/**/events.out.tfevents.*", recursive=True)
ea = EventAccumulator(sorted(paths)[-1], size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]
print("теги:", ", ".join(sorted(tags)))
for t in sorted(tags):
    if any(k in t for k in ("val", "action_accuracy", "loss")):
        ev = ea.Scalars(t)
        pts = [f"{e.step}:{e.value:.4f}" for e in ev[-14:]]
        print(f"\n{t}  ({len(ev)} точек)")
        print("  " + "  ".join(pts))
