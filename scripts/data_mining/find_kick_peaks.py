"""Derives kick_peaks.json from find_kicks.py's kick_history.jsonl output --
local maxima in ankle-lift-ratio, deduped so the same kick isn't counted
twice. Run find_kicks.py first."""
import json

import numpy as np

recs = [json.loads(l) for l in open("scripts/data_mining/work/kick_history.jsonl")]
recs = [r for r in recs if r["lift_ratio"] is not None]
recs.sort(key=lambda r: r["frame_idx"])

lifts = np.array([r["lift_ratio"] for r in recs])
frame_idxs = np.array([r["frame_idx"] for r in recs])

WINDOW = 15  # +/- ~1s at this script's frame stride
peaks = []
for i in range(WINDOW, len(lifts) - WINDOW):
    if lifts[i] < 0.15:
        continue
    seg = lifts[i - WINDOW : i + WINDOW + 1]
    if lifts[i] == seg.max():
        peaks.append(i)

kept = []
for i in peaks:
    if not kept or frame_idxs[i] - frame_idxs[kept[-1]] > 60:
        kept.append(i)

out = [{"frame_idx": int(frame_idxs[i]), "time_s": recs[i]["time_s"], "lift_ratio": float(lifts[i])} for i in kept]
with open("scripts/data_mining/work/kick_peaks.json", "w") as f:
    json.dump(out, f)
print(f"{len(peaks)} raw peaks -> {len(kept)} deduped kick events saved")
