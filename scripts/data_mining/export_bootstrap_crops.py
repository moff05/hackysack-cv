import json
import os

import cv2

VIDEO = "/Users/nmoff/Desktop/IMG_7251.mov"
CAND = "scripts/data_mining/work/bootstrap_candidates.jsonl"
OUT_DIR = "scripts/data_mining/work/bootstrap_review"
os.makedirs(OUT_DIR, exist_ok=True)

recs = [json.loads(l) for l in open(CAND)]
recs.sort(key=lambda r: r["time_s"])

# Same clustering approach as before: same physical sighting across a few
# consecutive sampled frames collapses to one representative.
clusters = []
for r in recs:
    x0, y0, x1, y1 = r["bbox"]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    placed = False
    for cl in clusters:
        last = cl[-1]
        lx0, ly0, lx1, ly1 = last["bbox"]
        lcx, lcy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
        if abs(r["time_s"] - last["time_s"]) <= 1.0 and abs(cx - lcx) < 200 and abs(cy - lcy) < 200:
            cl.append(r)
            placed = True
            break
    if not placed:
        clusters.append([r])

print(f"{len(recs)} raw candidates -> {len(clusters)} clusters")
reps = [max(cl, key=lambda r: r["conf"]) for cl in clusters]
reps.sort(key=lambda r: -r["conf"])  # highest confidence first -- most likely correct
print(f"{len(reps)} representatives")

with open(f"{OUT_DIR}/reps.json", "w") as f:
    json.dump(reps, f)

cap = cv2.VideoCapture(VIDEO)
PAD = 90
for i, r in enumerate(reps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame_idx"])
    ok, frame = cap.read()
    if not ok:
        continue
    x0, y0, x1, y1 = [int(v) for v in r["bbox"]]
    h, w = frame.shape[:2]
    cx0, cy0 = max(0, x0 - PAD), max(0, y0 - PAD)
    cx1, cy1 = min(w, x1 + PAD), min(h, y1 + PAD)
    crop = frame[cy0:cy1, cx0:cx1].copy()
    cv2.rectangle(crop, (x0 - cx0, y0 - cy0), (x1 - cx0, y1 - cy0), (0, 0, 255), 1)
    cv2.imwrite(f"{OUT_DIR}/rep_{i:04d}_t{r['time_s']:.1f}_c{r['conf']:.2f}.jpg", crop)

cap.release()
print("done exporting crops")
