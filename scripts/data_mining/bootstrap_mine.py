"""Bootstrap round 2: use the trained sack_detector.pt itself to find more
candidate sightings across the full video, instead of the old color/sky
heuristics. Should be far more precise now that the model has actually
learned the ball's shape/texture, not just color -- and critically, this
can find ground-level sightings the sky-only method structurally couldn't."""
import json
import time

import cv2
from ultralytics import YOLO

VIDEO = "/Users/nmoff/Desktop/IMG_7251.mov"
OUT = "scripts/data_mining/work/bootstrap_candidates.jsonl"

model = YOLO("/Users/nmoff/claudeprojects/hackysack-cv/models/sack_detector.pt")

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"fps={fps} total_frames={total_frames}")

STRIDE = 2
out_f = open(OUT, "w")
frame_idx = 0
processed = 0
found = 0
t0 = time.time()

while True:
    ok = cap.grab()
    if not ok:
        break
    if frame_idx % STRIDE != 0:
        frame_idx += 1
        continue
    ok, frame = cap.retrieve()
    if not ok:
        break
    processed += 1

    results = model.predict(frame, imgsz=1280, conf=0.03, verbose=False)
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            conf = float(b.conf[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            rec = {"frame_idx": frame_idx, "time_s": round(frame_idx / fps, 2), "bbox": [x1, y1, x2, y2], "conf": round(conf, 4)}
            out_f.write(json.dumps(rec) + "\n")
            found += 1

    frame_idx += 1
    if processed % 1000 == 0:
        out_f.flush()
        print(f"processed {processed} frames (t={frame_idx/fps:.0f}s), {found} candidates so far, {time.time()-t0:.0f}s elapsed")

cap.release()
out_f.close()
print(f"DONE. processed {processed} frames, {found} candidates, {time.time()-t0:.0f}s")
