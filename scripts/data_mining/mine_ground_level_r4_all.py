"""Round 3: hunt for ground-level sack sightings specifically, since every
positive example from rounds 1-2 is airborne against sky. Target: the
moment a foot is rising into a kick but hasn't launched the ball into open
sky yet -- pulled slightly BEFORE each detected kick-peak (see
kick_peaks.json), when the ball should still be near foot/ground height.
"""
import json
import sys

sys.path.insert(0, "/Users/nmoff/claudeprojects/hackysack-cv/src")
import cv2
from ultralytics import YOLO

VIDEO = "/Users/nmoff/Desktop/IMG_7251.mov"
OUT_DIR = "scripts/data_mining/work/ground_level_review_r4"
import os
os.makedirs(OUT_DIR, exist_ok=True)

peaks = json.load(open("scripts/data_mining/work/kick_peaks_remaining.json"))
print(f"{len(peaks)} kick events available")

# Sample a manageable, time-spread subset.
peaks.sort(key=lambda p: p["time_s"])
STEP = max(1, len(peaks) // 90)
sampled = peaks
print(f"sampling {len(sampled)} events")

pose_model = YOLO("/Users/nmoff/claudeprojects/hackysack-cv/yolov8n-pose.pt")
ANKLE_IDX = [15, 16]

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)

PRE_PEAK_OFFSET_FRAMES = 8  # ~0.27s before the peak, foot still rising
PAD = 140

manifest = []
for i, p in enumerate(sampled):
    frame_idx = max(0, p["frame_idx"] - PRE_PEAK_OFFSET_FRAMES)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        continue
    h, w = frame.shape[:2]

    results = pose_model.predict(frame, imgsz=1280, conf=0.4, iou=0.5, verbose=False)
    if not results or results[0].keypoints is None or results[0].boxes is None:
        continue
    kps = results[0].keypoints.xy.cpu().numpy()
    kconf = results[0].keypoints.conf
    kconf = kconf.cpu().numpy() if kconf is not None else None
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # Pick whichever tracked person has the highest ankle position (most
    # likely the one mid-kick at this exact moment) as the crop target.
    best = None  # (ankle_y, cx, cy)
    for person_idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[person_idx]
        bbox_h = y2 - y1
        if bbox_h <= 0:
            continue
        for kidx in ANKLE_IDX:
            if kconf is not None and kconf[person_idx, kidx] < 0.3:
                continue
            ax, ay = kps[person_idx, kidx]
            if ax <= 0 or ay <= 0:
                continue
            if best is None or ay < best[0]:
                best = (ay, ax, ay)
    if best is None:
        continue
    _, cx, cy = best
    x0, y0 = max(0, int(cx - PAD)), max(0, int(cy - PAD))
    x1, y1 = min(w, int(cx + PAD)), min(h, int(cy + PAD))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        continue
    name = f"gl_{i:04d}_t{p['time_s']:.1f}"
    cv2.imwrite(f"{OUT_DIR}/{name}.jpg", crop)
    manifest.append({"name": name, "frame_idx": frame_idx, "time_s": p["time_s"], "crop_origin": [x0, y0]})

    if (i + 1) % 20 == 0:
        print(f"processed {i+1}/{len(sampled)}")

cap.release()
with open(f"{OUT_DIR}/manifest.json", "w") as f:
    json.dump(manifest, f)
print(f"done, {len(manifest)} crops exported")
