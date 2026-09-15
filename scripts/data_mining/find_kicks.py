"""Find likely kick moments: frames where a tracked ankle rises sharply
(foot leaving the ground fast), which is when the sack is most likely
airborne and visually separable from background clutter. Much stronger
prior than raw color matching, which failed twice on this footage."""
import sys
import json

sys.path.insert(0, "/Users/nmoff/claudeprojects/hackysack-cv/src")
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO = "/Users/nmoff/Desktop/IMG_7251.mov"
pose_model = YOLO("/Users/nmoff/claudeprojects/hackysack-cv/yolov8n-pose.pt")

ANKLE_IDX = [15, 16]

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

STRIDE = 2
# Track each detected person's lowest ankle y over time (smaller y = higher/more airborne).
# We don't have persistent IDs here (no tracker), so just record the minimum ankle-y
# across ALL people per sampled frame -- a simple, cheap proxy: "someone's foot is
# unusually high this frame" is a fine signal for "something interesting is happening."
history = []  # (frame_idx, min_ankle_y_normalized_by_bbox_height)
out_f = open("scripts/data_mining/work/kick_history.jsonl", "w")

frame_idx = 0
processed = 0
import time
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
    results = pose_model.predict(frame, conf=0.4, verbose=False)
    processed += 1

    best_ratio = None
    if results and results[0].keypoints is not None and results[0].boxes is not None:
        kps = results[0].keypoints.xy.cpu().numpy()
        confs = results[0].keypoints.conf
        confs = confs.cpu().numpy() if confs is not None else None
        boxes = results[0].boxes.xyxy.cpu().numpy()
        for person_idx, person_kp in enumerate(kps):
            x1, y1, x2, y2 = boxes[person_idx]
            bbox_h = y2 - y1
            if bbox_h <= 0:
                continue
            for kidx in ANKLE_IDX:
                if confs is not None and confs[person_idx, kidx] < 0.4:
                    continue
                ay = person_kp[kidx][1]
                if ay <= 0:
                    continue
                # how far this ankle is above the player's own bbox bottom,
                # normalized by their bbox height (person-size invariant)
                lift_ratio = (y2 - ay) / bbox_h
                if best_ratio is None or lift_ratio > best_ratio:
                    best_ratio = lift_ratio

    record = {
        "frame_idx": frame_idx,
        "time_s": round(frame_idx / fps, 2),
        "lift_ratio": float(best_ratio) if best_ratio is not None else None,
    }
    history.append(record)
    out_f.write(json.dumps(record) + "\n")
    frame_idx += 1
    if processed % 500 == 0:
        out_f.flush()
        print(f"processed {processed} frames (t={frame_idx/fps:.0f}s), {time.time()-t0:.0f}s elapsed")

cap.release()
out_f.close()
print(f"done, {processed} frames, {time.time()-t0:.0f}s")
