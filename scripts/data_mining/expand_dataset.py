"""Round 2: add the 172 bootstrap-mined sightings (all manually verified by
eye against their crops, not trusted blind) as new positive training
examples, using the model's own predicted bboxes as labels -- a standard
pseudo-labeling bootstrap, valid specifically because every one was checked,
not sampled."""
import json
import os
import random

import cv2

VIDEO = "/Users/nmoff/Desktop/IMG_7251.mov"
DATASET_DIR = "/Users/nmoff/claudeprojects/hackysack-cv/dataset"
IMG_W, IMG_H = 1920, 1440

reps = json.load(open("scripts/data_mining/work/bootstrap_review/reps.json"))
print(f"{len(reps)} verified new positives to add")

random.seed(7)
random.shuffle(reps)
n_val = max(1, len(reps) // 6)
val_set = reps[:n_val]
train_set = reps[n_val:]
print(f"{len(train_set)} train, {len(val_set)} val")

cap = cv2.VideoCapture(VIDEO)


def save(rec, split):
    frame_idx = rec["frame_idx"]
    x0, y0, x1, y1 = rec["bbox"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print(f"WARN: could not read frame {frame_idx}")
        return
    name = f"boot2_{frame_idx}"
    cv2.imwrite(f"{DATASET_DIR}/images/{split}/{name}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cx = (x0 + x1) / 2 / IMG_W
    cy = (y0 + y1) / 2 / IMG_H
    w = (x1 - x0) / IMG_W
    h = (y1 - y0) / IMG_H
    with open(f"{DATASET_DIR}/labels/{split}/{name}.txt", "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


for r in train_set:
    save(r, "train")
for r in val_set:
    save(r, "val")

cap.release()
print("done")
