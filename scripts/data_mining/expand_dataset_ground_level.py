"""Round 3: add hand-verified ground-level/non-sky sightings, closing the
gap where every prior example (rounds 1-2) was the sack airborne against
open sky. Found via mine_ground_level.py (pre-kick-peak frames, cropped
around whichever player's ankle is highest) -- yield was much lower than
the sky bootstrap (5 confirmed out of 103 reviewed), expected since this
uses a coarser heuristic (timing offset from a kick peak) rather than the
model's own predictions."""
import os
import random

import cv2

DATASET_DIR = "/Users/nmoff/claudeprojects/hackysack-cv/dataset"
VIDEO = "/Users/nmoff/Desktop/IMG_7251.mov"
IMG_W, IMG_H = 1920, 1440

# (frame_idx, (x0, y0, x1, y1)) -- all manually verified against zoomed
# crops of the exact frame, not sampled.
POSITIVES = [
    (6790, (1006, 1053, 1031, 1075)),   # on the ground, right by a shoe
    (10344, (1597, 942, 1618, 965)),    # against a palm tree trunk + fence
    (31478, (1079, 1117, 1108, 1150)),  # close-up, resting against a shoe
    (32872, (1402, 1069, 1420, 1086)),  # on the ground, player reaching for it
    (34558, (1771, 805, 1787, 819)),    # mid-air against fence/tree clutter
]

random.seed(11)
random.shuffle(POSITIVES)
n_val = 1
val_set = POSITIVES[:n_val]
train_set = POSITIVES[n_val:]
print(f"{len(train_set)} train, {len(val_set)} val")

cap = cv2.VideoCapture(VIDEO)


def save(frame_idx, bbox, split):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print(f"WARN: could not read frame {frame_idx}")
        return
    name = f"ground_{frame_idx}"
    cv2.imwrite(f"{DATASET_DIR}/images/{split}/{name}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2 / IMG_W
    cy = (y0 + y1) / 2 / IMG_H
    w = (x1 - x0) / IMG_W
    h = (y1 - y0) / IMG_H
    with open(f"{DATASET_DIR}/labels/{split}/{name}.txt", "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


for fi, bbox in train_set:
    save(fi, bbox, "train")
for fi, bbox in val_set:
    save(fi, bbox, "val")

cap.release()
print("done")
